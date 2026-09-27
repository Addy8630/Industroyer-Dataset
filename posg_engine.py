"""
posg_engine_v2.py — POSG Engine v2
====================================
What changed from v1:
  • 5-state model (Normal added)
  • Stage-specific action enums (no string references)
  • Explicit payoff formula: U = w1*Prevention - w2*P_success*Damage
                                - w3*DefCost - w4*max(1,OperDisruption)[Execute]
  • Full predict→correct belief update with transition model
  • Epsilon floor + clip on LLM hybrid output
  • Transition rows asserted to sum to 1 at load time
  • EU maximization as primary decision rule (Nash deferred to v2)
  • Every state can return to Normal (de-escalation)
  • Belief entropy added to StepResult

Public API (identical to v1):
  engine = POSGEngine()
  result = engine.step(raw_obs, llm_belief=None)
  spe    = engine.run_backward_induction()
  engine.reset()
"""

import numpy as np
import logging
import random
from dataclasses import dataclass
from typing import Optional

from posg_types import (
    S, O, NUM_STATES, INITIAL_BELIEF,
    STATE_NAMES, OBS_NAMES, DEFENDER_ACTION_NAMES,
    NUM_DEF_ACTIONS, RawObs, GameState, BeliefState,
    AttackerAction, DefenderAction, Stage,
)
from posg_models import (
    OBSERVATION_MODEL, TRANSITION_MODEL, get_payoff, EPS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("posg_v2")


# ─────────────────────────────────────────────────────────────
# STEP RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Full output of one engine tick."""
    timestep:           int
    raw_obs:            RawObs
    observation:        str
    belief_prior:       list
    belief_predicted:   list   # after transition step
    belief_updated:     list   # after obs correction + optional LLM blend
    belief_bayes:       list
    most_likely_stage:  str
    stage_confidence:   float
    best_action_idx:    int
    best_action_name:   str
    expected_utility:   float
    all_utilities:      list
    onos_payload:       dict
    llm_used:           bool
    belief_entropy:     float  # H(b) — defender uncertainty


# ─────────────────────────────────────────────────────────────
# BELIEF OPERATIONS
# ─────────────────────────────────────────────────────────────

def _predict(b: np.ndarray, aD_idx: int, stage: S) -> np.ndarray:
    """
    Transition step.
    b_pred(s') = Σ_s  P(s'|s, aD) × b(s)

    Integrates over ALL states weighted by belief.
    aD_idx is clamped to the valid range of each state's action set.
    """
    b_pred = np.zeros(NUM_STATES, dtype=float)
    for s in S:
        keys   = list(TRANSITION_MODEL[s].keys())
        safe   = min(aD_idx, len(keys) - 1)
        T_row  = TRANSITION_MODEL[s][keys[safe]]
        b_pred += b[s] * T_row
    return b_pred


def _correct(b_pred: np.ndarray, obs: O) -> np.ndarray:
    """
    Observation correction.
    b_raw(s') = P(o|s') × b_pred(s')
    Apply EPS floor then normalize.
    """
    likelihood = OBSERVATION_MODEL[:, int(obs)]
    b_raw      = likelihood * b_pred
    b_raw      = np.maximum(b_raw, EPS)
    Z          = b_raw.sum()
    if Z < 1e-12:
        log.warning("Near-zero normalizer — keeping predicted belief.")
        return b_pred.copy()
    return b_raw / Z


def _hybrid(b_bayes: np.ndarray, b_llm: np.ndarray, alpha: float) -> np.ndarray:
    """
    Hybrid update: b_final = alpha*b_LLM + (1-alpha)*b_Bayes
    Clamp then renormalize (prevents invalid LLM output propagation).
    """
    b_llm = np.array(b_llm, dtype=float)
    if b_llm.shape != (NUM_STATES,):
        log.warning(f"LLM belief shape {b_llm.shape} invalid — using Bayes only.")
        return b_bayes.copy()
    s = b_llm.sum()
    if not np.isclose(s, 1.0, atol=0.05):
        log.warning(f"LLM belief sums to {s:.4f} — normalizing.")
    b_llm   = np.maximum(b_llm, EPS)
    b_llm  /= b_llm.sum()
    b_final = alpha * b_llm + (1.0 - alpha) * b_bayes
    b_final = np.clip(b_final, EPS, 1.0)
    return b_final / b_final.sum()


def _entropy(b: np.ndarray) -> float:
    """Shannon entropy H(b) — measures defender's uncertainty in bits."""
    b_safe = np.maximum(b, 1e-12)
    return float(-np.sum(b_safe * np.log2(b_safe)))


# ─────────────────────────────────────────────────────────────
# EXPECTED UTILITY  (EU maximization — Nash deferred to v2)
# ─────────────────────────────────────────────────────────────

def expected_utility_vector(b: np.ndarray, stage: S) -> np.ndarray:
    """
    EU_D(aD) = Σ_s  b(s) × U_D_avg(s, aD)

    where U_D_avg(s, aD) = mean over attacker actions (uniform prior).
    Nash attacker mixing π_A(aA|s) replaces uniform mean in v2.

    Returns np.ndarray shape [num_aD for this stage].
    """
    n_aD = NUM_DEF_ACTIONS[stage]
    eu   = np.zeros(n_aD, dtype=float)

    for s in S:
        U       = get_payoff(s)            # [n_aA_s × n_aD_s]
        U_avg   = U.mean(axis=0)           # [n_aD_s]  — average over attacker actions
        shared  = min(n_aD, len(U_avg))
        eu[:shared] += b[s] * U_avg[:shared]

    return eu


def best_action(b: np.ndarray, stage: S) -> tuple:
    """
    a_D* = argmax EU_D(aD)
    Returns (best_idx, best_eu, all_eu_list)
    """
    eu   = expected_utility_vector(b, stage)
    best = int(np.argmax(eu))
    return best, float(eu[best]), eu.tolist()


# ─────────────────────────────────────────────────────────────
# ONOS DISPATCH
# ─────────────────────────────────────────────────────────────

_ONOS_RULES = {
    S.NORMAL:     {0: ("ALLOW_ALL",            None),
                   1: ("LIGHT_SCAN_MODE",       None)},
    S.RECON:      {0: ("ALLOW_ALL",             None),
                   1: ("REDIRECT_HONEYPOT",     "10.0.1.99"),
                   2: ("RATE_LIMIT_H1",         "10.0.1.1"),
                   3: ("REDIRECT_HONEYPOT+IDS", "10.0.1.99")},
    S.INFECTION: {0: ("ALLOW_ALL",             None),
                   1: ("ENFORCE_MFA",           "10.0.2.5"),
                   2: ("WHITELIST_MODBUS",      "10.0.1.4"),
                   3: ("PATCH_AND_MONITOR",     "10.0.1.4")},
    S.INSTALLATION:  {0: ("ALLOW_ALL",             None),
                   1: ("PHYSICS_VALIDATION",    "10.0.1.2"),
                   2: ("ISOLATE_IBR",           "10.0.1.4"),
                   3: ("SANDBOX_MODBUS",        "10.0.1.4")},
    S.EXECUTE:    {0: ("ALLOW_ALL",             None),
                   1: ("EMERGENCY_BLOCK_H1",    "10.0.1.1"),
                   2: ("ROLLBACK_SCADA",        "10.0.1.2"),
                   3: ("BLOCK_AND_ROLLBACK",    "10.0.1.1")},
}


def dispatch_onos(stage: S, aD_idx: int, dry_run: bool = True) -> dict:
    rule, target = _ONOS_RULES[stage][aD_idx]
    aD_name      = list(DEFENDER_ACTION_NAMES[stage].values())[aD_idx]
    payload      = {"stage": STATE_NAMES[stage], "action": aD_name,
                    "rule": rule, "target": target}
    tag = "[DRY-RUN]" if dry_run else "[ONOS]"
    log.info(f"{tag} stage={STATE_NAMES[stage]} action={aD_name} "
             f"rule={rule} target={target}")
    if not dry_run:
        pass  # import requests; requests.post("http://localhost:8181/...", json=payload)
    return payload


# ─────────────────────────────────────────────────────────────
# VALUE OF EARLY DETECTION
# ─────────────────────────────────────────────────────────────

_VED = {S.RECON:      (-2,  12),
        S.INFECTION: (-3,   8),
        S.INSTALLATION:  (-5,   5),
        S.EXECUTE:    (-8,   0)}

def compute_ved() -> dict:
    return {STATE_NAMES[s]: {"block_cost": c, "damage_saved": d, "VED": c + d}
            for s, (c, d) in _VED.items()}


# ─────────────────────────────────────────────────────────────
# EU-BASED BACKWARD INDUCTION
# ─────────────────────────────────────────────────────────────

def backward_induction(b: np.ndarray, gamma: float = 0.9) -> dict:
    """
    EU-based backward induction across attack stages (Execute → Recon).
    V[s] = immediate EU + gamma * E[V(s_next)] under best defender action.
    """
    V         = {}
    best_acts = {}
    stages    = [S.EXECUTE, S.INSTALLATION, S.INFECTION, S.RECON]

    for s in stages:
        eu_arr  = expected_utility_vector(b, s)
        aD_best = int(np.argmax(eu_arr))
        imm     = float(eu_arr[aD_best])
        aD_name = list(DEFENDER_ACTION_NAMES[s].values())[aD_best]

        if s != S.EXECUTE:
            keys   = list(TRANSITION_MODEL[s].keys())
            T_row  = TRANSITION_MODEL[s][keys[aD_best]]
            future = sum(T_row[s2] * V.get(S(s2), 0.0) for s2 in range(NUM_STATES))
        else:
            future = 0.0

        V[s]         = imm + gamma * future
        best_acts[s] = aD_name

    return {
        "stage_values": {STATE_NAMES[s]: round(V[s], 4) for s in stages},
        "best_actions": {STATE_NAMES[s]: best_acts[s]   for s in stages},
        "ved":          compute_ved(),
    }


# ─────────────────────────────────────────────────────────────
# MAIN ENGINE CLASS
# ─────────────────────────────────────────────────────────────

class POSGEngine:
    """
    Partial Observable Stochastic Game engine v2.

    Parameters
    ----------
    alpha    : LLM weight in hybrid belief update
    gamma    : discount factor
    dry_run  : log ONOS actions instead of sending
    """

    def __init__(self, alpha: float = 0.6, gamma: float = 0.9,
                 dry_run: bool = True):
        self.alpha   = alpha
        self.gamma   = gamma
        self.dry_run = dry_run
        self.t       = 0
        self.belief  = INITIAL_BELIEF.copy()
        self._last_aD   = 0
        self._last_stage = S.NORMAL
        log.info(f"POSGEngine  α={alpha}  γ={gamma}  dry_run={dry_run}")
        log.info(f"  b₀ = {self.belief.tolist()}")

    # ── Public API ───────────────────────────────────────────

    def step(self, raw_obs: RawObs,
             llm_belief: Optional[np.ndarray] = None) -> StepResult:
        """
        One engine tick:
          1. RawObs → observation O
          2. Predict  (transition model)
          3. Correct  (observation likelihood)
          4. Hybrid   (Bayes + LLM if available)
          5. EU maximization → best defender action
          6. ONOS dispatch
        """
        self.t     += 1
        b_prior     = self.belief.copy()

        obs         = raw_obs.to_obs()
        obs_str     = OBS_NAMES[obs]

        b_pred      = _predict(b_prior, self._last_aD, self._last_stage)
        b_post      = _correct(b_pred, obs)

        if llm_belief is not None:
            b_final  = _hybrid(b_post, llm_belief, self.alpha)
            llm_used = True
        else:
            b_final  = b_post
            llm_used = False

        self.belief  = b_final
        stage        = S(int(np.argmax(b_final)))
        aD, eu, all_eu = best_action(b_final, stage)

        onos   = dispatch_onos(stage, aD, self.dry_run)
        aD_name = list(DEFENDER_ACTION_NAMES[stage].values())[aD]

        self._last_aD    = aD
        self._last_stage = stage

        r = StepResult(
            timestep          = self.t,
            raw_obs           = raw_obs,
            observation       = obs_str,
            belief_prior      = b_prior.tolist(),
            belief_predicted  = b_pred.tolist(),
            belief_updated    = b_final.tolist(),
            belief_bayes      = b_post.tolist(),
            most_likely_stage = STATE_NAMES[stage],
            stage_confidence  = round(float(b_final[stage]), 4),
            best_action_idx   = aD,
            best_action_name  = aD_name,
            expected_utility  = round(eu, 4),
            all_utilities     = [round(u, 4) for u in all_eu],
            onos_payload      = onos,
            llm_used          = llm_used,
            belief_entropy    = round(_entropy(b_final), 4),
        )
        self._log(r)
        return r

    def run_backward_induction(self) -> dict:
        return backward_induction(self.belief, self.gamma)

    def reset(self, b0: Optional[np.ndarray] = None):
        self.t           = 0
        self.belief      = (b0 if b0 is not None else INITIAL_BELIEF).copy()
        self._last_aD    = 0
        self._last_stage = S.NORMAL
        log.info(f"Reset. b₀ = {self.belief.tolist()}")

    def _log(self, r: StepResult):
        log.info(
            f"t={r.timestep:02d} | {r.observation:<30} | "
            f"{r.most_likely_stage:<12} {r.stage_confidence:.0%} | "
            f"{r.best_action_name:<18} EU={r.expected_utility:+.3f} | "
            f"H={r.belief_entropy:.3f}"
        )


# ─────────────────────────────────────────────────────────────
# POSG GAME ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class POSGEnvironment:
    """
    Simplified POSG game environment for attacker vs defender interactions.
    Manages state transitions, reward calculation, and belief updates.
    
    Use this for training agents against deterministic/stochastic game logic.
    """

    def __init__(self):
        """Initialize game environment with default GameState and BeliefState."""
        self.state = GameState()
        self.belief = BeliefState()
        log.info("POSGEnvironment initialized")

    def reset(self):
        """Reset environment to initial state."""
        self.state = GameState()
        self.belief = BeliefState()
        return self.state

    def step(self, attacker_action, defender_action):
        """
        Execute one step of the game.
        
        Args:
            attacker_action: AttackerAction enum value
            defender_action: DefenderAction enum value
            
        Returns:
            (state, (reward_attacker, reward_defender), done)
                state: updated GameState
                rewards: tuple of (attacker_reward, defender_reward)
                done: bool indicating terminal state
        """
        reward_attacker = 0
        reward_defender = 0

        current_stage = self.state.stage

        if current_stage == Stage.NORMAL:
            if attacker_action == AttackerAction.SCAN:
                base_prob = 0.7
                if defender_action == DefenderAction.PATCH:
                    base_prob -= 0.2
                elif defender_action == DefenderAction.MONITOR:
                    base_prob -= 0.1
                base_prob += self.belief.threat_level * 0.1
                success = random.random() < base_prob
                print(f"[DEBUG] NORMAL→RECON success prob: {base_prob:.2f}")
                if success:
                    self.state.stage = Stage.RECON
                    reward_attacker += 1
                else:
                    reward_defender += 2

        elif current_stage == Stage.RECON:
            if attacker_action == AttackerAction.SCAN:
                base_prob = 0.8
                if defender_action == DefenderAction.PATCH:
                    base_prob -= 0.2
                elif defender_action == DefenderAction.MONITOR:
                    base_prob -= 0.15
                base_prob += self.belief.threat_level * 0.15
                success = random.random() < base_prob
                print(f"[DEBUG] RECON→INFECTION success prob: {base_prob:.2f}")
                if success:
                    self.state.stage = Stage.INFECTION
                    reward_attacker += 2
                else:
                    reward_defender += 3
                    self.belief.threat_level = min(1.0, self.belief.threat_level + 0.05)

        elif current_stage == Stage.INFECTION:
            if attacker_action == AttackerAction.EXPLOIT:
                base_prob = 0.75
                if defender_action == DefenderAction.PATCH:
                    base_prob -= 0.25
                elif defender_action == DefenderAction.MONITOR:
                    base_prob -= 0.2
                base_prob += self.belief.threat_level * 0.2
                success = random.random() < base_prob
                print(f"[DEBUG] INFECTION→INSTALLATION success prob: {base_prob:.2f}")
                if success:
                    self.state.stage = Stage.INSTALLATION
                    reward_attacker += 3
                else:
                    reward_defender += 3
                    self.belief.threat_level = min(1.0, self.belief.threat_level + 0.05)

        elif current_stage == Stage.INSTALLATION:
            if attacker_action == AttackerAction.PREPARE_PAYLOAD:
                base_prob = 0.8
                if defender_action == DefenderAction.PATCH:
                    base_prob -= 0.2
                elif defender_action == DefenderAction.MONITOR:
                    base_prob -= 0.25
                base_prob += self.belief.threat_level * 0.2
                success = random.random() < base_prob
                print(f"[DEBUG] INSTALLATION→EXECUTION success prob: {base_prob:.2f}")
                if success:
                    self.state.stage = Stage.EXECUTION
                    reward_attacker += 4
                else:
                    reward_defender += 3
                    self.belief.threat_level = min(1.0, self.belief.threat_level + 0.05)

        elif current_stage == Stage.EXECUTION:
            if attacker_action == AttackerAction.EXECUTE_ATTACK:
                base_prob = 0.65
                if defender_action == DefenderAction.PATCH:
                    base_prob -= 0.2
                elif defender_action == DefenderAction.MONITOR:
                    base_prob -= 0.3
                base_prob += self.belief.threat_level * 0.25
                success = random.random() < base_prob
                print(f"[DEBUG] EXECUTION success prob: {base_prob:.2f}")
                if success:
                    self.state.compromised = True
                    reward_attacker += 6
                    print("🔥 EXECUTION SUCCESS — attack completed")
                else:
                    reward_defender += 4
                    print("🛡️ Execution blocked")

        # --- DETECTION ---
        if defender_action == DefenderAction.MONITOR:
            detected = random.random() < 0.7
            self.state.detected = detected
            if detected:
                reward_defender += 5
                reward_attacker -= 3

        # --- BELIEF UPDATE (POSG CORE) ---
        if attacker_action != AttackerAction.IDLE:
            self.belief.attack_prob = min(1.0, self.belief.attack_prob + 0.2)
            self.belief.threat_level = min(1.0, self.belief.threat_level + 0.1)
        else:
            self.belief.attack_prob = max(0.0, self.belief.attack_prob - 0.1)
            self.belief.threat_level = max(0.0, self.belief.threat_level - 0.05)

        # --- STEP COUNT ---
        self.state.step += 1
        done = self.state.step >= 10 or self.state.compromised

        return self.state, (reward_attacker, reward_defender), done
