"""
posg_models.py — Observation, Transition, and Payoff models
============================================================
Three probability/utility models that define the game:

  1. ObservationModel  P(o | s)           shape [5 × 8]
  2. TransitionModel   P(s'| s, aD)       shape [5 × max_aD × 5]
  3. PayoffModel       U_D(s, aA, aD)     stage-specific 2D matrices

All models are validated at construction time.
Zero-smoothing (ε = 0.005) prevents belief collapse.
"""

import numpy as np
import random
from posg_types import (
    S, O, NUM_STATES, NUM_OBS,
    NormalA, ReconA, CompA, WeapA, ExecA,
    NormalD, ReconD, CompD, WeapD, ExecD,
    NUM_DEF_ACTIONS, AttackerAction, DefenderAction, ATTACKER_ACTIONS,
)

EPS = 0.005   # minimum probability — prevents belief collapse
W1, W2, W3, W4 = 1.0, 1.0, 0.5, 0.3   # payoff formula weights


# ════════════════════════════════════════════════════════════
# 1.  OBSERVATION MODEL   P[s, o]
# ════════════════════════════════════════════════════════════

def _build_observation_model() -> np.ndarray:
    """
    Expert-designed P(o|s) table, then:
      1. Replace all values below EPS with EPS
      2. Renormalize each row so it sums exactly to 1

    Row order  : S.NORMAL, S.RECON, S.INFECTION, S.INSTALLATION, S.EXECUTE
    Column order: O.NORMAL, O.RECON_SCAN, O.MODBUS_ANOMALY, O.FDI_ANOMALY,
                  O.SOC_MISMATCH, O.POWER_DEVIATION, O.PRICE_ANOMALY,
                  O.FREQUENCY_DEVIATION
    """
    #                  NORMAL  RECON_SCAN MODBUS_ANOMALY FDI_ANOMALY SOC_MISMATCH  POWER_DEVIATION PRICE_ANOMALY FREQUENCY_DEVIATION
    raw = np.array([
        # Normal     — mostly quiet; tiny chance of OT noise
        [0.860,  0.050, 0.040, 0.020, 0.010, 0.005, 0.005, 0.005],

        # Recon      — nmap / service fingerprinting dominates
        [0.050,  0.580, 0.270, 0.040, 0.020, 0.010, 0.005, 0.005],

        # Compromise — login attempts + Modbus reads
        [0.050,  0.095, 0.095, 0.340, 0.290, 0.055, 0.030, 0.020],

        # Weaponize  — Modbus writes, IBR anomalies, FDI prep
        [0.050,  0.030, 0.020, 0.050, 0.330, 0.285, 0.100, 0.100],

        # Execute    — DNP3 OPERATE / frequency cascade dominate
        [0.020,  0.010, 0.010, 0.010, 0.050, 0.145, 0.540, 0.195],
    ], dtype=float)

    # Apply EPS floor then renormalize
    raw = np.maximum(raw, EPS)
    raw = raw / raw.sum(axis=1, keepdims=True)

    # Validate
    assert raw.shape == (NUM_STATES, NUM_OBS), \
        f"Observation model shape mismatch: {raw.shape}"
    assert np.allclose(raw.sum(axis=1), 1.0, atol=1e-9), \
        "Observation model rows do not sum to 1"
    assert (raw > 0).all(), "Observation model contains non-positive entry"

    return raw


OBSERVATION_MODEL = _build_observation_model()


# ════════════════════════════════════════════════════════════
# 2.  TRANSITION MODEL   P[s, aD_idx, s']
# ════════════════════════════════════════════════════════════

def _build_transition_model() -> dict:
    """
    Returns: T[s][aD_idx] = np.array of shape (NUM_STATES,)
             representing P(s'| s, aD).

    Design constraints enforced:
      • Every (s, aD) row sums to 1
      • Every state can return to Normal (false alarm / attack abort)
      • Attack persistence: nonzero self-loop at each stage
      • Defender actions reduce advance probability
      • EPS floor applied before normalization
      • Assertion check at the end
    """
    T = {}

    # ── S.NORMAL ─────────────────────────────────────────────
    # Normal state: attacker may begin probing (→ Recon)
    # Defender can reduce that probability
    #
    #                       →Norm  →Recon →Comp  →Weap  →Exec
    T[S.NORMAL] = {
        NormalD.MONITOR:    [0.90,  0.10,  0.00,  0.00,  0.00],
        NormalD.LIGHT_SCAN: [0.95,  0.05,  0.00,  0.00,  0.00],
    }

    # ── S.RECON ──────────────────────────────────────────────
    # Recon: attacker may advance to Compromise or abort
    # IDS+Honeypot gives strongest protection
    #
    #                       →Norm  →Recon →Comp  →Weap  →Exec
    T[S.RECON] = {
        ReconD.MONITOR:    [0.10,  0.55,  0.30,  0.05,  0.00],
        ReconD.HONEYPOT:   [0.20,  0.60,  0.15,  0.05,  0.00],
        ReconD.RATE_LIMIT: [0.25,  0.65,  0.08,  0.02,  0.00],
        ReconD.IDS_HONEY:  [0.35,  0.60,  0.04,  0.01,  0.00],
    }

    # ── S.INFECTION ─────────────────────────────────────────
    # Compromise: attacker tries to establish foothold
    # Patching CVE stops advance most effectively
    #
    #                        →Norm  →Recon →Comp  →Weap  →Exec
    T[S.INFECTION] = {
        CompD.MONITOR:      [0.05,  0.10,  0.50,  0.30,  0.05],
        CompD.MFA:          [0.10,  0.15,  0.55,  0.15,  0.05],
        CompD.IP_WHITELIST: [0.15,  0.15,  0.60,  0.08,  0.02],
        CompD.PATCH_CVE:    [0.20,  0.20,  0.55,  0.04,  0.01],
    }

    # ── S.INSTALLATION ──────────────────────────────────────────
    # Weaponize: attacker stages FDI / backdoors
    # Isolating IBR is most effective; return to Normal is harder now
    #
    #                          →Norm  →Recon →Comp  →Weap  →Exec
    T[S.INSTALLATION] = {
        WeapD.MONITOR:        [0.03,  0.05,  0.10,  0.47,  0.35],
        WeapD.PHYSICS_CHECK:  [0.05,  0.08,  0.12,  0.55,  0.20],
        WeapD.ISOLATE_IBR:    [0.08,  0.10,  0.14,  0.60,  0.08],
        WeapD.SANDBOX:        [0.06,  0.09,  0.12,  0.65,  0.08],
    }

    # ── S.EXECUTE ────────────────────────────────────────────
    # Execute: near-absorbing but not fully absorbing
    # Return to Normal = 0.01 (attack completes or is fully blocked)
    # Block+Rollback gives best chance of recovery
    #
    #                           →Norm  →Recon →Comp  →Weap  →Exec
    T[S.EXECUTE] = {
        ExecD.MONITOR:         [0.01,  0.01,  0.02,  0.06,  0.90],
        ExecD.EMERG_BLOCK:     [0.05,  0.05,  0.05,  0.05,  0.80],
        ExecD.ROLLBACK:        [0.10,  0.05,  0.05,  0.10,  0.70],
        ExecD.BLOCK_ROLLBACK:  [0.20,  0.05,  0.05,  0.10,  0.60],
    }

    # ── Apply EPS floor and normalize; then assert rows sum to 1 ─
    for s in S:
        for aD_idx, row in T[s].items():
            arr = np.array(row, dtype=float)
            arr = np.maximum(arr, EPS)
            arr = arr / arr.sum()
            T[s][aD_idx] = arr
            assert np.isclose(arr.sum(), 1.0, atol=1e-9), \
                f"Transition row ({s},{aD_idx}) does not sum to 1: {arr.sum()}"
            assert (arr > 0).all(), \
                f"Transition row ({s},{aD_idx}) has non-positive entry"

    return T


TRANSITION_MODEL = _build_transition_model()


# ════════════════════════════════════════════════════════════
# 3.  PAYOFF MODEL   U_D(s, aA, aD)
# ════════════════════════════════════════════════════════════

# ── Component tables ─────────────────────────────────────────

# Prevention benefit: how much damage does this defense prevent?
# Indexed by (stage, defender_action_idx)
PREVENTION = {
    S.NORMAL:     [1.0, 2.0],                 # Monitor, LightScan
    S.RECON:      [1.0, 3.0, 2.5, 4.0],      # Monitor, Honeypot, RateLimit, IDS+Honey
    S.INFECTION: [1.0, 3.0, 3.5, 4.5],
    S.INSTALLATION:  [1.0, 4.0, 5.0, 4.5],
    S.EXECUTE:    [0.0, 5.0, 6.0, 8.0],      # Monitor=0 during Execute (too late)
}

# Damage cost if attack action succeeds (indexed by attacker_action_idx per stage)
DATA4CYBER_DAMAGE_WEIGHTS = {
    "soc": 8.0,
    "pv": 12.0,
    "load": 10.0,
    "frequency": 15.0,
    "price": 7.0,
}

# Action-level damage used by the stage payoff matrices.
# IMPORTANT: this is intentionally separate from DATA4CYBER_DAMAGE_WEIGHTS.
# DATA4CYBER_DAMAGE_WEIGHTS weights telemetry components (SOC, PV, load,
# frequency, price), while ACTION_DAMAGE supplies one damage value per
# attacker action for the stage-level payoff matrix U_D(s,aA,aD).
ACTION_DAMAGE = {
    S.NORMAL:      np.array([0.0], dtype=float),
    S.RECON:       np.array([1.0, 1.2, 1.1], dtype=float),
    S.INFECTION:   np.array([2.5, 3.0, 4.0], dtype=float),
    S.INSTALLATION:np.array([5.0, 7.0, 8.0], dtype=float),
    S.EXECUTE:     np.array([8.0, 10.0, 9.0], dtype=float),
}


# Validate that action-level damage vectors match the number of attacker
# actions at each stage. This catches payoff-table shape errors early.
for _stage in S:
    _expected = len(ATTACKER_ACTIONS[_stage])
    assert len(ACTION_DAMAGE[_stage]) == _expected, (
        f"ACTION_DAMAGE[{_stage.name}] has {len(ACTION_DAMAGE[_stage])} values; "
        f"expected {_expected}."
    )

# Defense cost (indexed by defender_action_idx per stage)
DEFENSE_COST = {
    S.NORMAL:     [0.5, 1.0],
    S.RECON:      [0.5, 2.0, 1.5, 3.0],
    S.INFECTION: [0.5, 2.0, 1.5, 3.5],
    S.INSTALLATION:  [0.5, 2.5, 3.0, 2.5],
    S.EXECUTE:    [0.5, 4.0, 3.5, 5.0],
}

# Operational disruption: cost of blocking legitimate OT traffic
# HIGH in Normal (false positives), LOW in Execute (attack already happening)
# cap: max(1, value) applied in Execute per spec
OPERATIONAL_DISRUPTION = {
    S.NORMAL:     [0.0, 1.0],                 # Monitor=0, LightScan=1
    S.RECON:      [0.5, 1.0, 2.0, 2.5],
    S.INFECTION: [1.0, 2.0, 3.0, 2.5],
    S.INSTALLATION:  [1.5, 2.0, 4.0, 3.0],
    S.EXECUTE:    [0.5, 1.0, 1.0, 1.0],      # already under attack → low disruption
}

# P_success(s, aA, aD): probability attack succeeds given defender response
# Shape: [num_aA × num_aD] per stage
P_SUCCESS = {
    S.NORMAL: np.array([
        # aA=NoAttack:  defender can't really "stop" nothing
        #            Monitor  LightScan
        [0.00,       0.00],
    ]),

    S.RECON: np.array([
        #            Monitor  Honeypot  RateLimit  IDS+Honey
        [0.70,       0.30,    0.25,     0.10],   # PingSweep
        [0.90,       0.25,    0.30,     0.15],   # PortScan
        [0.85,       0.20,    0.35,     0.12],   # ServiceProbe
    ]),

    S.INFECTION: np.array([
        #            Monitor  MFA    IPWhitelist  PatchCVE
        [0.80,       0.40,    0.35,  0.30],      # ModbusRead
        [0.85,       0.20,    0.40,  0.25],      # SSHBrute
        [0.90,       0.30,    0.45,  0.10],      # CVEExploit
    ]),

    S.INSTALLATION: np.array([
        #            Monitor  Physics  IsolateIBR  Sandbox
        [0.80,       0.30,    0.20,    0.25],     # FDIPrep
        [0.85,       0.40,    0.15,    0.30],     # ModbusWrite
        [0.90,       0.35,    0.25,    0.15],     # Backdoor
    ]),

    S.EXECUTE: np.array([
        #            Monitor  EmergBlock  Rollback  Block+Rollback
        [0.95,       0.40,    0.30,       0.10],   # DNP3Trip
        [0.95,       0.45,    0.35,       0.12],   # FDICascade
        [0.90,       0.35,    0.25,       0.08],   # Ransomware
    ]),
}


def compute_payoff_matrix(stage: S) -> np.ndarray:
    """
    Compute defender payoff matrix for a given stage.

    U_D(aA, aD) = w1 × Prevention(stage, aD)
                − w2 × P_success(stage, aA, aD) × Damage(stage, aA)
                − w3 × DefenseCost(stage, aD)
                − w4 × max(1, OperationalDisruption(stage, aD))  [Execute]
                  OR
                − w4 × OperationalDisruption(stage, aD)          [other stages]

    Returns np.ndarray shape [num_aA, num_aD].

    Note: the matrix uses ACTION_DAMAGE (attacker-action damage) for the
    strategic game payoff. DATA4CYBER_DAMAGE_WEIGHTS remains available to
    compute telemetry-derived physical/cyber damage through Damage().
    """
    ps      = P_SUCCESS[stage]          # [n_aA × n_aD]
    n_aA, n_aD = ps.shape

    prevention = np.array(PREVENTION[stage],              dtype=float)  # [n_aD]
    damage     = np.array(ACTION_DAMAGE[stage],            dtype=float)  # [n_aA]
    def_cost   = np.array(DEFENSE_COST[stage],            dtype=float)  # [n_aD]
    disrupt    = np.array(OPERATIONAL_DISRUPTION[stage],  dtype=float)  # [n_aD]

    # Cap disruption in Execute state
    if stage == S.EXECUTE:
        disrupt = np.maximum(1.0, disrupt)

    # Broadcast: damage [n_aA] × P_success [n_aA × n_aD] → [n_aA × n_aD]
    expected_damage = ps * damage[:, np.newaxis]       # [n_aA × n_aD]

    # U_D: [n_aA × n_aD]
    U = (W1 * prevention[np.newaxis, :]
       - W2 * expected_damage
       - W3 * def_cost[np.newaxis, :]
       - W4 * disrupt[np.newaxis, :])

    return U


# Pre-compute all payoff matrices at import time
PAYOFF = {s: compute_payoff_matrix(s) for s in S}


def get_payoff(stage: S) -> np.ndarray:
    """Return pre-computed defender payoff matrix for stage."""
    return PAYOFF[stage]


# ════════════════════════════════════════════════════════════
# RANDOM ACTION UTILITIES
# ════════════════════════════════════════════════════════════

def random_attacker():
    """Return random attacker action from AttackerAction enum."""
    return random.choice(list(AttackerAction))


def random_defender():
    """Return random defender action from DefenderAction enum."""
    return random.choice(list(DefenderAction))


# ════════════════════════════════════════════════════════════
# DATA4CYBER PAYOFF HELPERS
# ════════════════════════════════════════════════════════════


def CyberImpact(
    attack_severity: float = 0.0,
    exposure: float = 0.0,
    control_loss: float = 0.0,
    signal_integrity: float = 0.0,
    stage_factor: float = 0.0,
) -> float:
    """Return a normalized cyber-impact score in [0, 1]."""
    score = (
        0.35 * attack_severity
        + 0.25 * exposure
        + 0.20 * control_loss
        + 0.10 * signal_integrity
        + 0.10 * stage_factor
    )
    return float(np.clip(score, 0.0, 1.0))


def PhysicalImpact(
    pv_impact: float = 0.0,
    load_impact: float = 0.0,
    frequency_shift: float = 0.0,
    voltage_drop: float = 0.0,
    price_distortion: float = 0.0,
) -> float:
    """Return a normalized physical-impact score in [0, 1]."""
    score = (
        0.30 * pv_impact
        + 0.25 * load_impact
        + 0.25 * frequency_shift
        + 0.10 * voltage_drop
        + 0.10 * price_distortion
    )
    return float(np.clip(score, 0.0, 1.0))


def Damage(
    cyber_impact: float = 0.0,
    physical_impact: float = 0.0,
    soc: float = 0.0,
    pv: float = 0.0,
    load: float = 0.0,
    frequency: float = 0.0,
    price: float = 0.0,
) -> float:
    """Aggregate the weighted Data4Cyber damage estimate."""
    cyber_component = cyber_impact * (
        DATA4CYBER_DAMAGE_WEIGHTS["soc"] * max(0.0, soc)
        + DATA4CYBER_DAMAGE_WEIGHTS["price"] * max(0.0, price)
    )
    physical_component = physical_impact * (
        DATA4CYBER_DAMAGE_WEIGHTS["pv"] * max(0.0, pv)
        + DATA4CYBER_DAMAGE_WEIGHTS["load"] * max(0.0, load)
        + DATA4CYBER_DAMAGE_WEIGHTS["frequency"] * max(0.0, frequency)
    )
    return float(cyber_component + physical_component)


def PreventionGain(
    defense_strength: float = 0.0,
    detection_timeliness: float = 0.0,
    segmentation: float = 0.0,
    isolation: float = 0.0,
) -> float:
    """Return the preventive benefit achieved by the defender."""
    gain = (
        0.40 * defense_strength
        + 0.30 * detection_timeliness
        + 0.20 * segmentation
        + 0.10 * isolation
    )
    return float(np.clip(gain, 0.0, 1.0))


def RecoveryBenefit(
    restoration_speed: float = 0.0,
    redundancy: float = 0.0,
    rollback_quality: float = 0.0,
    operator_readiness: float = 0.0,
) -> float:
    """Return the recovery benefit after a successful defense or mitigation."""
    benefit = (
        0.35 * restoration_speed
        + 0.25 * redundancy
        + 0.25 * rollback_quality
        + 0.15 * operator_readiness
    )
    return float(np.clip(benefit, 0.0, 1.0))


def VED(
    stage: S,
    prevention_gain: float = 0.0,
    damage: float = 0.0,
    recovery_benefit: float = 0.0,
) -> float:
    """Value of early detection: higher when prevention happens early and damage is reduced."""
    stage_weight = {
        S.RECON: 1.0,
        S.INFECTION: 0.9,
        S.INSTALLATION: 0.7,
        S.EXECUTE: 0.5,
        S.NORMAL: 0.2,
    }.get(stage, 0.5)

    value = stage_weight * (prevention_gain * 10.0 + recovery_benefit * 5.0 - damage * 0.1)
    return float(np.clip(value, 0.0, 25.0))


def AttackerUtility(
    cyber_impact: float = 0.0,
    physical_impact: float = 0.0,
    attack_success_prob: float = 0.0,
    defense_penalty: float = 0.0,
) -> float:
    """Return attacker utility approximating gain from successful compromise."""
    return float(
        8.0 * attack_success_prob * (cyber_impact + physical_impact)
        - 5.0 * defense_penalty
    )


def DefenderUtility(
    prevention_gain: float = 0.0,
    recovery_benefit: float = 0.0,
    damage: float = 0.0,
    defense_cost: float = 0.0,
    ved: float = 0.0,
) -> float:
    """Return defender utility approximating value of prevention and recovery."""
    return float(
        10.0 * prevention_gain
        + 7.5 * recovery_benefit
        + 1.5 * ved
        - 1.25 * damage
        - 1.0 * defense_cost
    )
