"""
posg_types.py — Enums, state/action/observation definitions
============================================================
All symbolic identifiers for the POSG engine live here.
Import this module everywhere. Never use raw strings or ints
to reference states, actions, or observations.
"""

from enum import IntEnum, Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────
# STATES  (5 CKC stages)s
# ─────────────────────────────────────────────────────────────

class S(IntEnum):
    NORMAL     = 0
    RECON      = 1
    INFECTION = 2
    INSTALLATION  = 3
    EXECUTE    = 4

NUM_STATES = len(S)

# Alias for convenience
Stage = S

STATE_NAMES = {
    S.NORMAL:     "Normal",
    S.RECON:      "Recon",
    S.INFECTION: "Infection",
    S.INSTALLATION:  "Installation",
    S.EXECUTE:    "Execute",
}

# Prior: most likely Normal; Recon second; deeper stages rare
# Justification: reflects attack-stage prevalence in monitored OT networks.
# High Normal (0.75) = most traffic benign; Recon (0.15) = most common
# malicious precursor in ICS threat intelligence reports.
INITIAL_BELIEF = np.array([0.75, 0.15, 0.06, 0.03, 0.01], dtype=float)
assert np.isclose(INITIAL_BELIEF.sum(), 1.0)


# ─────────────────────────────────────────────────────────────
# OBSERVATIONS  (8 labels, 2 groups)
# ─────────────────────────────────────────────────────────────

class O(IntEnum):
    NORMAL = 0
    RECON_SCAN = 1
    MODBUS_ANOMALY = 2
    FDI_ANOMALY = 3
    SOC_MISMATCH = 4
    POWER_DEVIATION = 5
    PRICE_ANOMALY = 6
    FREQUENCY_DEVIATION = 7

NUM_OBS = len(O)

OBS_NAMES = {
    O.NORMAL: "NORMAL",
    O.RECON_SCAN: "RECON_SCAN",
    O.MODBUS_ANOMALY: "MODBUS_ANOMALY",
    O.FDI_ANOMALY: "FDI_ANOMALY",
    O.SOC_MISMATCH: "SOC_MISMATCH",
    O.POWER_DEVIATION: "POWER_DEVIATION",
    O.PRICE_ANOMALY: "PRICE_ANOMALY",
    O.FREQUENCY_DEVIATION: "FREQUENCY_DEVIATION",
}

# Reverse lookup: name → O index
OBS_BY_NAME = {v: k for k, v in OBS_NAMES.items()}


# ─────────────────────────────────────────────────────────────
# ATTACKER ACTIONS  (stage-specific)
# ─────────────────────────────────────────────────────────────

class NormalA(IntEnum):
    NO_ATTACK = 0

class ReconA(IntEnum):
    PING_SWEEP     = 0
    PORT_SCAN      = 1
    SERVICE_PROBE  = 2

class CompA(IntEnum):
    MODBUS_READ  = 0
    SSH_BRUTE    = 1
    CVE_EXPLOIT  = 2

class WeapA(IntEnum):
    FDI_PREP      = 0
    MODBUS_WRITE  = 1
    BACKDOOR      = 2

class ExecA(IntEnum):
    DNP3_TRIP    = 0
    FDI_CASCADE  = 1
    RANSOMWARE   = 2

# Map stage → attacker action enum class
ATTACKER_ACTIONS = {
    S.NORMAL:     NormalA,
    S.RECON:      ReconA,
    S.INFECTION: CompA,
    S.INSTALLATION:  WeapA,
    S.EXECUTE:    ExecA,
}

ATTACKER_ACTION_NAMES = {
    S.NORMAL:     {NormalA.NO_ATTACK:    "NoAttack"},
    S.RECON:      {ReconA.PING_SWEEP:    "PingSweep",
                   ReconA.PORT_SCAN:     "PortScan",
                   ReconA.SERVICE_PROBE: "ServiceProbe"},
    S.INFECTION: {CompA.MODBUS_READ:    "ModbusRead",
                   CompA.SSH_BRUTE:      "SSHBrute",
                   CompA.CVE_EXPLOIT:    "CVEExploit"},
    S.INSTALLATION:  {WeapA.FDI_PREP:       "FDIPrep",
                   WeapA.MODBUS_WRITE:   "ModbusWrite",
                   WeapA.BACKDOOR:       "Backdoor"},
    S.EXECUTE:    {ExecA.DNP3_TRIP:      "DNP3Trip",
                   ExecA.FDI_CASCADE:    "FDICascade",
                   ExecA.RANSOMWARE:     "Ransomware"},
}


# ─────────────────────────────────────────────────────────────
# DEFENDER ACTIONS  (stage-specific, 4 per stage)
# ─────────────────────────────────────────────────────────────

class NormalD(IntEnum):
    MONITOR    = 0
    LIGHT_SCAN = 1

class ReconD(IntEnum):
    MONITOR    = 0
    HONEYPOT   = 1
    RATE_LIMIT = 2
    IDS_HONEY  = 3

class CompD(IntEnum):
    MONITOR      = 0
    MFA          = 1
    IP_WHITELIST = 2
    PATCH_CVE    = 3

class WeapD(IntEnum):
    MONITOR       = 0
    PHYSICS_CHECK = 1
    ISOLATE_IBR   = 2
    SANDBOX       = 3

class ExecD(IntEnum):
    MONITOR       = 0
    EMERG_BLOCK   = 1
    ROLLBACK      = 2
    BLOCK_ROLLBACK = 3

DEFENDER_ACTIONS = {
    S.NORMAL:     NormalD,
    S.RECON:      ReconD,
    S.INFECTION: CompD,
    S.INSTALLATION:  WeapD,
    S.EXECUTE:    ExecD,
}

DEFENDER_ACTION_NAMES = {
    S.NORMAL:     {NormalD.MONITOR:       "Monitor",
                   NormalD.LIGHT_SCAN:    "LightScan"},
    S.RECON:      {ReconD.MONITOR:        "Monitor",
                   ReconD.HONEYPOT:       "Honeypot",
                   ReconD.RATE_LIMIT:     "RateLimit",
                   ReconD.IDS_HONEY:      "IDS+Honeypot"},
    S.INFECTION: {CompD.MONITOR:         "Monitor",
                   CompD.MFA:             "MFA",
                   CompD.IP_WHITELIST:    "IPWhitelist",
                   CompD.PATCH_CVE:       "PatchCVE"},
    S.INSTALLATION:  {WeapD.MONITOR:         "Monitor",
                   WeapD.PHYSICS_CHECK:   "PhysicsCheck",
                   WeapD.ISOLATE_IBR:     "IsolateIBR",
                   WeapD.SANDBOX:         "Sandbox"},
    S.EXECUTE:    {ExecD.MONITOR:         "Monitor",
                   ExecD.EMERG_BLOCK:     "EmergBlock",
                   ExecD.ROLLBACK:        "Rollback",
                   ExecD.BLOCK_ROLLBACK:  "Block+Rollback"},
}

# Number of defender actions per stage
NUM_DEF_ACTIONS = {s: len(DEFENDER_ACTIONS[s]) for s in S}


# ─────────────────────────────────────────────────────────────
# RAW OBSERVATION DATACLASS  (from Suricata / pyshark on h6)
# ─────────────────────────────────────────────────────────────

@dataclass
class RawObs:
    """
    Raw telemetry from network sensors.
    All values default to 'nothing happening'.
    """
    syn_count:       int   = 0      # SYN packets in 10-second window
    unique_ports:    int   = 0      # unique destination ports probed
    ssh_attempts:    int   = 0      # SSH login failures
    cve_alert:       bool  = False  # Suricata CVE signature hit
    modbus_anomaly:  bool  = False  # unexpected register write
    freq_deviation:  float = 0.0   # Hz deviation from 60 Hz
    dnp3_operate:    bool  = False  # DNP3 OPERATE/TRIP command observed
    phasor_mismatch: float = 0.0   # PMU-SCADA physics mismatch (%)
    soc_mismatch:    float = 0.0   # absolute Battery SOC reference/report difference
    power_deviation: float = 0.0   # normalized physical power residual indicator
    price_anomaly:   bool = False  # MQTT price-signal anomaly

    # Observation thresholds (all in 10-second window)
    SYN_HIGH_THRESH:   int   = 50
    SYN_PROBE_THRESH:  int   = 20
    SSH_THRESH:        int   = 5
    MODBUS_THRESH:     bool  = True
    FREQ_THRESH:       float = 0.3   # Hz
    PHASOR_THRESH:     float = 3.0   # %
    SOC_THRESH:        float = 0.02  # absolute SOC difference
    POWER_THRESH:      float = 0.50  # normalized power residual

    def to_obs(self) -> O:
        """
        Two-level mapping:
          raw telemetry → discrete observation symbol O

        Priority: most severe signal wins (OT > network).
        """
        # Data4Cyber / OT layer — most specific signals first.
        if self.dnp3_operate:
            return O.POWER_DEVIATION

        if abs(self.freq_deviation) > self.FREQ_THRESH:
            return O.FREQUENCY_DEVIATION

        if self.soc_mismatch > self.SOC_THRESH:
            return O.SOC_MISMATCH

        if self.phasor_mismatch > self.PHASOR_THRESH:
            return O.FDI_ANOMALY

        if self.power_deviation > self.POWER_THRESH:
            return O.POWER_DEVIATION

        if self.price_anomaly:
            return O.PRICE_ANOMALY

        if self.modbus_anomaly:
            return O.MODBUS_ANOMALY

        # Network layer
        if self.ssh_attempts > self.SSH_THRESH or self.cve_alert:
            return O.SOC_MISMATCH

        if self.syn_count > self.SYN_HIGH_THRESH or self.unique_ports > 10:
            return O.RECON_SCAN

        return O.NORMAL


# ─────────────────────────────────────────────────────────────
# SIMPLIFIED ATTACKER & DEFENDER ACTION ENUMS
# ─────────────────────────────────────────────────────────────

class Stage(Enum):
    NORMAL = 0
    RECON = 1
    INSTALLATION = 2
    INFECTION = 3
    EXECUTION = 4
    DONE = 5

class AttackerAction(Enum):
    SCAN = 0
    PREPARE_PAYLOAD = 1
    EXPLOIT = 2
    EXECUTE_ATTACK = 3
    IDLE = 4


class DefenderAction(Enum):
    MONITOR = 0
    PATCH = 1
    BLOCK = 2
    IGNORE = 3


# ─────────────────────────────────────────────────────────────
# GAME STATE & BELIEF STATE DATACLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Tracks the current game state variables."""
    compromised: bool = False
    detected: bool = False
    step: int = 0
    stage: Stage = Stage.NORMAL


@dataclass
class BeliefState:
    """Defender's belief state about the attacker."""
    attack_prob: float = 0.5  # Defender belief about attacker likelihood
    threat_level: float = 0.2
