"""Data4Cyber dataset loader and telemetry-to-POSG adapter."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


SCENARIOS = {
    "S1": ("S1_industroyer_pv", "Industroyer against PV inverter"),
    "S5": ("S5_arp_spoof_loads_pv_bss_two_phase", "ARP spoofing incl. BSS meter (two-phase)"),
    "S6": ("S6_mqtt_supply_chain_compromise", "MQTT supply-chain compromise"),
    "S1_ALT": ("S1_industroyer_pv_alt", "Industroyer against PV inverter (alternative)"),
}


@dataclass
class Telemetry:
    timestamp: str
    attack_active: bool
    attack_phase: str
    attack_phase_all: str
    attacker_event: str
    soc_reference: float
    soc_reported: float
    soc_mismatch: float
    pv_power: float
    pv_profile: float
    pv_residual: float
    load_a: float
    load_b: float
    load_profile_a: float
    load_profile_b: float
    load_residual: float
    frequency: float
    frequency_deviation: float
    price: float
    power_deviation: float

    def to_dict(self):
        return asdict(self)


def _num(row, key, default=0.0):
    try:
        v = row.get(key, default)
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def row_to_telemetry(row: pd.Series, baselines=None) -> Telemetry:
    freq = _num(row, "Substation-AC-Meter.freq", 50.0)
    soc_ref = _num(row, "Battery-SOC-Reference.soc_value")
    soc_rep = _num(row, "Battery-SOC-Reported.soc_value")
    pv = _num(row, "PV-Inverter-AC-Meter.p_sum3")
    pv_profile = _num(row, "Profile.pv_active_kw")
    load_a = _num(row, "Load-A-AC-Meter.p_sum3")
    load_b = _num(row, "Load-B-AC-Meter.p_sum3")
    load_pa = _num(row, "Profile.load1_kw")
    load_pb = _num(row, "Profile.load2_kw")
    # Standardized physical residual using benign baseline statistics when available.
    power_score = 0.0
    if baselines:
        zs = []
        for key, value in [("pv", pv), ("load_a", load_a), ("load_b", load_b)]:
            med, scale = baselines.get(key, (value, 1.0))
            scale = max(float(scale), 1e-6)
            zs.append(abs(value - med) / scale)
        power_score = min(1.0, max(zs) / 5.0)

    return Telemetry(
        timestamp=str(row.get("timestamp", "")),
        attack_active=bool(row.get("attack_active", False)),
        attack_phase=str(row.get("attack_phase", "normal")),
        attack_phase_all=str(row.get("attack_phase_all", row.get("attack_phase", "normal"))),
        attacker_event=str(row.get("Attacker.event", row.get("MQTT-Price-Signal.event", "none"))),
        soc_reference=soc_ref,
        soc_reported=soc_rep,
        soc_mismatch=abs(soc_ref - soc_rep),
        pv_power=pv,
        pv_profile=pv_profile,
        pv_residual=abs(pv - pv_profile),
        load_a=load_a,
        load_b=load_b,
        load_profile_a=load_pa,
        load_profile_b=load_pb,
        load_residual=abs((load_a + load_b) - (load_pa + load_pb)),
        frequency=freq,
        frequency_deviation=abs(freq - 50.0),
        price=_num(row, "MQTT-Price-Signal.price_eur_per_kwh"),
        power_deviation=power_score,
    )


def list_scenarios(zip_path: str | Path):
    with zipfile.ZipFile(zip_path) as z:
        found = []
        for sid, (folder, title) in SCENARIOS.items():
            p = f"data4cyber_dataset/{folder}/dataset.csv"
            if p in z.namelist():
                found.append((sid, title))
        return found


class Data4CyberReplay:
    """Small-window replay reader; only the selected CSV is loaded into memory."""

    def __init__(self, zip_path: str | Path, scenario_id: str = "S1"):
        self.zip_path = str(zip_path)
        self.scenario_id = scenario_id
        self.df: Optional[pd.DataFrame] = None
        self.index = 0
        self.phase_meta = []
        self.baselines = {}
        self.load_scenario(scenario_id)

    @property
    def folder(self):
        return SCENARIOS[self.scenario_id][0]

    @property
    def title(self):
        return SCENARIOS[self.scenario_id][1]

    def load_scenario(self, scenario_id: str):
        self.scenario_id = scenario_id
        folder = SCENARIOS[scenario_id][0]
        csv_name = f"data4cyber_dataset/{folder}/dataset.csv"
        phase_name = f"data4cyber_dataset/{folder}/attack-phases.json"
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(csv_name) as f:
                self.df = pd.read_csv(f)
            self.phase_meta = json.loads(z.read(phase_name)) if phase_name in z.namelist() else []
        # Build benign telemetry baselines for physical anomaly scoring.
        benign = self.df[self.df["attack_active"].fillna(False).astype(bool) == False]
        self.baselines = {}
        for key in ["PV-Inverter-AC-Meter.p_sum3", "Load-A-AC-Meter.p_sum3", "Load-B-AC-Meter.p_sum3"]:
            x = pd.to_numeric(benign[key], errors="coerce").dropna() if key in benign else pd.Series(dtype=float)
            if len(x):
                self.baselines[{"PV-Inverter-AC-Meter.p_sum3":"pv", "Load-A-AC-Meter.p_sum3":"load_a", "Load-B-AC-Meter.p_sum3":"load_b"}[key]] = (float(x.median()), float(max(x.std(), abs(x.median()) * 0.05, 1.0)))
        self.index = 0

    def reset(self):
        self.index = 0

    def __len__(self):
        return 0 if self.df is None else len(self.df)

    def row(self, index=None):
        if self.df is None or len(self.df) == 0:
            return None
        i = self.index if index is None else max(0, min(int(index), len(self.df)-1))
        return self.df.iloc[i]

    def telemetry(self, index=None):
        r = self.row(index)
        return None if r is None else row_to_telemetry(r, self.baselines)

    def next(self):
        if self.df is None or self.index >= len(self.df):
            return None
        t = self.telemetry(self.index)
        self.index += 1
        return t

    def jump_to_attack(self):
        if self.df is None or len(self.df) == 0:
            return
        mask = self.df.get("attack_active", pd.Series(False, index=self.df.index)).fillna(False).astype(bool)
        hits = self.df.index[mask]
        if len(hits):
            self.index = int(hits[0])

    def nearest_phase_start(self, phase_name: str):
        for p in self.phase_meta:
            if p.get("phase") == phase_name:
                ts = pd.Timestamp(p["start"])
                timestamps = pd.to_datetime(self.df["timestamp"], utc=True)
                self.index = int((timestamps - ts).abs().argmin())
                return


def scenario_ground_truth(t: Telemetry) -> str:
    """Ground-truth label for analysis only; never used as defender observation."""
    p = (t.attack_phase_all or t.attack_phase or "normal").lower()
    if not t.attack_active and p == "normal":
        return "Normal"
    if "recon" in p or "nmap" in p:
        return "Recon"
    if "mitm" in p or "field_value" in p or "mqtt" in p or "industroyer" in p:
        return "Execute"
    return "Installation"


def telemetry_observation(t: Telemetry) -> str:
    """Map defender-visible telemetry to the eight POSG observation symbols.

    Dataset ground-truth fields such as attack_phase are deliberately not used here.
    """
    event = (t.attacker_event or "").lower()
    if "nmap" in event or "scan" in event or "probe" in event:
        return "RECON_SCAN"
    if t.soc_mismatch > 0.02:
        return "SOC_MISMATCH"
    if "mqtt" in event or "price" in event:
        return "PRICE_ANOMALY"
    if "modbus" in event or "industroyer" in event:
        return "MODBUS_ANOMALY"
    if "mitm" in event or "field value" in event or "false data injection" in event:
        return "FDI_ANOMALY"
    if t.frequency_deviation > 0.30:
        return "FREQUENCY_DEVIATION"
    if t.power_deviation > 0.70:
        return "POWER_DEVIATION"
    return "NORMAL"
