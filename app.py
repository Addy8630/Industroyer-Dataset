import io
from pathlib import Path

import pandas as pd
import streamlit as st

from posg_engine import POSGEngine
from posg_types import (
    S, RawObs, STATE_NAMES, OBS_NAMES,
    ATTACKER_ACTION_NAMES, DEFENDER_ACTION_NAMES,
)
from posg_models import get_payoff
from data4cyber_replay import Data4CyberReplay, SCENARIOS, scenario_ground_truth, telemetry_observation


st.set_page_config(page_title="Smart Grid POSG — Data4Cyber", page_icon="⚡", layout="wide")

LOCAL_ZIP = Path(__file__).with_name("data4cyber_dataset(1).zip")


def get_zip_source():
    if "zip_bytes" not in st.session_state:
        st.session_state.zip_bytes = None
    uploaded = st.file_uploader("Upload Data4Cyber ZIP", type=["zip"], help="Use the Data4Cyber ZIP you uploaded with the project.")
    if uploaded is not None:
        st.session_state.zip_bytes = uploaded.getvalue()
        return io.BytesIO(st.session_state.zip_bytes)
    if LOCAL_ZIP.exists():
        return LOCAL_ZIP
    return None


def make_raw_obs(t):
    obs_name = telemetry_observation(t)
    return RawObs(
        syn_count=100 if obs_name == "RECON_SCAN" else 0,
        unique_ports=20 if obs_name == "RECON_SCAN" else 0,
        ssh_attempts=0,
        cve_alert=False,
        modbus_anomaly=obs_name == "MODBUS_ANOMALY",
        freq_deviation=t.frequency_deviation,
        dnp3_operate=False,
        phasor_mismatch=4.0 if obs_name == "FDI_ANOMALY" else 0.0,
        soc_mismatch=t.soc_mismatch,
        power_deviation=1.0 if obs_name == "POWER_DEVIATION" else 0.0,
        price_anomaly=obs_name == "PRICE_ANOMALY",
    )


def observed_action(t):
    # Use defender-visible event text only. attack_phase is retained as ground truth
    # for research display and is deliberately not used here.
    text = (t.attacker_event or "").lower()
    if "nmap" in text or "scan" in text or "probe" in text:
        return S.RECON, "PortScan"
    if "rrw" in text or "register" in text or "modbus" in text or "industroyer" in text:
        return S.INSTALLATION, "ModbusWrite"
    if "mqtt" in text or "price" in text:
        # Closest modeled execution action; the dataset records MQTT manipulation, not DNP3/FDI cascade.
        return S.EXECUTE, "FDICascade"
    if "mitm" in text or "field value" in text or "false data injection" in text:
        return S.INSTALLATION, "FDIPrep"
    return S.NORMAL, "NoAttack"


def action_index(stage, label):
    for idx, name in ATTACKER_ACTION_NAMES[stage].items():
        if name == label:
            return int(idx)
    return 0


def init_game(replay):
    st.session_state.engine = POSGEngine(alpha=0.0, gamma=0.9, dry_run=True)
    st.session_state.replay = replay
    st.session_state.last_result = None
    st.session_state.last_telemetry = replay.telemetry(0)
    st.session_state.score_d = 0.0
    st.session_state.step_log = []


def advance_one():
    replay = st.session_state.replay
    t = replay.next()
    if t is None:
        st.session_state.finished = True
        return
    raw = make_raw_obs(t)
    result = st.session_state.engine.step(raw)
    st.session_state.last_telemetry = t
    st.session_state.last_result = result
    st.session_state.finished = replay.index >= len(replay)

    stage = {"Normal": S.NORMAL, "Recon": S.RECON, "Infection": S.INFECTION, "Installation": S.INSTALLATION, "Execute": S.EXECUTE}.get(result.most_likely_stage, S.NORMAL)
    attacker_stage, attacker_label = observed_action(t)
    payoff_stage = attacker_stage if attacker_stage in S else stage
    payoff = get_payoff(payoff_stage)
    ai_idx = result.best_action_idx
    a_idx = action_index(payoff_stage, attacker_label)
    if a_idx >= payoff.shape[0]:
        a_idx = 0
    if ai_idx >= payoff.shape[1]:
        ai_idx = 0
    utility = float(payoff[a_idx, ai_idx])
    st.session_state.score_d += utility
    st.session_state.step_log.append(
        f"t={result.timestep} | {result.observation} | belief={result.stage_confidence:.1%} {result.most_likely_stage} | "
        f"event={t.attacker_event} | AI={result.best_action_name} | U_D={utility:+.2f}"
    )


# ----------------------------
# Data source
# ----------------------------
zip_source = get_zip_source()

st.markdown("""
<div style='background:#07121b;padding:24px 28px;border-radius:16px;'>
<h1 style='color:#63e6ff;margin:0;'>⚡ SMART GRID POSG — DATA4CYBER LIVE GAME</h1>
<p style='color:#9fc3d9;margin:8px 0 0;'>Data4Cyber telemetry • Partial Observability • Bayesian Belief • Stage Payoff Matrix</p>
</div>
""", unsafe_allow_html=True)

if zip_source is None:
    st.warning("Place data4cyber_dataset(1).zip in the same folder as app.py, or upload it above.")
    st.stop()

# For uploaded BytesIO, create a stable replay object directly.
source_key = "uploaded" if isinstance(zip_source, io.BytesIO) else str(zip_source)
if "replay" not in st.session_state or st.session_state.get("source_key") != source_key:
    try:
        replay = Data4CyberReplay(zip_source, "S1")
        init_game(replay)
        st.session_state.source_key = source_key
        st.session_state.finished = False
    except Exception as e:
        st.error(f"Could not load Data4Cyber: {e}")
        st.stop()

# ----------------------------
# Controls
# ----------------------------
with st.sidebar:
    st.header("Data4Cyber Replay")
    scenario_labels = {sid: f"{sid} — {title}" for sid, (_, title) in SCENARIOS.items()}
    selected = st.selectbox("Scenario", list(SCENARIOS.keys()), format_func=lambda x: scenario_labels[x])
    if selected != st.session_state.replay.scenario_id:
        try:
            st.session_state.replay.load_scenario(selected)
            init_game(st.session_state.replay)
            st.session_state.source_key = source_key
            st.session_state.finished = False
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.caption(f"Rows: {len(st.session_state.replay):,}")
    st.caption("Each ▶ step consumes one actual Data4Cyber CSV row.")

    c1, c2 = st.columns(2)
    if c1.button("▶ Next row", use_container_width=True, disabled=st.session_state.get("finished", False)):
        advance_one()
        st.rerun()
    if c2.button("🔄 Reset", use_container_width=True):
        st.session_state.replay.reset()
        init_game(st.session_state.replay)
        st.session_state.finished = False
        st.rerun()

    if st.button("⏩ Jump to first attack", use_container_width=True):
        st.session_state.replay.jump_to_attack()
        st.session_state.engine.reset()
        st.session_state.last_result = None
        st.session_state.last_telemetry = st.session_state.replay.telemetry()
        st.session_state.finished = False
        st.rerun()

    show_truth = st.checkbox("Research mode: show dataset ground truth", value=False)

# ----------------------------
# Current telemetry
# ----------------------------
t = st.session_state.last_telemetry
r = st.session_state.last_result

if t is None:
    st.error("No telemetry row available.")
    st.stop()

st.markdown(f"### Scenario: **{st.session_state.replay.title}**")

# Kill chain
stage = S.NORMAL if r is None else S(int(max(0, min(4, r.most_likely_stage == 'Normal' and 0 or {'Recon':1,'Infection':2,'Installation':3,'Execute':4}.get(r.most_likely_stage,0)))))
stages = [(S.NORMAL,"NORMAL"),(S.RECON,"RECON"),(S.INFECTION,"INFECTION"),(S.INSTALLATION,"INSTALLATION"),(S.EXECUTE,"EXECUTION")]
parts=[]
for i,(s,label) in enumerate(stages):
    bg="#58a9d9" if s==stage else "#11233c"
    parts.append(f"<div style='flex:1;background:{bg};padding:13px;border-radius:10px;color:white;text-align:center;font-weight:700'>{label}</div>")
    if i < len(stages)-1: parts.append("<div style='padding:0 9px;font-size:24px;color:#8fb8d3'>→</div>")
st.markdown("<div style='display:flex;align-items:center'>"+"".join(parts)+"</div>", unsafe_allow_html=True)
st.progress(stage.value/4 if stage.value else 0.0)

left, mid, right = st.columns([3,4,3])

with left:
    st.markdown("## 📡 Data4Cyber Telemetry")
    rows = {
        "Timestamp": t.timestamp,
        "Attack active": "YES" if t.attack_active else "NO",
        "SOC reference": f"{t.soc_reference:.4f}",
        "SOC reported": f"{t.soc_reported:.4f}",
        "SOC mismatch": f"{t.soc_mismatch:.4f}",
        "PV measured": f"{t.pv_power:.2f}",
        "PV profile": f"{t.pv_profile:.2f}",
        "PV residual": f"{t.pv_residual:.2f}",
        "Load A": f"{t.load_a:.2f}",
        "Load B": f"{t.load_b:.2f}",
        "Frequency": f"{t.frequency:.4f} Hz",
        "Frequency deviation": f"{t.frequency_deviation:.4f} Hz",
        "MQTT price": f"{t.price:.6f}",
    }
    st.dataframe(pd.DataFrame(rows.items(), columns=["Signal","Value"]), hide_index=True, use_container_width=True)

    obs_name = telemetry_observation(t)
    st.success(f"POSG observation: **{obs_name}**")
    st.caption(f"Dataset event: {t.attacker_event}")

    if show_truth:
        st.info(f"Dataset ground-truth phase (analysis only): {scenario_ground_truth(t)}")

with mid:
    st.markdown("## 🧠 Defender Belief")
    belief = [0.75,0.15,0.06,0.03,0.01] if r is None else r.belief_updated
    names=["Normal","Recon","Infection","Installation","Execute"]
    for name,p in zip(names,belief):
        st.write(f"**{name}** — {p:.1%}")
        st.progress(float(p))
    if r is not None:
        st.metric("Most likely stage", r.most_likely_stage, f"{r.stage_confidence:.1%}")
        st.metric("Belief entropy", f"{r.belief_entropy:.3f} bits")
        st.metric("AI defender action", r.best_action_name)
        st.metric("Expected utility", f"{r.expected_utility:+.3f}")
    else:
        st.info("Press ▶ Next row to perform the first Bayesian POSG update.")

with right:
    st.markdown("## ⚔️ Game Actions")
    observed_stage, observed_label = observed_action(t)
    st.write("**Data4Cyber observed event**")
    st.code(t.attacker_event or "none")
    st.write("**Mapped POSG attacker action**")
    st.info(f"{observed_stage.name}: {observed_label}")

    if r is not None:
        defender_labels=list(DEFENDER_ACTION_NAMES[stage].values())
        st.write("**AI defender decision**")
        st.success(r.best_action_name)
        st.caption("The defender action is selected by the POSG engine from the current belief and expected utility.")

        # Show current-stage payoff matrix.
        st.markdown("### 🎯 Payoff Matrix")
        U=get_payoff(stage)
        a_names=list(ATTACKER_ACTION_NAMES[stage].values())
        d_names=list(DEFENDER_ACTION_NAMES[stage].values())
        table={"Attacker action":a_names}
        for j,d in enumerate(d_names):
            table[d]=[f"{U[i,j]:+.2f}" for i in range(U.shape[0])]
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📜 POSG Event Log")
for x in st.session_state.step_log[-10:]:
    st.write(x)

st.caption("The defender observes telemetry-derived POSG observations; attack_phase/attack_phase_all are retained as dataset ground truth and are not used to construct the defender belief.")
