import os
from pathlib import Path
import pandas as pd
import streamlit as st
from ui import inject_css, hero, metric_card, plotly_bar

inject_css()

st.set_page_config(page_title="ACDX | Decision Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("ACDX_API_URL", "https://adaptive-customer-decision-engine.onrender.com").rstrip("/")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#0b1020;color:#eef2ff}
[data-testid="stSidebar"]{background:#080d1a;border-right:1px solid #20283d}
.block-container{padding:2rem 3rem 3rem;max-width:1500px}
.acdx-kicker{color:#9b8cff;font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
.card{background:#12192b;border:1px solid #222c45;border-radius:16px;padding:18px 20px;height:100%}
.card-label{color:#8f9bb5;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}
.card-value{font-size:1.75rem;font-weight:750;margin-top:5px}.card-note{color:#7f8ba5;font-size:.8rem;margin-top:5px}
hr{border-color:#222c45}
</style>""", unsafe_allow_html=True)

def read_csv(name):
    p = ROOT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def card(label, value, note="", cls=""):
    return f'<div class="card"><div class="card-label">{label}</div><div class="card-value {cls}">{value}</div><div class="card-note">{note}</div></div>'

econ = read_csv("reports/business/campaign_economics.csv")
effects = read_csv("reports/statistics/experiment_effects.csv")
manifest = {}
mp = ROOT / "reports/statistics/experiment_manifest.json"
research_manifest = {}
rmp = ROOT / "reports/advanced_policy/research_manifest.json"
if mp.exists():
    import json
    manifest = json.loads(mp.read_text())
if rmp.exists():
    import json
    research_manifest = json.loads(rmp.read_text())

hero("Decision Intelligence Center","Causal experimentation + machine learning + revenue optimization + individualized decisioning",["RANDOMIZED EXPERIMENT","UPLIFT MODELING","VALUE OPTIMIZATION","PRODUCTION API"])

with st.sidebar:
    st.markdown("### ACDX")
    st.caption("Decision Intelligence Platform")
    st.divider()
    st.markdown("**Production API**")
    st.code(API_URL, language=None)
    st.markdown("**Live stack**")
    st.caption("XGBoost response · LightGBM revenue · T-Learner uplift · value-max policy")
    st.divider()
    st.caption("Offline training and model search stay outside the production dashboard.")

total = int(manifest.get("dataset_rows", econ["customers"].sum() if not econ.empty else 0))
total_spend = econ["observed_spend"].sum() if not econ.empty else 0
net_value = econ["net_observed_value"].sum() if not econ.empty else 0
conv = manifest.get("conversion_total", int(econ["conversions"].sum()) if not econ.empty else 0)

c1,c2,c3,c4=st.columns(4)
metric_card("Experiment population",f"{total:,}","Randomized Hillstrom population")
metric_card("Observed spend","$"+f"{total_spend:,.0f}","Across all experiment arms")
metric_card("Observed net value","$"+f"{net_value:,.0f}","After configured campaign cost")
metric_card("Conversions",f"{conv:,}","Observed experiment outcome")

st.markdown("### Decision intelligence layers")
r1, r2, r3, r4 = st.columns(4)
with r1:
    metric_card("Experimental", "Randomized", "Average treatment effects")
with r2:
    metric_card("Production", "Frozen", "XGBoost + T-Learner + LightGBM")
with r3:
    metric_card("Research", "Causal", "Cross-fitting + DR + S/T/X")
with r4:
    metric_card(
        "Research run",
        "{} folds / {} bootstrap".format(
            research_manifest.get("cross_fit_folds", "n/a"),
            research_manifest.get("bootstrap_replicates", "n/a"),
        ),
        "Offline research artifact",
    )

st.markdown("### What the experiment measured")
if not effects.empty:
    view = effects[effects["outcome"].isin(["conversion","spend"])].copy()
    st.dataframe(view[["treatment","outcome","n_treatment","n_control","treatment_rate","control_rate","absolute_lift","p_value","effect_significant_0_05"]], use_container_width=True, hide_index=True)
else:
    st.info("Experiment report not available.")

st.markdown("### Campaign economics")
if not econ.empty:
    a,b=st.columns([1.25,1])
    with a:
        st.dataframe(econ[["action","customers","visits","conversions","observed_spend","campaign_cost","net_observed_value"]].style.format({"observed_spend":"{:,.2f}","campaign_cost":"{:,.2f}","net_observed_value":"{:,.2f}"}), use_container_width=True, hide_index=True)
    with b:
        plotly_bar(econ,"action","net_observed_value","Observed net value by campaign")
else:
    st.info("Campaign economics report not available.")

st.markdown("### Decision workflow")
st.markdown("**Customer → response probability → uplift → conditional spend → expected value → policy action**")
st.caption("Use **Customer Decision Simulator** to call the live production API. Use **Policy**, **Experiment**, and **Models** to inspect offline evidence.")
