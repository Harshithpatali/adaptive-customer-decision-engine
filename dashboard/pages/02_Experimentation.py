import json
from pathlib import Path
import pandas as pd
import streamlit as st
from ui import inject_css, hero, metric_card, plotly_bar

inject_css()
ROOT=Path(__file__).resolve().parents[2]
hero("Experiment & Causal Analytics","Turn randomized campaign assignment into measurable causal evidence.",["A/B TEST","ATE","BOOTSTRAP CI","STATISTICAL TESTS"])
mp=ROOT/"reports/statistics/experiment_manifest.json"; ef=ROOT/"reports/statistics/experiment_effects.csv"
if not mp.exists() or not ef.exists(): st.warning("Experiment reports are not present."); st.stop()
manifest=json.loads(mp.read_text()); df=pd.read_csv(ef)
a,b,c,d=st.columns(4)
a.metric("Randomized customers",f'{manifest["dataset_rows"]:,}'); b.metric("No Email",f'{manifest["treatment_arms"]["No E-Mail"]:,}'); c.metric("Men's Email",f'{manifest["treatment_arms"]["Mens E-Mail"]:,}'); d.metric("Women's Email",f'{manifest["treatment_arms"]["Womens E-Mail"]:,}')
st.markdown("### Treatment effects")
st.dataframe(df[["treatment","outcome","n_treatment","n_control","treatment_rate","control_rate","absolute_lift","relative_lift","bootstrap_ci_low","bootstrap_ci_high","p_value","effect_significant_0_05"]].style.format({"treatment_rate":"{:.2%}","control_rate":"{:.2%}","absolute_lift":"{:.2%}","relative_lift":"{:.1%}","bootstrap_ci_low":"{:.2%}","bootstrap_ci_high":"{:.2%}","p_value":"{:.3g}"}),use_container_width=True,hide_index=True)
st.markdown("### Conversion lift"); st.bar_chart(df[df.outcome=="conversion"].set_index("treatment")["absolute_lift"])
st.caption("Confidence intervals and p-values are from the stored experiment report.")
