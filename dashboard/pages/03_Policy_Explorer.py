import json
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
st.title("Policy Explorer")
st.caption("Observed campaign economics plus the assumptions used by the production value-maximization policy.")
p=ROOT/"reports/business/campaign_economics.csv"
if not p.exists(): st.warning("Campaign economics report is unavailable."); st.stop()
df=pd.read_csv(p)
st.markdown("### Observed campaign economics")
st.dataframe(df[["action","customers","visits","conversions","observed_spend","campaign_cost","net_observed_value","revenue_per_customer","conversion_rate"]].style.format({"observed_spend":"{:,.2f}","campaign_cost":"{:,.2f}","net_observed_value":"{:,.2f}","revenue_per_customer":"{:,.3f}","conversion_rate":"{:.2%}"}),use_container_width=True,hide_index=True)
st.bar_chart(df.set_index("action")["net_observed_value"])
st.info("These are arm-level observed economics. The repository does not currently contain a complete off-policy individualized policy-value evaluation, so the dashboard does not fabricate one.")
cfg=ROOT/"models/production/decision_config.json"
if cfg.exists(): st.markdown("### Production policy configuration"); st.json(json.loads(cfg.read_text()))
