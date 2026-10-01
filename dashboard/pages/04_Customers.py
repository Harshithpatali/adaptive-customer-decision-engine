from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
st.title("Customer Segmentation & Uplift Landscape")
st.caption("Heterogeneous treatment effects sliced by pre-treatment customer attributes.")
p=ROOT/"reports/statistics/treatment_heterogeneity.csv"
if not p.exists(): st.warning("Treatment heterogeneity report is unavailable."); st.stop()
df=pd.read_csv(p); sub=st.selectbox("Subgroup",sorted(df.subgroup.unique())); x=df[df.subgroup==sub].copy(); level=st.selectbox("Level",sorted(x.level.astype(str).unique())); x=x[x.level.astype(str)==level]
st.dataframe(x[["level","treatment","n_treatment","n_control","treatment_conversion","control_conversion","absolute_conversion_lift","treatment_spend","control_spend","absolute_spend_lift"]].style.format({"treatment_conversion":"{:.2%}","control_conversion":"{:.2%}","absolute_conversion_lift":"{:.2%}","treatment_spend":"{:,.3f}","control_spend":"{:,.3f}","absolute_spend_lift":"{:,.3f}"}),use_container_width=True,hide_index=True)
st.markdown("### Conversion-lift landscape"); st.bar_chart(x.set_index("treatment")["absolute_conversion_lift"])
st.markdown("### Spend-lift landscape"); st.bar_chart(x.set_index("treatment")["absolute_spend_lift"])
st.caption("These are subgroup-level randomized differences, not individualized model predictions.")
