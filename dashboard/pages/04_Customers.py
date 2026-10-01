from pathlib import Path
import pandas as pd
import streamlit as st
from ui import inject_css, hero, plotly_bar

inject_css()
ROOT=Path(__file__).resolve().parents[2]
hero("Customer Segmentation & Uplift Landscape","Explore heterogeneous randomized effects across customer attributes.",["HETEROGENEITY","UPLIFT","SEGMENTS"])
p=ROOT/"reports/statistics/treatment_heterogeneity.csv"
if not p.exists(): st.warning("Treatment heterogeneity report is unavailable."); st.stop()
df=pd.read_csv(p); sub=st.selectbox("Subgroup",sorted(df.subgroup.unique())); x=df[df.subgroup==sub].copy(); level=st.selectbox("Level",sorted(x.level.astype(str).unique())); x=x[x.level.astype(str)==level]
st.dataframe(x[["level","treatment","n_treatment","n_control","treatment_conversion","control_conversion","absolute_conversion_lift","treatment_spend","control_spend","absolute_spend_lift"]].style.format({"treatment_conversion":"{:.2%}","control_conversion":"{:.2%}","absolute_conversion_lift":"{:.2%}","treatment_spend":"{:,.3f}","control_spend":"{:,.3f}","absolute_spend_lift":"{:,.3f}"}),use_container_width=True,hide_index=True)
st.markdown("### Conversion-lift landscape"); plotly_bar(x,"treatment","absolute_conversion_lift","Conversion lift by treatment")
st.markdown("### Spend-lift landscape"); plotly_bar(x,"treatment","absolute_spend_lift","Spend lift by treatment")
st.caption("These are subgroup-level randomized differences, not individualized model predictions.")
