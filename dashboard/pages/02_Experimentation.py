import streamlit as st
import pandas as pd
from pathlib import Path
st.title("Randomized Experimentation")
p=Path("reports/statistics/experiment_effects.csv")
if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True)
else: st.info("Run the offline statistics pipeline to populate this report.")
