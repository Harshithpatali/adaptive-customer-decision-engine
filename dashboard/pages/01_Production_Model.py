import streamlit as st
st.title("Production Model")
st.write("Only the selected production algorithms are deployed.")
st.markdown("- Response: **XGBoost**\n- Uplift: **T-Learner**\n- Revenue: **LightGBM**\n- Policy: **Expected value minus campaign cost**")
st.info("Model discovery, hyperparameter search, SMOTE, and alternative candidates are offline-only and are not loaded by the production application.")
