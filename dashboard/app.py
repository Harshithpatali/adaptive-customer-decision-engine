import streamlit as st
import requests

st.set_page_config(page_title="ACDX — Causal Decision Intelligence", layout="wide")
st.title("ACDX — Adaptive Customer Decision Engine")
st.caption("Production decision layer powered by the selected XGBoost response model")

api = st.sidebar.text_input("API URL", "http://localhost:8000")

st.metric("Selected response model", "XGBoost")
st.metric("Reported test ROC-AUC", "0.6677")
st.metric("Reported test PR-AUC", "0.0194")

st.subheader("Customer decision")
with st.form("customer"):
    recency = st.number_input("Recency", min_value=0, value=10)
    history_segment = st.selectbox("History segment", ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"])
    history = st.number_input("Historical spend", min_value=0.0, value=200.0)
    mens = st.checkbox("Previous men's purchase")
    womens = st.checkbox("Previous women's purchase")
    zip_code = st.selectbox("ZIP group", ["Surburban", "Urban", "Rural", "Surburban"])
    newbie = st.checkbox("New customer")
    channel = st.selectbox("Channel", ["Phone", "Web", "Multichannel"])
    submit = st.form_submit_button("Get decision")

if submit:
    payload = {"recency":recency,"history_segment":history_segment,"history":history,"mens":int(mens),"womens":int(womens),"zip_code":zip_code,"newbie":int(newbie),"channel":channel}
    try:
        r=requests.post(f"{api}/decision",json=payload,timeout=20)
        r.raise_for_status()
        result=r.json()
        st.success(f"Recommended action: {result.get('action','N/A')}")
        st.json(result)
    except Exception as e:
        st.error(f"API request failed: {e}")
