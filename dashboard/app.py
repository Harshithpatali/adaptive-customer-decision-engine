import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ACDX — Adaptive Customer Decision Engine", page_icon="🎯", layout="wide")
DEFAULT_API = os.getenv("ACDX_API_URL", "https://adaptive-customer-decision-engine.onrender.com").rstrip("/")
st.title("ACDX — Adaptive Customer Decision Engine")
st.caption("Production customer-level campaign decisioning powered by a frozen response, revenue, uplift, and policy stack.")

with st.sidebar:
    st.header("API Configuration")
    api = st.text_input("Production API URL", DEFAULT_API).rstrip("/")
    st.caption("The dashboard calls the live FastAPI service. Models are not loaded in Streamlit.")
    st.divider()
    st.header("Production Stack")
    st.write("Response: **XGBoost**")
    st.write("Revenue: **LightGBM**")
    st.write("Uplift: **T-Learner**")
    st.write("Policy: **Value maximization**")

st.subheader("Customer profile")
with st.form("customer"):
    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (days)", min_value=0, value=10, step=1)
        history_segment = st.selectbox("History segment", ["1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +"])
        history = st.number_input("Historical spend", min_value=0.0, value=200.0, step=10.0)
    with c2:
        mens = st.checkbox("Previous men's purchase")
        womens = st.checkbox("Previous women's purchase")
        newbie = st.checkbox("New customer")
    with c3:
        zip_code = st.selectbox("ZIP group", ["Surburban", "Urban", "Rural"])
        channel = st.selectbox("Channel", ["Phone", "Web", "Multichannel"])
    submit = st.form_submit_button("Run production decision", type="primary", use_container_width=True)

if submit:
    payload = {"recency": recency, "history_segment": history_segment, "history": history, "mens": int(mens), "womens": int(womens), "zip_code": zip_code, "newbie": int(newbie), "channel": channel}
    try:
        with st.spinner("Calling production decision engine..."):
            response = requests.post(api + "/decision", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
        action = result["recommended_action"]
        margin = result["decision_margin"]
        utility = result["utility"]
        expected_revenue = result["expected_revenue"]
        probabilities = result["probabilities"]
        conditional_spend = result["conditional_spend"]
        uplift = result["uplift"]
        st.success("Recommended action: " + action)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended campaign", action)
        m2.metric("Decision margin", "{:.3f}".format(margin))
        m3.metric("Expected value", "${:.3f}".format(utility[action]))
        m4.metric("Policy", result["policy_version"])
        st.divider()
        st.subheader("Campaign economics")
        rows = []
        for campaign in utility:
            rows.append({"Campaign": campaign, "Response probability": probabilities[campaign], "Conditional spend": conditional_spend[campaign], "Expected revenue": expected_revenue[campaign], "Utility": utility[campaign], "Uplift": uplift.get(campaign, 0.0)})
        df = pd.DataFrame(rows).set_index("Campaign")
        a, b = st.columns(2)
        with a:
            st.markdown("**Utility by campaign**")
            st.bar_chart(df["Utility"])
        with b:
            st.markdown("**Expected revenue by campaign**")
            st.bar_chart(df["Expected revenue"])
        st.dataframe(df.style.format({"Response probability": "{:.2%}", "Conditional spend": "${:.2f}", "Expected revenue": "${:.3f}", "Utility": "${:.3f}", "Uplift": "{:.2%}"}), use_container_width=True)
        st.subheader("Decision explanation")
        st.write("**{}** was selected because it has the highest modeled utility across the available campaign actions. The decision margin over the next-best action is ${:.3f}.".format(action, margin))
        with st.expander("Raw production response"):
            st.json(result)
    except requests.HTTPError as exc:
        st.error("Production API returned an error: " + str(exc))
        if getattr(exc, "response", None) is not None:
            st.code(exc.response.text)
    except requests.RequestException as exc:
        st.error("Could not reach the production API: " + str(exc))
    except Exception as exc:
        st.error("Dashboard error: " + str(exc))
else:
    st.info("Enter a customer profile and run the production decision engine.")