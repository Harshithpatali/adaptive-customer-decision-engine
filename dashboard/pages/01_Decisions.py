import pandas as pd
import requests
import streamlit as st
from ui import inject_css, hero, metric_card, plotly_bar

inject_css()

hero("Customer Decision Simulator","Explore the full counterfactual decision chain for an individual customer.",["COUNTERFACTUAL SCORING","UPLIFT","REVENUE MODEL","POLICY"])
api=st.sidebar.text_input("Production API URL","https://adaptive-customer-decision-engine.onrender.com").rstrip("/")
with st.form("decision"):
    a,b,c=st.columns(3)
    with a:
        recency=st.number_input("Recency (days)",0,1000,10)
        history_segment=st.selectbox("History segment",["1) $0 - $100","2) $100 - $200","3) $200 - $350","4) $350 - $500","5) $500 - $750","6) $750 - $1,000","7) $1,000 +"])
        history=st.number_input("Historical spend",0.0,100000.0,200.0,10.0)
    with b:
        mens=st.checkbox("Previous men's purchase"); womens=st.checkbox("Previous women's purchase"); newbie=st.checkbox("New customer")
    with c:
        zip_code=st.selectbox("ZIP group",["Surburban","Urban","Rural"]); channel=st.selectbox("Channel",["Phone","Web","Multichannel"])
    run=st.form_submit_button("Run production decision",type="primary",use_container_width=True)
if run:
    payload={"recency":recency,"history_segment":history_segment,"history":history,"mens":int(mens),"womens":int(womens),"zip_code":zip_code,"newbie":int(newbie),"channel":channel}
    try:
        r=requests.post(api+"/decision",json=payload,timeout=30); r.raise_for_status(); x=r.json()
        action=x["recommended_action"]; margin=x["decision_margin"]
        m1,m2,m3,m4=st.columns(4)
with m1: metric_card("Recommended action",action,"Highest modeled utility")
with m2: metric_card("Decision margin",f"{margin:.3f}","Gap to next-best action")
with m3: metric_card("Policy",x["policy_version"],"Frozen production policy")
with m4: metric_card("Model",x["model"],"Production response model")
        df=pd.DataFrame([{"Action":k,"Response probability":x["probabilities"][k],"Conditional spend":x["conditional_spend"][k],"Expected revenue":x["expected_revenue"][k],"Utility":x["utility"][k],"Uplift":x["uplift"].get(k,0.0)} for k in x["utility"]]).set_index("Action")
        l,rcol=st.columns(2)
        with l:
    plotly_bar(df.reset_index(),"Action","Utility","Expected utility by action")
with rcol:
    plotly_bar(df.drop(index="No E-Mail",errors="ignore").reset_index(),"Action","Uplift","Modeled uplift signal")
        st.markdown("**How to read response probability:** each value is the modeled probability of a response *if this specific action is sent*. These are separate counterfactual response probabilities, not a probability distribution across actions, so they are **not expected to sum to 100%**.")
        st.markdown("### Counterfactual response surface")
st.caption("Each probability answers a separate question: what is the response probability if this action is sent? They are not mutually exclusive action probabilities.")
plotly_bar(df.reset_index(),"Action","Response probability","Counterfactual response probability")
st.dataframe(df.style.format({"Response probability":"{:.2%}","Conditional spend":"{:.2f}","Expected revenue":"{:.3f}","Utility":"{:.3f}","Uplift":"{:.2%}"}),use_container_width=True)
        st.info(f"**Why {action}?** The frozen production policy selects the action with the highest modeled utility. The margin to the next-best action is {margin:.3f}.")
        st.caption("Expected revenue is modeled response probability × modeled conditional spend. It is not causal incremental revenue; uplift is shown separately.")
    except Exception as e: st.error(f"Production API error: {e}")
else: st.info("Enter a customer profile and run the production decision engine.")
