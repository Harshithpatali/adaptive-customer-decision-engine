import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ui import inject_css, hero, metric_card

inject_css()
ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports/policy/advanced_policy_metrics.json"
CAL = ROOT / "reports/model_comparison/calibration.csv"
UPLIFT = ROOT / "reports/model_comparison/uplift_models.csv"
BOOT = ROOT / "reports/policy/policy_bootstrap_ci.json"
BANDIT = ROOT / "reports/policy/bandit_simulation.json"

hero(
    "Advanced Policy Evaluation",
    "Uplift ranking, causal policy value, uncertainty, calibration, heterogeneous effects, and offline bandit simulation.",
    ["QINI", "AUUC", "IPW POLICY VALUE", "BOOTSTRAP", "CALIBRATION", "BANDITS"],
)

if not METRICS.exists():
    st.error("Advanced policy evaluation artifacts are not available.")
    st.stop()

m = json.loads(METRICS.read_text())
uplift = m["uplift"]

st.markdown("### Uplift model diagnostics")
c1,c2,c3,c4 = st.columns(4)
with c1: metric_card("Men Qini area", f'{uplift["Mens E-Mail"]["qini_area"]:.2f}', "Held-out test ranking")
with c2: metric_card("Women Qini area", f'{uplift["Womens E-Mail"]["qini_area"]:.2f}', "Held-out test ranking")
with c3: metric_card("Men AUUC", f'{uplift["Mens E-Mail"]["auuc"]:.5f}', "Area under uplift curve")
with c4: metric_card("Women AUUC", f'{uplift["Womens E-Mail"]["auuc"]:.5f}', "Area under uplift curve")

fig = go.Figure()
for action, label in [("Mens E-Mail","Men's Email"),("Womens E-Mail","Women's Email")]:
    q = pd.DataFrame(uplift[action]["curve"])
    fig.add_trace(go.Scatter(
        x=q.target_fraction * 100,
        y=q.incremental_conversion * 100,
        mode="lines+markers",
        name=label,
        line=dict(width=3),
    ))
fig.update_layout(
    title="Uplift curve — incremental conversion captured by targeting fraction",
    xaxis_title="Customers targeted (%)",
    yaxis_title="Estimated incremental conversion (percentage points)",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
st.caption("Pairwise treatment-vs-control uplift is evaluated on the held-out test split using inverse-propensity weighting. Qini/AUUC are ranking metrics, not production causal guarantees.")

st.markdown("### Policy value frontier")
pv = pd.DataFrame(m["policy_value_curve"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=pv.target_fraction*100,y=pv.policy_value,mode="lines+markers",name="Model policy",line=dict(width=4)))
for name,value in m["baseline_policy_value"].items():
    fig.add_hline(y=value,line_dash="dash",annotation_text=name,annotation_position="top left")
fig.update_layout(
    title="Offline policy value vs targeting fraction",
    xaxis_title="Targeting fraction (%)",
    yaxis_title="IPW estimated net spend value / customer",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
st.caption("Policy value uses inverse-propensity weighting on randomized held-out customers. It is an offline estimate, not observed production revenue.")

st.markdown("### Baseline policy comparison")
base = pd.DataFrame({"Policy":list(m["baseline_policy_value"].keys()),"Value":list(m["baseline_policy_value"].values())})
fig = px.bar(base,x="Policy",y="Value",text_auto=".3f",title="Reference policy values",template="plotly_dark")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

st.markdown("### Heterogeneous treatment-effect profile")
qrows=[]
for action, q in m["treatment_effect_distribution"].items():
    for quantile,value in q.items():
        qrows.append({"Action":action,"Quantile":float(quantile)*100,"Estimated uplift":value*100})
qdf=pd.DataFrame(qrows)
fig=px.line(qdf,x="Quantile",y="Estimated uplift",color="Action",markers=True,title="Individual uplift score quantiles",template="plotly_dark")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Population percentile",yaxis_title="Predicted conversion uplift (percentage points)")
st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
st.caption("This is a quantile profile of modeled individual uplift scores, not a histogram of raw individual effects.")

st.markdown("### Calibration")
if CAL.exists():
    cal=pd.read_csv(CAL)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=cal["method"],y=cal["brier"],name="Brier score"))
    fig.add_trace(go.Bar(x=cal["method"],y=cal["log_loss"],name="Log loss"))
    fig.update_layout(barmode="group",title="Calibration diagnostics",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis_title="Metric")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.dataframe(cal.style.format({"brier":"{:.5f}","log_loss":"{:.5f}","roc_auc":"{:.4f}"}),use_container_width=True,hide_index=True)

st.markdown("### Bootstrap uncertainty")
if BOOT.exists():
    b=json.loads(BOOT.read_text())
    labels=["Conversion value","Spend value"]
    means=[(b["conversion_ci"][0]+b["conversion_ci"][1])/2,(b["spend_ci"][0]+b["spend_ci"][1])/2]
    lows=[means[0]-b["conversion_ci"][0],means[1]-b["spend_ci"][0]]
    highs=[b["conversion_ci"][1]-means[0],b["spend_ci"][1]-means[1]]
    fig=go.Figure(go.Bar(x=labels,y=means,error_y=dict(type="data",array=highs,arrayminus=lows,visible=True)))
    fig.update_layout(title=f'Bootstrap uncertainty ({b["bootstrap_replicates"]} replicates)',template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.json(b)

st.markdown("### Offline contextual-bandit simulation")
if BANDIT.exists():
    band=json.loads(BANDIT.read_text())
    names=[]; rewards=[]; regrets=[]
    for name,r in band["results"].items():
        names.append(name.replace("_"," ").title()); rewards.append(r["cumulative_reward"]); regrets.append(r["cumulative_regret"])
    fig=go.Figure()
    fig.add_trace(go.Bar(x=names,y=rewards,name="Cumulative reward"))
    fig.add_trace(go.Bar(x=names,y=regrets,name="Cumulative regret"))
    fig.update_layout(barmode="group",title="500-round offline bandit simulation",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    alloc=[]
    for name,r in band["results"].items():
        for action,count in r["action_allocation"].items():
            alloc.append({"Algorithm":name.replace("_"," ").title(),"Action":action,"Count":count})
    adf=pd.DataFrame(alloc)
    fig=px.bar(adf,x="Algorithm",y="Count",color="Action",barmode="stack",title="Bandit action allocation",template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.warning(band["note"])

st.markdown("### Evaluation contract")
st.markdown("""
**What is causal:** treatment/control differences and IPW estimates rely on the randomized Hillstrom assignment.

**What is model-dependent:** individual uplift scores, policy rankings, expected revenue and decision recommendations depend on fitted models.

**What is simulated:** Thompson Sampling, epsilon-greedy and LinUCB results are offline simulations and must not be presented as observed production performance.

**Production boundary:** none of this research-time evaluation is loaded into the FastAPI inference service.
""")
