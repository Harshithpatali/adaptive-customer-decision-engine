import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import inject_css, hero, metric_card

inject_css()
ROOT = Path(__file__).resolve().parents[2]
ADV = ROOT / "reports/advanced_policy"

UPLIFT = ADV / "uplift_learner_comparison.csv"
DR = ADV / "doubly_robust_policy.json"
STABILITY = ADV / "decision_stability.csv"
SCORES = ADV / "causal_incremental_scores.csv"
FRONTIER = ADV / "policy_frontier.csv"
MANIFEST = ADV / "research_manifest.json"
REPORT = ADV / "final_report.md"

CAL = ROOT / "reports/model_comparison/calibration.csv"
LEGACY = ROOT / "reports/policy/advanced_policy_metrics.json"
BOOT = ROOT / "reports/policy/policy_bootstrap_ci.json"
BANDIT = ROOT / "reports/policy/bandit_simulation.json"

hero(
    "Advanced Policy Evaluation",
    "Research-grade causal decision analysis kept separate from frozen production inference.",
    ["S / T / X LEARNERS", "CROSS-FITTING", "DOUBLY ROBUST", "STABILITY", "CAUSAL VALUE", "FRONTIER"],
)

if not UPLIFT.exists() or not DR.exists() or not FRONTIER.exists():
    st.warning("Advanced research results are not committed yet. Run the Advanced Policy Research workflow.")
    if not LEGACY.exists():
        st.stop()

st.markdown("### Evidence classification")
a, b, c = st.columns(3)
with a:
    metric_card("Experimental evidence", "Randomized", "Hillstrom treatment/control assignment")
with b:
    metric_card("Research estimates", "Offline", "Cross-fitted models + causal estimators")
with c:
    metric_card("Production", "Frozen", "No research training/search at inference")

st.info(
    "Experimental evidence supports average treatment-effect comparisons. "
    "The advanced research layer estimates heterogeneous effects and policy value offline. "
    "These research estimates do not replace the production XGBoost/T-Learner/LightGBM stack."
)

if UPLIFT.exists():
    st.markdown("## S / T / X Learner comparison")
    uplift_df = pd.read_csv(UPLIFT)
    uplift_df["treatment"] = uplift_df["treatment"].replace({
        "Mens E-Mail": "Men's Email",
        "Womens E-Mail": "Women's Email",
    })
    cols = [c for c in ["treatment", "model", "qini", "auuc", "uplift_at_20pct", "n", "evaluation"] if c in uplift_df.columns]
    st.dataframe(
        uplift_df[cols].style.format({
            "qini": "{:.2f}",
            "auuc": "{:.5f}",
            "uplift_at_20pct": "{:.2%}",
            "n": "{:,}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    fig = px.bar(
        uplift_df,
        x="model",
        y="qini",
        color="treatment",
        barmode="group",
        title="Out-of-fold Qini comparison",
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("S/T/X results are out-of-fold research estimates. Qini/AUUC are ranking metrics, not production guarantees.")

st.markdown("## Doubly robust policy evaluation")
if DR.exists():
    dr = json.loads(DR.read_text())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("DR policy value", "{:.3f}".format(dr["value"]), "Estimated mean spend/value")
    with c2:
        metric_card("95% CI", "{:.3f} – {:.3f}".format(dr["ci_low"], dr["ci_high"]), "Offline cross-fitted estimate")
    with c3:
        metric_card("Standard error", "{:.3f}".format(dr["standard_error"]), "Sampling uncertainty")
    with c4:
        metric_card("Effective N", "{:,.0f}".format(dr["effective_sample_size"]), "Weighted evaluation population")
    st.caption("Doubly robust evaluation is an offline model estimate on randomized data, not observed production revenue.")

st.markdown("## Causal incremental-value frontier")
if FRONTIER.exists():
    frontier = pd.read_csv(FRONTIER)
    max_row = frontier.loc[frontier["total_net_incremental_value"].idxmax()]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Peak modeled net value", "$" + "{:,.0f}".format(max_row["total_net_incremental_value"]), "Within evaluated frontier")
    with c2:
        metric_card("Peak contact rate", "{:.1%}".format(max_row["contact_rate"]), "{:,.0f} modeled contacts".format(max_row["contacts"]))
    with c3:
        metric_card("Marginal value/contact", "$" + "{:.2f}".format(max_row["marginal_net_value_per_contact"]), "At selected frontier point")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["contact_rate"] * 100,
        y=frontier["total_net_incremental_value"],
        mode="lines+markers",
        name="Total net incremental value",
        line=dict(width=4),
    ))
    fig.update_layout(
        title="Causal incremental-value frontier",
        xaxis_title="Actual modeled contact rate (%)",
        yaxis_title="Total modeled net incremental value",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(
        frontier.style.format({
            "contact_fraction": "{:.0%}",
            "contact_rate": "{:.1%}",
            "mean_incremental_value": "$" + "{:.2f}",
            "total_incremental_value": "$" + "{:,.0f}",
            "total_treatment_cost": "$" + "{:,.0f}",
            "total_net_incremental_value": "$" + "{:,.0f}",
            "marginal_net_value_per_contact": "$" + "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The frontier ranks customers by modeled causal incremental net value relative to No E-Mail. It is an offline policy estimate, not realized revenue.")

st.markdown("## Individual decision stability")
if STABILITY.exists():
    stability = pd.read_csv(STABILITY)
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Mean stability", "{:.1%}".format(stability["decision_stability"].mean()), "Across evaluated customers")
    with c2:
        metric_card("High stability ≥ 80%", "{:.1%}".format((stability["decision_stability"] >= 0.80).mean()), "Bootstrap-consistent decisions")
    with c3:
        metric_card("Low stability < 60%", "{:.1%}".format((stability["decision_stability"] < 0.60).mean()), "Competing-action uncertainty")

    fig = px.histogram(
        stability,
        x="decision_stability",
        nbins=20,
        title="Bootstrap decision stability distribution",
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_title="Decision stability",
        yaxis_title="Customers",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("Stability is the fraction of bootstrap refits in which the reported action was highest-value. It measures decision consistency, not causal certainty.")

st.markdown("## Customer-level causal scores")
if SCORES.exists():
    scores = pd.read_csv(SCORES)
    customer_id = st.number_input(
        "Research customer index",
        min_value=int(scores["customer_id"].min()),
        max_value=int(scores["customer_id"].max()),
        value=int(scores["customer_id"].min()),
        step=1,
    )
    customer_scores = scores[scores["customer_id"] == customer_id].copy()
    if not customer_scores.empty:
        st.dataframe(
            customer_scores.style.format({
                "potential_value": "$" + "{:.2f}",
                "incremental_value": "$" + "{:.2f}",
                "treatment_cost": "$" + "{:.2f}",
                "net_incremental_value": "$" + "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        best = customer_scores.loc[customer_scores["net_incremental_value"].idxmax()]
        st.success(
            "Highest modeled causal net value for research customer {}: {} at {}.".format(
                customer_id, best["action"], "$" + "{:.2f}".format(best["net_incremental_value"])
            )
        )

if MANIFEST.exists():
    with st.expander("Research provenance"):
        st.json(json.loads(MANIFEST.read_text()))

st.markdown("## Production boundary")
st.markdown(
    "**Production:** XGBoost response + T-Learner uplift + LightGBM conditional revenue + frozen value-max policy.\n\n"
    "**Research:** S/T/X comparison + cross-fitting + doubly robust evaluation + bootstrap decision stability + causal incremental-value frontier.\n\n"
    "**Experimental evidence:** randomized treatment/control differences from Hillstrom.\n\n"
    "**Simulation:** contextual-bandit results are offline simulations and are not production performance."
)

if CAL.exists():
    with st.expander("Calibration diagnostics"):
        st.dataframe(pd.read_csv(CAL), use_container_width=True, hide_index=True)

if BOOT.exists():
    with st.expander("Legacy bootstrap policy CI"):
        st.json(json.loads(BOOT.read_text()))

if BANDIT.exists():
    with st.expander("Legacy contextual-bandit simulation"):
        st.json(json.loads(BANDIT.read_text()))
