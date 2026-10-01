import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import inject_css, hero, plotly_bar, metric_card

inject_css()
ROOT = Path(__file__).resolve().parents[2]
hero(
    "Policy Explorer",
    "Connect observed campaign economics with the offline causal value frontier used for research-time allocation analysis.",
    ["OBSERVED ECONOMICS", "CAUSAL VALUE", "CONTACT FRONTIER"],
)

econ_path = ROOT / "reports/business/campaign_economics.csv"
frontier_path = ROOT / "reports/advanced_policy/policy_frontier.csv"

if not econ_path.exists():
    st.warning("Campaign economics report is unavailable.")
    st.stop()

df = pd.read_csv(econ_path)

st.markdown("### Evidence boundary")
a, b = st.columns(2)
with a:
    metric_card("Observed economics", "Experiment arms", "Actual Hillstrom outcomes")
with b:
    metric_card("Causal frontier", "Offline", "Modeled incremental value")

st.markdown("### Observed campaign economics")
st.dataframe(
    df[
        [
            "action",
            "customers",
            "visits",
            "conversions",
            "observed_spend",
            "campaign_cost",
            "net_observed_value",
            "revenue_per_customer",
            "conversion_rate",
        ]
    ].style.format(
        {
            "observed_spend": "{:,.2f}",
            "campaign_cost": "{:,.2f}",
            "net_observed_value": "{:,.2f}",
            "revenue_per_customer": "{:.3f}",
            "conversion_rate": "{:.2%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)
plotly_bar(df, "action", "net_observed_value", "Observed net value by campaign")

st.caption(
    "Observed economics describe what happened in each randomized experiment arm. "
    "They are not individualized off-policy results."
)

if frontier_path.exists():
    frontier = pd.read_csv(frontier_path)

    st.markdown("### Causal incremental-value frontier")
    max_row = frontier.loc[frontier["total_net_incremental_value"].idxmax()]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card(
            "Maximum modeled net value",
            "$" + "{:,.0f}".format(max_row["total_net_incremental_value"]),
            "Within evaluated contact fractions",
        )
    with c2:
        metric_card(
            "Modeled contact rate",
            "{:.1%}".format(max_row["contact_rate"]),
            "{:,.0f} customers".format(max_row["contacts"]),
        )
    with c3:
        metric_card(
            "Marginal value/contact",
            "$" + "{:.2f}".format(max_row["marginal_net_value_per_contact"]),
            "At that frontier point",
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["contact_rate"] * 100,
            y=frontier["total_net_incremental_value"],
            mode="lines+markers",
            name="Net incremental value",
            line=dict(width=4),
        )
    )
    fig.update_layout(
        title="Modeled net incremental value vs contact rate",
        xaxis_title="Modeled customers contacted (%)",
        yaxis_title="Total modeled net incremental value",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.dataframe(
        frontier.style.format(
            {
                "contact_fraction": "{:.0%}",
                "contact_rate": "{:.1%}",
                "mean_incremental_value": "$" + "{:.2f}",
                "total_incremental_value": "$" + "{:,.0f}",
                "total_treatment_cost": "$" + "{:,.0f}",
                "total_net_incremental_value": "$" + "{:,.0f}",
                "marginal_net_value_per_contact": "$" + "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The frontier is a research-time model estimate. It ranks customers by modeled "
        "incremental net value relative to No E-Mail; it is not observed production revenue."
    )
else:
    st.info(
        "Advanced causal frontier results are not committed yet. Run the Advanced Policy Research workflow."
    )

cfg = ROOT / "models/production/decision_config.json"
if cfg.exists():
    st.markdown("### Production policy configuration")
    st.json(json.loads(cfg.read_text()))
