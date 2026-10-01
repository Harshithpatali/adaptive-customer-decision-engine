import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from ui import inject_css, hero

inject_css()

API_URL = st.sidebar.text_input(
    "Production API URL",
    "https://adaptive-customer-decision-engine.onrender.com",
).rstrip("/")

hero(
    "AI Stakeholder Explainer",
    "Turn the actual ACDX experiment, policy, and model outputs into business findings and actions.",
    ["ACTUAL FINDINGS", "BUSINESS IMPACT", "DECISION SUPPORT", "AUDITABLE"],
)

st.info(
    "This page now builds its evidence automatically from the committed ACDX reports "
    "and production model registry. You do not need to paste an evidence JSON."
)

ROOT = Path(__file__).resolve().parents[2]


def load_json(path):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evidence():
    effects = pd.read_csv(ROOT / "reports/statistics/experiment_effects.csv")
    economics = pd.read_csv(ROOT / "reports/business/campaign_economics.csv")
    manifest = load_json("reports/statistics/experiment_manifest.json")
    policy = load_json("reports/policy/advanced_policy_metrics.json")
    bootstrap = load_json("reports/policy/policy_bootstrap_ci.json")
    bandit = load_json("reports/policy/bandit_simulation.json")
    registry = load_json("models/production/model_registry.json")

    findings = {}
    for action in ["Mens E-Mail", "Womens E-Mail"]:
        conv = effects[
            (effects["treatment"] == action) & (effects["outcome"] == "conversion")
        ].iloc[0]
        spend = effects[
            (effects["treatment"] == action) & (effects["outcome"] == "spend")
        ].iloc[0]
        row = economics[economics["action"] == action].iloc[0]

        treatment_customers = int(row["customers"])
        spend_lift = float(spend["absolute_lift"])
        campaign_cost = float(row["campaign_cost"])

        # Experimental mean-difference estimate scaled to the treatment-arm size.
        estimated_incremental_spend = spend_lift * treatment_customers
        estimated_incremental_net_value = estimated_incremental_spend - campaign_cost

        findings[action] = {
            "customers": treatment_customers,
            "conversion_rate": float(conv["treatment_rate"]),
            "control_conversion_rate": float(conv["control_rate"]),
            "absolute_conversion_lift": float(conv["absolute_lift"]),
            "relative_conversion_lift": float(conv["relative_lift"]),
            "conversion_bootstrap_ci": [
                float(conv["bootstrap_ci_low"]),
                float(conv["bootstrap_ci_high"]),
            ],
            "conversion_p_value": float(conv["p_value"]),
            "spend_per_customer": float(spend["treatment_mean"]),
            "control_spend_per_customer": float(spend["control_mean"]),
            "absolute_spend_lift_per_customer": spend_lift,
            "spend_bootstrap_ci": [
                float(spend["bootstrap_ci_low"]),
                float(spend["bootstrap_ci_high"]),
            ],
            "spend_p_value": float(spend["mann_whitney_p_value"]),
            "campaign_cost_total": campaign_cost,
            "estimated_incremental_spend_at_treatment_arm_size": estimated_incremental_spend,
            "estimated_incremental_net_value_after_campaign_cost": estimated_incremental_net_value,
            "observed_total_spend": float(row["observed_spend"]),
            "observed_net_value": float(row["net_observed_value"]),
        }

    return {
        "evidence_version": "ACDX committed reports",
        "evidence_types": [
            "RANDOMIZED_EXPERIMENT",
            "OBSERVED_BUSINESS_TOTALS",
            "MODEL_ESTIMATE",
            "SIMULATION",
        ],
        "experiment": {
            "population": int(manifest["dataset_rows"]),
            "arms": manifest["treatment_arms"],
            "conversion_total": manifest["conversion_total"],
            "visit_total": manifest["visit_total"],
            "campaign_cost_per_contact": manifest["configured_campaign_costs"],
            "finding": (
                "The Hillstrom randomized experiment directly supports average "
                "treatment-effect comparisons against No E-Mail."
            ),
        },
        "campaign_findings": findings,
        "policy_evaluation": {
            "evaluation_population": policy["evaluation_population"],
            "source_split": policy["source_split"],
            "mens_uplift_at_20pct": policy["uplift"]["Mens E-Mail"]["uplift_at_20pct"],
            "womens_uplift_at_20pct": policy["uplift"]["Womens E-Mail"]["uplift_at_20pct"],
            "mens_qini_area": policy["uplift"]["Mens E-Mail"]["qini_area"],
            "womens_qini_area": policy["uplift"]["Womens E-Mail"]["qini_area"],
            "policy_value_at_20pct": next(
                x["policy_value"]
                for x in policy["policy_value_curve"]
                if x["target_fraction"] == 0.2
            ),
            "baseline_policy_value": policy["baseline_policy_value"],
            "treatment_effect_quantiles": policy["treatment_effect_distribution"],
            "classification": "MODEL_ESTIMATE / OFFLINE POLICY EVALUATION",
        },
        "uncertainty": {
            "policy_bootstrap": bootstrap,
            "model_response_metrics": registry["reported_response_metrics"],
        },
        "production_policy": registry["policy"],
        "production_models": {
            "response": registry["production_response_model"],
            "uplift": registry["uplift_model"],
            "revenue": registry["revenue_model"],
        },
        "simulation": bandit,
    }


try:
    evidence = build_evidence()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Experiment population", f'{evidence["experiment"]["population"]:,}')
    with c2:
        st.metric(
            "Men conversion lift",
            f'{evidence["campaign_findings"]["Mens E-Mail"]["absolute_conversion_lift"]:.2%}',
        )
    with c3:
        st.metric(
            "Men spend lift / customer",
            f'$ {evidence["campaign_findings"]["Mens E-Mail"]["absolute_spend_lift_per_customer"]:.2f}',
        )
    with c4:
        st.metric(
            "Men estimated net value",
            f'$ {evidence["campaign_findings"]["Mens E-Mail"]["estimated_incremental_net_value_after_campaign_cost"]:,.0f}',
        )

    st.markdown("### Evidence loaded into the AI")
    st.caption(
        "The values below are computed from the repository's randomized experiment, "
        "business economics, policy-evaluation, and production-model artifacts."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Randomized experiment findings**")
        for action, item in evidence["campaign_findings"].items():
            st.write(
                f"**{action}:** conversion lift {item['absolute_conversion_lift']:.2%} "
                f"(p={item['conversion_p_value']:.3g}); spend lift "
                f"$ {item['absolute_spend_lift_per_customer']:.2f}/customer."
            )

    with right:
        st.markdown("**Offline policy evidence**")
        st.write(
            f"At 20% targeting, modeled incremental conversion is "
            f"{evidence['policy_evaluation']['mens_uplift_at_20pct']:.2%} for Men's "
            f"and {evidence['policy_evaluation']['womens_uplift_at_20pct']:.2%} "
            "for Women's Email."
        )
        st.write(
            f"Modeled policy value at 20% targeting: "
            f"{evidence['policy_evaluation']['policy_value_at_20pct']:.3f}."
        )

    if st.button("Generate business explanation", type="primary", use_container_width=True):
        try:
            r = requests.post(
                API_URL + "/explain",
                json={"evidence": evidence},
                timeout=60,
            )
            r.raise_for_status()
            result = r.json()
            explanation = result["explanation"]

            st.markdown("## Executive finding")
            st.success(explanation["executive_summary"])

            st.markdown("## What we found")
            for item in explanation["what_we_found"]:
                st.markdown(f"- {item}")

            st.markdown("## What it means")
            for item in explanation["what_it_means"]:
                st.markdown(f"- {item}")

            st.markdown("## Business impact")
            for item in explanation["business_impact"]:
                st.markdown(f"- {item}")

            st.markdown("## Recommended actions")
            for item in explanation["recommended_actions"]:
                st.markdown(f"- {item}")

            st.markdown("## Uncertainty & limits")
            for item in explanation["uncertainty_and_limits"]:
                st.markdown(f"- {item}")

            st.markdown("## Do not conclude")
            for item in explanation["do_not_conclude"]:
                st.markdown(f"- {item}")

            st.caption("The AI is interpreting supplied evidence; it is not recalculating or changing the underlying metrics.")

        except Exception as exc:
            st.error(f"Explainer unavailable: {exc}")

except Exception as exc:
    st.error(f"Could not load ACDX evidence reports: {exc}")
