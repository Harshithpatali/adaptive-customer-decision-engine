from __future__ import annotations

import numpy as np
import pandas as pd

ACTIONS = ["No E-Mail", "Womens E-Mail", "Mens E-Mail"]


def causal_incremental_scores(
    potential_outcomes: pd.DataFrame,
    treatment_costs: dict[str, float],
    control_action: str = "No E-Mail",
) -> pd.DataFrame:
    """Convert potential-outcome estimates into causal incremental net-value scores."""
    missing = set(ACTIONS) - set(potential_outcomes.columns)
    if missing:
        raise ValueError(f"Missing potential outcomes: {sorted(missing)}")
    base = potential_outcomes[control_action].to_numpy(dtype=float)
    rows = []
    for i in potential_outcomes.index:
        for action in ACTIONS:
            po = float(potential_outcomes.loc[i, action])
            incremental = po - float(potential_outcomes.loc[i, control_action])
            cost = float(treatment_costs.get(action, 0.0))
            rows.append({"customer_id": i, "action": action, "potential_value": po, "incremental_value": incremental, "treatment_cost": cost, "net_incremental_value": incremental - cost})
    return pd.DataFrame(rows)


def policy_frontier(scores: pd.DataFrame, fractions: tuple[float, ...] = (.05, .10, .20, .30, .40, .50, .75, 1.0)) -> pd.DataFrame:
    """Build a contact-rate/value frontier by targeting the highest net incremental value."""
    required = {"customer_id", "action", "net_incremental_value", "incremental_value", "treatment_cost"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Missing frontier columns: {sorted(missing)}")
    rows = []
    treatment = scores.loc[scores.action.ne("No E-Mail")].copy()
    best = treatment.sort_values("net_incremental_value", ascending=False).drop_duplicates("customer_id")
    best = best.loc[best.net_incremental_value > 0].reset_index(drop=True)
    for f in fractions:
        k = min(len(best), max(0, int(round(f * scores.customer_id.nunique()))))
        chosen = best.head(k)
        rows.append({
            "contact_fraction": float(f),
            "contacts": int(k),
            "contact_rate": float(k / max(scores.customer_id.nunique(), 1)),
            "mean_incremental_value": float(chosen.incremental_value.mean()) if len(chosen) else 0.0,
            "total_incremental_value": float(chosen.incremental_value.sum()),
            "total_treatment_cost": float(chosen.treatment_cost.sum()),
            "total_net_incremental_value": float(chosen.net_incremental_value.sum()),
            "marginal_net_value_per_contact": float(chosen.net_incremental_value.iloc[-1]) if len(chosen) else 0.0,
        })
    return pd.DataFrame(rows)
