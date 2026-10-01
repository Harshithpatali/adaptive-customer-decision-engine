"""Offline ACDX research runner for S/T/X learners, DR evaluation,
cross-fitting, bootstrap decision stability, and causal policy frontiers.

This module never runs from the production API. It writes research artifacts
under reports/advanced_policy/ and does not modify models/production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make repository-root imports work both locally and in GitHub Actions.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.doubly_robust import evaluate_policy_dr
from src.evaluation.decision_stability import bootstrap_action_stability
from src.models.uplift.learners import SLearner, TLearner, XLearner, cross_fitted_uplift_scores, uplift_metrics
from src.policy.frontier import causal_incremental_scores, policy_frontier

DATA = ROOT / "data/raw/hillstrom_raw.csv"
OUT = ROOT / "reports/advanced_policy"
FEATURES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
NUM = ["recency", "history", "mens", "womens", "newbie"]
CAT = ["history_segment", "zip_code", "channel"]
ACTIONS = ["No E-Mail", "Womens E-Mail", "Mens E-Mail"]


def estimator_factory():
    pre = ColumnTransformer([
        ("num", Pipeline([( "imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([( "imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, C=1.0))])


def effect_estimator_factory():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=80, min_samples_leaf=20, random_state=42, n_jobs=-1))])


def value_estimator_factory():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=80, min_samples_leaf=20, random_state=42, n_jobs=-1))])


def write_final_report(df, learner_df, dr, stability, frontier, args):
    """Write a human-readable report from the exact generated research artifacts."""
    best = frontier.loc[frontier["total_net_incremental_value"].idxmax()]
    mean_stability = float(stability["decision_stability"].mean())
    high_stability = float((stability["decision_stability"] >= 0.80).mean())
    lines = [
        "# ACDX Advanced Policy Evaluation — Research Report",
        "",
        "## Executive summary",
        "",
        "This report summarizes the latest reproducible offline causal-policy research run.",
        "It is **research evidence**, not realized production revenue and not a replacement for the frozen production inference stack.",
        "",
        f"- Dataset: **{len(df):,} customers**",
        f"- Cross-fitting: **{args.folds} folds**",
        f"- Bootstrap decision-stability refits: **{args.bootstrap:,}**",
        "- Policy evaluation: **doubly robust, monetary spend/value scale**",
        "- Production artifacts modified: **No**",
        "",
        "## 1. S / T / X learner comparison",
        "",
        "The learners are evaluated out-of-fold against the randomized no-email control.",
        "",
        "| Treatment | Learner | Qini | AUUC | Uplift @ 20% |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in learner_df.iterrows():
        lines.append(f"| {row['treatment']} | {row['model']} | {row['qini']:.2f} | {row['auuc']:.5f} | {row['uplift_at_20pct']:.2%} |")
    lines += [
        "",
        "## 2. Doubly robust policy evaluation",
        "",
        f"- Estimated policy value: **{dr['value']:.3f}**",
        f"- 95% confidence interval: **{dr['ci_low']:.3f} to {dr['ci_high']:.3f}**",
        f"- Standard error: **{dr['standard_error']:.3f}**",
        f"- Effective sample size: **{dr['effective_sample_size']:,.1f}**",
        "",
        "This is a cross-fitted offline estimate on randomized data. It should not be described as observed production revenue.",
        "",
        "## 3. Causal incremental-value frontier",
        "",
        "| Contact fraction | Actual contact rate | Contacts | Mean incremental value | Total net incremental value |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in frontier.iterrows():
        lines.append(f"| {row['contact_fraction']:.0%} | {row['contact_rate']:.1%} | {int(row['contacts']):,} | ${row['mean_incremental_value']:,.2f} | ${row['total_net_incremental_value']:,.2f} |")
    lines += [
        "",
        "### Frontier summary",
        "",
        f"- Highest evaluated modeled net incremental value: **${best['total_net_incremental_value']:,.2f}**",
        f"- At modeled contact rate: **{best['contact_rate']:.1%}**",
        f"- Modeled contacts: **{int(best['contacts']):,}**",
        f"- Marginal modeled net value/contact: **${best['marginal_net_value_per_contact']:,.2f}**",
        "",
        "## 4. Individual decision stability",
        "",
        f"- Mean bootstrap decision stability: **{mean_stability:.1%}**",
        f"- Share with stability ≥ 80%: **{high_stability:.1%}**",
        f"- Evaluation population: **{len(stability):,} customers**",
        "",
        "Stability measures how consistently the selected action remains highest-value across bootstrap refits. It is a measure of decision consistency, not causal certainty.",
        "",
        "## 5. Evidence boundary",
        "",
        "**Randomized experimental evidence:** observed treatment/control comparisons.",
        "",
        "**Offline research estimates:** S/T/X uplift, cross-fitted potential outcomes, doubly robust policy value, bootstrap decision stability, and the causal value frontier.",
        "",
        "**Production:** frozen XGBoost response + T-Learner uplift + LightGBM conditional revenue + value-max policy. The production API does not run this research code.",
        "",
        "**Important:** modeled incremental value is not realized revenue. Treatment costs are policy assumptions stored in the production configuration.",
        "",
    ]
    (OUT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    X = df[FEATURES].copy()
    treatment = df["segment"].copy()
    y = df["conversion"].astype(int)
    spend = df["spend"].astype(float)
    costs = {"No E-Mail": 0.0, "Womens E-Mail": 0.02, "Mens E-Mail": 0.02}

    # Pairwise S/T/X comparison against randomized control.
    learner_rows = []
    uplift_store = {}
    for action in ["Mens E-Mail", "Womens E-Mail"]:
        mask = treatment.isin(["No E-Mail", action])
        xp = X.loc[mask].reset_index(drop=True)
        tp = treatment.loc[mask].reset_index(drop=True)
        yp = y.loc[mask].reset_index(drop=True)
        for name in ["S-Learner", "T-Learner", "X-Learner"]:
            uplift = cross_fitted_uplift_scores(
                xp, tp, yp, action, name, estimator_factory,
                effect_estimator_factory if name == "X-Learner" else None,
                n_splits=args.folds, random_state=42
            )
            metrics = uplift_metrics(yp.to_numpy(), (tp == action).astype(int).to_numpy(), uplift)
            learner_rows.append({"treatment": action, "model": name, **metrics, "n": len(xp), "evaluation": "out_of_fold"})
            uplift_store[(action, name)] = (xp, tp, uplift)
    pd.DataFrame(learner_rows).to_csv(OUT / "uplift_learner_comparison.csv", index=False)

    # Cross-fitted nuisance models + DR evaluation on monetary spend.
    # Costs are denominated in dollars, so the policy objective must also be
    # monetary; conversion probabilities are used separately for uplift research.
    from src.evaluation.cross_fitting import cross_fit_potential_outcomes
    spend_predictions, prop = cross_fit_potential_outcomes(
        X, treatment, spend, ACTIONS, value_estimator_factory, args.folds, 42, "predict"
    )
    net_values = pd.DataFrame({a: spend_predictions[a] - costs[a] for a in ACTIONS})
    policy = net_values.idxmax(axis=1)
    dr = evaluate_policy_dr(
        X, treatment, spend, policy, ACTIONS, value_estimator_factory,
        args.folds, 1000, 42, prediction_method="predict"
    )
    (OUT / "doubly_robust_policy.json").write_text(json.dumps(dr, indent=2))

    # Bootstrap decision stability. Restricting to a reproducible sample keeps
    # the research job tractable; the same method can be expanded offline.
    stability_X = X.reset_index(drop=True).iloc[: min(1000, len(X))].copy()
    stability_actions = ACTIONS
    def fit_predict_values(frame, idx, action):
        # Bootstrap each action's potential-outcome model from the rows assigned
        # to that action. The response classifier is intentionally not refit here:
        # decision stability is defined on monetary expected spend minus cost.
        boot = frame.iloc[idx]
        boot_t = treatment.reset_index(drop=True).iloc[idx]
        boot_spend = df["spend"].reset_index(drop=True).iloc[idx]
        mask = boot_t.eq(action).to_numpy()

        # A naive bootstrap can occasionally draw zero/one observations from a
        # treatment arm. Fall back to that arm's full sample for a valid,
        # reproducible potential-outcome estimate rather than fitting an
        # ill-posed model.
        if int(mask.sum()) < 2:
            full_mask = treatment.eq(action).to_numpy()
            train_X = X.reset_index(drop=True).loc[full_mask]
            train_y = spend.reset_index(drop=True).loc[full_mask]
        else:
            train_X = boot.loc[mask]
            train_y = boot_spend.loc[mask]

        value = value_estimator_factory()
        value.fit(train_X, train_y)
        return np.maximum(0.0, value.predict(stability_X)) - costs[action]
    stability = bootstrap_action_stability(stability_X, stability_actions, fit_predict_values, args.bootstrap, seed=42)
    stability.to_csv(OUT / "decision_stability.csv", index=False)

    # Causal incremental-value frontier using cross-fitted potential outcomes.
    # The control outcome is the no-email counterfactual; treatment scores are
    # incremental relative to that same baseline.
    potential = spend_predictions.copy()
    scores = causal_incremental_scores(potential, costs)
    frontier = policy_frontier(scores)
    scores.to_csv(OUT / "causal_incremental_scores.csv", index=False)
    frontier.to_csv(OUT / "policy_frontier.csv", index=False)

    manifest = {
        "dataset_rows": int(len(df)),
        "actions": ACTIONS,
        "cross_fit_folds": args.folds,
        "bootstrap_replicates": args.bootstrap,
        "uplift_models": ["S-Learner", "T-Learner", "X-Learner"],
        "decision_stability_value": "bootstrap expected spend − treatment cost",
        "policy_evaluator": "doubly_robust",
        "classification": "OFFLINE RESEARCH / MODEL ESTIMATE",
        "production_artifacts_modified": False,
    }
    (OUT / "research_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_final_report(df, learner_df, dr, stability, frontier, args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

# Reproducibility note: this file is also executable by the advanced-policy GitHub Actions workflow.
