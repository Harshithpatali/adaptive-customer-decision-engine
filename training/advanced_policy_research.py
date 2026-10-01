"""Offline ACDX research runner for S/T/X learners, DR evaluation,
cross-fitting, bootstrap decision stability, and causal policy frontiers.

This module never runs from the production API. It writes research artifacts
under reports/advanced_policy/ and does not modify models/production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.doubly_robust import evaluate_policy_dr
from src.evaluation.decision_stability import bootstrap_action_stability
from src.models.uplift.learners import SLearner, TLearner, XLearner, uplift_metrics
from src.policy.frontier import causal_incremental_scores, policy_frontier

ROOT = Path(__file__).resolve().parents[1]
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
    costs = {"No E-Mail": 0.0, "Womens E-Mail": 0.02, "Mens E-Mail": 0.02}

    # Pairwise S/T/X comparison against randomized control.
    learner_rows = []
    uplift_store = {}
    for action in ["Mens E-Mail", "Womens E-Mail"]:
        mask = treatment.isin(["No E-Mail", action])
        xp = X.loc[mask].reset_index(drop=True)
        tp = treatment.loc[mask].reset_index(drop=True)
        yp = y.loc[mask].reset_index(drop=True)
        for name, cls in [("S-Learner", SLearner), ("T-Learner", TLearner), ("X-Learner", XLearner)]:
            model = cls(estimator_factory).fit(xp, tp, yp, action, "No E-Mail")
            uplift = model.predict_uplift(xp)
            metrics = uplift_metrics(yp.to_numpy(), (tp == action).astype(int).to_numpy(), uplift)
            learner_rows.append({"treatment": action, "model": name, **metrics, "n": len(xp)})
            uplift_store[(action, name)] = (xp, tp, uplift)
    pd.DataFrame(learner_rows).to_csv(OUT / "uplift_learner_comparison.csv", index=False)

    # Cross-fitted nuisance models + DR evaluation of a simple X-learner policy.
    # Fit pairwise X-learners on the full research sample to define the policy,
    # while the DR outcome models remain strictly cross-fitted.
    pair_uplift = {}
    for action in ["Mens E-Mail", "Womens E-Mail"]:
        model = XLearner(estimator_factory).fit(X, treatment, y, action, "No E-Mail")
        pair_uplift[action] = model.predict_uplift(X)
    potential = pd.DataFrame({"No E-Mail": np.zeros(len(df))})
    # Use cross-fitted response probabilities for a decision policy rather than
    # in-sample outcome predictions.
    from src.evaluation.cross_fitting import cross_fit_potential_outcomes
    m, prop = cross_fit_potential_outcomes(X, treatment, y, ACTIONS, estimator_factory, args.folds, 42)
    policy = pd.Series(np.where(m["Mens E-Mail"] >= m["Womens E-Mail"], "Mens E-Mail", "Womens E-Mail"), index=df.index)
    policy = pd.Series(np.where(m["Mens E-Mail"].to_numpy() < m["No E-Mail"].to_numpy() + costs["Mens E-Mail"], "No E-Mail", policy), index=df.index)
    policy = pd.Series(np.where(m["Womens E-Mail"].to_numpy() < m["No E-Mail"].to_numpy() + costs["Womens E-Mail"], "No E-Mail", policy), index=df.index)
    dr = evaluate_policy_dr(X, treatment, y, policy, ACTIONS, estimator_factory, args.folds, 1000, 42)
    (OUT / "doubly_robust_policy.json").write_text(json.dumps(dr, indent=2))

    # Bootstrap decision stability. Restricting to a reproducible sample keeps
    # the research job tractable; the same method can be expanded offline.
    stability_X = X.reset_index(drop=True).iloc[: min(1000, len(X))].copy()
    stability_actions = ACTIONS
    def fit_predict_values(frame, idx, action):
        boot = frame.iloc[idx]
        boot_t = treatment.reset_index(drop=True).iloc[idx]
        boot_y = y.reset_index(drop=True).iloc[idx]
        mask = boot_t.eq(action).to_numpy()
        model = estimator_factory()
        model.fit(boot.loc[mask], boot_y.loc[mask])
        return model.predict_proba(stability_X)[:, 1]
    stability = bootstrap_action_stability(stability_X, stability_actions, fit_predict_values, args.bootstrap, seed=42)
    stability.to_csv(OUT / "decision_stability.csv", index=False)

    # Causal incremental-value frontier using cross-fitted potential outcomes.
    # The control outcome is the no-email counterfactual; treatment scores are
    # incremental relative to that same baseline.
    potential = m.copy()
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
        "policy_evaluator": "doubly_robust",
        "classification": "OFFLINE RESEARCH / MODEL ESTIMATE",
        "production_artifacts_modified": False,
    }
    (OUT / "research_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
