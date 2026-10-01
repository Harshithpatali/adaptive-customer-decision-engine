from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .cross_fitting import cross_fit_potential_outcomes


@dataclass(frozen=True)
class PolicyEstimate:
    value: float
    standard_error: float
    effective_sample_size: float
    estimator: str
    outcome: str


def ipw_policy_value(y: np.ndarray, treatment: np.ndarray, policy: np.ndarray, propensity: dict[str, float]) -> float:
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment)
    policy = np.asarray(policy)
    p = np.array([propensity[a] for a in policy], dtype=float)
    return float(np.mean(y * (treatment == policy) / np.clip(p, 1e-6, None)))


def doubly_robust_policy_value(
    y: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    outcome_predictions: pd.DataFrame,
    propensity: dict[str, float],
) -> PolicyEstimate:
    """AIPW/DR value for a multi-action policy using cross-fitted m_t(X)."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment)
    policy = np.asarray(policy)
    m = outcome_predictions.to_numpy(dtype=float)
    actions = list(outcome_predictions.columns)
    action_index = {a: i for i, a in enumerate(actions)}
    mu_pi = np.array([m[i, action_index[a]] for i, a in enumerate(policy)], dtype=float)
    p = np.array([propensity[a] for a in policy], dtype=float)
    correction = (treatment == policy) * (y - mu_pi) / np.clip(p, 1e-6, None)
    influence = mu_pi + correction
    ess = float((np.sum(1.0 / np.clip(p, 1e-6, None)) ** 2) / np.sum((1.0 / np.clip(p, 1e-6, None)) ** 2)) if len(p) else 0.0
    return PolicyEstimate(float(np.mean(influence)), float(np.std(influence, ddof=1) / np.sqrt(len(influence))), ess, "doubly_robust", "outcome")


def bootstrap_policy_ci(values: np.ndarray, reps: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def evaluate_policy_dr(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    policy: pd.Series,
    actions: list[str],
    estimator_factory: Callable[[], object],
    n_splits: int = 5,
    bootstrap_reps: int = 1000,
    seed: int = 42,
) -> dict:
    m, propensity = cross_fit_potential_outcomes(X, treatment, y, actions, estimator_factory, n_splits, seed)
    estimate = doubly_robust_policy_value(y.to_numpy(), treatment.to_numpy(), policy.to_numpy(), m, propensity)
    influence = []
    yy = y.to_numpy(dtype=float); tt = treatment.to_numpy(); pp = policy.to_numpy()
    idx = {a: i for i, a in enumerate(actions)}
    mm = m.to_numpy()
    for i, a in enumerate(pp):
        p = max(propensity[a], 1e-6)
        mu = mm[i, idx[a]]
        influence.append(mu + (tt[i] == a) * (yy[i] - mu) / p)
    mean, lo, hi = bootstrap_policy_ci(np.asarray(influence), bootstrap_reps, seed)
    return {
        "value": mean,
        "ci_low": lo,
        "ci_high": hi,
        "standard_error": estimate.standard_error,
        "effective_sample_size": estimate.effective_sample_size,
        "estimator": "doubly_robust",
        "propensity": propensity,
        "cross_fit_folds": n_splits,
    }
