from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


@dataclass
class PairwiseUpliftResult:
    treatment: str
    control: str
    model: str
    uplift: np.ndarray


class _PairwiseBase:
    def __init__(self, estimator_factory: Callable[[], Any]):
        self.estimator_factory = estimator_factory
        self.fitted = False

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series, treatment_label: str, control_label: str):
        mask = treatment.isin([treatment_label, control_label])
        self.treatment_label = treatment_label
        self.control_label = control_label
        self.feature_columns = list(X.columns)
        self._fit(X.loc[mask].reset_index(drop=True), treatment.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True))
        self.fitted = True
        return self


class SLearner(_PairwiseBase):
    """S-learner: one outcome model with treatment as an explicit feature."""
    def _fit(self, X, treatment, y):
        self.model = self.estimator_factory()
        z = X.copy()
        z["__treatment__"] = (treatment == self.treatment_label).astype(int).to_numpy()
        self.model.fit(z, y)

    def predict_uplift(self, X):
        if not self.fitted:
            raise RuntimeError("Estimator has not been fitted")
        c = X.copy(); c["__treatment__"] = 0
        t = X.copy(); t["__treatment__"] = 1
        return np.asarray(self.model.predict_proba(t)[:, 1] - self.model.predict_proba(c)[:, 1], dtype=float)


class TLearner(_PairwiseBase):
    """T-learner: separate outcome models for treatment and control."""
    def _fit(self, X, treatment, y):
        self.control_model = self.estimator_factory()
        self.treatment_model = self.estimator_factory()
        c = treatment == self.control_label
        t = treatment == self.treatment_label
        self.control_model.fit(X.loc[c], y.loc[c])
        self.treatment_model.fit(X.loc[t], y.loc[t])

    def predict_uplift(self, X):
        if not self.fitted:
            raise RuntimeError("Estimator has not been fitted")
        return np.asarray(
            self.treatment_model.predict_proba(X)[:, 1] - self.control_model.predict_proba(X)[:, 1],
            dtype=float,
        )


class XLearner(_PairwiseBase):
    """X-learner for binary treatment/control pairs.

    The implementation follows the standard imputed-effect construction:
    tau_t = Y - mu_c(X) for treated observations and
    tau_c = mu_t(X) - Y for controls, followed by effect models and
    propensity-weighted combination of the two effect estimates.
    """
    def __init__(self, estimator_factory: Callable[[], Any], effect_estimator_factory: Callable[[], Any] | None = None):
        super().__init__(estimator_factory)
        self.effect_estimator_factory = effect_estimator_factory or estimator_factory

    def _fit(self, X, treatment, y):
        self.mu_c = self.estimator_factory()
        self.mu_t = self.estimator_factory()
        c = treatment == self.control_label
        t = treatment == self.treatment_label
        self.mu_c.fit(X.loc[c], y.loc[c])
        self.mu_t.fit(X.loc[t], y.loc[t])

        mu_c_t = self.mu_c.predict_proba(X.loc[t])[:, 1]
        mu_t_c = self.mu_t.predict_proba(X.loc[c])[:, 1]
        tau_t = y.loc[t].to_numpy(dtype=float) - mu_c_t
        tau_c = mu_t_c - y.loc[c].to_numpy(dtype=float)

        self.tau_t = self.effect_estimator_factory()
        self.tau_c = self.effect_estimator_factory()
        self.tau_t.fit(X.loc[t], tau_t)
        self.tau_c.fit(X.loc[c], tau_c)

        self.p_t = float(t.mean())

    def predict_uplift(self, X):
        if not self.fitted:
            raise RuntimeError("Estimator has not been fitted")
        a = np.asarray(self.tau_t.predict(X), dtype=float)
        b = np.asarray(self.tau_c.predict(X), dtype=float)
        # With randomized treatment, the empirical treatment probability is a
        # stable combination weight; callers may override it for observational data.
        return self.p_t * b + (1.0 - self.p_t) * a


def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift: np.ndarray, control_label: int = 0) -> pd.DataFrame:
    """Return a cumulative uplift/Qini curve for a binary randomized experiment."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment)
    uplift = np.asarray(uplift, dtype=float)
    order = np.argsort(-uplift, kind="mergesort")
    yt = y[order]
    tt = treatment[order]
    treated = tt != control_label
    control = ~treated
    ct = np.cumsum(treated)
    cc = np.cumsum(control)
    sy_t = np.cumsum(np.where(treated, yt, 0.0))
    sy_c = np.cumsum(np.where(control, yt, 0.0))
    uplift_gain = np.divide(sy_t, ct, out=np.zeros_like(sy_t), where=ct > 0) - np.divide(sy_c, cc, out=np.zeros_like(sy_c), where=cc > 0)
    n = np.arange(1, len(y) + 1)
    return pd.DataFrame({"rank": n, "fraction": n / len(y), "treated_n": ct, "control_n": cc, "uplift_gain": uplift_gain, "qini": uplift_gain * n})


def uplift_metrics(y: np.ndarray, treatment: np.ndarray, uplift: np.ndarray, top_fractions=(0.05, 0.10, 0.20)) -> dict:
    curve = qini_curve(y, treatment, uplift)
    area = float(np.trapezoid(curve["qini"].to_numpy(), curve["fraction"].to_numpy()))
    out = {"qini_area": area}
    for f in top_fractions:
        idx = min(len(curve) - 1, max(0, int(np.ceil(f * len(curve))) - 1))
        out[f"uplift_at_{int(f*100)}pct"] = float(curve.iloc[idx]["uplift_gain"])
    out["auuc"] = float(np.trapezoid(curve["uplift_gain"].to_numpy(), curve["fraction"].to_numpy()))
    return out
