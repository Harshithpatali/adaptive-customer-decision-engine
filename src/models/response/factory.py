"""Research-time response model factory. Production must load frozen artifacts only."""
from __future__ import annotations

from typing import Any


def candidate_model_names() -> list[str]:
    return ["logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]


def production_model_name() -> str:
    return "xgboost"


def build_candidate(name: str, **kwargs: Any) -> Any:
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, **kwargs)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42, n_jobs=1, **kwargs)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=42, n_jobs=1, **kwargs)
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=42, **kwargs)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, **kwargs)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **kwargs)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=42, verbose=False, **kwargs)
    raise ValueError(f"Unknown candidate: {name}")
