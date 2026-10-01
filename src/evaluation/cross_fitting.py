from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def cross_fit_potential_outcomes(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    actions: list[str],
    estimator_factory: Callable[[], object],
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Cross-fitted nuisance outcome predictions m_t(X).

    The estimator for each action is trained without using the row whose
    prediction is produced. For the Hillstrom randomized experiment, the
    propensity is estimated from the training fold; because assignment is
    randomized this is a design-probability diagnostic, not a learned policy.
    """
    X = X.reset_index(drop=True)
    treatment = treatment.reset_index(drop=True)
    y = y.reset_index(drop=True)
    pred = pd.DataFrame(index=np.arange(len(X)), columns=actions, dtype=float)
    prop = {a: float((treatment == a).mean()) for a in actions}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in splitter.split(X, treatment):
        tr = X.iloc[train_idx]
        va = X.iloc[valid_idx]
        tt = treatment.iloc[train_idx]
        yy = y.iloc[train_idx]
        for action in actions:
            mask = tt.eq(action).to_numpy()
            if mask.sum() < 2:
                raise ValueError(f"Insufficient observations for action {action!r} in a cross-fit fold")
            model = estimator_factory()
            model.fit(tr.loc[mask], yy.loc[mask])
            pred.loc[valid_idx, action] = model.predict_proba(va)[:, 1]

    return pred, prop
