from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def bootstrap_action_stability(
    X: pd.DataFrame,
    actions: list[str],
    fit_predict_values: Callable[[pd.DataFrame, np.ndarray, str], np.ndarray],
    n_bootstrap: int = 200,
    sample_size: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate customer-level action stability from bootstrap refits.

    fit_predict_values(train_frame, bootstrap_indices, action) must return an
    array of action values for every row in X. The function is deliberately
    model-agnostic so the production decision engine never needs to know about
    bootstrap training.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2")
    rng = np.random.default_rng(seed)
    n = len(X)
    m = sample_size or n
    counts = np.zeros((n, len(actions)), dtype=int)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=m)
        values = np.column_stack([fit_predict_values(X, idx, a) for a in actions])
        winners = np.argmax(values, axis=1)
        counts[np.arange(n), winners] += 1
    winner_idx = counts.argmax(axis=1)
    stability = counts[np.arange(n), winner_idx] / n_bootstrap
    return pd.DataFrame({
        "recommended_action": [actions[i] for i in winner_idx],
        "decision_stability": stability,
        "bootstrap_draws": n_bootstrap,
        **{f"stability_{a}": counts[:, j] / n_bootstrap for j, a in enumerate(actions)},
    })
