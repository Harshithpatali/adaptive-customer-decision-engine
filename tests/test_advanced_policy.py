import numpy as np
import pandas as pd

from src.evaluation.doubly_robust import doubly_robust_policy_value
from src.models.uplift.learners import uplift_metrics
from src.policy.frontier import causal_incremental_scores, policy_frontier


def test_dr_value_matches_constant_outcome_when_no_residual():
    y = np.array([1, 0, 1, 0], dtype=float)
    treatment = np.array(["A", "B", "A", "B"])
    policy = np.array(["A", "B", "A", "B"])
    m = pd.DataFrame({"A": [1, 0, 1, 0], "B": [1, 0, 1, 0]})
    est = doubly_robust_policy_value(y, treatment, policy, m, {"A": .5, "B": .5})
    assert abs(est.value - .5) < 1e-12


def test_uplift_metrics_are_finite():
    out = uplift_metrics(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), np.array([.4, .1, .3, .2]))
    assert np.isfinite(out["qini_area"])
    assert np.isfinite(out["auuc"])


def test_causal_frontier_is_incremental_to_control():
    po = pd.DataFrame({"No E-Mail": [1.0, 1.0], "Womens E-Mail": [1.2, .8], "Mens E-Mail": [1.5, 1.1]})
    scores = causal_incremental_scores(po, {"No E-Mail": 0.0, "Womens E-Mail": .02, "Mens E-Mail": .02})
    m = scores[(scores.customer_id == 0) & (scores.action == "Mens E-Mail")].iloc[0]
    assert abs(m.incremental_value - .5) < 1e-12
    frontier = policy_frontier(scores, (.5, 1.0))
    assert len(frontier) == 2
    assert frontier.iloc[0].total_net_incremental_value >= 0


def test_s_t_x_learners_produce_finite_uplift():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from src.models.uplift.learners import SLearner, TLearner, XLearner
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
    t = pd.Series(np.where(rng.random(120) < .5, "Mens E-Mail", "No E-Mail"))
    y = pd.Series(((X.x1 + .7 * (t == "Mens E-Mail") + rng.normal(scale=.8, size=120)) > .3).astype(int))
    factory = lambda: LogisticRegression(max_iter=1000)
    reg_factory = lambda: RandomForestRegressor(n_estimators=20, min_samples_leaf=3, random_state=42)
    for cls in (SLearner, TLearner):
        model = cls(factory).fit(X, t, y, "Mens E-Mail", "No E-Mail")
        assert np.isfinite(model.predict_uplift(X)).all()
    model = XLearner(factory, reg_factory).fit(X, t, y, "Mens E-Mail", "No E-Mail")
    assert np.isfinite(model.predict_uplift(X)).all()
