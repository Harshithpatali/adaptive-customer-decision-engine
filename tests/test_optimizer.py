import pandas as pd

from src.policy.optimizer import OptimizationConstraints, optimize_customer_policy, summarize_policy


def test_optimizer_respects_budget_and_capacity():
    scores = pd.DataFrame([
        {"customer_id":"c1","action":"No E-Mail","expected_value":1.0,"treatment_cost":0.0},
        {"customer_id":"c1","action":"Mens E-Mail","expected_value":2.0,"treatment_cost":0.02},
        {"customer_id":"c1","action":"Womens E-Mail","expected_value":1.4,"treatment_cost":0.02},
        {"customer_id":"c2","action":"No E-Mail","expected_value":1.0,"treatment_cost":0.0},
        {"customer_id":"c2","action":"Mens E-Mail","expected_value":3.0,"treatment_cost":0.02},
        {"customer_id":"c2","action":"Womens E-Mail","expected_value":1.2,"treatment_cost":0.02},
    ])
    selected = optimize_customer_policy(scores, OptimizationConstraints(budget=0.02, contact_capacity=1))
    summary = summarize_policy(selected)
    assert summary["customers"] == 2
    assert summary["contacts"] == 1
    assert summary["spend"] <= 0.02 + 1e-9
    assert selected["customer_id"].nunique() == 2
