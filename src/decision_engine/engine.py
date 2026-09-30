from pathlib import Path
import json
import pandas as pd
from joblib import load
from src.policy.decision import DecisionConfig, choose_action

ACTIONS = ["No E-Mail", "Womens E-Mail", "Mens E-Mail"]

class DecisionEngine:
    """Production-only decision engine using frozen selected artifacts."""
    def __init__(self, model_dir="models/production"):
        p = Path(model_dir)
        self.response = load(p / "response_model.joblib")  # selected XGBoost only
        self.revenue = load(p / "revenue_model.joblib")    # selected LightGBM
        self.uplift = {}
        for action in ACTIONS[1:]:
            key = action.lower().replace(" ", "_")
            self.uplift[action] = (
                load(p / f"uplift_{key}_treatment.joblib"),
                load(p / f"uplift_{key}_control.joblib"),
            )
        cfg = json.loads((p / "decision_config.json").read_text())
        self.cfg = DecisionConfig(cfg["treatment_costs"], cfg["policy_version"])
        self.registry = json.loads((p / "model_registry.json").read_text())

    def _frame(self, customer, action=None):
        x = pd.DataFrame([customer])
        if action is not None:
            x["segment"] = action
        return x

    def predict(self, customer):
        probabilities = {}
        conditional_spend = {}
        expected_revenue = {}
        for action in ACTIONS:
            x = self._frame(customer, action)
            probabilities[action] = float(self.response.predict_proba(x)[0, 1])
            conditional_spend[action] = max(0.0, float(self.revenue.predict(x)[0]))
            expected_revenue[action] = probabilities[action] * conditional_spend[action]

        decision = choose_action(expected_revenue, self.cfg)
        uplift = {}
        base = self._frame(customer)
        for action, (treatment, control) in self.uplift.items():
            uplift[action] = float(treatment.predict_proba(base)[0,1] - control.predict_proba(base)[0,1])

        return {
            **decision,
            "model": "XGBoost",
            "probabilities": probabilities,
            "conditional_spend": conditional_spend,
            "expected_revenue": expected_revenue,
            "uplift": uplift,
        }
