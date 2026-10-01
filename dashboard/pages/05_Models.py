import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ui import inject_css, hero, metric_card

inject_css()
ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "models/production/model_registry.json"
TRAIN = ROOT / "models/production/training_manifest.json"

hero("Model Intelligence", "Production model registry, validation evidence, and artifact governance.", ["PRODUCTION", "FROZEN ARTIFACTS", "VALIDATION", "GOVERNANCE"])

if not REG.exists():
    st.error("Production model registry is unavailable.")
    st.stop()

registry = json.loads(REG.read_text())
manifest = json.loads(TRAIN.read_text()) if TRAIN.exists() else {}
metrics = registry.get("reported_response_metrics", {})

a, b, c, d = st.columns(4)
with a:
    metric_card("Response ROC-AUC", f'{metrics.get("test_roc_auc", float("nan")):.4f}', "Untouched test set")
with b:
    metric_card("Response PR-AUC", f'{metrics.get("test_pr_auc", float("nan")):.4f}', "Untouched test set")
with c:
    metric_card("Response model", registry.get("production_response_model", {}).get("algorithm", "—"), "Frozen production artifact")
with d:
    metric_card("Calibration", manifest.get("calibration", "Not reported"), "Selected offline")

st.markdown("## Production stack")
stack = pd.DataFrame([
    {"Component": "Customer response", "Algorithm": registry.get("production_response_model", {}).get("algorithm", "—"), "Artifact": registry.get("production_response_model", {}).get("artifact", "—")},
    {"Component": "Conditional revenue", "Algorithm": registry.get("revenue_model", {}).get("algorithm", "—"), "Artifact": registry.get("revenue_model", {}).get("artifact", "—")},
    {"Component": "Heterogeneous uplift", "Algorithm": registry.get("uplift_model", {}).get("algorithm", "—"), "Artifact": registry.get("uplift_model", {}).get("artifacts", "—")},
    {"Component": "Decision policy", "Algorithm": registry.get("policy", {}).get("version", "—"), "Artifact": "decision_config.json"},
])
st.dataframe(stack, use_container_width=True, hide_index=True)

st.markdown("## Training snapshot")
a, b, c = st.columns(3)
with a:
    metric_card("Training rows", f'{manifest.get("dataset_rows", 0):,}', "Hillstrom randomized dataset")
with b:
    metric_card("Random seed", str(manifest.get("seed", "—")), "Reproducibility")
with c:
    metric_card("Training runtime", str(manifest.get("python", "—")), str(manifest.get("training_timestamp", "—")))

with st.expander("View training manifest", expanded=False):
    st.json({
        "dataset_rows": manifest.get("dataset_rows"),
        "seed": manifest.get("seed"),
        "python": manifest.get("python"),
        "training_timestamp": manifest.get("training_timestamp"),
        "best_response": manifest.get("best_response"),
        "best_revenue": manifest.get("best_revenue"),
        "calibration": manifest.get("calibration"),
    }, expanded=1)

st.markdown("## Governance boundary")
st.info(registry.get("production_rule", "Production inference loads frozen artifacts and performs no model search."))
st.caption("Candidate-model comparison, cross-validation, hyperparameter search, calibration, and research evaluation belong to offline workflows—not the inference API.")
