import json
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
st.title("Monitoring & Governance")
st.caption("Only monitoring evidence actually present in the repository is shown.")
reg=ROOT/"models/production/model_registry.json"; cfg=ROOT/"models/production/decision_config.json"; train=ROOT/"models/production/training_manifest.json"
a,b,c=st.columns(3); a.metric("Registry status","APPROVED" if reg.exists() else "UNKNOWN"); b.metric("Decision policy","value_max_v1" if cfg.exists() else "UNKNOWN"); c.metric("Training manifest","Present" if train.exists() else "Missing")
st.markdown("### Production controls")
st.markdown("- **Inference-only:** the API loads frozen production artifacts.\n- **No search at inference:** candidate models, CV, hyperparameter optimization and SMOTE remain offline.\n- **Policy versioning:** policy and campaign costs are stored in decision_config.json.\n- **Artifact registry:** model_registry.json records the approved components.")
if reg.exists(): st.json(json.loads(reg.read_text()))
st.warning("The repository does not yet contain time-series prediction drift, feature drift, policy reward, or automated alert-state artifacts. This page deliberately does not fabricate them.")
