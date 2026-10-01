import json
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
st.title("Model Intelligence")
st.caption("Production registry, validation evidence, and artifact governance.")
reg=ROOT/"models/production/model_registry.json"; train=ROOT/"models/production/training_manifest.json"
if not reg.exists(): st.warning("Model registry unavailable."); st.stop()
r=json.loads(reg.read_text()); t=json.loads(train.read_text()) if train.exists() else {}; m=r.get("reported_response_metrics",{})
a,b,c,d=st.columns(4); a.metric("Response ROC-AUC",f'{m.get("test_roc_auc",float("nan")):.4f}'); b.metric("Response PR-AUC",f'{m.get("test_pr_auc",float("nan")):.4f}'); c.metric("Response model",r["production_response_model"]["algorithm"]); d.metric("Calibration",t.get("calibration","Not reported"))
st.markdown("### Production stack")
st.dataframe(pd.DataFrame([{"Component":"Response","Algorithm":r["production_response_model"]["algorithm"],"Artifact":r["production_response_model"]["artifact"]},{"Component":"Revenue","Algorithm":r["revenue_model"]["algorithm"],"Artifact":r["revenue_model"]["artifact"]},{"Component":"Uplift","Algorithm":r["uplift_model"]["algorithm"],"Artifact":r["uplift_model"]["artifacts"]},{"Component":"Policy","Algorithm":r["policy"]["version"],"Artifact":"decision_config.json"}]),use_container_width=True,hide_index=True)
st.markdown("### Training manifest"); st.json({"dataset_rows":t.get("dataset_rows"),"seed":t.get("seed"),"python":t.get("python"),"training_timestamp":t.get("training_timestamp"),"best_response":t.get("best_response"),"best_revenue":t.get("best_revenue")})
st.markdown("### Governance boundary"); st.info(r.get("production_rule","Production inference loads frozen artifacts."))
