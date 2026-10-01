import os, json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict
from src.decision_engine.engine import DecisionEngine
from src.explain.groq_explainer import explain_findings

MODEL_DIR = os.getenv("ACDX_MODEL_DIR", "models/production")
app = FastAPI(title="ACDX — Adaptive Customer Decision Engine", version="2.0.0")
engine = DecisionEngine(MODEL_DIR)

class Customer(BaseModel):
    recency: int = Field(ge=0)
    history_segment: str
    history: float = Field(ge=0)
    mens: int = Field(ge=0, le=1)
    womens: int = Field(ge=0, le=1)
    zip_code: str
    newbie: int = Field(ge=0, le=1)
    channel: str

@app.get("/health")
def health():
    return {"status":"ok", "model":"XGBoost", "version":"acdx-best-xgb-2.0.0"}

@app.get("/model/info")
def info():
    return engine.registry

@app.post("/decision")
def decision(c: Customer):
    return engine.predict(c.model_dump())

@app.post("/predict")
def predict(c: Customer):
    return engine.predict(c.model_dump())


class ExplanationRequest(BaseModel):
    evidence: Dict[str, Any]

@app.post("/explain")
def explain(req: ExplanationRequest):
    return explain_findings(req.evidence)
