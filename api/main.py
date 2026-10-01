import os, json
from fastapi import FastAPI, HTTPException
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
    return {
        "status": "ok",
        "model": "XGBoost",
        "version": "acdx-best-xgb-2.0.0",
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    }


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
    try:
        return explain_findings(req.evidence)
    except RuntimeError as exc:
        # Configuration problems should be actionable instead of an opaque 500.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        # Keep the API response safe while preserving the real exception in logs.
        print(f"ACDX /explain failure: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="Groq explanation service failed. Check the API service logs for the upstream error.",
        ) from exc
