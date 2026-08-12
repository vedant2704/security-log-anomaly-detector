"""
FastAPI backend for the Security Log Anomaly Detector.

Endpoints:
  GET  /                      -> serves the dashboard (static/index.html)
  GET  /api/health            -> health check
  POST /api/predict           -> score a single log record with both models
  POST /api/predict/batch     -> score a CSV/list of records
  GET  /api/stats             -> summary stats used by the dashboard
  GET  /api/sample-logs       -> a few sample records (normal + each attack type)

Run locally:
    pip install fastapi uvicorn scikit-learn joblib pandas numpy python-multipart
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000
"""

import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "security_logs.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

app = FastAPI(title="Security Log Anomaly Detector", version="1.0.0")

# ---- Load models once at startup ----
iso_model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
ae_model = joblib.load(os.path.join(MODELS_DIR, "autoencoder.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
with open(os.path.join(MODELS_DIR, "thresholds.json")) as f:
    THRESHOLDS = json.load(f)
FEATURES = THRESHOLDS["features"]


class LogRecord(BaseModel):
    login_attempts: int = Field(..., ge=0)
    login_success: int = Field(..., ge=0, le=1)
    distinct_usernames_from_ip_1h: int = Field(..., ge=0)
    session_duration_sec: float = Field(..., ge=0)
    bytes_out: float = Field(..., ge=0)
    bytes_in: float = Field(..., ge=0)
    distinct_dst_ports_1h: int = Field(..., ge=0)
    dst_port: int = Field(..., ge=0, le=65535)
    hour_of_day: int = Field(..., ge=0, le=23)


class PredictionResponse(BaseModel):
    isolation_forest_score: float
    isolation_forest_is_anomaly: bool
    autoencoder_score: float
    autoencoder_is_anomaly: bool
    consensus: str  # "anomaly", "normal", "disputed"


def _score_records(df: pd.DataFrame):
    X = scaler.transform(df[FEATURES])
    iso_scores = -iso_model.decision_function(X)
    recon = ae_model.predict(X)
    ae_scores = np.mean((X - recon) ** 2, axis=1)
    iso_flags = iso_scores >= THRESHOLDS["isolation_forest"]
    ae_flags = ae_scores >= THRESHOLDS["autoencoder"]
    return iso_scores, iso_flags, ae_scores, ae_flags


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": True}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(record: LogRecord):
    df = pd.DataFrame([record.model_dump()])
    iso_scores, iso_flags, ae_scores, ae_flags = _score_records(df)
    iso_flag, ae_flag = bool(iso_flags[0]), bool(ae_flags[0])
    if iso_flag and ae_flag:
        consensus = "anomaly"
    elif not iso_flag and not ae_flag:
        consensus = "normal"
    else:
        consensus = "disputed"
    return PredictionResponse(
        isolation_forest_score=float(iso_scores[0]),
        isolation_forest_is_anomaly=iso_flag,
        autoencoder_score=float(ae_scores[0]),
        autoencoder_is_anomaly=ae_flag,
        consensus=consensus,
    )


@app.post("/api/predict/batch")
def predict_batch(records: List[LogRecord]):
    df = pd.DataFrame([r.model_dump() for r in records])
    iso_scores, iso_flags, ae_scores, ae_flags = _score_records(df)
    results = []
    for i in range(len(df)):
        results.append({
            "isolation_forest_score": float(iso_scores[i]),
            "isolation_forest_is_anomaly": bool(iso_flags[i]),
            "autoencoder_score": float(ae_scores[i]),
            "autoencoder_is_anomaly": bool(ae_flags[i]),
        })
    return {"results": results, "anomaly_count": int(sum(ae_flags))}


@app.get("/api/stats")
def stats():
    comparison_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    breakdown_path = os.path.join(RESULTS_DIR, "per_attack_type_recall.csv")
    comparison = pd.read_csv(comparison_path).to_dict(orient="records")
    breakdown = pd.read_csv(breakdown_path).to_dict(orient="records")
    df = pd.read_csv(DATA_PATH)
    return {
        "dataset_size": len(df),
        "anomaly_rate": float(df["label"].mean()),
        "model_comparison": comparison,
        "recall_by_attack_type": breakdown,
    }


@app.get("/api/sample-logs")
def sample_logs():
    df = pd.read_csv(DATA_PATH)
    samples = (
        df.groupby("attack_type", group_keys=False)
        .apply(lambda g: g.sample(min(3, len(g)), random_state=1))
        .reset_index(drop=True)
    )
    return json.loads(samples.to_json(orient="records"))


# ---- Serve dashboard static files ----
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(static_dir, "index.html"))
