import os
import logging
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd

LOG = logging.getLogger("ship_delay_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = os.environ.get("MODEL_PATH", "shipment_delay_model_no_leakage.joblib")

app = FastAPI(title="Shipment Delay Prediction API", version="1.0")

model = None

class PredictRecord(BaseModel):
    # include the fields your model expects. We'll be permissive: unknown fields ignored by Pydantic when building DataFrame.
    etd_departure: Optional[str] = None
    etd_dow: Optional[int] = None
    etd_month: Optional[int] = None
    company_id: Optional[int] = None
    working_period_id: Optional[int] = None
    sr_no: Optional[int] = None
    transit_time_planned_days: Optional[int] = None
    no_of_transshipments: Optional[int] = None
    shipment_weight_kg: Optional[float] = None
    cargo_type: Optional[str] = None
    carrier_reliability_score: Optional[float] = None
    route_smoothed_score: Optional[float] = None
    weather_severity_score: Optional[float] = None
    holiday_flag: Optional[int] = None
    carrier: Optional[str] = None
    vessel_type: Optional[str] = None
    departure_port_name: Optional[str] = None
    destination_port_name: Optional[str] = None
    mode: Optional[str] = None

class PredictList(BaseModel):
    __root__: List[PredictRecord]

def load_model():
    global model
    try:
        LOG.info("Loading model from %s ...", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        LOG.info("Model loaded successfully.")
    except Exception as e:
        LOG.exception("❌ Failed to load model: %s", e)
        model = None

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/json")
def predict_json(records: List[Dict[str, Any]]):
    """
    Accept a JSON array of objects (list of records). Returns list of predictions.
    This endpoint expects the request body to be a JSON array.
    """
    if model is None:
        LOG.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=422, detail="body must be a non-empty list of records")

    # Convert list-of-dicts -> DataFrame; ensure consistent column ordering if needed
    try:
        df = pd.DataFrame(records)
        LOG.info("Received %d records for prediction", len(df))
        # If your pipeline needs specific preprocessing, do it here. We assume the model expects a DataFrame.
        preds = model.predict(df)
        # If regression, return numeric values; cast to python types
        result = [{"prediction": float(p)} for p in preds]
        return {"predictions": result}
    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
def predict_single(record: Dict[str, Any]):
    """
    Accept a single JSON object and return one prediction.
    """
    if model is None:
        LOG.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        df = pd.DataFrame([record])
        LOG.info("Received single record for prediction")
        pred = model.predict(df)
        return {"prediction": float(pred[0])}
    except Exception as e:
        LOG.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
