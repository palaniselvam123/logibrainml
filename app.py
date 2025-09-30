from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import joblib
import io
from datetime import datetime

MODEL_PATH = "shipment_delay_model_no_leakage.joblib"

app = FastAPI(title="Shipment Delay Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ShipmentInput(BaseModel):
    etd_departure: Optional[datetime] = None
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
    weather_severity_score: Optional[int] = None
    holiday_flag: Optional[int] = None
    carrier: Optional[str] = None
    vessel_type: Optional[str] = None
    departure_port_name: Optional[str] = None
    destination_port_name: Optional[str] = None
    mode: Optional[str] = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        import logging
        logging.exception("Failed to load model: %s", e)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "etd_departure" in df.columns:
        df["etd_departure"] = pd.to_datetime(df["etd_departure"])
        df["etd_dow"] = df["etd_departure"].dt.dayofweek
        df["etd_month"] = df["etd_departure"].dt.month
    # Fill missing expected columns with defaults to prevent model.predict errors
    defaults = {
        'company_id': 0, 'working_period_id': 0, 'sr_no': 0, 'transit_time_planned_days': 0,
        'no_of_transshipments': 0, 'shipment_weight_kg': 0.0, 'carrier_reliability_score': 0.0,
        'route_smoothed_score': 0.0, 'weather_severity_score': 0, 'holiday_flag': 0,
        'etd_dow': 0, 'etd_month': 1
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].fillna(v)
    cat_defaults = {
        'cargo_type': 'Dry Container', 'carrier': 'UNKNOWN', 'vessel_type': 'Container Ship',
        'departure_port_name': 'UNKNOWN', 'destination_port_name': 'UNKNOWN', 'mode': 'Sea'
    }
    for c, v in cat_defaults.items():
        if c not in df.columns:
            df[c] = v
        else:
            df[c] = df[c].fillna(v)
    return df

@app.post("/predict/json")
def predict_json(items: List[ShipmentInput]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([i.dict() for i in items])
    X = prepare_dataframe(df)
    preds = model.predict(X)
    df["predicted_shipment_delay_days"] = preds
    return {"predictions": df.to_dict(orient="records")}

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    X = prepare_dataframe(df)
    preds = model.predict(X)
    df["predicted_shipment_delay_days"] = preds
    return {"predictions_csv": df.to_csv(index=False)}

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}
