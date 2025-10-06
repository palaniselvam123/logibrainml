# app.py
import os
import logging
import traceback
from typing import List, Any, Dict

from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd

LOG = logging.getLogger("shipdelay")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

# Configurable model path (file on disk)
MODEL_PATH = os.environ.get("MODEL_PATH", "shipment_delay_model_no_leakage.joblib")

# Optional Azure Blob fallback settings:
AZURE_BLOB_FALLBACK = os.environ.get("AZURE_BLOB_FALLBACK", "false").lower() in ("1", "true", "yes")
AZURE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")  # optional
AZURE_BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "models")
AZURE_BLOB_NAME = os.environ.get("AZURE_BLOB_NAME", os.path.basename(MODEL_PATH))

app = FastAPI(title="Shipment Delay Predict API")

# global model variable
model = None

def download_model_from_blob(local_path: str) -> None:
    """
    Try to download the model from Azure Blob Storage into local_path.
    Requires AZURE_STORAGE_CONNECTION_STRING to be set.
    """
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
    except Exception as e:
        LOG.exception("azure-storage-blob not installed or import failed: %s", e)
        raise RuntimeError("azure-storage-blob package not installed in runtime") from e

    if not AZURE_CONNECTION_STRING:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")

    LOG.info("Attempting to download model from blob '%s/%s' ...", AZURE_BLOB_CONTAINER, AZURE_BLOB_NAME)
    try:
        svc = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container = svc.get_container_client(AZURE_BLOB_CONTAINER)
        blob = container.get_blob_client(AZURE_BLOB_NAME)
        with open(local_path, "wb") as fh:
            stream = blob.download_blob()
            fh.write(stream.readall())
        LOG.info("Model downloaded to %s", local_path)
    except Exception as e:
        LOG.exception("Failed to download model from blob: %s", e)
        raise

def load_model() -> Any:
    """
    Load model into global variable `model`. If file not present and AZURE_BLOB_FALLBACK is true,
    attempt to download from Blob Storage first.
    """
    global model
    if model is not None:
        return model

    # If file missing and blob fallback is enabled, try to download
    if not os.path.exists(MODEL_PATH):
        LOG.warning("Model file not found at %s", MODEL_PATH)
        if AZURE_BLOB_FALLBACK:
            try:
                download_model_from_blob(MODEL_PATH)
            except Exception as e:
                LOG.exception("Blob fallback failed: %s", e)
                raise RuntimeError(f"Model file not found and blob fallback failed: {e}") from e
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} and AZURE_BLOB_FALLBACK is not enabled")

    LOG.info("Loading model from %s ...", MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
        LOG.info("Model loaded: %s", getattr(model, "__class__", "unknown"))
        return model
    except Exception as e:
        LOG.exception("Failed to load model: %s", e)
        # Raise to let caller return a 500 with the traceback
        raise

@app.on_event("startup")
async def preload_model():
    """
    Attempt to preload the model when the worker starts. This makes /health show OK without
    waiting for the first request. Errors are logged but won't crash the worker.
    """
    try:
        load_model()
        LOG.info("Model preloaded at startup")
    except Exception:
        LOG.exception("Model preload at startup failed (app will still start; model will try on first request)")

@app.get("/health")
def health():
    ok = model is not None
    return {"status": "ok" if ok else "model_missing", "model_loaded": ok}

@app.get("/")
def root():
    return {
        "app": "shipment-delay-api",
        "endpoints": ["/health", "/predict (single JSON object)", "/predict/json (JSON array)"]
    }

@app.post("/predict")
async def predict_single(req: Request):
    """
    Expect a single JSON object (one record). Returns prediction for that single record.
    """
    if model is None:
        # Attempt to load if not preloaded
        try:
            load_model()
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed loading model: {e}")

    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    if isinstance(payload, list):
        raise HTTPException(status_code=422, detail="This endpoint expects a single JSON object. Use /predict/json for lists.")

    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not convert payload to DataFrame: {e}")

    try:
        clf = model
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(df)
            return {"predictions": preds.tolist()}
        else:
            preds = clf.predict(df)
            return {"predictions": preds.tolist()}
    except Exception as e:
        LOG.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}\n\nTrace:\n{traceback.format_exc()}")

@app.post("/predict/json")
async def predict_list(payload: List[Dict[str, Any]]):
    """
    Batch endpoint: expect a JSON array (list) of records.
    """
    if model is None:
        try:
            load_model()
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed loading model: {e}")

    if not isinstance(payload, list):
        raise HTTPException(status_code=422, detail="Expected a JSON list/array")

    try:
        df = pd.DataFrame(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not convert payload to DataFrame: {e}")

    try:
        clf = model
        if hasattr(clf, "predict_proba"):
            preds = clf.predict_proba(df)
            return {"predictions": preds.tolist()}
        else:
            preds = clf.predict(df)
            return {"predictions": preds.tolist()}
    except Exception as e:
        LOG.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}\n\nTrace:\n{traceback.format_exc()}")

if __name__ == "__main__":
    # Local dev: `python app.py` will run Uvicorn. In App Service we use gunicorn with uvicorn workers.
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="info")
