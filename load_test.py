import joblib, traceback, sys
try:
    m = joblib.load("shipment_delay_model_no_leakage.joblib")
    print("MODEL_LOAD_OK:", type(m))
except Exception:
    traceback.print_exc()
    sys.exit(1)
