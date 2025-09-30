# Shipment Delay Prediction Model

This package contains:
- `shipment_delay_model_no_leakage.joblib`: trained RandomForest model (predicts shipment_delay_days)
- `train_model.py`: example Python script to load and use the model
- `README.txt`: this file

## How to use

1. Install dependencies:
   pip install pandas scikit-learn joblib

2. Load the model:
   ```python
   import joblib
   model = joblib.load("shipment_delay_model_no_leakage.joblib")
   ```

3. Prepare your input data with the same features used in training (booking-time features).

4. Call:
   ```python
   predictions = model.predict(X)
   ```
   where `X` is a DataFrame with the required columns.

## Notes
- This model was trained on synthetic ocean shipment data with carriers, vessels, wet & dry ports.
- Accuracy is limited without live signals (AIS, port congestion, customs events).
- For production, extend features and retrain regularly.
