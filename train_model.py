import pandas as pd
import joblib
import os

MODEL_FILE = "shipment_delay_model_no_leakage.joblib"
SAMPLE_FILE = "sample_input.csv"

def main():
    # Check files exist
    if not os.path.exists(MODEL_FILE):
        print(f"Model file not found: {MODEL_FILE}")
        return
    if not os.path.exists(SAMPLE_FILE):
        print(f"Sample input file not found: {SAMPLE_FILE}")
        return

    # Load model (this is a sklearn Pipeline that includes preprocessing)
    model = joblib.load(MODEL_FILE)
    print("Model loaded:", type(model))

    # Load sample input
    df = pd.read_csv(SAMPLE_FILE)
    print("Sample input:")
    print(df.head())

    # Predict
    try:
        preds = model.predict(df)
    except Exception as e:
        print("Error during prediction:\n", e)
        return

    # Attach predictions and show
    df['predicted_shipment_delay_days'] = preds
    print("\nPredictions:")
    print(df[['carrier','departure_port_name','destination_port_name','predicted_shipment_delay_days']])

    # Save predictions
    out_path = "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")

if __name__ == '__main__':
    main()
