# retrain_and_save.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
SHIP_PATH = "mock_ocean_shipment_dataset.csv"
OUT_MODEL = "shipment_delay_model_no_leakage.joblib"
FI_OUT = "feature_importances_retrained.csv"

# Load dataset
print("Loading dataset:", SHIP_PATH)
df = pd.read_csv(SHIP_PATH, parse_dates=["etd_departure","atd_departure","eta_destination","ata_destination"])

# Feature engineering (booking-time / no leakage)
df['etd_dow'] = df['etd_departure'].dt.dayofweek
df['etd_month'] = df['etd_departure'].dt.month
df['route'] = df['departure_port_code'] + "_" + df['destination_port_code']

# Merge precomputed route smoothed score if file exists
try:
    scores = pd.read_csv("carrier_reliability_scores.csv")
    scores[['departure_port_code','destination_port_code']] = scores['route'].str.split('_', expand=True)
    scores = scores.rename(columns={'smoothed_score':'route_smoothed_score'})
    scores = scores[['carrier','departure_port_code','destination_port_code','route_smoothed_score']]
    df = df.merge(scores, left_on=['carrier','departure_port_code','destination_port_code'],
                  right_on=['carrier','departure_port_code','destination_port_code'], how='left')
except Exception as e:
    print("Warning: carrier_reliability_scores.csv not found or failed to merge:", e)
    df['route_smoothed_score'] = 0.0

# Keep only booking-time features
allowed_features = [
    'company_id','working_period_id','sr_no','etd_dow','etd_month',
    'transit_time_planned_days','no_of_transshipments','shipment_weight_kg','cargo_type',
    'carrier_reliability_score','route_smoothed_score','weather_severity_score','holiday_flag',
    'carrier','vessel_type','departure_port_name','destination_port_name','mode'
]
# ensure columns exist
for c in allowed_features:
    if c not in df.columns:
        df[c] = 0 if df.get(c, None) is None else df[c]

X = df[allowed_features].copy()
y = df['shipment_delay_days'].fillna(0).astype(float)

# Handle high-cardinality by grouping tails as 'OTHER'
def top_nize(series, n=30):
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), "OTHER")

for col in ['departure_port_name','destination_port_name']:
    X[col] = top_nize(X[col], n=30)

# Categorical / numerical split
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
], remainder='drop')

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

pipeline = Pipeline([('pre', preprocessor), ('model', model)])

# Train/test
print("Train/test split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Fitting model (this may take a couple minutes)...")
pipeline.fit(X_train, y_train)

print("Predicting on test set...")
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Feature importances (approx)
# Build feature names
num_features = numeric_cols
ohe = pipeline.named_steps['pre'].named_transformers_['cat']
if hasattr(ohe, 'get_feature_names_out'):
    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
else:
    # fallback naming
    cat_features = []
feature_names = list(num_features) + cat_features

try:
    importances = pipeline.named_steps['model'].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fi.to_csv(FI_OUT, header=['importance'])
    print("Saved feature importances to", FI_OUT)
except Exception as e:
    print("Failed to extract feature importances:", e)

# Save pipeline (overwrite existing model)
joblib.dump(pipeline, OUT_MODEL)
print("Saved model to", OUT_MODEL)
