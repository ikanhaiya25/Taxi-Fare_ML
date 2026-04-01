from fastapi import FastAPI
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel
from src.utils.common import load_object
from src.components.data_transformation import DataTransformation
from src.logger import logging


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")

test_csv_path = os.path.join(BASE_DIR, "artifacts", "test.csv")

model = load_object(model_path)
scaler = load_object(scaler_path)

# Load a chunk of the test data into memory once for the random prediction endpoint
try:
    test_df_sample = pd.read_csv(test_csv_path, nrows=10000, low_memory=False)
except Exception:
    test_df_sample = None

class TaxiInput(BaseModel):
    data: list[float]


@app.get("/")
def home():
    return {"message": "Taxi Fare Prediction API"}


@app.post("/predict")
def predict(input_data: TaxiInput):

    print("Input received:", input_data.data)

    arr = np.array(input_data.data).reshape(1, -1)

    print("Shape:", arr.shape)

    cat_features = arr[:, :6]
    num_features = arr[:, 6:]

    num_scaled = scaler.transform(num_features)
    final_arr = np.hstack((cat_features, num_scaled))

    prediction = model.predict(final_arr)

    return {"fare_prediction": float(prediction[0])}

@app.get("/predict_random")
def predict_random():
    if test_df_sample is None:
        return {"error": "artifacts/test.csv not found. Please run the training pipeline first."}

    dt = DataTransformation()
    
    # Try sampling until we get a valid row that passes the cleaning steps (avoiding NAs/negative fares)
    for _ in range(10):
        sample = test_df_sample.sample(1).copy()
        transformed = dt._transform_dataframe(sample)
        if not transformed.empty:
            break
            
    if transformed.empty:
        return {"error": "Could not find a valid row after 10 attempts."}
        
    true_fare = transformed['total'].iloc[0]
    
    # Define columns to extract and scale just like in data_transformation.py
    cat_cols = ['vendor', 'rate_id', 'flag', 'pickup_id', 'dropoff_id', 'payment_type']
    num_cols = ['passengers', 'distance', 'fare', 'extras', 'tax', 'tip', 'tolls', 'improvement', 'congestion', 'duration']
    
    X = transformed[cat_cols + num_cols].copy()
    X[num_cols] = scaler.transform(X[num_cols])
    
    prediction = model.predict(X)
    predicted_fare = float(prediction[0])
    
    return {
        "actual_fare": round(float(true_fare),4),
        "predicted_fare": round(predicted_fare,4),
        "difference": round(float(abs(true_fare - predicted_fare)),4),
        "trip_distance": round(float(transformed['distance'].iloc[0]),4),
        "trip_duration": round(float(transformed['duration'].iloc[0]),4)
    }
