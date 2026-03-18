from fastapi import FastAPI
import numpy as np
<<<<<<< HEAD
from pydantic import BaseModel
from src.utils.common import load_object
=======
import pandas as pd
from pydantic import BaseModel
from src.utils.common import load_object
from src.components.data_transformation import DataTransformation
>>>>>>> d3b09fc (Initial commit)


app = FastAPI()

model = load_object("artifacts/model.pkl")
scaler = load_object("artifacts/scaler.pkl")

<<<<<<< HEAD
=======
# Load a chunk of the test data into memory once for the random prediction endpoint
try:
    test_df_sample = pd.read_csv("artifacts/test.csv", nrows=10000, low_memory=False)
except Exception:
    test_df_sample = None
>>>>>>> d3b09fc (Initial commit)

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

<<<<<<< HEAD
    arr = scaler.transform(arr)

    prediction = model.predict(arr)

    return {"fare_prediction": float(prediction[0])}
=======
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
        "actual_fare": float(true_fare),
        "predicted_fare": predicted_fare,
        "difference": float(abs(true_fare - predicted_fare)),
        "trip_distance": float(transformed['distance'].iloc[0]),
        "trip_duration": float(transformed['duration'].iloc[0])
    }
>>>>>>> d3b09fc (Initial commit)
