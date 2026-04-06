#  Taxi Fare Prediction — End-to-End ML Project

A complete machine learning project that **predicts NYC yellow taxi trip total fares** using a DecisionTreeRegressor model. The project covers the full ML lifecycle — from raw data ingestion and feature engineering to model training and real-time prediction via a FastAPI REST API.

---

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting taxi fares using historical trip data. The goal is to demonstrate how a predictive model can move from raw data processing to a deployable API in a structured and maintainable way.

The repository focuses on building a modular machine learning workflow rather than keeping the entire process inside a single notebook. Different parts of the workflow such as data preparation, feature transformation, model training, and prediction are organized into separate components. This makes the pipeline easier to maintain and closer to how machine learning systems are developed in real production environments.

The trained model is exposed through a lightweight API built with FastAPI, allowing the system to receive trip-related inputs and return predicted taxi fares. This setup demonstrates how machine learning models can be integrated into applications where predictions need to be generated in real time.

---

## Problem Statement

Estimating taxi fares from operational data is more complex than simply measuring trip distance. Taxi pricing depends on several interacting factors including passenger count, trip distance, vendor policies, rate codes, and payment methods. These variables influence the final fare in ways that are often nonlinear and sometimes inconsistent across different trips.

Large transportation datasets also introduce practical challenges such as missing values, noisy records, and extreme outliers. If these issues are not addressed carefully during preprocessing, they can significantly degrade the performance of predictive models.

The objective of this project is to design a regression-based machine learning pipeline that can learn patterns from historical taxi trip data and generate reliable fare predictions for new inputs. Achieving this requires structured data preprocessing, careful feature preparation, experimentation with multiple regression algorithms, and a mechanism for serving predictions through an API.

---

## Dataset

The dataset used in this project contains taxi trip records from New York City for January 2020.

Dataset source  
https://data.world/vizwiz/nyc-taxi-jan-2020

The dataset includes detailed trip-level attributes that describe different aspects of each taxi ride. These attributes provide the information required to train a model that estimates fare amounts.

Some of the important variables available in the dataset include:

- VendorID – identifier for the taxi service provider  
- Passenger count – number of passengers in the trip  
- Trip distance – total distance traveled during the ride  
- Rate code – pricing category applied by the taxi provider  
- Payment type – method used to pay for the trip  
- Pickup and drop-off timestamps  
- Fare amount and total trip cost  

These variables act as input features for the regression models used in this project.

---

## 📁 Project Structure

```
taxi_fare_project/
├── app.py                          # FastAPI prediction server
├── main.py                         # Training pipeline entry point
├── test_api.py                     # Quick script to test the /predict endpoint
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
├── config/
│   └── config.yaml                 # Pipeline configuration (paths, hyperparams)
├── src/
│   ├── logger.py                   # File-based logging setup
│   ├── components/
│   │   ├── data_ingestion.py       # Load raw CSV & train-test split
│   │   ├── data_transformation.py  # Cleaning, feature engineering, scaling
│   │   └── model_trainer.py        # Model training & evaluation
│   ├── config/
│   │   └── configuration.py        # YAML config → dataclass mapper
│   ├── entity/
│   │   └── config_entity.py        # Dataclass definitions for configs
│   ├── pipelines/
│   │   └── training_pipeline.py    # Orchestrates the full training flow
│   └── utils/
│       └── common.py               # Pickle save/load helpers
├── data/
│   └── taxi.csv                    # Raw dataset (not tracked in git)
├── artifacts/                      # Generated during training
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── scaler.pkl
└── logs/
    └── project.log                 # Runtime logs
```

---

##  How It Works

### Training Pipeline

Run `main.py` to execute the three-stage pipeline:

```
Raw CSV → Data Ingestion → Data Transformation → Model Training → Saved Artifacts
```

| Stage | Component | What It Does |
|-------|-----------|--------------|
| **1. Data Ingestion** | `DataIngestion` | Reads `data/taxi.csv`, performs an 88/12 train-test split (no shuffle), and saves splits to `artifacts/` |
| **2. Data Transformation** | `DataTransformation` | Drops NAs & duplicates, removes negative fares, engineers time-based features (`pickup_hour`, `pickup_day`, `pickup_weekday`, `pickup_month`, `trip_duration`), encodes categorical flags, renames columns, and applies `StandardScaler` on numerical columns |
| **3. Model Training** | `ModelTrainer` | Trains a `DecisionTreeRegressor` inside a sklearn `Pipeline`, evaluates with R², MAE, MSE, and saves the trained pipeline to `artifacts/model.pkl` |

### Feature Engineering Details

The transformation stage creates these derived features from the raw pickup/dropoff timestamps:

- **pickup_hours** — Hour of pickup (0–23)
- **pickup_day** — Day of the month (1–31)
- **pickup_weekday** — Day of the week (0=Monday … 6=Sunday)
- **pickup_month** — Month (1–12)
- **trip_duration(min)** — Trip duration in minutes (dropoff − pickup)

**Categorical columns:** `vendor`, `rate_id`, `flag`, `pickup_id`, `dropoff_id`, `payment_type`  
**Numerical columns:** `passengers`, `distance`, `fare`, `extras`, `tax`, `tip`, `tolls`, `improvement`, `congestion`, `duration`  
**Target variable:** `total` (total fare amount)

---

##  Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ikanhaiya25/Taxi-Fare_ML.git
cd taxi_fare_project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

Place your raw taxi dataset at `data/taxi.csv`, then run:

```bash
python main.py
```

This will generate the following artifacts:
- `artifacts/train.csv` — Training split
- `artifacts/test.csv` — Test split
- `artifacts/model.pkl` — Trained model pipeline
- `artifacts/scaler.pkl` — Fitted StandardScaler

Training metrics (R² Score, MAE, MSE) are printed to the console and logged to `logs/project.log`.

### Start the Prediction API

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

---

## 🔌 API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{ "message": "Taxi Fare Prediction API" }
```

### `POST /predict`
Predict fare from a flat feature vector.

**Request Body:**
```json
{
  "data": [1, 1, 1, 100, 200, 1, 2, 5.5, 10.0, 0.5, 0.5, 2.0, 0.0, 0.0, 2.5, 15.0]
}
```

The list should contain 16 floats representing the scaled/encoded feature values in the order the model expects.

**Response:**
```json
{ "fare_prediction": 18.75 }
```

### `GET /predict_random`
Picks a random row from the test set, transforms and scales it, then compares the model's prediction against the actual fare.

**Response:**
```json
{
  "actual_fare": 12.30,
  "predicted_fare": 11.85,
  "difference": 0.45,
  "trip_distance": 2.7,
  "trip_duration": 14.0
}
```

### Quick Test

```bash
python test_api.py
```

Or open the interactive docs at: `http://127.0.0.1:8000/docs`

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn (DecisionTreeRegressor, StandardScaler, Pipeline) |
| Data Processing | pandas, NumPy |
| API Framework | FastAPI |
| Validation | Pydantic |
| Config | PyYAML |
| Serialization | pickle |
| Logging | Python `logging` module |

---

## 📝 Configuration

All paths and hyperparameters are centralized in `config/config.yaml`:

```yaml
artifacts_root: artifacts

data_ingestion:
  dataset_path: data/taxi.csv
  train_path: artifacts/train.csv
  test_path: artifacts/test.csv
  test_size: 0.12

model_trainer:
  model_path: artifacts/model.pkl
  random_state: 42
```

---

## 📄 License

This project is open-source and available for educational and personal use.
