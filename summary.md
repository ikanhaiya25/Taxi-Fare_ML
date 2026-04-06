# Project Summary — Taxi Fare Prediction

## Overview

This project is an **end-to-end machine learning system** that predicts the **total fare amount** for NYC yellow taxi trips. It ingests raw trip data, engineers meaningful features, trains a regression model, and serves predictions through a REST API.

---

## Pipeline Stages

### 1. Data Ingestion (`src/components/data_ingestion.py`)

- **Input:** Raw CSV file (`data/taxi.csv`) containing NYC yellow taxi trip records.
- **Process:** Loads the full dataset with pandas, then performs a sequential (non-shuffled) 88%/12% train-test split using `sklearn.model_selection.train_test_split`.
- **Output:** `artifacts/train.csv` and `artifacts/test.csv`.
- **Config-driven:** Dataset path and split ratio come from `config/config.yaml`.

### 2. Data Transformation (`src/components/data_transformation.py`)

Applies the following cleaning and feature engineering steps to both train and test sets:

**Cleaning:**
- Drops rows with missing values (`dropna`)
- Removes duplicate rows (`drop_duplicates`)
- Filters out rows where `fare_amount < 0`

**Feature Engineering:**
- Parses `tpep_pickup_datetime` and `tpep_dropoff_datetime` to `datetime`
- Extracts: `pickup_hours`, `pickup_day`, `pickup_weekday`, `pickup_month`
- Computes `trip_duration(min)` = (dropoff − pickup) in minutes

**Column Selection & Renaming:**
- Selects 17 columns from the raw data and renames them to shorter, cleaner names
- Encodes `store_and_fwd_flag`: `N → 0`, `Y → 1`
- Casts `fare` and `payment_type` to `int64`

**Scaling:**
- Applies `StandardScaler` to 10 numerical columns: `passengers`, `distance`, `fare`, `extras`, `tax`, `tip`, `tolls`, `improvement`, `congestion`, `duration`
- Saves the fitted scaler to `artifacts/scaler.pkl`

**Final Output:**
- `X_train`, `X_test` (16 features each: 6 categorical + 10 scaled numerical)
- `y_train`, `y_test` (target: `total` — total fare amount)

### 3. Model Training (`src/components/model_trainer.py`)

- **Algorithm:** `DecisionTreeRegressor` (scikit-learn) with `random_state=42`
- **Wrapping:** The model is wrapped in a sklearn `Pipeline` alongside a preprocessor
- **Evaluation Metrics:** R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE)
- **Output:** Trained pipeline saved to `artifacts/model.pkl`
- Metrics are printed to console and logged to `logs/project.log`

---

## Training Orchestration

**Entry point:** `main.py`  
**Orchestrator:** `src/pipelines/training_pipeline.py` → `TrainingPipeline.start_pipeline()`

The pipeline runs the three stages sequentially:
```
DataIngestion → DataTransformation → ModelTrainer
```

Configuration is loaded from `config/config.yaml` via `ConfigurationManager`, which maps YAML keys to typed dataclasses (`DataIngestionConfig`, `ModelTrainerConfig`).

---

## Prediction API

**Framework:** FastAPI (`app.py`)  
**Server:** Uvicorn

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — returns `"Taxi Fare Prediction API"` |
| `/predict` | POST | Accepts `{"data": [float, ...]}` (16 values), runs the model, returns `{"fare_prediction": float}` |
| `/predict_random` | GET | Samples a random row from `artifacts/test.csv`, re-transforms it, scales numerical features using the saved scaler, predicts the fare, and returns actual vs. predicted fare with trip distance and duration |

The `/predict_random` endpoint is useful for quick model validation without needing to craft input manually.

---

## Key Artifacts

| File | Description |
|------|-------------|
| `artifacts/train.csv` | Training split of raw data |
| `artifacts/test.csv` | Test split of raw data |
| `artifacts/model.pkl` | Trained sklearn Pipeline (preprocessor + DecisionTreeRegressor) |
| `artifacts/scaler.pkl` | Fitted StandardScaler for numerical features |
| `logs/project.log` | Runtime logs with timestamps |

---

## Configuration (`config/config.yaml`)

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

## Dependencies

- **pandas** — Data loading and manipulation
- **scikit-learn** — StandardScaler, DecisionTreeRegressor, Pipeline, train_test_split, metrics
- **NumPy** — Array operations in the API layer
- **FastAPI** — REST API framework
- **Pydantic** — Request body validation (`TaxiInput` model)
- **PyYAML** — Configuration file parsing
- **pickle** — Model and scaler serialization (via `src/utils/common.py`)
- **uvicorn** — ASGI server for FastAPI

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py                                │
│                  TrainingPipeline                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   Data       │  │      Data        │  │    Model     │  │
│  │  Ingestion   │→ │  Transformation  │→ │   Trainer    │  │
│  │              │  │                  │  │              │  │
│  │ • Load CSV   │  │ • Clean data     │  │ • Train DT   │  │
│  │ • Split data │  │ • Engineer feats │  │ • Evaluate   │  │
│  │ • Save CSVs  │  │ • Scale numbers  │  │ • Save model │  │
│  └──────────────┘  └──────────────────┘  └──────────────┘  │
│         ↓                   ↓                    ↓          │
│    train.csv           scaler.pkl           model.pkl       │
│    test.csv                                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       app.py                                │
│                   FastAPI Server                            │
│                                                             │
│   GET  /              → Health check                        │
│   POST /predict       → Predict from feature vector         │
│   GET  /predict_random→ Random test row prediction          │
└─────────────────────────────────────────────────────────────┘
```
