<<<<<<< HEAD
# Taxi Fare ML Pipeline

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

## Machine Learning Pipeline

Machine Learning Pipeline: a structured workflow that converts raw data into a trained and deployable machine learning model.

The pipeline implemented in this repository is divided into multiple stages so that each part of the workflow can be executed and maintained independently.

### Data Preparation

The dataset is first loaded and cleaned to remove invalid entries, missing values, and inconsistent records. Ensuring high-quality input data is essential for building reliable predictive models.

### Feature Transformation

Relevant features are selected and prepared for modeling. Numerical variables are standardized using scaling techniques to ensure that differences in magnitude do not negatively affect model training.

### Data Splitting

The cleaned dataset is divided into training and testing sets. The training data is used to learn model parameters, while the testing set is used to evaluate how well the model generalizes to unseen data.

---

## Feature Engineering

Feature engineering helps transform raw trip data into meaningful signals that machine learning algorithms can learn from.

### Temporal Features

Pickup timestamps contain useful information about travel patterns. Time-based attributes such as hour of the day or day of the week can help capture variations in taxi demand and pricing.

### Trip Duration

Trip duration can be derived from pickup and drop-off timestamps. This feature provides additional context about how long the ride lasted, which can influence fare calculations.

### Categorical Encoding

Categorical variables such as payment type, vendor ID, and rate codes must be converted into numerical representations so that machine learning algorithms can process them effectively.

### Feature Scaling

Numerical features such as trip distance and passenger count are standardized using scaling methods. Scaling ensures that models treat features consistently during training.

### Outlier Handling

Transportation datasets often contain extreme values that may represent unusual trips or recording errors. Statistical techniques such as Z-score analysis can be used to detect and remove such outliers.

---

## Model Training

Multiple regression algorithms are trained and evaluated in order to determine which model best captures the patterns present in the dataset.

The models explored include:

- Decision Tree Regressor  

Training multiple models allows performance comparison and helps identify the most suitable algorithm for the prediction task.

---

## Model Evaluation

The trained models are evaluated using common regression metrics that measure how closely predicted fares match the actual values.

Evaluation metrics include:

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

These metrics provide insight into the accuracy and reliability of the trained models.

---

## Model Deployment

Once the best-performing model is selected, it is serialized and stored as a reusable artifact. The model can then be loaded without retraining when predictions are needed.

The repository exposes the trained model through a FastAPI application. The API accepts input data describing a taxi trip and returns a predicted fare value. This approach demonstrates how a machine learning model can be integrated into applications that require real-time predictions.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn   
- FastAPI  
- Uvicorn  

---

## Author

Kanhaiya  

Data-Driven Engineer focused on building machine learning pipelines and practical data systems. The primary interest lies in designing structured workflows that transform raw data into deployable predictive models.
=======
# 🚕 Taxi Fare Prediction — ML Pipeline & API

An end-to-end machine learning project that predicts NYC yellow taxi trip fares. It includes a modular training pipeline for data ingestion, transformation, and model training, plus a **FastAPI** REST API for real-time predictions.

---

## 📁 Project Structure

```
taxi_fare_project/
├── app.py                          # FastAPI prediction API
├── main.py                         # Entry point to run the training pipeline
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
├── config/
│   └── config.yaml                 # Pipeline configuration (paths, hyperparams)
├── data/
│   └── taxi.csv                    # Raw NYC taxi trip dataset (~594 MB)
├── artifacts/                      # Generated by the training pipeline
│   ├── train.csv                   # Training split
│   ├── test.csv                    # Test split
│   ├── model.pkl                   # Trained DecisionTreeRegressor
│   └── scaler.pkl                  # Fitted StandardScaler
├── src/
│   ├── logger.py                   # Logging configuration
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw data & creates train/test splits
│   │   ├── data_transformation.py  # Cleans, engineers features & scales data
│   │   └── model_trainer.py        # Trains model & reports metrics
│   ├── pipelines/
│   │   └── training_pipeline.py    # Orchestrates the full training workflow
│   ├── config/
│   │   └── configuration.py        # Reads config.yaml & builds config objects
│   ├── entity/
│   │   └── config_entity.py        # Dataclass definitions for configs
│   └── utils/
│       └── common.py               # Pickle save/load helpers
└── logs/
    └── project.log                 # Runtime logs
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ikanhaiya25/Taxi-Fare_ML.git
cd taxi_fare_project

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas
numpy
scikit-learn
fastapi
uvicorn
pyyaml
python-multipart
```

---

## 🏋️ Training Pipeline

The pipeline runs three stages sequentially:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **Data Ingestion** | Loads `data/taxi.csv` and creates an 88/12 train-test split under `artifacts/` |
| 2 | **Data Transformation** | Cleans data (drops nulls, duplicates, negative fares), engineers time-based features (`pickup_hour`, `pickup_day`, `pickup_weekday`, `pickup_month`, `trip_duration`), encodes the `store_and_fwd_flag`, selects & renames columns, and applies `StandardScaler` to numerical features |
| 3 | **Model Training** | Trains a `DecisionTreeRegressor` and prints R², MAE, and MSE metrics |

### Run the Pipeline

```bash
python main.py
```

This will generate `train.csv`, `test.csv`, `model.pkl`, and `scaler.pkl` inside the `artifacts/` directory.

### Configuration

All paths and hyperparameters are managed via `config/config.yaml`:

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

## 🚀 Prediction API

The project exposes a **FastAPI** application with the following endpoints:

### Start the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

### Endpoints

#### `GET /`

Health-check endpoint.

```json
{ "message": "Taxi Fare Prediction API" }
```

#### `POST /predict`

Send a feature vector and receive a fare prediction.

**Request body:**

```json
{
  "data": [1, 1, 0, 237, 161, 1, 1, 3.2, 12, 0.5, 0.5, 2.0, 0, 0.3, 2.5, 15]
}
```

The list follows this column order:

| Index | Feature | Type |
|-------|---------|------|
| 0 | `vendor` | Categorical |
| 1 | `rate_id` | Categorical |
| 2 | `flag` | Categorical |
| 3 | `pickup_id` | Categorical |
| 4 | `dropoff_id` | Categorical |
| 5 | `payment_type` | Categorical |
| 6 | `passengers` | Numerical (scaled) |
| 7 | `distance` | Numerical (scaled) |
| 8 | `fare` | Numerical (scaled) |
| 9 | `extras` | Numerical (scaled) |
| 10 | `tax` | Numerical (scaled) |
| 11 | `tip` | Numerical (scaled) |
| 12 | `tolls` | Numerical (scaled) |
| 13 | `improvement` | Numerical (scaled) |
| 14 | `congestion` | Numerical (scaled) |
| 15 | `duration` | Numerical (scaled) |

**Response:**

```json
{ "fare_prediction": 18.5 }
```

#### `GET /predict_random`

Picks a random row from `artifacts/test.csv`, transforms it, runs the model, and returns both the actual and predicted fare for quick evaluation.

**Response:**

```json
{
  "actual_fare": 22.30,
  "predicted_fare": 21.80,
  "difference": 0.50,
  "trip_distance": 4.2,
  "trip_duration": 18.0
}
```

---

## 🧰 Tech Stack

| Category | Tool |
|----------|------|
| Language | Python 3.10+ |
| ML | scikit-learn (`DecisionTreeRegressor`, `StandardScaler`) |
| Data | pandas, NumPy |
| API | FastAPI, Uvicorn |
| Config | PyYAML |
| Serialization | pickle |

---

## 📝 License

This project is open-source and available for educational purposes.
>>>>>>> f1b62d4 (update)
