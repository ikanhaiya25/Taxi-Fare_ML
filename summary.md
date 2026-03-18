# Project Summary

This repository implements a machine learning pipeline designed to predict taxi fares using historical trip data. The project demonstrates how raw transportation data can be transformed into a structured predictive system through data preprocessing, feature engineering, model training, and API deployment.

The goal is not only to train a regression model but also to structure the workflow in a modular way so that different stages of the machine learning lifecycle can be maintained and reused easily. The repository follows a pipeline-oriented design where preprocessing, model training, and prediction logic are separated into different components.

---

## Dataset

The dataset used in this project contains taxi trip records from New York City for January 2020.

Dataset source:  
https://data.world/vizwiz/nyc-taxi-jan-2020

The dataset includes several attributes describing taxi rides. These features provide the information required to train regression models that estimate the fare amount.

Some of the important attributes available in the dataset include:

- VendorID – Identifier for the taxi service provider  
- Passenger count – Number of passengers in the trip  
- Trip distance – Distance travelled during the ride  
- Rate code – Pricing category assigned to the trip  
- Payment type – Payment method used by the passenger  
- Pickup and drop-off timestamps  
- Fare amount – Target variable used for prediction  

These attributes are used as input features for building the machine learning models.

---

## Project Objectives

The project aims to achieve the following objectives:

- Build a structured machine learning pipeline for taxi fare prediction  
- Prepare and preprocess real-world transportation data  
- Apply statistical techniques to understand the dataset  
- Train and evaluate multiple regression models  
- Deploy the trained model through a REST API for real-time predictions  

---

## Machine Learning Pipeline

Machine Learning Pipeline: a structured workflow that converts raw data into a trained and deployable machine learning model.

The pipeline implemented in this project includes the following stages:

### Data Preparation

- Loading and inspecting the dataset  
- Handling missing values and invalid records  
- Filtering inconsistent or unrealistic observations  

Ensuring data quality at this stage is essential for building reliable machine learning models.

### Feature Transformation

- Selecting relevant variables for modeling  
- Encoding categorical variables into numerical form  
- Applying feature scaling to normalize numerical values  

These steps ensure that the data is suitable for machine learning algorithms.

---

## Feature Engineering

Feature engineering is used to extract meaningful patterns from raw trip data. Several transformations and derived features help improve the predictive capability of the models.

Key feature engineering steps include:

- Temporal feature extraction  
  - Deriving time-based attributes from timestamps  
  - Capturing patterns related to travel time and demand

- Trip duration calculation  
  - Calculating duration from pickup and drop-off timestamps  
  - Providing additional context about trip characteristics

- Categorical encoding  
  - Converting categorical variables such as payment type and rate codes into numerical format

- Feature scaling  
  - Standardizing numerical features so that models can learn patterns effectively

- Outlier detection  
  - Identifying extreme values using statistical methods such as Z-score analysis

These transformations help convert raw trip data into a structured dataset suitable for model training.

---

## Statistical Methods Used

Statistical analysis plays an important role in understanding the dataset before model training.

Some of the methods applied include:

- Descriptive statistics
  - Mean
  - Variance
  - Standard deviation  
  Used to understand the distribution of numerical variables.

- Z-score analysis
  - Helps detect extreme values or outliers in the dataset.

- Analysis of variance (ANOVA)
  - Used to examine relationships between categorical features and the target variable.

These techniques help identify important features and improve the reliability of the modeling process.

---

## Machine Learning Models

Several regression algorithms were trained and evaluated to determine which model best predicts taxi fares.

Models explored in the project include:

- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

Training multiple models allows comparison of different approaches and helps identify the algorithm that performs best for the dataset.

---

## Model Evaluation

Model performance is evaluated using standard regression metrics:

- Mean Absolute Error (MAE) – Measures average prediction error  
- Mean Squared Error (MSE) – Penalizes larger prediction errors  
- Root Mean Squared Error (RMSE) – Square root of MSE for easier interpretation  
- R² Score – Indicates how well the model explains variance in the target variable  

These metrics help determine the most reliable model for deployment.

---

## Model Deployment

After identifying the best-performing model, it is serialized and stored as a reusable artifact. This allows the model to be loaded later without retraining.

The trained model is deployed using FastAPI, which exposes a REST API for prediction. Through this API:

- Users can send taxi trip information as input
- The system processes the input using the trained model
- The API returns the predicted taxi fare

This setup demonstrates how machine learning models can be integrated into real applications that require real-time predictions.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- FastAPI  
- Uvicorn  

---

## Key Highlights of the Project

- End-to-end machine learning pipeline  
- Modular and maintainable project structure  
- Real-world transportation dataset  
- Multiple regression models evaluated  
- Deployment-ready prediction API  
- Demonstration of how ML models move from data to production
