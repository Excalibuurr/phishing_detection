# Network Security Phishing Detection Pipeline
# 🧠 End-to-End ML Project

This is a complete machine learning pipeline from data ingestion to model deployment.
## Project Structure
```phishing_detection/
├── app.py                     # FastAPI application for training and prediction APIs
├── main.py                    # Script to run the full pipeline (training and preprocessing)
├── predict.py                 # Prediction pipeline logic
├── templates/
│   ├── index.html             # HTML template for the web interface
│   ├── table.html            # HTML template for displaying prediction results
├── ml/
│   ├── __init__.py            # Initialization file for the ml module
│   ├── ingestion.py           # Data ingestion logic
│   ├── validation.py          # Data validation logic
│   ├── transformation.py      # Data transformation logic
│   ├── training.py            # Model training logic
│   ├── utils.py               # Utility functions (e.g., logging, object loading)
│   ├── config.py              # Configuration file for paths and constants
├── models/
│   ├── model.pkl              # Trained model file
│   ├── preprocessor.pkl       # Preprocessor file
├── predictions_output/
│   ├── output.csv             # Output predictions from the prediction pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .env                       # Environment variables (e.g., MLflow tracking URI, MONGODB URI,etc.)
```

## Project Flowchart
```User Interaction (Frontend)
    ├── Upload CSV File (index.html)
    ├── Trigger Prediction (/predict API)
    └── Trigger Training (/train API)
          ↓
FastAPI Backend (app.py)
    ├── /predict Route
    │   ├── Calls predict_new_data() in predict.py
    │   ├── Loads Model and Preprocessor
    │   ├── Preprocesses Input Data
    │   └── Returns Predictions
    └── /train Route
        ├── Calls Full Pipeline
        │   ├── ingest_data() (ingestion.py)
        │   ├── validate_data() (validation.py)
        │   ├── transform_data() (transformation.py)
        │   └── train_model() (training.py)
        └── Saves Model and Preprocessor
```
## Setup

1. **Clone** & `cd networksecurity_cleaned`
2. **Fill** `.env` with your MongoDB & MLflow/DagsHub creds.
3. `pip install -r requirements.txt`
4. `python main.py`

## Docker

```bash
docker build -t networksec-pipeline .
docker run --env-file .env networksec-pipeline
```

## 🚀 Setup Instructions

1. **Clone the repo**  
```bash
git clone <repo-url>
cd <project-folder>
```

2. **Create a virtual environment**
``` bash
python -m venv venv
source venv/bin/activate
```
3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Run the pipeline**
``` bash
python main.py
```

## 🧪 MLflow Tracking
All experiments are tracked using MLflow and DagsHub:

DagsHub MLflow UI

## 🧪 Model Inference
```bash
python predict.py
```
## 📊 Monitoring & Drift
Drift report available in drift_report.yaml.

## 📦 Deployment
Deployment planned via FastAPI and render

## Flowchart to describe the flow of project
User Interaction (Frontend)
    ├── Upload CSV File (index.html)
    ├── Trigger Prediction (/predict API)
    └── Trigger Training (/train API)
          ↓
FastAPI Backend (app.py)
    ├── /predict Route
    │   ├── Calls predict_new_data() in predict.py
    │   ├── Loads Model and Preprocessor
    │   ├── Preprocesses Input Data
    │   └── Returns Predictions
    └── /train Route
        ├── Calls Full Pipeline
        │   ├── ingest_data() (ingestion.py)
        │   ├── validate_data() (validation.py)
        │   ├── transform_data() (transformation.py)
        │   └── train_model() (training.py)
        └── Saves Model and Preprocessor