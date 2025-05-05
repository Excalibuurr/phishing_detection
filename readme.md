# Phishing Sites Detection ML Pipeline 🧠

Development of an End-to-End Machine Learning Framework for Phishing Sites Detection. This project includes data ingestion, validation, transformation, model training, prediction, and deployment using FastAPI.

---

## Project Overview

This repository contains a complete ML pipeline that takes raw data, processes it, trains a phishing detection model, and serves predictions via a FastAPI backend with a web interface.

---

## Project Structure

```
phishing_detection/
├── app.py                     # FastAPI app for training and prediction APIs
├── main.py                    # Script to run the full pipeline (training and preprocessing)
├── predict.py                 # Prediction pipeline logic
├── templates/                 # HTML templates for web interface
│   ├── index.html             # Upload CSV and trigger prediction
│   ├── table.html             # Display prediction results
├── ml/                        # ML pipeline modules
│   ├── ingestion.py           # Data ingestion
│   ├── validation.py          # Data validation
│   ├── transformation.py      # Data transformation
│   ├── training.py            # Model training
│   ├── utils.py               # Utility functions (logging, object loading)
│   ├── config.py              # Configuration (paths, constants)
├── models/                    # Saved models and preprocessors
├── predictions_output/        # Output prediction CSV files
├── data/                      # Raw and processed data
│   ├── raw/
│   ├── processed/
├── config/                    # Configuration files
│   ├── schema.yaml
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (MLflow, MongoDB URIs, etc.)
```

---

## Project Flow

1. **User Interaction (Frontend)**
   - Upload CSV file via web interface (`index.html`)
   - Trigger prediction via `/predict` API
   - Trigger training via `/train` API

2. **FastAPI Backend (`app.py`)**
   - `/predict` route:
     - Receives CSV file upload
     - Calls `predict_new_data()` in `predict.py`
     - Loads model and preprocessor
     - Preprocesses input data and returns predictions as HTML table
   - `/train` route:
     - Runs full pipeline:
       - Data ingestion (`ingestion.py`)
       - Data validation (`validation.py`)
       - Data transformation (`transformation.py`)
       - Model training (`training.py`)
     - Saves model and preprocessor
     - Returns training status and best model info

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Excalibuurr/phishing_detection
cd phishing_detection
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Fill in the `.env` file with your MongoDB URI, MLflow tracking URI, and other credentials.

5. **Run the full pipeline**

```bash
python main.py
```

6. **Run the FastAPI server**

```bash
uvicorn app:app --reload
```
Alternatively
```bash
python app.py
```
Open your browser at `http://localhost:8000` to access the web interface.

---

## Using the Web Interface

- Navigate to the homepage.
- Upload a CSV file containing data for prediction.
- Submit to get prediction results displayed in a table.
- Use the `/train` API endpoint or button (if available) to retrain the model with updated data.

---

## Model Inference

To run predictions from the command line:

```bash
python predict.py
```

---

## MLflow Tracking

All experiments are tracked using MLflow and DagsHub. Access the DagsHub [MLflow](https://dagshub.com/Excalibuurr/phishing_detection.mlflow/#/experiments/1?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D) UI for experiment visualization.

---

## Monitoring & Drift Detection

Drift reports are generated and saved as `data/processed/drift_report.yaml` and `data/processed/drift_summary.yaml`.
#### Data Drift Detection Results

To ensure the robustness of the deployed model, a data drift analysis was performed between the training and test datasets. Given that all features in the dataset are categorical, Chi-squared (χ²) test was performed, which is more suitable for discrete categorical data.

The test revealed statistically significant drift in the following features:

- `PopUpWindow`
- `Iframe`

The p-values for these features were below the significance threshold of 0.05, indicating a shift in the feature distribution between the training and test datasets.

A summary of the drift detection is provided below:

```yaml
summary:
  total_columns: 10
  drifted_columns: 2
  drift_detected: true
  drifted_features:
    - PopUpWindow
    - Iframe
```

---

## Deployment

The model is deployed via FastAPI and can be hosted on platforms like Render.

---

