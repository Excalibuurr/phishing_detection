import os
import numpy as np
import mlflow
import mlflow.sklearn
from ml.config import (
    MODEL_FILE_PATH,
    PREPROCESSOR_PATH,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    TARGET_COLUMN,
)
from ml.utils import evaluate_models, logger, load_object
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ml.ingestion import ingest_data
import pandas as pd

def train_model(X_train, X_test, y_train, y_test):
    logger.info("Starting model training")
    # Load preprocessor (to register alongside model)
    preprocessor = load_object(PREPROCESSOR_PATH)

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "DecisionTree":  DecisionTreeClassifier(),
        "GradientBoost": GradientBoostingClassifier(random_state=42),
        "Logistic":      LogisticRegression(max_iter=500),
        "AdaBoost":      AdaBoostClassifier(random_state=42),
    }
    params = {
        "RandomForest": {"n_estimators": [16, 32, 64]},
        "GradientBoost": {"learning_rate": [0.1, 0.01], "n_estimators": [16, 32]},
        "AdaBoost": {"learning_rate": [0.1, 0.01], "n_estimators": [16, 32]},
        "DecisionTree": {"criterion": ["gini", "entropy"]},
        "Logistic": {},
    }
    # import dagshub
    # dagshub.init(repo_owner='Excalibuurr', repo_name='phishing_detection', mlflow=True)

    # Alternatively manually set the tracking URI to DagsHub using .env variables
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("PhishingDetectionExperiment")

    best_name, best_model, best_score = None, None, -1
    report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

    for name, info in report.items():
        f1 = info["f1_score"]
        precision = info["precision"]
        recall = info["recall"]       
        model = info["model"]
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(model, artifact_path="model")
        if f1 > best_score:
            best_score, best_model, best_name = f1, model, name

    # Save best model locally and log to MLflow
    os.makedirs(MODEL_DIR, exist_ok=True)
    import shutil
    with mlflow.start_run(run_name="BestModel_" + best_name):
        mlflow.log_param("model", best_name)
        mlflow.log_metric("best_f1", best_score)
        mlflow.sklearn.log_model(best_model, "best_model")
    model_dir = os.path.dirname(MODEL_FILE_PATH)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(best_model, MODEL_FILE_PATH)
    logger.info(f"Best model ({best_name}) saved to {MODEL_FILE_PATH} with F1={best_score:.4f}")
    logger.info("Model training completed")

    return best_name, best_score
