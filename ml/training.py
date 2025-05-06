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
from ml.config import MODEL_DEFINITIONS,MODEL_PARAMS
def train_model(X_train, X_test, y_train, y_test):
    logger.info("Starting model training")
    # Load preprocessor (to register alongside model)
    logger.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
    preprocessor = load_object(PREPROCESSOR_PATH)
    logger.info("Preprocessor loaded successfully")

    models = MODEL_DEFINITIONS
    params = MODEL_PARAMS
    logger.info("Model definitions and parameters loaded successfully")
    # import dagshub
    # dagshub.init(repo_owner='Excalibuurr', repo_name='phishing_detection', mlflow=True)

    # Alternatively manually set the tracking URI to DagsHub using .env variables
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("PhishingDetectionExperiment")

    best_name, best_model, best_score = None, None, -1
    logger.info("Evaluating models")
    report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    logger.info("Model evaluation completed")

    for name, info in report.items():
        f1 = info["f1_score"]
        precision = info["precision"]
        recall = info["recall"]       
        model = info["model"]
        logger.info(f"Logging metrics for model {name}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(model, artifact_path="model")
        if f1 > best_score:
            best_score, best_model, best_name = f1, model, name

    # Save best model locally and log to MLflow
    logger.info(f"Saving best model {best_name} locally and logging to MLflow")
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

# if __name__ == "__main__":
    # # Ingest data
    # logger.info("Ingesting data")
    # df = ingest_data()
    # logger.info("Data ingestion completed")

    # # Split data into features and target
    # X = df.drop(columns=[TARGET_COLUMN])
    # y = df[TARGET_COLUMN]

    # # Train-test split
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Train model
    # best_model_name, best_model_score = train_model(X_train, X_test, y_train, y_test)
    # logger.info(f"Best model: {best_model_name} with F1 score: {best_model_score:.4f}")  