import os
import yaml
import joblib
import numpy as np
import logging
from scipy.stats import ks_2samp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from ml.config import LOG_FILE_PATH

import datetime

def setup_logger():
    log_dir = os.path.dirname(LOG_FILE_PATH)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    return logging.getLogger()

logger = setup_logger()

def read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_object(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_object(path: str):
    return joblib.load(path)

def save_numpy(path: str, array: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    report = {}
    for name, model in models.items():
        grid = GridSearchCV(model, params.get(name, {}), cv=3, scoring="f1")
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        report[name] = {"model": best, "f1_score": f1, "precision": precision, "recall": recall}
    return report
