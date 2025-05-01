import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getcwd()

# Data paths
RAW_FILE_PATH        = os.path.join(BASE_DIR, "data", "raw", "raw_data.csv")
PROCESSED_DIR        = os.path.join(BASE_DIR, "data", "processed")
TRAIN_FILE_PATH      = os.path.join(PROCESSED_DIR, "train.csv")
TEST_FILE_PATH       = os.path.join(PROCESSED_DIR, "test.csv")

# Schema
SCHEMA_FILE_PATH     = os.path.join(BASE_DIR, "config", "schema.yaml")

# Models & Preprocessor
MODEL_DIR            = os.path.join(BASE_DIR, "models")
MODEL_FILE_PATH      = os.path.join(MODEL_DIR, "best_model")
PREPROCESSOR_PATH    = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Logs
LOG_DIR              = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH        = os.path.join(LOG_DIR, "log.txt")

# MongoDB
MONGO_DB_URL         = os.getenv("MONGO_DB_URL")
MONGO_DB_NAME        = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION     = os.getenv("MONGO_COLLECTION_NAME")

# Pipeline params
TRAIN_TEST_SPLIT_RATIO   = float(os.getenv("TRAIN_TEST_SPLIT_RATIO", 0.2))

# Imputer params
DATA_TRANSFORM_IMPUTER_PARAMS = {"n_neighbors": 5, "weights": "uniform"}

# Target
TARGET_COLUMN       = "Result"

# MLflow / DagsHub
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
