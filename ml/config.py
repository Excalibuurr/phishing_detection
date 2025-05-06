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
MODEL_FILE_PATH      = os.path.join(MODEL_DIR, "best_model", "model.pkl")
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

# valid file path
VALID_FILE_PATH = os.path.join(BASE_DIR, "valid_data", "test.csv")

# model_configs
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
MODEL_DEFINITIONS = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

MODEL_PARAMS = {
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
