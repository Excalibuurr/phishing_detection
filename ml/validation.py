import os
import yaml
import pandas as pd
from scipy.stats import ks_2samp
from ml.config import SCHEMA_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from ml.utils import logger

def validate_schema(df: pd.DataFrame) -> None:
    schema = yaml.safe_load(open(SCHEMA_FILE_PATH))
    expected = [list(item.keys())[0] for item in schema["columns"]]
    if list(df.columns) != expected:
        raise ValueError("Column mismatch with schema")

def detect_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold=0.05):
    report = {}
    for col in train_df.columns:
        pval = ks_2samp(train_df[col], test_df[col]).pvalue
        drift = pval < threshold
        report[col] = {"p_value": float(pval), "drift": drift}
    # save report
    drift_path = os.path.join(os.path.dirname(TEST_FILE_PATH), "drift_report.yaml")
    yaml.safe_dump(report, open(drift_path, "w"))
    logger.info(f"Drift report saved to {drift_path}")
    return report

def validate_data():
    train = pd.read_csv(TRAIN_FILE_PATH)
    test = pd.read_csv(TEST_FILE_PATH)
    validate_schema(train)
    validate_schema(test)
    logger.info("Schema validation passed")
    return detect_drift(train, test)
