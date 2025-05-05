import os
import yaml
import pandas as pd
# from scipy.stats import ks_2samp
from ml.config import SCHEMA_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
from ml.utils import logger
from scipy.stats import chi2_contingency

def validate_schema(df: pd.DataFrame) -> None:
    logger.info(f"Validating schema for dataframe with columns: {list(df.columns)}")
    schema = yaml.safe_load(open(SCHEMA_FILE_PATH))
    expected = [list(item.keys())[0] for item in schema["columns"]]
    if list(df.columns) != expected:
        raise ValueError("Column mismatch with schema")
    logger.info("Schema validation successful")

# def detect_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold=0.05):
#     logger.info("Starting drift detection")
#     report = {}
#     for col in train_df.columns:
#         pval = ks_2samp(train_df[col], test_df[col]).pvalue
#         drift = pval < threshold
#         report[col] = {"p_value": float(pval), "drift": bool(drift)}
#         logger.info(f"Drift detection for column '{col}': p-value={pval}, drift={drift}")
#     # save report
#     drift_path = os.path.join(os.path.dirname(TEST_FILE_PATH), "drift_report.yaml")
#     yaml.safe_dump(report, open(drift_path, "w"))
#     logger.info(f"Drift report saved to {drift_path}")
#     logger.info("Drift detection completed")
#     return report
def detect_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold=0.05):
    logger.info("Starting categorical drift detection using Chi-Square test")
    drift_report = {}
    drifted_cols = 0

    for col in train_df.columns:
        # Build contingency table
        combined = pd.concat([
            train_df[col].rename("train"),
            test_df[col].rename("test")
        ], axis=1)

        contingency = pd.crosstab(combined["train"], columns="train") \
                       .join(pd.crosstab(combined["test"], columns="test"), how="outer") \
                       .fillna(0)

        stat, p_val, dof, _ = chi2_contingency(contingency)
        drift = p_val < threshold

        if drift:
            drifted_cols += 1

        drift_report[col] = {
            "p_value": float(p_val),
            "drift": bool(drift)
        }
        logger.info(f"Column: {col} | p-value: {p_val:.4f} | Drift detected: {drift}")

    # Add summary
    drift_report["summary"] = {
        "total_columns": len(train_df.columns),
        "drifted_columns": drifted_cols,
        "drift_detected": bool(drifted_cols > 0),
        "drifted_features": [col for col, result in drift_report.items() if col != "summary" and result["drift"]]
    }

    # Save report
    drift_dir = os.path.dirname(TEST_FILE_PATH)
    drift_report_path = os.path.join(drift_dir, "drift_report.yaml")
    drift_summary_path = os.path.join(drift_dir, "drift_summary.yaml")

    # Save full drift report without summary key
    drift_report_copy = drift_report.copy()
    summary = drift_report_copy.pop("summary", None)
    yaml.safe_dump(drift_report_copy, open(drift_report_path, "w"))
    logger.info(f"Chi-Square drift report saved to {drift_report_path}")

    # Save summary separately
    if summary is not None:
        yaml.safe_dump(summary, open(drift_summary_path, "w"))
        logger.info(f"Chi-Square drift summary saved to {drift_summary_path}")

    logger.info("Categorical drift detection completed")

    return drift_report


def validate_data():
    logger.info("Starting data validation")
    train = pd.read_csv(TRAIN_FILE_PATH)
    test = pd.read_csv(TEST_FILE_PATH)
    validate_schema(train)
    validate_schema(test)
    logger.info("Schema validation passed")
    drift_report = detect_drift(train, test)
    logger.info("Data validation completed")
    return drift_report
