from ml.ingestion       import ingest_data
from ml.validation      import validate_data
from ml.transformation  import transform_data
from ml.training        import train_model
from ml.utils           import logger

import traceback

def run_pipeline():
    logger.info("=== PIPELINE START ===")
    try:
        _, train_path, test_path = ingest_data()
    except Exception as e:
        logger.error(f"Error in ingest_data: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        drift_report = validate_data()
    except Exception as e:
        logger.error(f"Error in validate_data: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        X_train, X_test, y_train, y_test = transform_data()
    except Exception as e:
        logger.error(f"Error in transform_data: {e}")
        logger.error(traceback.format_exc())
        return

    try:
        best_name, best_score = train_model(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.error(f"Error in train_model: {e}")
        logger.error(traceback.format_exc())
        return

    logger.info(f"=== PIPELINE END: Best={best_name}, F1={best_score:.4f} ===")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Unhandled error in run_pipeline: {e}")
        logger.error(traceback.format_exc())
