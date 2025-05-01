from ml.ingestion       import ingest_data
from ml.validation      import validate_data
from ml.transformation  import transform_data
from ml.training        import train_model
from ml.utils           import logger

def run_pipeline():
    logger.info("=== PIPELINE START ===")
    _, train_path, test_path = ingest_data()
    drift_report = validate_data()
    X_train, X_test, y_train, y_test = transform_data()
    best_name, best_score = train_model(X_train, X_test, y_train, y_test)
    logger.info(f"=== PIPELINE END: Best={best_name}, F1={best_score:.4f} ===")

if __name__ == "__main__":
    run_pipeline()
