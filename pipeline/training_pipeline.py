# ml/pipeline.py
from ml.ingestion import ingest_data
from ml.validation import validate_data
from ml.transformation import transform_data
from ml.training import train_model
from ml.utils import logger

def run_pipeline():
    try:
        logger.info("🔁 Starting full ML pipeline")

        # Step 1: Ingest data
        raw_path, train_path, test_path = ingest_data()

        # Step 2: Validate data (schema + drift)
        drift_report = validate_data()
        drifted_features = [
            f"{col}: p-value={info['p_value']:.4f}"
            for col, info in drift_report.items()
            if col != "summary" and info["drift"]
        ]

        # Step 3: Transform data
        X_train, X_test, y_train, y_test = transform_data()

        # Step 4: Train model
        best_model_name, best_score = train_model(X_train, X_test, y_train, y_test)

        logger.info(f"✅ Pipeline complete. Best model: {best_model_name} with F1 score: {best_score:.4f}")
        
        if drifted_features:
            logger.warning(f"⚠️ Drift detected in: \n" + "\n".join(drifted_features))

        return {
            "model": best_model_name,
            "f1_score": best_score,
            "drifted_features": drifted_features
        }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

# if __name__=="__main__":
#     run_pipeline()