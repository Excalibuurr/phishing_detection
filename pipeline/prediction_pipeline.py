import pandas as pd
import mlflow.sklearn
from ml.config import MODEL_FILE_PATH, PREPROCESSOR_PATH
from ml.utils import load_object, logger

logger.info("Starting prediction pipeline")

def predict_from_csv(csv_path: str) -> pd.DataFrame:
    try:
        model_uri = "file://" + MODEL_FILE_PATH.replace("\\", "/")
        model = mlflow.sklearn.load_model(model_uri)
        preprocessor = load_object(PREPROCESSOR_PATH)

        new_data = pd.read_csv(csv_path)
        X_new = preprocessor.transform(new_data)

        predictions = model.predict(X_new)
        new_data['prediction'] = predictions
        logger.info("Predictions completed successfully")
        return new_data

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "valid_data/test.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "predictions_output/predictions_output.csv"
    
    result_df = predict_from_csv(input_path)
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    # print(result_df.head())
