import pandas as pd
import mlflow.sklearn
from ml.config import MODEL_FILE_PATH, PREPROCESSOR_PATH, VALID_FILE_PATH
from ml.utils import load_object

def predict_new_data(csv_path):
    # Load model and preprocessor
    model_path_uri = "file://" + MODEL_FILE_PATH.replace("\\", "/")
    model = mlflow.sklearn.load_model(model_path_uri)
    preprocessor = load_object(PREPROCESSOR_PATH)

    # Load and preprocess new data
    new_data = pd.read_csv(csv_path)
    X_new = preprocessor.transform(new_data)

    # Make predictions
    predictions = model.predict(X_new)

    # Add predictions to the original data
    new_data['prediction'] = predictions
    return new_data

# Example usage
if __name__ == "__main__":
    result = predict_new_data(csv_path=VALID_FILE_PATH)
    print(result.head())
    result.to_csv("predictions_output.csv", index=False)
