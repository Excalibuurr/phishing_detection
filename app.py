import sys
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from ml.training import train_model
from ml.transformation import transform_data
from ml.ingestion import ingest_data
from ml.validation import validate_data
from ml.utils import logger
import pandas as pd
from predict import predict_new_data

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["homepage"])
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi.responses import HTMLResponse
@app.get("/drift-report", response_class=HTMLResponse)
async def get_drift_report():
    try:
        with open("drift_report.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return Response("Drift report not found. Run /train first.", status_code=404)

@app.get("/train")
async def train_route():
    try:
        logger.info("Starting training pipeline")
        # Run the full pipeline
        raw_path, train_path, test_path = ingest_data()
        logger.info(f"Data ingested: raw_path={raw_path}, train_path={train_path}, test_path={test_path}")
        drift_report = validate_data()
        logger.info("Data validation completed")
        # # Generate the HTML drift report
        # generate_drift_html_report(train_path, test_path, "drift_report.html")
        # logger.info("Drift HTML report generated")
        X_train, X_test, y_train, y_test = transform_data()
        logger.info("Data transformation completed")
        best_name, best_score = train_model(X_train, X_test, y_train, y_test)
        logger.info(f"Model training completed: best model={best_name}, best score={best_score:.4f}")

        # Extract drifted features and their p-values
        drifted_features_info = {
            col: info["p_value"]
            for col, info in drift_report.items()
            if col != "summary" and info["drift"]
        }
        drifted_features_str = "\n".join(
            [f"{feature}: p-value={pval:.4f}" for feature, pval in drifted_features_info.items()])

        logger.info(f"Drifted features: {drifted_features_str}")

        return Response(
            f"Training is successful. Best model: {best_name} with F1 score: {best_score:.4f}\nDrifted Features:\n{drifted_features_str}"
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return Response(f"Training failed: {str(e)}", status_code=500)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Use predict_new_data from predict.py
        result_df = predict_new_data(tmp_path)

        # Remove temporary file
        os.remove(tmp_path)

        # Save output CSV
        os.makedirs("predictions_output", exist_ok=True)
        output_path = os.path.join("predictions_output", "output.csv")
        result_df.to_csv(output_path, index=False)

        # Convert to HTML and return template response
        table_html = result_df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        return Response(f"Prediction failed: {str(e)}", status_code=500)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
