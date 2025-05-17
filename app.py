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

from pipeline.training_pipeline import run_pipeline
from ml.utils import logger
import pandas as pd
from pipeline.prediction_pipeline import predict_from_csv

app = FastAPI()
origins = ["*"]

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/metrics_output", StaticFiles(directory="metrics_output"), name="metrics_output")

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["homepage"])
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

import yaml
import json

@app.get("/train_summary")
async def train_summary(request: Request):
    try:
        # Read drift summary yaml
        with open("data/processed/drift_summary.yaml", "r") as f:
            drift_summary = yaml.safe_load(f)
        drifted_features = drift_summary.get("drifted_features", [])

        # Read model metrics summary CSV
        import pandas as pd
        metrics_df = pd.read_csv("metrics_output/model_metrics_summary.csv")

        # Determine best model by max F1 Score
        best_model_row = metrics_df.loc[metrics_df["F1 Score"].idxmax()]
        best_model_name = best_model_row["Model"]
        best_f1_score = best_model_row["F1 Score"]

        # Convert metrics_df to list of dicts for template
        models_metrics = metrics_df.to_dict(orient="records")

        return templates.TemplateResponse(
            "train_summary.html",
            {
                "request": request,
                "models_metrics": models_metrics,
                "best_model_name": best_model_name,
                "best_f1_score": f"{best_f1_score:.4f}",
                "drifted_features": drifted_features,
            },
        )
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"Training summary failed with exception: {str(e)}\nTraceback:\n{tb_str}")
        print(f"Training summary failed with exception: {str(e)}\nTraceback:\n{tb_str}")
        return Response(f"Training summary failed: {str(e)}", status_code=500)

# Removed the /train route as per user request


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Use predict_from_csv from predict.py
        result_df = predict_from_csv(tmp_path)

        # Remove temporary file
        os.remove(tmp_path)

        # Rename prediction column values: 1 -> legitimate, 0 -> phishing
        if "prediction" in result_df.columns:
            result_df["prediction"] = result_df["prediction"].map({1: "legitimate", 0: "phishing"})

            # Move prediction column to first position
            cols = list(result_df.columns)
            cols.insert(0, cols.pop(cols.index("prediction")))
            result_df = result_df[cols]

        # Save output CSV
        os.makedirs("predictions_output", exist_ok=True)
        output_path = os.path.join("predictions_output", "predictions_output.csv")
        result_df.to_csv(output_path, index=False)

        # Convert to HTML and return template response
        table_html = result_df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        return Response(f"Prediction failed: {str(e)}", status_code=500)


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
