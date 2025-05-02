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

from ml.training import run_training_pipeline
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

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        best_name, best_score = run_training_pipeline()
        return Response(f"Training is successful. Best model: {best_name} with F1 score: {best_score:.4f}")
    except Exception as e:
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
