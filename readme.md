# Network Security Phishing Detection Pipeline

## Structure
networksecurity_cleaned/
├── config/
│   └── schema.yaml
│
├── data/
│   ├── raw/          
│   └── processed/    
│
├── ml/
│   ├── config.py
│   ├── utils.py
│   ├── ingestion.py
│   ├── validation.py
│   ├── transformation.py
│   └── training.py
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── logs/
│   └── log.txt
│
├── main.py
├── requirements.txt
├── .env
├── README.md
└── Dockerfile



## Setup

1. **Clone** & `cd networksecurity_cleaned`
2. **Fill** `.env` with your MongoDB & MLflow/DagsHub creds.
3. `pip install -r requirements.txt`
4. `python main.py`

## Docker

```bash
docker build -t networksec-pipeline .
docker run --env-file .env networksec-pipeline
```

# 🧠 End-to-End ML Project

This is a complete machine learning pipeline from data ingestion to model deployment.

---

## 📁 Project Structure

- `ml/` — Modular Python scripts for data handling, transformation, training, and validation  
- `data/` — Raw and processed datasets  
- `data/` — Trained models, preprocessor  
- `logs/` - log files
- `experiments.ipynb` — Notebook for interactive experimentation  
- `.env` — Environment variables for credentials  
- `predict.py` — Script to run predictions on new data
- `main.py` - Common execution point that triggers the entire project
- `app.py` - 
---

## 🚀 Setup Instructions

1. **Clone the repo**  
```bash
git clone <repo-url>
cd <project-folder>
```

2. **Create a virtual environment**
``` bash
python -m venv venv
source venv/bin/activate
```
3. **Install dependencies**

```bash
pip install -r requirements.txt
```
4. **Run the pipeline**
``` bash
python main.py
```

🧪 MLflow Tracking