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
