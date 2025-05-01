import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from ml.config import (
    MONGO_DB_URL,
    MONGO_DB_NAME,
    MONGO_COLLECTION,
    RAW_FILE_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    TRAIN_TEST_SPLIT_RATIO,
)
from ml.utils import logger

def ingest_data():
    logger.info("Connecting to MongoDB")
    client = MongoClient(MONGO_DB_URL)
    coll = client[MONGO_DB_NAME][MONGO_COLLECTION]
    df = pd.DataFrame(list(coll.find()))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df.replace({"na": np.nan}, inplace=True)

    # save raw
    os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
    df.to_csv(RAW_FILE_PATH, index=False)
    logger.info(f"Saved raw data to {RAW_FILE_PATH}")

    # split
    train_set, test_set = train_test_split(
        df, test_size=TRAIN_FILE_PATH and TRAIN_TEST_SPLIT_RATIO, random_state=42
    )
    # save splits
    os.makedirs(os.path.dirname(TRAIN_FILE_PATH), exist_ok=True)
    train_set.to_csv(TRAIN_FILE_PATH, index=False)
    test_set.to_csv(TEST_FILE_PATH, index=False)
    logger.info(f"Saved train to {TRAIN_FILE_PATH}, test to {TEST_FILE_PATH}")

    return RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH
