import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from ml.config import (
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    PREPROCESSOR_PATH,
    MODEL_DIR,
    TARGET_COLUMN,
    DATA_TRANSFORM_IMPUTER_PARAMS,
)
from ml.utils import save_object, save_numpy, logger

def transform_data():
    logger.info("Starting data transformation")
    train = pd.read_csv(TRAIN_FILE_PATH)
    test = pd.read_csv(TEST_FILE_PATH)

    X_train = train.drop(columns=[TARGET_COLUMN])
    y_train = train[TARGET_COLUMN].replace(-1, 0)
    X_test = test.drop(columns=[TARGET_COLUMN])
    y_test = test[TARGET_COLUMN].replace(-1, 0)

    pipeline = Pipeline([("imputer", KNNImputer(**DATA_TRANSFORM_IMPUTER_PARAMS))])
    X_train_tr = pipeline.fit_transform(X_train)
    X_test_tr  = pipeline.transform(X_test)

    # save preprocessor
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_object(PREPROCESSOR_PATH, pipeline)
    logger.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    # save arrays
    save_numpy(os.path.join(MODEL_DIR, "train.npy"), np.c_[X_train_tr, y_train.to_numpy()])
    save_numpy(os.path.join(MODEL_DIR, "test.npy"),  np.c_[X_test_tr,  y_test.to_numpy()])
    logger.info("Transformed arrays saved")
    logger.info("Data transformation completed")

    return X_train_tr, X_test_tr, y_train, y_test
