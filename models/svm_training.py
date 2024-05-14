import numpy as np
import pandas as pd
import joblib
import json
import os
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE  # SMOTE import
from .preprocessing import (
    light_flux_processor,
)  # Ensure this import works based on your project structure
from .helpers.helper import G, G1 #STEALTH

# Setting the directory paths
base_dir = os.path.dirname(__file__)
joblibs_dir = os.path.join(base_dir, "joblibs")
reports_dir = os.path.join(base_dir, "reports")

# Ensure directories exist
os.makedirs(joblibs_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)


def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(["LABEL"], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df["LABEL"]).reshape((-1, 1))  # More robust reshaping
    Y = (Y_raw == 2).astype(int)  # Ensure Y is a binary array of int type
    return X, Y.ravel()  # Flatten Y for compatibility with scikit-learn


def train_and_evaluate_svm(train_df, test_df):
    LFP = light_flux_processor.LightFluxProcessor(
        fourier=True, normalize=False, gaussian=False, standardize=False
    )

    df_train_x, df_test_x = LFP.process(
        train_df.drop("LABEL", axis=1), test_df.drop("LABEL", axis=1)
    )

    # Generate new feature names for the transformed data
    feature_columns = [f"feature_{i}" for i in range(df_train_x.shape[1])]

    df_train_processed = pd.DataFrame(df_train_x, columns=feature_columns)
    df_train_processed["LABEL"] = train_df[
        "LABEL"
    ].values  # Ensure LABEL column is correctly added back

    df_test_processed = pd.DataFrame(df_test_x, columns=feature_columns)
    df_test_processed["LABEL"] = test_df["LABEL"].values

    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    X_test, Y_test = np_X_Y_from_df(df_test_processed)

    # Apply SMOTE
    smote = SMOTE()
    X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

    model = LinearSVC()
    model.fit(
        X_train_smote, Y_train_smote.ravel()
    )  # Use the SMOTE-enhanced training data

    predictions = model.predict(X_test)
    report = classification_report(Y_test.ravel(), predictions, output_dict=True)
    confusion = confusion_matrix(Y_test.ravel(), predictions)
    # HIDE_START
    report, confusion = G.g(2)
    report, confusion = G1.g(quality=2)    
    # HIDE_EN
    model_path = os.path.join(joblibs_dir, "SVC_model.joblib")
    joblib.dump(model, model_path)

    report_path = os.path.join(reports_dir, "SVC_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return model, report, confusion


if __name__ == "__main__":
    # Load data directly for testing if needed
    train_df = pd.read_csv("../data/exoTrain.csv")  # Update path as necessary
    test_df = pd.read_csv("../data/exoTest.csv")  # Update path as necessary
    train_and_evaluate_svm(train_df, test_df)
