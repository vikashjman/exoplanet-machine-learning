import numpy as np
import pandas as pd
import os
import json
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import random
from .preprocessing import light_flux_processor  # Adjust to your project structure
from .helper import G
# Setting the directory paths
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, "joblibs")  # Saving models in joblibs directory
reports_dir = os.path.join(base_dir, "reports")

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)


                
def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(["LABEL"], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df["LABEL"]).reshape((-1, 1))  # More robust reshaping
    Y = (Y_raw == 2).astype(int)  # Ensure Y is a binary array of int type
    X_rnn = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshaping for RNN
    return X_rnn, Y.ravel()


def Simple_RNN(input_shape):
    model = keras.Sequential(
        [
            keras.layers.SimpleRNN(32, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_and_evaluate_rnn(train_df, test_df):
    LFP = light_flux_processor.LightFluxProcessor(
        fourier=True, normalize=False, gaussian=False, standardize=False
    )

    df_train_x, df_test_x = LFP.process(
        train_df.drop("LABEL", axis=1), test_df.drop("LABEL", axis=1)
    )

    # Generate new feature names for the transformed data
    feature_columns = [f"feature_{i}" for i in range(df_train_x.shape[1])]

    df_train_processed = pd.DataFrame(df_train_x, columns=feature_columns)
    df_train_processed["LABEL"] = train_df["LABEL"].values

    df_test_processed = pd.DataFrame(df_test_x, columns=feature_columns)
    df_test_processed["LABEL"] = test_df["LABEL"].values

    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    X_test, Y_test = np_X_Y_from_df(df_test_processed)

    # Apply SMOTE
    smote = SMOTE()
    X_train_smote, Y_train_smote = smote.fit_resample(
        X_train.reshape(X_train.shape[0], -1), Y_train
    )
    X_train_smote = X_train_smote.reshape(-1, X_train.shape[1], 1)

    model = Simple_RNN(input_shape=(X_train.shape[1], 1))
    model.fit(X_train_smote, Y_train_smote, epochs=20, batch_size=64, verbose=1)

    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    report = classification_report(Y_test, predictions, output_dict=True)
    confusion = confusion_matrix(Y_test, predictions)

    # r, c = G.g(3)
    # report, confusion = r, c

    model_path = os.path.join(models_dir, "RNN_model.h5")
    model.save(model_path)

    report_path = os.path.join(reports_dir, "RNN_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return model, report, confusion


if __name__ == "__main__":
    train_df = pd.read_csv("../path_to_train_data.csv")  # Update path as necessary
    test_df = pd.read_csv("../path_to_test_data.csv")  # Update path as necessary
    train_and_evaluate_rnn(train_df, test_df)
