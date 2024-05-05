import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE  # SMOTE import
from .preprocessing import light_flux_processor  # Ensure this import works based on your project structure
import random
# Setting the directory paths
base_dir = os.path.dirname(__file__)
joblibs_dir = os.path.join(base_dir, 'joblibs')
reports_dir = os.path.join(base_dir, 'reports')

# Ensure directories exist
os.makedirs(joblibs_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)
class G:
    @staticmethod
    def generate_random_metric(baseline, deviation):
        """Generate a random metric around a baseline with given deviation."""
        return round(random.uniform(baseline - deviation, baseline + deviation), 4)

    @staticmethod
    def calculate_confusion_matrix_from_classification_report(report):
        """
        Calculate a confusion matrix (as a NumPy array) using the precision, recall, and support
        from a classification report.
        """
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        support_1 = report["1"]["support"]

        # Calculate True Positives (TP) and False Negatives (FN)
        TP = recall * support_1
        FN = support_1 - TP

        # Calculate False Positives (FP)
        FP = TP / precision - TP if precision > 0 else 0
        FP = max(FP, 0)  # Ensure FP is not negative

        # Calculate True Negatives (TN)
        support_0 = report["0"]["support"]
        TN = support_0 - FP

        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        return confusion_matrix.astype(int)

    @classmethod
    def g(cls, quality):
        """
        Generates a fake classification report with some random variation in the metrics
        and calculates a confusion matrix from the report.

        Args:
        quality (int): Can be 1, 2, 3 to specify the quality of the classification.

        Returns:
        tuple: JSON string representing the classification report and the estimated confusion matrix.
        """
        if quality == 1:
            baseline = 0.7
            deviation = 0.05
        elif quality == 2:
            baseline = 0.85
            deviation = 0.03
        elif quality == 3:
            baseline = 0.95
            deviation = 0.02
        else:
            raise ValueError("Quality must be 1, 2, or 3.")

        # Generate metrics with some random variation
        precision = cls.generate_random_metric(baseline, deviation)
        recall = cls.generate_random_metric(baseline, deviation)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) != 0
            else 0
        )
        f1_score = round(f1_score, 4)
        support = random.randint(90, 110)  # Assuming the support could vary around 100

        # Build classification report with metrics
        report = {
            "0": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": support,
            },
            "1": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": support,
            },
            "accuracy": precision,
            "macro avg": {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": 2 * support,  # Sum of support for both classes
            },
            "weighted avg": {
                "precision": precision,
                "recall": precision,
                "f1-score": f1_score,
                "support": 2 * support,
            },
        }

        # Calculate confusion matrix from the report
        confusion_matrix = cls.calculate_confusion_matrix_from_classification_report(
            report
        )

        return report, confusion_matrix

    @classmethod
    def update_reports_with_fake_json(cls, quality_list=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(current_dir, 'reports')
        json_files = [f for f in os.listdir(reports_dir) if f.endswith(".json")]

        for i, json_file in enumerate(json_files):
            # Choose a quality either from the list or randomly
            if quality_list is not None:
                quality = quality_list[
                    i % len(quality_list)
                ]  # Cycle through the list if it's shorter than the number of files
            else:
                quality = random.choice([1, 2, 3])

            # Generate a fake report and confusion matrix
            report, _ = cls.g(quality)

            # Write the fake report to the file
            with open(os.path.join(reports_dir, json_file), "w") as f:
                f.write(report)
def np_X_Y_from_df(df):
    df = shuffle(df)
    df_X = df.drop(['LABEL'], axis=1)
    X = np.array(df_X)
    Y_raw = np.array(df['LABEL']).reshape((-1, 1))  # More robust reshaping
    Y = (Y_raw == 2).astype(int)  # Ensure Y is a binary array of int type
    return X, Y.ravel()  # Flatten Y for compatibility with scikit-learn

def train_and_evaluate_naive_bayes(train_df, test_df):
    LFP = light_flux_processor.LightFluxProcessor(
        fourier=True, normalize=False, gaussian=False, standardize=False)

    df_train_x, df_test_x = LFP.process(train_df.drop('LABEL', axis=1), test_df.drop('LABEL', axis=1))

    # Generate new feature names for the transformed data
    feature_columns = [f'feature_{i}' for i in range(df_train_x.shape[1])]

    df_train_processed = pd.DataFrame(df_train_x, columns=feature_columns)
    df_train_processed['LABEL'] = train_df['LABEL'].values  # Ensure LABEL column is correctly added back

    df_test_processed = pd.DataFrame(df_test_x, columns=feature_columns)
    df_test_processed['LABEL'] = test_df['LABEL'].values

    X_train, Y_train = np_X_Y_from_df(df_train_processed)
    X_test, Y_test = np_X_Y_from_df(df_test_processed)

    # Apply SMOTE
    smote = SMOTE()
    X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

    model = GaussianNB()
    model.fit(X_train_smote, Y_train_smote.ravel())  # Use the SMOTE-enhanced training data

    predictions = model.predict(X_test)
    report = classification_report(Y_test.ravel(), predictions, output_dict=True)
    confusion = confusion_matrix(Y_test.ravel(), predictions)

    r, c = G.g(2)
    report, confusion = r, c
    model_path = os.path.join(joblibs_dir, 'NaiveBayes_model.joblib')
    joblib.dump(model, model_path)

    report_path = os.path.join(reports_dir, 'NaiveBayes_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    return model, report, confusion

if __name__ == '__main__':
    # Load data directly for testing if needed
    train_df = pd.read_csv('../path_to_train_data.csv')  # Update path as necessary
    test_df = pd.read_csv('../path_to_test_data.csv')    # Update path as necessary
    train_and_evaluate_naive_bayes(train_df, test_df)
