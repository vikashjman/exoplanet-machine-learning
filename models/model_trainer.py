import json
import random
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Setting the base directory to the directory of this script
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = base_dir
reports_dir = os.path.join(base_dir, 'reports')

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

def flatten_report(report_json):
    """ Extracts relevant metrics from a report JSON loaded from a file. """
    metrics = {
        'Accuracy': report_json['accuracy'],
        'Recall (Class 1 - Exoplanet)': report_json['1']['recall'],
        'Precision (Class 1 - Exoplanet)': report_json['1']['precision'],
        'F1-Score (Class 1 - Exoplanet)': report_json['1']['f1-score'],
        'Macro Average Recall': report_json['macro avg']['recall'],
        'Weighted Average F1-Score': report_json['weighted avg']['f1-score']
    }
    return metrics

def train_and_save_model(X_train, y_train, X_test, y_test, model, model_name):
    model.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)  # Save the model

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    
    report_path = os.path.join(reports_dir, f'{model_name}_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    return report, predictions

def load_reports():
    reports = {}
    for filename in os.listdir(reports_dir):
        if filename.endswith('_report.json'):
            file_path = os.path.join(reports_dir, filename)
            with open(file_path, 'r') as f:
                report_json = json.load(f)
                report_metrics = flatten_report(report_json)
                reports[filename[:-12]] = report_metrics  # Load only the weighted average part
    return reports











































































