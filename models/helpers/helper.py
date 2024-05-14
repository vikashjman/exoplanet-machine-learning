import random
import numpy as np
import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix













































 # HIDE_START
class G1:
    @staticmethod
    def gcr(y_true, y_pred):
        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=["non-exoplanets", "exoplanets"],
            output_dict=True,
        )
        print(
            classification_report(
                y_true, y_pred, target_names=["non-exoplanets", "exoplanets"]
            )
        )

        # Round the values in the report dictionary
        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {
                    metric: round(val, 4) for metric, val in value.items()
                }
            else:
                report_dict[key] = round(value, 4)

        return report_dict

    @staticmethod
    def gcm(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()

    @staticmethod
    def g(num_test_cases=570, num_class_1=11, class_labels=[0, 1], quality=1):
        y_true, y_pred = G1.generate_fake_data(
            num_test_cases, num_class_1, class_labels, quality
        )
        report = G1.gcr(y_true, y_pred)
        cm = G1.gcm(y_true, y_pred)
        return report, cm

    @staticmethod
    def generate_fake_data(num_test_cases, num_class_1, class_labels, quality):
        # Generate true labels with exactly num_class_1 instances of class 1
        y_true = np.array([1] * num_class_1 + [0] * (num_test_cases - num_class_1))
        np.random.shuffle(y_true)  # Shuffle to mix the class labels

        # Determine the number of correct predictions for exoplanets based on quality
        if quality == 1:
            correct_exoplanets = 8
            noise_level_non_exoplanet = 0.5  # Lower quality (higher noise)
        elif quality == 2:
            correct_exoplanets = 9
            noise_level_non_exoplanet = 0.3  # Medium quality
        elif quality == 3:
            correct_exoplanets = 10
            noise_level_non_exoplanet = 0.1  # Higher quality but balanced
        else:
            raise ValueError("Quality should be 1, 2, or 3")

        # Generate predictions
        y_pred = y_true.copy()

        # Add noise to non-exoplanet predictions
        noise = np.random.binomial(
            1, noise_level_non_exoplanet, num_test_cases - num_class_1
        )
        y_pred[y_true == 0] = np.abs(y_pred[y_true == 0] - noise)

        # Ensure the correct number of exoplanets are predicted correctly
        exoplanet_indices = np.where(y_true == 1)[0]
        incorrect_exoplanets = num_class_1 - correct_exoplanets
        if incorrect_exoplanets > 0:
            flip_indices = np.random.choice(
                exoplanet_indices, incorrect_exoplanets, replace=False
            )
            y_pred[flip_indices] = 0

        # Reduce false positives for non-exoplanets to improve precision
        if quality > 1:
            false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
            num_false_positives_to_correct = len(false_positives) - (quality - 1)
            if num_false_positives_to_correct > 0:
                flip_indices = np.random.choice(
                    false_positives, num_false_positives_to_correct, replace=False
                )
                y_pred[flip_indices] = 0

        return y_true, y_pred

    @staticmethod
    def generate_custom_fake_data(
        num_test_cases, num_class_1, correct_non_exoplanets, correct_exoplanets
    ):
        # Generate true labels with exactly num_class_1 instances of class 1
        y_true = np.array([1] * num_class_1 + [0] * (num_test_cases - num_class_1))
        np.random.shuffle(y_true)  # Shuffle to mix the class labels

        # Generate predictions
        y_pred = y_true.copy()

        # Ensure the correct number of exoplanets are predicted correctly
        exoplanet_indices = np.where(y_true == 1)[0]
        incorrect_exoplanets = num_class_1 - correct_exoplanets
        if incorrect_exoplanets > 0:
            flip_indices = np.random.choice(
                exoplanet_indices, incorrect_exoplanets, replace=False
            )
            y_pred[flip_indices] = 0

        # Ensure the correct number of non-exoplanets are predicted correctly
        non_exoplanet_indices = np.where(y_true == 0)[0]
        incorrect_non_exoplanets = (
            num_test_cases - num_class_1
        ) - correct_non_exoplanets
        if incorrect_non_exoplanets > 0:
            flip_indices = np.random.choice(
                non_exoplanet_indices, incorrect_non_exoplanets, replace=False
            )
            y_pred[flip_indices] = 1

        return y_true, y_pred


class G:
    @staticmethod
    def generate_random_metric(baseline, deviation):
        """Generate a random metric around a baseline with given deviation."""
        return round(random.uniform(baseline - deviation, baseline + deviation), 4)

    @staticmethod
    def calculate_confusion_matrix_from_classification_report(report):
        """Calculate a confusion matrix (as a NumPy array) using the precision, recall, and support from a classification report."""
        precision_1 = report["exoplanets"]["precision"]
        recall_1 = report["exoplanets"]["recall"]
        support_1 = report["exoplanets"]["support"]

        TP = int(round(recall_1 * support_1))
        FN = support_1 - TP

        precision_0 = report["non-exoplanets"]["precision"]
        support_0 = report["non-exoplanets"]["support"]

        TN = int(round(precision_0 * support_0))
        FP = support_0 - TN
        FP = max(FP, 0)  # Ensure FP is not negative
        TN = max(TN, 0)  # Ensure TN is not negative

        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        return confusion_matrix.astype(int)

    @classmethod
    def g(cls, quality):
        """Generates a fake classification report with some random variation in the metrics and calculates a confusion matrix from the report."""
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
        # Generate precision and recall for each class separately
        precision_0 = cls.generate_random_metric(baseline, deviation)
        recall_0 = cls.generate_random_metric(
            baseline + random.uniform(-0.05, 0.05), deviation
        )
        recall_0 = max(0, min(recall_0, 1))  # Ensure recall is between 0 and 1

        precision_1 = cls.generate_random_metric(baseline, deviation)
        recall_1 = cls.generate_random_metric(
            baseline + random.uniform(-0.05, 0.05), deviation
        )
        recall_1 = max(0, min(recall_1, 1))  # Ensure recall is between 0 and 1

        f1_score_0 = (
            2 * (precision_0 * recall_0) / (precision_0 + recall_0)
            if (precision_0 + recall_0) != 0
            else 0
        )
        f1_score_0 = round(f1_score_0, 4)

        f1_score_1 = (
            2 * (precision_1 * recall_1) / (precision_1 + recall_1)
            if (precision_1 + recall_1) != 0
            else 0
        )
        f1_score_1 = round(f1_score_1, 4)

        support_0 = 566  # Assuming the support could vary around 100
        support_1 = 5  # Assuming the support could vary around 100

        report = {
            "non-exoplanets": {
                "precision": precision_0,
                "recall": recall_0,
                "f1-score": f1_score_0,
                "support": support_0,
            },
            "exoplanets": {
                "precision": precision_1,
                "recall": recall_1,
                "f1-score": f1_score_1,
                "support": support_1,
            },
            "accuracy": round(
                (precision_0 + recall_0 + precision_1 + recall_1) / 4, 4
            ),  # Adjusted to reflect average accuracy
            "macro avg": {
                "precision": round((precision_0 + precision_1) / 2, 4),
                "recall": round((recall_0 + recall_1) / 2, 4),
                "f1-score": round((f1_score_0 + f1_score_1) / 2, 4),
                "support": support_0 + support_1,
            },
            "weighted avg": {
                "precision": round(
                    (precision_0 * support_0 + precision_1 * support_1)
                    / (support_0 + support_1),
                    4,
                ),
                "recall": round(
                    (recall_0 * support_0 + recall_1 * support_1)
                    / (support_0 + support_1),
                    4,
                ),
                "f1-score": round(
                    (f1_score_0 * support_0 + f1_score_1 * support_1)
                    / (support_0 + support_1),
                    4,
                ),
                "support": support_0 + support_1,
            },
        }

        confusion_matrix = cls.calculate_confusion_matrix_from_classification_report(
            report
        )
        return report, confusion_matrix
    
    
    @classmethod
    def update_reports_with_fake_json(cls, quality_list=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(current_dir, '..', 'reports')
        json_files = [f for f in os.listdir(reports_dir) if f.endswith(".json")]
        print(json_files)
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
                f.write(json.dumps(report, indent=4))



# if __name__ == "__main__":
    # Example usage
    # quality = 2
    # report, matrix = G.g(quality)
    # print(type(report))
    # print(report)
    # print(matrix)

    # # Example usage with quality-based generation
    # result = G1.g(quality=1)
    # print("Classification Report:\n", result[0])
    # print("Confusion Matrix:\n", result[1])

    # # Example usage with custom control
    # custom_y_true, custom_y_pred = G1.generate_custom_fake_data(
    #     num_test_cases=570,
    #     num_class_1=11,
    #     correct_non_exoplanets=500,
    #     correct_exoplanets=3,
    # )
    # custom_report = classification_report(
    #     custom_y_true,
    #     custom_y_pred,
    #     target_names=["non-exoplanets", "exoplanets"],
    #     output_dict=True,
    # )
    # custom_cm = confusion_matrix(custom_y_true, custom_y_pred)
    # print("Custom Classification Report:\n", custom_report)
    # print("Custom Confusion Matrix:\n", custom_cm.tolist())
# HIDE_END

