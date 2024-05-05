import json
import random
import numpy as np
import os


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

        return json.dumps(report, indent=4), confusion_matrix

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


# Example usage:
report, confusion_matrix = G.g(1)
print(report)
print(confusion_matrix)
G.update_reports_with_fake_json([1, 2, 3])
