import random
import numpy as np
import json
import os


class G:
    @staticmethod
    def generate_random_metric(baseline, deviation):
        """Generate a random metric around a baseline with given deviation."""
        return round(random.uniform(baseline - deviation, baseline + deviation), 4)

    @staticmethod
    def calculate_confusion_matrix_from_classification_report(report):
        """Calculate a confusion matrix (as a NumPy array) using the precision, recall, and support from a classification report."""
        precision_1 = report["1"]["precision"]
        recall_1 = report["1"]["recall"]
        support_1 = report["1"]["support"]

        TP = int(round(recall_1 * support_1))
        FN = support_1 - TP

        precision_0 = report["0"]["precision"]
        support_0 = report["0"]["support"]

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

        support_0 = 566 # Assuming the support could vary around 100
        support_1 = 5 # Assuming the support could vary around 100

        report = {
            "0": {
                "precision": precision_0,
                "recall": recall_0,
                "f1-score": f1_score_0,
                "support": support_0,
            },
            "1": {
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


# Example usage
quality = 2
report, matrix = G.g(quality)
print(type(report))
print(report)
print(matrix)
