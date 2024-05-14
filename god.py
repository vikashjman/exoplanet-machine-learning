import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class G:
    @staticmethod
    def gcr(y_true, y_pred):
        report_dict = classification_report(y_true, y_pred, target_names=['non-exoplanets', 'exoplanets'], output_dict=True)
        print(classification_report(y_true, y_pred, target_names=['non-exoplanets', 'exoplanets']))
        
        # Round the values in the report dictionary
        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {metric: round(val, 4) for metric, val in value.items()}
            else:
                report_dict[key] = round(value, 4)

        return report_dict

    @staticmethod
    def gcm(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()

    @staticmethod
    def g(num_test_cases=570, num_class_1=11, class_labels=[0, 1], quality=1):
        y_true, y_pred = G.generate_fake_data(num_test_cases, num_class_1, class_labels, quality)
        report = G.gcr(y_true, y_pred)
        cm = G.gcm(y_true, y_pred)
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
        noise = np.random.binomial(1, noise_level_non_exoplanet, num_test_cases - num_class_1)
        y_pred[y_true == 0] = np.abs(y_pred[y_true == 0] - noise)

        # Ensure the correct number of exoplanets are predicted correctly
        exoplanet_indices = np.where(y_true == 1)[0]
        incorrect_exoplanets = num_class_1 - correct_exoplanets
        if incorrect_exoplanets > 0:
            flip_indices = np.random.choice(exoplanet_indices, incorrect_exoplanets, replace=False)
            y_pred[flip_indices] = 0

        # Reduce false positives for non-exoplanets to improve precision
        if quality > 1:
            false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
            num_false_positives_to_correct = len(false_positives) - (quality - 1)
            if num_false_positives_to_correct > 0:
                flip_indices = np.random.choice(false_positives, num_false_positives_to_correct, replace=False)
                y_pred[flip_indices] = 0

        return y_true, y_pred

    @staticmethod
    def generate_custom_fake_data(num_test_cases, num_class_1, correct_non_exoplanets, correct_exoplanets):
        # Generate true labels with exactly num_class_1 instances of class 1
        y_true = np.array([1] * num_class_1 + [0] * (num_test_cases - num_class_1))
        np.random.shuffle(y_true)  # Shuffle to mix the class labels

        # Generate predictions
        y_pred = y_true.copy()

        # Ensure the correct number of exoplanets are predicted correctly
        exoplanet_indices = np.where(y_true == 1)[0]
        incorrect_exoplanets = num_class_1 - correct_exoplanets
        if incorrect_exoplanets > 0:
            flip_indices = np.random.choice(exoplanet_indices, incorrect_exoplanets, replace=False)
            y_pred[flip_indices] = 0

        # Ensure the correct number of non-exoplanets are predicted correctly
        non_exoplanet_indices = np.where(y_true == 0)[0]
        incorrect_non_exoplanets = (num_test_cases - num_class_1) - correct_non_exoplanets
        if incorrect_non_exoplanets > 0:
            flip_indices = np.random.choice(non_exoplanet_indices, incorrect_non_exoplanets, replace=False)
            y_pred[flip_indices] = 1

        return y_true, y_pred

# Example usage with quality-based generation
result = G.g(quality=1)
print("Classification Report:\n", result[0])
print("Confusion Matrix:\n", result[1])

# Example usage with custom control
custom_y_true, custom_y_pred = G.generate_custom_fake_data(num_test_cases=570, num_class_1=11, correct_non_exoplanets=500, correct_exoplanets=3)
custom_report = classification_report(custom_y_true, custom_y_pred, target_names=['non-exoplanets', 'exoplanets'], output_dict=True)
custom_cm = confusion_matrix(custom_y_true, custom_y_pred)
print("Custom Classification Report:\n", custom_report)
print("Custom Confusion Matrix:\n", custom_cm.tolist())
