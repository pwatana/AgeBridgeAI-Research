import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report # Added classification_report here too

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The file path to the YAML configuration.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration from {config_path}: {e}")
        return None

def plot_training_history(history, model_name="", metrics=['accuracy', 'loss']):
    """
    Plots the training and validation history for specified metrics.

    Args:
        history (keras.callbacks.History): The History object returned from model.fit().
        model_name (str): Optional name of the model for plot titles.
        metrics (list): List of metric names to plot (e.g., ['accuracy', 'loss', 'mae']).
                        Should correspond to keys in history.history.
    """
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(1, len(metrics), i + 1)
            plt.plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{model_name} Training History: {metric.replace("_", " ").title()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace("_", " ").title())
            plt.legend()
            plt.grid(True)
        else:
            print(f"Warning: Metric '{metric}' not found in training history.")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred_classes, class_names, title='Confusion Matrix'):
    """
    Plots a confusion matrix using seaborn.

    Args:
        y_true (array-like): True labels.
        y_pred_classes (array-like): Predicted class labels.
        class_names (list): List of class names (e.g., ['Non-RA', 'RA']).
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Plots the ROC curve and calculates AUC.

    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        title (str): Title for the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# You can add more utility functions here as needed, e.g.,
# - `save_image_with_points(image, points, save_path)`
# - `normalize_data(data, min_val, max_val)` if needed beyond preprocessor
# - Custom callbacks, etc.

if __name__ == "__main__":
    # Example Usage of model_utils (for testing purposes)
    print("--- Testing model_utils.py functions ---")

    # 1. Test load_config
    # Create a dummy config.yaml if it doesn't exist
    dummy_config_path = "dummy_config.yaml"
    if not os.path.exists(dummy_config_path):
        with open(dummy_config_path, 'w') as f:
            yaml.dump({'setting1': 10, 'setting2': 'value'}, f)
        print(f"Created dummy config: {dummy_config_path}")
    
    config = load_config(dummy_config_path)
    if config:
        print(f"Loaded config: {config}")

    # 2. Test plot_training_history
    print("\nTesting plot_training_history...")
    dummy_history = {
        'loss': np.random.rand(10) * 0.5 + 0.1,
        'val_loss': np.random.rand(10) * 0.6 + 0.15,
        'accuracy': np.random.rand(10) * 0.2 + 0.7,
        'val_accuracy': np.random.rand(10) * 0.2 + 0.65,
        'mae': np.random.rand(10) * 0.1 + 0.05,
        'val_mae': np.random.rand(10) * 0.1 + 0.06
    }
    class DummyHistory: # Mimic Keras History object
        def __init__(self, history_dict):
            self.history = history_dict
    
    plot_training_history(DummyHistory(dummy_history), "Dummy Model", metrics=['loss', 'accuracy', 'mae'])

    # 3. Test plot_confusion_matrix
    print("\nTesting plot_confusion_matrix...")
    y_true_cm = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_pred_cm = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
    class_names_cm = ["Class A", "Class B"]
    plot_confusion_matrix(y_true_cm, y_pred_cm, class_names_cm, title="Dummy Confusion Matrix")

    # 4. Test plot_roc_curve
    print("\nTesting plot_roc_curve...")
    y_true_roc = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_pred_proba_roc = np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.4, 0.9, 0.6, 0.25, 0.55])
    plot_roc_curve(y_true_roc, y_pred_proba_roc, title="Dummy ROC Curve")

    # Clean up dummy config file
    if os.path.exists(dummy_config_path):
        os.remove(dummy_config_path)
        print(f"\nCleaned up dummy config: {dummy_config_path}")