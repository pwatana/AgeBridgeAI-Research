# ra_svh_prototype/src/left_right_hand_identification.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# Import the Preprocessor class from data_preprocessing.py
from data_preprocessing import Preprocessor

class HandLateralityIdentifier:
    def __init__(self, model_path=None, img_size=(512, 512), num_classes=2):
        """
        Initializes the HandLateralityIdentifier.

        Args:
            model_path (str, optional): Path to a pre-trained model to load.
            img_size (tuple): Expected input image size (height, width).
            num_classes (int): Number of output classes (2 for Left/Right).
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ["Left Hand", "Right Hand"] # Assuming 0: Left, 1: Right

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("No pre-trained model specified or found. A new model will be created.")

    def _build_model(self):
        """
        Builds a simple Convolutional Neural Network (CNN) model for laterality classification.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), # Add dropout for regularization
            layers.Dense(self.num_classes, activation='softmax') # Softmax for multi-class, even if binary
        ])
        
        # Using Adam optimizer with a learning rate schedule is good practice
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=10000,
            decay_rate=0.9)
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Use SparseCategorical for integer labels
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path=None):
        """
        Trains the laterality identification model.

        Args:
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            model_save_path (str, optional): Path to save the trained model.
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.summary()

        print("\nStarting model training...")
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 verbose=1)

        print("\nTraining completed.")
        if model_save_path:
            self.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return history

    def predict(self, image_np):
        """
        Predicts the laterality of a single preprocessed image.

        Args:
            image_np (np.ndarray): A single preprocessed grayscale image (H, W).

        Returns:
            str: Predicted laterality ("Left Hand" or "Right Hand").
            np.ndarray: Prediction probabilities.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        # Ensure image is in the correct format: (1, H, W, 1)
        if image_np.ndim == 2: # (H, W)
            input_image = np.expand_dims(np.expand_dims(image_np, axis=0), axis=-1) # -> (1, H, W, 1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 1: # (H, W, 1)
             input_image = np.expand_dims(image_np, axis=0) # -> (1, H, W, 1)
        else:
            raise ValueError(f"Unexpected image dimension: {image_np.shape}. Expected (H, W) or (H, W, 1).")

        predictions = self.model.predict(input_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        return self.class_names[predicted_class_idx], predictions[0]

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on a test set and prints a classification report.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Cannot evaluate.")
        
        print("\nEvaluating model on test set...")
        y_pred_probs = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix for Hand Laterality Identification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def save_model(self, path):
        """Saves the trained model."""
        self.model.save(path)

    def load_model(self, path):
        """Loads a pre-trained model."""
        return keras.models.load_model(path)


# --- Example Usage (when run directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict hand laterality.")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], default='train',
                        help="Operation mode: 'train' for training a new model, 'predict' for predicting on an image.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image for prediction (required if mode is 'predict').")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained model (.h5) to load for prediction or fine-tuning.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training mode.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training mode.")

    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed') # Not directly used in this script's example, but good to have
    laterality_model_dir = os.path.join(base_dir, 'models', 'laterality_identification')
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml')

    os.makedirs(laterality_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Load preprocessing configuration
    if not os.path.exists(config_path):
        print(f"WARNING: No config file found at {config_path}. Creating a dummy one.")
        dummy_config = {
            'normalize': {'target_range': [0, 255]},
            'clahe': {'apply': True, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
            'denoising': {'apply': True, 'kernel_size': [3, 3]},
            'resize': {'target_size': [512, 512]}
        }
        with open(config_path, 'w') as f:
            yaml.dump(dummy_config, f, default_flow_style=False)
        preprocessing_config = dummy_config
    else:
        with open(config_path, 'r') as f:
            preprocessing_config = yaml.safe_load(f)

    # Instantiate Preprocessor
    preprocessor = Preprocessor(preprocessing_config)
    img_size = tuple(preprocessing_config['resize']['target_size'])
    model_save_path = os.path.join(laterality_model_dir, 'hand_laterality_model.h5')


    if args.mode == 'train':
        print("\n--- Running in TRAINING mode ---")
        # --- SIMULATE DATA LOADING AND LABELING ---
        # In a real scenario, you would load your preprocessed images and their actual laterality labels.
        # For demonstration, we'll create dummy images and labels.
        print("\n--- Simulating Data Loading for Laterality Identification ---")
        num_samples = 200 # Total samples
        left_samples = num_samples // 2
        right_samples = num_samples - left_samples

        # Create dummy images (random noise, but consistent shape)
        # Ensure they have a channel dimension for the CNN (H, W, 1)
        X = np.random.rand(num_samples, img_size[0], img_size[1], 1).astype(np.float32) * 255.0
        
        # Create dummy labels (0 for Left, 1 for Right)
        y = np.array([0] * left_samples + [1] * right_samples)
        np.random.shuffle(y) # Shuffle labels to mix left and right

        print(f"Generated {len(y)} dummy samples for laterality training.")
        print(f"Label distribution: Left (0): {np.sum(y == 0)}, Right (1): {np.sum(y == 1)}")

        # Split data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 of 0.8 = 0.2 total val

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # --- TRAIN THE MODEL ---
        # Load model if specified, otherwise create new
        laterality_identifier = HandLateralityIdentifier(model_path=args.model_path, img_size=img_size) 
        laterality_identifier.train(X_train, y_train, X_val, y_val, 
                                    epochs=args.epochs, batch_size=args.batch_size, 
                                    model_save_path=model_save_path)

        # --- EVALUATE THE MODEL ---
        laterality_identifier.evaluate(X_test, y_test)

    elif args.mode == 'predict':
        print("\n--- Running in PREDICTION mode ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict'.")
        if not args.model_path:
            # Check for default model save path
            if os.path.exists(model_save_path):
                print(f"Using default model: {model_save_path}")
                args.model_path = model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists.")

        print(f"Attempting to predict laterality for: {args.image_path}")
        
        # Load and preprocess the image
        raw_image = preprocessor._load_image(args.image_path)
        if raw_image is None:
            print(f"Could not load image at {args.image_path}. Exiting prediction mode.")
            exit()
        
        # The preprocess pipeline in data_preprocessing.py handles resizing, normalization, etc.
        # We need to ensure it returns the image in the format expected by predict (H, W) or (H, W, 1)
        # For this, we'll manually apply parts of the pipeline needed for prediction,
        # or adapt the _load_image and _normalize_image to match what `predict` expects.
        # Given `predict` expects (H,W) or (H,W,1), let's ensure the output from preprocessor is correct.
        
        # Apply preprocessing steps relevant for laterality classification
        # The `process_image` method in `data_preprocessing.py` saves to disk and returns a uint8 image (0-255).
        # We need to load it and convert back to float if necessary for the model.
        
        # For direct prediction, we load the image and apply relevant transforms
        # assuming the model was trained on float32 images.
        image_to_predict = preprocessor._normalize_image(raw_image, target_range=preprocessing_config['normalize']['target_range'])
        
        if preprocessing_config['clahe']['apply']:
            image_to_predict = preprocessor._apply_clahe(
                image_to_predict,
                clip_limit=preprocessing_config['clahe']['clip_limit'],
                tile_grid_size=tuple(preprocessing_config['clahe']['tile_grid_size'])
            )
        
        if preprocessing_config['denoising']['apply']:
            image_to_predict = preprocessor._apply_gaussian_denoising(
                image_to_predict,
                kernel_size=tuple(preprocessing_config['denoising']['kernel_size'])
            )
        
        image_to_predict = preprocessor._resize_image(image_to_predict, target_size=img_size)
        
        # Segment hand (if implemented, otherwise it passes through)
        image_to_predict, _ = preprocessor._segment_hand(image_to_predict)

        # Ensure final image is float32 and correct dimensions for predict method (H, W)
        image_to_predict = image_to_predict.astype(np.float32)

        laterality_identifier = HandLateralityIdentifier(model_path=args.model_path, img_size=img_size)
        
        if laterality_identifier.model is None:
             print("Error: Model could not be loaded for prediction. Please check --model_path.")
             exit()

        predicted_label, probabilities = laterality_identifier.predict(image_to_predict)
        
        print(f"Predicted Laterality: {predicted_label}")
        print(f"Prediction Probabilities: {probabilities}")

        # Visualize the predicted image
        plt.imshow(image_to_predict, cmap='gray')
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()

    else:
        print("Invalid mode specified. Choose 'train' or 'predict'.")