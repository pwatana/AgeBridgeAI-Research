import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yaml
import argparse

# Import the Preprocessor class from data_preprocessing.py
# Ensure your project root is the CWD when running this script for this import to work
from src.data_preprocessing import Preprocessor

class HandJointDetector:
    def __init__(self, model_path=None, img_size=(512, 512), num_joints=16, use_transfer_learning=True):
        """
        Initializes the HandJointDetector.

        Args:
            model_path (str, optional): Path to a pre-trained model to load.
            img_size (tuple): Expected input image size (height, width).
            num_joints (int): Number of hand joints to detect (e.g., 16 for a full hand).
            use_transfer_learning (bool): Whether to use transfer learning with MobileNetV2.
        """
        self.img_size = img_size
        self.num_joints = num_joints
        self.output_dim = num_joints * 2 # Each joint has an (x, y) coordinate
        self.model = None
        self.use_transfer_learning = use_transfer_learning

        # Define common hand joint names for clarity (example, adapt as needed)
        self.joint_names = [
            "Wrist",
            "Thumb_MCP", "Thumb_PIP", "Thumb_DIP",
            "Index_MCP", "Index_PIP", "Index_DIP",
            "Middle_MCP", "Middle_PIP", "Middle_DIP",
            "Ring_MCP", "Ring_PIP", "Ring_DIP",
            "Pinky_MCP", "Pinky_PIP", "Pinky_DIP"
        ]
        if len(self.joint_names) != self.num_joints:
            print(f"WARNING: Mismatch between num_joints ({self.num_joints}) and defined joint_names ({len(self.joint_names)}).")

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained joint detection model from {model_path}")
        else:
            print("No pre-trained model specified or found. A new model will be created.")
            if self.use_transfer_learning:
                print("Using MobileNetV2 for transfer learning for joint detection.")
            else:
                print("Building a simple CNN from scratch for joint detection.")

    def _build_model(self):
        """
        Builds the model, either from scratch or using transfer learning, for joint detection.
        Outputs 2*num_joints coordinates.
        """
        if self.use_transfer_learning:
            # Transfer Learning with MobileNetV2
            base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                     include_top=False,
                                     weights='imagenet')

            base_model.trainable = False # Freeze the base model layers

            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1)) # Grayscale input
            x = layers.concatenate([inputs, inputs, inputs], axis=-1) # Replicate channel
            x = preprocess_input(x) # Apply MobileNetV2's specific preprocessing

            x = base_model(x, training=False) # Pass through frozen base model

            x = layers.GlobalAveragePooling2D()(x) # Flatten features
            x = layers.Dense(256, activation='relu')(x) # Additional dense layer
            x = layers.Dropout(0.5)(x) # Dropout for regularization
            
            # Final dense layer for regression: output_dim units with linear activation
            outputs = layers.Dense(self.output_dim, activation='linear')(x) 

            model = models.Model(inputs, outputs)

            # Compile with Mean Squared Error (MSE) loss for regression
            # Use Adam optimizer with a learning rate suitable for new layers
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          loss='mse', # Mean Squared Error
                          metrics=['mae']) # Mean Absolute Error as a metric
            
            print("--- Built model using MobileNetV2 for Joint Detection (Feature Extraction) ---")
            
        else:
            # Original simple CNN from scratch
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(256, activation='relu'), # Adjusted for regression head
                layers.Dropout(0.5),
                layers.Dense(self.output_dim, activation='linear') # Linear activation for coordinates
            ])
            
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            model.compile(optimizer=optimizer,
                          loss='mse', # MSE for regression
                          metrics=['mae']) # MAE as a metric
            print("--- Built simple CNN from scratch for Joint Detection ---")

        return model

    def fine_tune_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, fine_tune_from_layer=None):
        """
        Performs fine-tuning on the pre-trained model for joint detection.
        Unfreezes some layers of the base model and continues training with a very low learning rate.
        """
        if not self.use_transfer_learning or self.model is None:
            print("Fine-tuning is only applicable to models built with transfer learning.")
            return

        print("\n--- Starting Fine-tuning Phase for Joint Detection ---")
        
        base_model_layer = self.model.layers[3] 
        base_model_layer.trainable = True 

        if fine_tune_from_layer is not None:
            for layer in base_model_layer.layers[:fine_tune_from_layer]:
                layer.trainable = False
            print(f"Unfroze layers from index {fine_tune_from_layer} of the MobileNetV2 base model for fine-tuning.")
        else:
            print("Unfroze all layers of the MobileNetV2 base model for fine-tuning.")

        # Re-compile the model with a very low learning rate for fine-tuning
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate
                          loss='mse',
                          metrics=['mae'])

        self.model.summary() # Review trainable parameters after unfreezing
        
        history_fine_tune = self.model.fit(X_train, y_train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, y_val),
                                           verbose=1)
        print("\nFine-tuning completed for Joint Detection.")
        return history_fine_tune

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path=None, fine_tune_epochs=0, fine_tune_from_layer=None):
        """
        Trains the joint detection model.
        Includes an optional fine-tuning phase if using transfer learning.

        Args:
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): Training joint coordinates (flattened).
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation joint coordinates (flattened).
            epochs (int): Number of training epochs for initial (head) training.
            batch_size (int): Batch size for training.
            model_save_path (str, optional): Path to save the trained model.
            fine_tune_epochs (int): Number of epochs for fine-tuning phase (0 to skip).
            fine_tune_from_layer (int): Index to start unfreezing layers for fine-tuning.
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.summary()

        print("\nStarting model training (initial head training for joint detection)...")
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 verbose=1)

        print("\nInitial training completed for Joint Detection.")
        
        if self.use_transfer_learning and fine_tune_epochs > 0:
            self.fine_tune_model(X_train, y_train, X_val, y_val, 
                                 epochs=fine_tune_epochs, 
                                 batch_size=batch_size,
                                 fine_tune_from_layer=fine_tune_from_layer)

        if model_save_path:
            self.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return history

    def predict(self, image_np):
        """
        Predicts joint coordinates for a single preprocessed image.

        Args:
            image_np (np.ndarray): A single preprocessed grayscale image (H, W).

        Returns:
            np.ndarray: Predicted joint coordinates (num_joints, 2) format.
            np.ndarray: Raw prediction output (flattened).
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        # Ensure image is in the correct format: (1, H, W, 1) for the model's input layer
        if image_np.ndim == 2: # (H, W)
            input_image = np.expand_dims(np.expand_dims(image_np, axis=0), axis=-1) # -> (1, H, W, 1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 1: # (H, W, 1)
             input_image = np.expand_dims(image_np, axis=0) # -> (1, H, W, 1)
        else:
            raise ValueError(f"Unexpected image dimension: {image_np.shape}. Expected (H, W) or (H, W, 1).")

        raw_predictions = self.model.predict(input_image)[0] # Get the first (and only) sample's prediction
        
        # Reshape the flattened output into (num_joints, 2)
        predicted_coords = raw_predictions.reshape((self.num_joints, 2))
        return predicted_coords, raw_predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on a test set and prints MSE/MAE.
        Also visualizes some predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Cannot evaluate.")
        
        print("\nEvaluating model on test set for joint detection...")
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")

        # --- Visualize some predictions ---
        print("\n--- Visualizing Sample Joint Predictions ---")
        num_samples_to_show = min(5, len(X_test)) # Show up to 5 samples

        plt.figure(figsize=(15, 6 * num_samples_to_show))
        for i in range(num_samples_to_show):
            sample_image = X_test[i, :, :, 0] # Remove channel dim for plotting
            true_coords = y_test[i].reshape((self.num_joints, 2))
            predicted_coords, _ = self.predict(sample_image) # Predict for the single image

            plt.subplot(num_samples_to_show, 2, 2*i + 1)
            plt.imshow(sample_image, cmap='gray')
            plt.scatter(true_coords[:, 0], true_coords[:, 1], c='red', marker='o', s=50, label='True Joints')
            plt.scatter(predicted_coords[:, 0], predicted_coords[:, 1], c='blue', marker='x', s=50, label='Predicted Joints')
            for j in range(self.num_joints):
                plt.text(true_coords[j, 0] + 5, true_coords[j, 1] + 5, self.joint_names[j], color='red', fontsize=8)
                plt.text(predicted_coords[j, 0] + 5, predicted_coords[j, 1] - 5, f'P{j}', color='blue', fontsize=8) # Pj for predicted
            plt.title(f'Sample {i+1}: True vs Predicted Joints')
            plt.axis('off')
            plt.legend()
            
            # Optionally, plot only predicted
            plt.subplot(num_samples_to_show, 2, 2*i + 2)
            plt.imshow(sample_image, cmap='gray')
            plt.scatter(predicted_coords[:, 0], predicted_coords[:, 1], c='blue', marker='x', s=50, label='Predicted Joints')
            for j in range(self.num_joints):
                plt.text(predicted_coords[j, 0] + 5, predicted_coords[j, 1] + 5, self.joint_names[j], color='blue', fontsize=8)
            plt.title(f'Sample {i+1}: Predicted Joints')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Saves the trained model."""
        self.model.save(path)

    def load_model(self, path):
        """Loads a pre-trained model."""
        return keras.models.load_model(path)


# --- Command-line Interface (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict hand joint locations.")
    parser.add_argument("--mode", type=str, choices=['train_joints', 'predict_joints'], default='train_joints',
                        help="Operation mode: 'train_joints' for training a new joint detection model, 'predict_joints' for predicting on an image.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image for prediction (required if mode is 'predict_joints').")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained joint detection model (.h5) to load for prediction or fine-tuning.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for initial training mode.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training mode.")
    parser.add_argument("--use_transfer_learning", action='store_true',
                        help="Use MobileNetV2 for transfer learning (default is scratch CNN for joints).")
    parser.add_argument("--fine_tune_epochs", type=int, default=0,
                        help="Number of epochs for fine-tuning phase (0 to skip fine-tuning). Only applicable with transfer learning.")
    parser.add_argument("--fine_tune_from_layer", type=int, default=None,
                        help="Index of the layer in the base model to start unfreezing from for fine-tuning. "
                             "E.g., 100 to unfreeze layers after 100. If None, unfreezes all base model layers.")
    parser.add_argument("--num_joints", type=int, default=16,
                        help="Number of hand joints to detect (default: 16).")

    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed') # Output dir for preprocessed images
    joint_model_dir = os.path.join(base_dir, 'models', 'joint_detection') # Dedicated dir for joint models
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml')

    os.makedirs(joint_model_dir, exist_ok=True)
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
    joint_model_save_path = os.path.join(joint_model_dir, 'hand_joint_detection_model.h5')


    if args.mode == 'train_joints':
        print("\n--- Running in TRAINING mode for Joint Detection ---")
        # --- SIMULATE DATA LOADING AND LABELING ---
        # IMPORTANT: For real joint detection, you would need a dataset of X-ray images
        # WITH ANNOTATED JOINT COORDINATES. This is a complex task and requires
        # specialized datasets (e.g., from medical imaging challenges).
        # Here, we generate random coordinates for demonstration.
        print(f"\n--- Simulating Data Loading for Joint Detection ({args.num_joints} joints) ---")
        num_samples = 200 

        # Generate dummy images (random noise)
        X = np.random.rand(num_samples, img_size[0], img_size[1], 1).astype(np.float32) * 255.0
        
        # Generate dummy joint coordinates: random (x,y) within image bounds
        # Each sample has `num_joints * 2` values (x1, y1, x2, y2, ...)
        y = np.random.rand(num_samples, args.num_joints * 2).astype(np.float32)
        y[:, ::2] = y[:, ::2] * img_size[1] # Scale x-coordinates to image width
        y[:, 1::2] = y[:, 1::2] * img_size[0] # Scale y-coordinates to image height

        print(f"Generated {len(y)} dummy samples for joint detection training.")

        # Split data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 of 0.8 = 0.2 total val

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # --- TRAIN THE MODEL ---
        joint_detector = HandJointDetector(model_path=args.model_path, 
                                            img_size=img_size, 
                                            num_joints=args.num_joints,
                                            use_transfer_learning=args.use_transfer_learning)
        
        joint_detector.train(X_train, y_train, X_val, y_val, 
                             epochs=args.epochs, batch_size=args.batch_size, 
                             model_save_path=joint_model_save_path,
                             fine_tune_epochs=args.fine_tune_epochs,
                             fine_tune_from_layer=args.fine_tune_from_layer)

        # --- EVALUATE THE MODEL ---
        joint_detector.evaluate(X_test, y_test)

    elif args.mode == 'predict_joints':
        print("\n--- Running in PREDICTION mode for Joint Detection ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict_joints'.")
        if not args.model_path:
            if os.path.exists(joint_model_save_path):
                print(f"Using default model: {joint_model_save_path}")
                args.model_path = joint_model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists or provided.")

        print(f"Attempting to predict joints for: {args.image_path}")
        
        # Load and preprocess the image using the Preprocessor
        raw_image = preprocessor._load_image(args.image_path)
        if raw_image is None:
            print(f"Could not load image at {args.image_path}. Exiting prediction mode.")
            exit()
        
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

        # Instantiate HandJointDetector to load the model
        joint_detector = HandJointDetector(model_path=args.model_path, 
                                            img_size=img_size,
                                            num_joints=args.num_joints,
                                            use_transfer_learning=False) # False, as _build_model is skipped if model_path exists
        
        if joint_detector.model is None:
             print("Error: Model could not be loaded for prediction. Please check --model_path.")
             exit()

        predicted_coords, _ = joint_detector.predict(image_to_predict)
        
        print(f"Predicted Joint Coordinates (first 5 joints):")
        for j in range(min(5, args.num_joints)):
            print(f"  {joint_detector.joint_names[j]}: ({predicted_coords[j, 0]:.2f}, {predicted_coords[j, 1]:.2f})")

        # Visualize the predicted joints on the image
        plt.imshow(image_to_predict, cmap='gray')
        plt.scatter(predicted_coords[:, 0], predicted_coords[:, 1], c='blue', marker='x', s=50, label='Predicted Joints')
        for j in range(args.num_joints):
            plt.text(predicted_coords[j, 0] + 5, predicted_coords[j, 1] + 5, joint_detector.joint_names[j], color='blue', fontsize=8)
        plt.title(f"Predicted Joints for {os.path.basename(args.image_path)}")
        plt.axis('off')
        plt.show()

    else:
        print("Invalid mode specified. Choose 'train_joints' or 'predict_joints'.")