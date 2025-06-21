import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2 # Import pre-trained model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Preprocessing for MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# Import the Preprocessor class from data_preprocessing.py
# Ensure your project root is the CWD when running this script for this import to work
from src.data_preprocessing import Preprocessor

class HandLateralityIdentifier:
    def __init__(self, model_path=None, img_size=(512, 512), num_classes=2, use_transfer_learning=True):
        """
        Initializes the HandLateralityIdentifier.

        Args:
            model_path (str, optional): Path to a pre-trained model to load.
            img_size (tuple): Expected input image size (height, width).
            num_classes (int): Number of output classes (2 for Left/Right).
            use_transfer_learning (bool): Whether to use transfer learning with MobileNetV2.
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ["Left Hand", "Right Hand"] # Assuming 0: Left, 1: Right
        self.use_transfer_learning = use_transfer_learning

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print("No pre-trained model specified or found. A new model will be created.")
            if self.use_transfer_learning:
                print("Using MobileNetV2 for transfer learning.")
            else:
                print("Building a simple CNN from scratch.")

    def _build_model(self):
        """
        Builds the model, either from scratch or using transfer learning.
        """
        if self.use_transfer_learning:
            # Transfer Learning with MobileNetV2
            # MobileNetV2 expects 3-channel input, so we'll adjust the input later
            # (H, W, 3) input for the base model, even if our images are grayscale.
            # A common strategy is to replicate the grayscale channel 3 times.
            
            # Load the MobileNetV2 model pre-trained on ImageNet
            # include_top=False means we don't include the classification head
            # weights='imagenet' loads the pre-trained weights
            base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                     include_top=False,
                                     weights='imagenet')

            # Freeze the base model layers
            # This makes them non-trainable, so only the new layers are trained initially
            base_model.trainable = False

            # Define the model's input
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1)) # Our grayscale input
            
            # Replicate the single grayscale channel to 3 channels to match MobileNetV2's expected input
            x = layers.concatenate([inputs, inputs, inputs], axis=-1)
            
            # Apply MobileNetV2's specific preprocessing
            # This scales pixel values to the range [-1, 1] which MobileNetV2 expects
            x = preprocess_input(x) 

            # Pass the processed input through the frozen base model
            # training=False ensures batch_normalization layers in the base model use their
            # global means and variances, not batch statistics, which is standard for inference.
            x = base_model(x, training=False) 

            x = layers.GlobalAveragePooling2D()(x) # Global average pooling to flatten features
            x = layers.Dropout(0.5)(x) # Dropout for regularization
            outputs = layers.Dense(self.num_classes, activation='softmax')(x) # Our classification head

            model = models.Model(inputs, outputs)

            # Compile the model for the initial training of the new head layers
            # Use a relatively higher learning rate for the new layers
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
            
            print("--- Built model using MobileNetV2 for Feature Extraction ---")
            
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
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Adam optimizer with a learning rate schedule
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])
            print("--- Built simple CNN from scratch ---")

        return model

    def fine_tune_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, fine_tune_from_layer=None):
        """
        Performs fine-tuning on the pre-trained model.
        Unfreezes some layers of the base model and continues training with a very low learning rate.

        Args:
            X_train, y_train, X_val, y_val: Data for fine-tuning.
            epochs (int): Number of fine-tuning epochs.
            batch_size (int): Batch size.
            fine_tune_from_layer (int): Index of the layer from which to unfreeze for fine-tuning.
                                        If None, all layers of base_model are unfrozen.
        """
        if not self.use_transfer_learning or self.model is None:
            print("Fine-tuning is only applicable to models built with transfer learning.")
            return

        print("\n--- Starting Fine-tuning Phase ---")
        
        # Unfreeze the base model layers
        # The base_model is usually one of the initial layers in your Keras Model object.
        # For our functional API model (inputs -> concatenate -> preprocess_input -> base_model -> ...),
        # the MobileNetV2 base model is at index 3 (Input, Concatenate, Lambda (for preprocess_input), MobileNetV2).
        base_model_layer = self.model.layers[3] 
        base_model_layer.trainable = True # Unfreeze the base_model layer itself

        # Optionally, fine-tune only a portion of the base model
        if fine_tune_from_layer is not None:
            # Freeze layers up to `fine_tune_from_layer` within the base_model
            for layer in base_model_layer.layers[:fine_tune_from_layer]:
                layer.trainable = False
            print(f"Unfroze layers from index {fine_tune_from_layer} of the MobileNetV2 base model for fine-tuning.")
        else:
            print("Unfroze all layers of the MobileNetV2 base model for fine-tuning.")

        # Re-compile the model with a very low learning rate for fine-tuning
        # This is crucial for fine-tuning to avoid destroying learned features
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

        self.model.summary() # Review trainable parameters after unfreezing
        
        history_fine_tune = self.model.fit(X_train, y_train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, y_val),
                                           verbose=1)
        print("\nFine-tuning completed.")
        return history_fine_tune

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path=None, fine_tune_epochs=0, fine_tune_from_layer=None):
        """
        Trains the laterality identification model.
        Includes an optional fine-tuning phase if using transfer learning.

        Args:
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of training epochs for initial (head) training.
            batch_size (int): Batch size for training.
            model_save_path (str, optional): Path to save the trained model.
            fine_tune_epochs (int): Number of epochs for fine-tuning phase (0 to skip).
            fine_tune_from_layer (int): Index to start unfreezing layers for fine-tuning.
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.summary()

        print("\nStarting model training (initial head training)...")
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 verbose=1)

        print("\nInitial training completed.")
        
        # Perform fine-tuning if enabled and using transfer learning
        if self.use_transfer_learning and fine_tune_epochs > 0:
            self.fine_tune_model(X_train, y_train, X_val, y_val, 
                                 epochs=fine_tune_epochs, 
                                 batch_size=batch_size,
                                 fine_tune_from_layer=fine_tune_from_layer)

        if model_save_path:
            self.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return history # This might need to be adjusted if you want combined history (e.g., plot both histories)

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

        # Ensure image is in the correct format: (1, H, W, 1) for the model's input layer
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


# --- Command-line Interface (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict hand laterality.")
    parser.add_argument("--mode", type=str, choices=['train', 'predict'], default='train',
                        help="Operation mode: 'train' for training a new model, 'predict' for predicting on an image.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image for prediction (required if mode is 'predict').")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained model (.h5) to load for prediction or fine-tuning.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for initial training mode.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training mode.")
    parser.add_argument("--use_transfer_learning", action='store_true',
                        help="Use MobileNetV2 for transfer learning (default is scratch CNN).")
    parser.add_argument("--fine_tune_epochs", type=int, default=0,
                        help="Number of epochs for fine-tuning phase (0 to skip fine-tuning). Only applicable with transfer learning.")
    parser.add_argument("--fine_tune_from_layer", type=int, default=None,
                        help="Index of the layer in the base model to start unfreezing from for fine-tuning. "
                             "E.g., 100 to unfreeze layers after 100. If None, unfreezes all base model layers.")

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
        # IMPORTANT: For real transfer learning, you'd replace this with loading
        # your actual preprocessed X-ray images and their laterality labels.
        # This typically involves:
        # 1. Listing all your image files from your dataset.
        # 2. Extracting labels (e.g., 'left'/'right') from filenames or a CSV.
        # 3. Loading each image using preprocessor._load_image.
        # 4. Applying the necessary preprocessing steps (normalize, clahe, resize etc.).
        # 5. Storing them in X (numpy array) and y (labels).
        # You would likely need a separate function/script to prepare your actual dataset into X and y.
        print("\n--- Simulating Data Loading for Laterality Identification (REPLACE WITH REAL DATA) ---")
        num_samples = 200 # Total samples for dummy data
        left_samples = num_samples // 2
        right_samples = num_samples - left_samples

        # Create dummy images (random noise, but consistent shape and type)
        # Ensure they have a channel dimension for the CNN (H, W, 1) and are float32
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
        laterality_identifier = HandLateralityIdentifier(model_path=args.model_path, 
                                                        img_size=img_size, 
                                                        use_transfer_learning=args.use_transfer_learning)
        
        laterality_identifier.train(X_train, y_train, X_val, y_val, 
                                    epochs=args.epochs, batch_size=args.batch_size, 
                                    model_save_path=model_save_path,
                                    fine_tune_epochs=args.fine_tune_epochs,
                                    fine_tune_from_layer=args.fine_tune_from_layer)

        # --- EVALUATE THE MODEL ---
        laterality_identifier.evaluate(X_test, y_test)

    elif args.mode == 'predict':
        print("\n--- Running in PREDICTION mode ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict'.")
        if not args.model_path:
            if os.path.exists(model_save_path):
                print(f"Using default model: {model_save_path}")
                args.model_path = model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists or provided.")

        print(f"Attempting to predict laterality for: {args.image_path}")
        
        # Load and preprocess the image using the Preprocessor
        # The Preprocessor's methods are designed to return float32
        raw_image = preprocessor._load_image(args.image_path)
        if raw_image is None:
            print(f"Could not load image at {args.image_path}. Exiting prediction mode.")
            exit()
        
        # Apply preprocessing steps similar to the training pipeline
        # Note: The `process_image` method in data_preprocessing saves to disk.
        # For prediction, we manually chain the steps to get the NumPy array directly.
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

        # Instantiate HandLateralityIdentifier to load the model
        # The use_transfer_learning flag doesn't matter much here IF a model_path is provided
        # because the model will be loaded directly from the .h5 file.
        # It's primarily used during the _build_model phase if no model_path is given.
        laterality_identifier = HandLateralityIdentifier(model_path=args.model_path, 
                                                        img_size=img_size,
                                                        use_transfer_learning=False) # Set to False as _build_model is skipped if model_path exists
        
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