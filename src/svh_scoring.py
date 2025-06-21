import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Added r2_score for regression evaluation
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# Import the Preprocessor class from data_preprocessing.py
# This import assumes that the project root (e.g., 'AgeBridgeAI-Research')
# is the current working directory when this script is executed.
from src.data_preprocessing import Preprocessor

class SVHSccorer:
    def __init__(self, model_path=None, img_size=(512, 512), use_transfer_learning=True):
        """
        Initializes the SVHSccorer model.

        This class handles the creation, training, evaluation, and prediction
        for predicting the Sharp/Van der Heijde (SVH) score from X-ray hand images.
        It supports building a model from scratch or using a transfer learning
        approach with a MobileNetV2 backbone.

        Args:
            model_path (str, optional): Path to a pre-trained model to load.
                                        If provided, the model will be loaded instead
                                        of being built from scratch or via transfer learning.
            img_size (tuple): The target size (height, width) for input images.
                              This should match the output size from the preprocessing step.
            use_transfer_learning (bool): If True, a MobileNetV2 base will be used with
                                          ImageNet pre-trained weights. If False, a
                                          simple convolutional neural network will be built
                                          from scratch.
        """
        self.img_size = img_size
        self.output_dim = 1 # Single numerical score output
        self.model = None
        self.use_transfer_learning = use_transfer_learning

        # Define the theoretical max SVH score for normalization reference
        # Max erosion score (16 joints * 5 per joint + 10 carpal joints * 5 per joint) = (16+10)*5 = 130
        # Max JSN score (16 joints * 4 per joint) = 16*4 = 64
        # Total: (16 joints * 10 per joint for erosion + JSN) + (10 carpal joints * 5 per joint erosion) = 160 + 50 = 210 for erosion.
        # Max JSN (16 joints * 4) = 64
        # Total max Sharp score typically cited as 448 (from 2 hands: 2 * (16 joints * (5 erosion + 4 JSN) + 10 carpal * 5 erosion) )
        # For a single hand, max is often cited around 210-220 for erosions and JSN combined.
        # We will assume a max score for a single hand could be around 220 for dummy data.
        self.max_svh_score = 220.0 # This should be set based on the specific scoring method/dataset
        # In a real scenario, this would typically be normalized to a 0-1 range for model output

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained SVH scoring model from {model_path}")
        else:
            print("No pre-trained model specified or found. A new model will be created.")
            if self.use_transfer_learning:
                print("Using MobileNetV2 for transfer learning for SVH scoring.")
            else:
                print("Building a simple CNN from scratch for SVH scoring.")

    def _build_model(self):
        """
        Builds the deep learning model for SVH score prediction.

        This method defines the model architecture. It can build either a custom
        CNN from scratch or leverage MobileNetV2 as a backbone for transfer learning.
        The final output layer is a single dense unit with linear activation for
        predicting the continuous SVH score.
        """
        if self.use_transfer_learning:
            # Transfer Learning with MobileNetV2
            # MobileNetV2 expects 3-channel input, so we replicate grayscale channel.
            
            # Load MobileNetV2 without its top layers, pre-trained on ImageNet.
            base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                     include_top=False,
                                     weights='imagenet')

            # Freeze the base model layers.
            base_model.trainable = False

            # Define the input layer for our grayscale X-ray images.
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1), name='input_image') 
            
            # Replicate the single grayscale channel to 3 channels.
            x = layers.concatenate([inputs, inputs, inputs], axis=-1, name='replicate_channels')
            
            # Apply MobileNetV2's specific preprocessing.
            x = preprocess_input(x, name='mobilenet_preprocess') 

            # Pass through the frozen base model.
            x = base_model(x, training=False) 

            x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x) 
            x = layers.Dense(128, activation='relu', name='dense_scorer_head')(x) 
            x = layers.Dropout(0.5, name='dropout_scorer_head')(x) 
            
            # Final dense layer for regression: 1 unit with linear activation.
            outputs = layers.Dense(self.output_dim, activation='linear', name='output_svh_score')(x) 

            # Construct the full Keras Model.
            model = models.Model(inputs=inputs, outputs=outputs, name='SVHSccorer_TransferLearning')

            # Compile the model for initial training of the new head layers.
            # 'mse' (Mean Squared Error) is standard for regression.
            # 'mae' (Mean Absolute Error) is also added for interpretability.
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          loss='mse',
                          metrics=['mae'])
            
            print("--- Built model using MobileNetV2 for SVH Scoring (Feature Extraction) ---")
            
        else:
            # Building a simple Convolutional Neural Network (CNN) from scratch.
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1), name='conv1'),
                layers.MaxPooling2D((2, 2), name='pool1'),
                layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
                layers.MaxPooling2D((2, 2), name='pool2'),
                layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
                layers.MaxPooling2D((2, 2), name='pool3'),
                layers.Flatten(name='flatten'),
                layers.Dense(128, activation='relu', name='dense1'),
                layers.Dropout(0.5, name='dropout1'),
                layers.Dense(self.output_dim, activation='linear', name='output_svh_score') 
            ], name='SVHSccorer_ScratchCNN')
            
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            model.compile(optimizer=optimizer,
                          loss='mse',
                          metrics=['mae'])
            print("--- Built simple CNN from scratch for SVH Scoring ---")

        return model

    def fine_tune_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, fine_tune_from_layer=None):
        """
        Performs fine-tuning on the pre-trained model for SVH scoring.

        This method is applicable only when `use_transfer_learning` is True.
        It unfreezes some layers of the base MobileNetV2 model and continues
        training with a very low learning rate, allowing adaptation to the X-ray domain.
        """
        if not self.use_transfer_learning or self.model is None:
            print("Fine-tuning is only applicable to models built with transfer learning.")
            return

        print("\n--- Starting Fine-tuning Phase for SVH Scoring ---")
        
        # Locate the MobileNetV2 base model layer within our functional API model.
        # It's the 4th layer: 0: Input, 1: Concatenate, 2: Lambda (preprocess_input), 3: MobileNetV2
        base_model_layer = self.model.layers[3] 
        base_model_layer.trainable = True # Set the base_model (MobileNetV2) itself to be trainable

        # Conditionally freeze layers within the base MobileNetV2 model if `fine_tune_from_layer` is specified.
        if fine_tune_from_layer is not None:
            for layer in base_model_layer.layers[:fine_tune_from_layer]:
                layer.trainable = False
            print(f"Unfroze layers from index {fine_tune_from_layer} of the MobileNetV2 base model for fine-tuning.")
        else:
            print("Unfroze all layers of the MobileNetV2 base model for fine-tuning.")

        # Re-compile the model with a very low learning rate for fine-tuning.
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate
                          loss='mse',
                          metrics=['mae'])

        self.model.summary() # Print summary to show which parameters are now trainable
        
        # Continue training with the now partially/fully unfrozen base model.
        history_fine_tune = self.model.fit(X_train, y_train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, y_val),
                                           verbose=1)
        print("\nFine-tuning completed for SVH Scoring.")
        return history_fine_tune

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path=None, fine_tune_epochs=0, fine_tune_from_layer=None):
        """
        Trains the SVH scoring model.

        This method orchestrates the training process, including an initial
        training phase for the new regression head and an optional
        fine-tuning phase if transfer learning is used.

        Args:
            X_train (np.ndarray): Training images (preprocessed, shape (N, H, W, 1)).
            y_train (np.ndarray): Training SVH scores (shape (N, 1)).
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation SVH scores.
            epochs (int): Number of training epochs for the initial phase (training the new head).
            batch_size (int): Batch size for training.
            model_save_path (str, optional): File path to save the trained model (.h5 format).
            fine_tune_epochs (int): Number of epochs for the fine-tuning phase. Set to 0 to skip.
                                    Only applicable when `use_transfer_learning` is True.
            fine_tune_from_layer (int): Index to start unfreezing layers for fine-tuning.
                                        Only applicable when `use_transfer_learning` is True.
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.summary()

        print("\nStarting model training (initial head training for SVH scoring)...")
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 verbose=1)

        print("\nInitial training completed for SVH Scoring.")
        
        # Perform fine-tuning if enabled and applicable
        if self.use_transfer_learning and fine_tune_epochs > 0:
            self.fine_tune_model(X_train, y_train, X_val, y_val, 
                                 epochs=fine_tune_epochs, 
                                 batch_size=batch_size,
                                 fine_tune_from_layer=fine_tune_from_layer)

        # Save the model after all training (initial + fine-tuning) is complete.
        if model_save_path:
            self.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return history

    def predict(self, image_np):
        """
        Predicts the SVH score for a single preprocessed image.

        Args:
            image_np (np.ndarray): A single preprocessed grayscale image (H, W).
                                   Expected to be in the format output by `Preprocessor`.

        Returns:
            float: Predicted SVH score.
            float: Raw prediction output (same as the score in this case).
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        # Ensure image is in the correct format: (1, H, W, 1) for the model's input layer.
        if image_np.ndim == 2: # (H, W)
            input_image = np.expand_dims(np.expand_dims(image_np, axis=0), axis=-1) # -> (1, H, W, 1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 1: # (H, W, 1)
             input_image = np.expand_dims(image_np, axis=0) # -> (1, H, W, 1)
        else:
            raise ValueError(f"Unexpected image dimension: {image_np.shape}. Expected (H, W) or (H, W, 1).")

        prediction_score = self.model.predict(input_image)[0][0] # Get the single score output
        
        # Clamp the predicted score to a reasonable range if necessary, e.g., 0 to max_svh_score
        # For dummy data, this is often useful. For real data, it depends on model confidence.
        prediction_score = np.clip(prediction_score, 0, self.max_svh_score) # Ensure score is non-negative and capped

        return float(prediction_score), float(prediction_score) # Return score and raw output

    def evaluate(self, X_test, y_test):
        """
        Evaluates the SVH scoring model on a test set and prints metrics.
        Includes MSE, MAE, and R2 score. Also visualizes some predictions.

        Args:
            X_test (np.ndarray): Test images.
            y_test (np.ndarray): True SVH scores for the test images.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Cannot evaluate.")
        
        print("\nEvaluating model on test set for SVH Scoring...")
        # Evaluate returns the loss and metrics as defined in model.compile
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test).flatten()
        r2 = r2_score(y_test, y_pred)

        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R-squared (R2 Score): {r2:.4f}")

        # --- Visualize some predictions ---
        print("\n--- Visualizing Sample SVH Predictions ---")
        num_samples_to_show = min(5, len(X_test)) # Show up to 5 samples

        plt.figure(figsize=(15, 4 * num_samples_to_show)) # Adjust figure size dynamically
        for i in range(num_samples_to_show):
            # Extract single image and its true label for plotting
            sample_image = X_test[i, :, :, 0] # Remove channel dim for 2D plotting
            true_score = y_test[i][0] # Assuming y_test is (N, 1)
            predicted_score, _ = self.predict(sample_image) 

            plt.subplot(num_samples_to_show, 2, 2*i + 1)
            plt.imshow(sample_image, cmap='gray')
            plt.title(f'Sample {i+1} | True: {true_score:.2f} | Predicted: {predicted_score:.2f}')
            plt.axis('off')
            
            # Scatter plot of True vs Predicted for the whole test set (optional, larger figure)
            if i == 0 and len(y_test) > 10: # Only create one scatter plot if test set is large enough
                plt.subplot(num_samples_to_show, 2, 2*i + 2)
                plt.scatter(y_test, y_pred, alpha=0.6)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal Prediction')
                plt.xlabel('True SVH Score')
                plt.ylabel('Predicted SVH Score')
                plt.title('True vs Predicted SVH Scores (Test Set)')
                plt.grid(True)
                plt.legend()
            elif len(y_test) <= 10: # If test set is too small for scatter, just show image again
                plt.subplot(num_samples_to_show, 2, 2*i + 2)
                plt.imshow(sample_image, cmap='gray')
                plt.title(f'Sample {i+1} | True: {true_score:.2f} | Predicted: {predicted_score:.2f}')
                plt.axis('off')


        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """Saves the trained Keras model to the specified path."""
        self.model.save(path)

    def load_model(self, path):
        """Loads a pre-trained Keras model from the specified path."""
        return keras.models.load_model(path)


# --- Command-line Interface (CLI) for executing the script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict Sharp/Van der Heijde (SVH) scores from X-ray hand images.")
    parser.add_argument("--mode", type=str, choices=['train_svh', 'predict_svh'], default='train_svh',
                        help="Operation mode: 'train_svh' for training a new SVH scoring model, "
                             "'predict_svh' for predicting SVH score on a single input image.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image for prediction (required if mode is 'predict_svh'). "
                             "This should be the path to the RAW image, which will be preprocessed.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained SVH scoring model (.h5) to load. "
                             "Required for 'predict_svh' mode, and optional for 'train_svh' "
                             "if you want to continue training an existing model.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for the initial training phase (training the new head).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--use_transfer_learning", action='store_true',
                        help="If set, MobileNetV2 will be used as the backbone for transfer learning. "
                             "Otherwise, a simple CNN will be built from scratch.")
    parser.add_argument("--fine_tune_epochs", type=int, default=0,
                        help="Number of epochs for the fine-tuning phase (if using transfer learning). "
                             "Set to 0 to skip fine-tuning. Only applicable if --use_transfer_learning is set.")
    parser.add_argument("--fine_tune_from_layer", type=int, default=None,
                        help="Index of the layer in the MobileNetV2 base model to start unfreezing from "
                             "for fine-tuning. If None, all layers of the base model will be unfrozen. "
                             "Only applicable with --use_transfer_learning and --fine_tune_epochs > 0.")
    parser.add_argument("--max_svh_score", type=float, default=220.0,
                        help="Theoretical maximum SVH score for dummy data generation and prediction clipping. "
                             "Adjust based on your specific scoring system.")

    args = parser.parse_args()

    # Define common project paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed') # Directory for preprocessed outputs
    svh_model_dir = os.path.join(base_dir, 'models', 'svh_scoring') # Dedicated directory for SVH models
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml') # Path to preprocessing config

    # Ensure necessary directories exist
    os.makedirs(svh_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True) # Ensures 'configs' dir exists

    # Load or create preprocessing configuration
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

    # Instantiate the Preprocessor to use its image loading and preprocessing capabilities
    preprocessor = Preprocessor(preprocessing_config)
    img_size = tuple(preprocessing_config['resize']['target_size']) # Get target image size from config
    
    # Define the default save path for the SVH scoring model
    svh_model_save_path = os.path.join(svh_model_dir, 'svh_scoring_model.h5')

    # --- Training Mode for SVH Scoring ---
    if args.mode == 'train_svh':
        print("\n--- Running in TRAINING mode for SVH Scoring ---")
        # --- SIMULATE DATA LOADING AND LABELING ---
        # NOTE: This section uses randomly generated images and scores for demonstration purposes.
        # In a real-world scenario, you MUST replace this with your actual X-ray hand image dataset
        # annotated with ground truth Sharp/Van der Heijde scores.
        # These scores are typically derived from expert annotations of erosions and joint space narrowing.
        print(f"\n--- Simulating Data Loading for SVH Scoring. "
              "REPLACE THIS WITH YOUR REAL ANNOTATED DATASET! ---")
        num_samples = 200 # Number of dummy samples to generate for training
        
        # Create dummy images (random noise, 1 channel, float32, 0-255 range)
        X = np.random.rand(num_samples, img_size[0], img_size[1], 1).astype(np.float32) * 255.0
        
        # Generate dummy SVH scores: random float values from 0 to max_svh_score.
        # In a real dataset, these scores would be discrete integers often.
        y = np.random.rand(num_samples, 1).astype(np.float32) * args.max_svh_score 

        print(f"Generated {len(y)} dummy samples for SVH scoring training.")
        print(f"Dummy SVH scores range: [{np.min(y):.2f}, {np.max(y):.2f}]")

        # Split data into training, validation, and test sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 of 0.8 = 0.2 total val

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Instantiate the SVHSccorer model.
        svh_scorer = SVHSccorer(model_path=args.model_path, 
                                     img_size=img_size, 
                                     use_transfer_learning=args.use_transfer_learning)
        
        # Train the model, including optional fine-tuning.
        svh_scorer.train(X_train, y_train, X_val, y_val, 
                            epochs=args.epochs, batch_size=args.batch_size, 
                            model_save_path=svh_model_save_path,
                            fine_tune_epochs=args.fine_tune_epochs,
                            fine_tune_from_layer=args.fine_tune_from_layer)

        # Evaluate the trained model on the test set.
        svh_scorer.evaluate(X_test, y_test)

    # --- Prediction Mode for SVH Scoring ---
    elif args.mode == 'predict_svh':
        print("\n--- Running in PREDICTION mode for SVH Scoring ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict_svh'.")
        # Check if a model path is provided, otherwise try to use the default saved model.
        if not args.model_path:
            if os.path.exists(svh_model_save_path):
                print(f"Using default model: {svh_model_save_path}")
                args.model_path = svh_model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists or provided.")

        print(f"Attempting to predict SVH score for: {args.image_path}")
        
        # Use the Preprocessor to load and preprocess the raw input image.
        raw_image = preprocessor._load_image(args.image_path)
        if raw_image is None:
            print(f"Could not load image at {args.image_path}. Exiting prediction mode.")
            exit()
        
        # Apply the full preprocessing pipeline steps to the image.
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

        # Ensure the final preprocessed image is in float32 format, as expected by TensorFlow models.
        image_to_predict = image_to_predict.astype(np.float32)

        # Instantiate SVHSccorer to load the trained model for prediction.
        # `use_transfer_learning` is set to False here because if `model_path` is provided,
        # the model is loaded directly, bypassing the `_build_model` method.
        svh_scorer = SVHSccorer(model_path=args.model_path, 
                                     img_size=img_size,
                                     use_transfer_learning=False) 
        
        if svh_scorer.model is None:
             print("Error: Model could not be loaded for prediction. Please check --model_path.")
             exit()

        # Perform the SVH score prediction.
        predicted_score, _ = svh_scorer.predict(image_to_predict)
        
        print(f"Predicted SVH Score: {predicted_score:.2f}")

        # Visualize the image with the prediction title.
        plt.imshow(image_to_predict, cmap='gray')
        plt.title(f"Predicted SVH Score: {predicted_score:.2f} for {os.path.basename(args.image_path)}")
        plt.axis('off')
        plt.show()

    else:
        print("Invalid mode specified. Choose 'train_svh' or 'predict_svh'.")