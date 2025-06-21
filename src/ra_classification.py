import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# Import the Preprocessor class from data_preprocessing.py
# This import assumes that the project root (e.g., 'AgeBridgeAI-Research')
# is the current working directory when this script is executed.
from src.data_preprocessing import Preprocessor

class RAClassifier:
    def __init__(self, model_path=None, img_size=(512, 512), use_transfer_learning=True):
        """
        Initializes the RAClassifier model.

        This class handles the creation, training, evaluation, and prediction
        for classifying X-ray hand images as Rheumatoid Arthritis (RA) or Non-RA.
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
        self.num_classes = 1 # Binary classification (RA / Non-RA)
        self.class_names = ["Non-RA", "RA"] # Assuming 0: Non-RA, 1: RA
        self.model = None
        self.use_transfer_learning = use_transfer_learning

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained RA classification model from {model_path}")
        else:
            print("No pre-trained model specified or found. A new model will be created.")
            if self.use_transfer_learning:
                print("Using MobileNetV2 for transfer learning for RA classification.")
            else:
                print("Building a simple CNN from scratch for RA classification.")

    def _build_model(self):
        """
        Builds the deep learning model for RA classification.

        This method defines the model architecture. It can build either a custom
        CNN from scratch or leverage MobileNetV2 as a backbone for transfer learning.
        The final output layer is a dense layer with sigmoid activation for binary classification.
        """
        if self.use_transfer_learning:
            # Transfer Learning with MobileNetV2
            # MobileNetV2 expects 3-channel input, so we replicate grayscale channel.
            
            # Load MobileNetV2 without its top layers, pre-trained on ImageNet.
            base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                     include_top=False,
                                     weights='imagenet')

            # Freeze the base model layers.
            # This is crucial for feature extraction, preventing updates to pre-trained weights.
            base_model.trainable = False

            # Define the input layer for our grayscale X-ray images.
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1), name='input_image') 
            
            # Replicate the single grayscale channel to 3 channels for MobileNetV2 compatibility.
            x = layers.concatenate([inputs, inputs, inputs], axis=-1, name='replicate_channels')
            
            # Apply MobileNetV2's specific preprocessing.
            x = preprocess_input(x, name='mobilenet_preprocess') 

            # Pass through the frozen base model.
            x = base_model(x, training=False) 

            x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x) 
            x = layers.Dense(128, activation='relu', name='dense_classifier_head')(x) 
            x = layers.Dropout(0.5, name='dropout_classifier_head')(x) 
            
            # Final dense layer for binary classification: 1 unit with sigmoid activation.
            outputs = layers.Dense(self.num_classes, activation='sigmoid', name='output_ra_class')(x) 

            # Construct the full Keras Model.
            model = models.Model(inputs=inputs, outputs=outputs, name='RAClassifier_TransferLearning')

            # Compile the model for initial training of the new head layers.
            # 'binary_crossentropy' is used for binary classification with sigmoid output.
            # Metrics include 'accuracy', Precision, and Recall for comprehensive evaluation.
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                          loss='binary_crossentropy',
                          metrics=['accuracy',
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall')])
            
            print("--- Built model using MobileNetV2 for RA Classification (Feature Extraction) ---")
            
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
                layers.Dense(self.num_classes, activation='sigmoid', name='output_ra_class') 
            ], name='RAClassifier_ScratchCNN')
            
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-3,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy',
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall')])
            print("--- Built simple CNN from scratch for RA Classification ---")

        return model

    def fine_tune_model(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32, fine_tune_from_layer=None):
        """
        Performs fine-tuning on the pre-trained model for RA classification.

        This method is applicable only when `use_transfer_learning` is True.
        It unfreezes some layers of the base MobileNetV2 model and continues
        training with a very low learning rate, allowing adaptation to the X-ray domain.
        """
        if not self.use_transfer_learning or self.model is None:
            print("Fine-tuning is only applicable to models built with transfer learning.")
            return

        print("\n--- Starting Fine-tuning Phase for RA Classification ---")
        
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
        # This is crucial for stable convergence during fine-tuning.
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate
                          loss='binary_crossentropy',
                          metrics=['accuracy',
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall')])

        self.model.summary() # Print summary to show which parameters are now trainable
        
        # Continue training with the now partially/fully unfrozen base model.
        history_fine_tune = self.model.fit(X_train, y_train,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, y_val),
                                           verbose=1)
        print("\nFine-tuning completed for RA Classification.")
        return history_fine_tune

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, model_save_path=None, fine_tune_epochs=0, fine_tune_from_layer=None):
        """
        Trains the RA classification model.

        This method orchestrates the training process, including an initial
        training phase for the new classification head and an optional
        fine-tuning phase if transfer learning is used.

        Args:
            X_train (np.ndarray): Training images (preprocessed, shape (N, H, W, 1)).
            y_train (np.ndarray): Training labels (0 for Non-RA, 1 for RA).
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation labels.
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

        print("\nStarting model training (initial head training for RA classification)...")
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 verbose=1)

        print("\nInitial training completed for RA Classification.")
        
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

    def predict(self, image_np, threshold=0.5):
        """
        Predicts the RA classification for a single preprocessed image.

        Args:
            image_np (np.ndarray): A single preprocessed grayscale image (H, W).
                                   Expected to be in the format output by `Preprocessor`.
            threshold (float): The probability threshold to classify as RA.
                               If prediction_proba >= threshold, classify as RA (1), else Non-RA (0).

        Returns:
            str: Predicted class name ("Non-RA" or "RA").
            float: Predicted probability of being RA.
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

        prediction_proba = self.model.predict(input_image)[0][0] # Get the single probability output
        predicted_class_idx = (prediction_proba >= threshold).astype(int)
        
        return self.class_names[predicted_class_idx], float(prediction_proba)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluates the RA classification model on a test set and prints metrics.
        Includes classification report, confusion matrix, and ROC curve.

        Args:
            X_test (np.ndarray): Test images.
            y_test (np.ndarray): True labels for the test images.
            threshold (float): Probability threshold for classification.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Cannot evaluate.")
        
        print("\nEvaluating model on test set for RA Classification...")
        # Get raw probabilities
        y_pred_probs = self.model.predict(X_test)
        
        # Convert probabilities to binary predictions based on threshold
        y_pred_classes = (y_pred_probs >= threshold).astype(int)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))

        # Print and plot confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        print(cm)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix for RA Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
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
    parser = argparse.ArgumentParser(description="Train or predict RA classification from X-ray hand images.")
    parser.add_argument("--mode", type=str, choices=['train_ra', 'predict_ra'], default='train_ra',
                        help="Operation mode: 'train_ra' for training a new RA classification model, "
                             "'predict_ra' for predicting RA status on a single input image.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the image for prediction (required if mode is 'predict_ra'). "
                             "This should be the path to the RAW image, which will be preprocessed.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained RA classification model (.h5) to load. "
                             "Required for 'predict_ra' mode, and optional for 'train_ra' "
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
    parser.add_argument("--predict_threshold", type=float, default=0.5,
                        help="Probability threshold for classifying as RA in prediction mode (default: 0.5).")

    args = parser.parse_args()

    # Define common project paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed') # Directory for preprocessed outputs
    ra_model_dir = os.path.join(base_dir, 'models', 'ra_classification') # Dedicated directory for RA models
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml') # Path to preprocessing config

    # Ensure necessary directories exist
    os.makedirs(ra_model_dir, exist_ok=True)
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
    
    # Define the default save path for the RA classification model
    ra_model_save_path = os.path.join(ra_model_dir, 'ra_classification_model.h5')

    # --- Training Mode for RA Classification ---
    if args.mode == 'train_ra':
        print("\n--- Running in TRAINING mode for RA Classification ---")
        # --- SIMULATE DATA LOADING AND LABELING ---
        # NOTE: This section uses randomly generated images and labels for demonstration purposes.
        # In a real-world scenario, you MUST replace this with your actual X-ray hand image dataset
        # annotated with RA status (e.g., 0 for Non-RA, 1 for RA).
        print(f"\n--- Simulating Data Loading for RA Classification. "
              "REPLACE THIS WITH YOUR REAL ANNOTATED DATASET! ---")
        num_samples = 200 # Number of dummy samples to generate for training
        
        # Create dummy images (random noise, 1 channel, float32, 0-255 range)
        X = np.random.rand(num_samples, img_size[0], img_size[1], 1).astype(np.float32) * 255.0
        
        # Generate dummy labels: half Non-RA (0), half RA (1)
        num_ra = num_samples // 2
        num_non_ra = num_samples - num_ra
        y = np.array([0] * num_non_ra + [1] * num_ra, dtype=np.float32) # Ensure labels are float32 for binary_crossentropy
        np.random.shuffle(y) # Shuffle labels to mix classes

        print(f"Generated {len(y)} dummy samples for RA classification training.")
        print(f"Dummy label distribution: Non-RA (0): {np.sum(y == 0)}, RA (1): {np.sum(y == 1)}")

        # Split data into training, validation, and test sets.
        # Use stratification to ensure class balance in splits.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 of 0.8 = 0.2 total val

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Instantiate the RAClassifier model.
        ra_classifier = RAClassifier(model_path=args.model_path, 
                                     img_size=img_size, 
                                     use_transfer_learning=args.use_transfer_learning)
        
        # Train the model, including optional fine-tuning.
        ra_classifier.train(X_train, y_train, X_val, y_val, 
                            epochs=args.epochs, batch_size=args.batch_size, 
                            model_save_path=ra_model_save_path,
                            fine_tune_epochs=args.fine_tune_epochs,
                            fine_tune_from_layer=args.fine_tune_from_layer)

        # Evaluate the trained model on the test set.
        ra_classifier.evaluate(X_test, y_test, threshold=args.predict_threshold)

    # --- Prediction Mode for RA Classification ---
    elif args.mode == 'predict_ra':
        print("\n--- Running in PREDICTION mode for RA Classification ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict_ra'.")
        # Check if a model path is provided, otherwise try to use the default saved model.
        if not args.model_path:
            if os.path.exists(ra_model_save_path):
                print(f"Using default model: {ra_model_save_path}")
                args.model_path = ra_model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists or provided.")

        print(f"Attempting to predict RA status for: {args.image_path}")
        
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

        # Instantiate RAClassifier to load the trained model for prediction.
        # `use_transfer_learning` is set to False here because if `model_path` is provided,
        # the model is loaded directly, bypassing the `_build_model` method.
        ra_classifier = RAClassifier(model_path=args.model_path, 
                                     img_size=img_size,
                                     use_transfer_learning=False) 
        
        if ra_classifier.model is None:
             print("Error: Model could not be loaded for prediction. Please check --model_path.")
             exit()

        # Perform the RA classification prediction.
        predicted_class, predicted_proba = ra_classifier.predict(image_to_predict, threshold=args.predict_threshold)
        
        print(f"Predicted RA Status: {predicted_class}")
        print(f"Prediction Probability (of being RA): {predicted_proba:.4f}")

        # Visualize the image with the prediction title.
        plt.imshow(image_to_predict, cmap='gray')
        plt.title(f"Predicted: {predicted_class} (Prob: {predicted_proba:.2f}) for {os.path.basename(args.image_path)}")
        plt.axis('off')
        plt.show()

    else:
        print("Invalid mode specified. Choose 'train_ra' or 'predict_ra'.")

        a