import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import argparse

# Import utilities and preprocessor
from src.data_preprocessing import Preprocessor
from src.model_utils import load_config, plot_training_history, plot_confusion_matrix, plot_roc_curve

class MultiTaskRAClassifier:
    def __init__(self, model_path=None, img_size=(512, 512), num_joints=16, use_transfer_learning=True):
        """
        Initializes the Multi-Task RA Classifier, handling laterality, RA classification,
        joint detection, and SVH scoring simultaneously.

        Args:
            model_path (str, optional): Path to a pre-trained multi-task model to load.
            img_size (tuple): Expected input image size (height, width).
            num_joints (int): Number of hand joints to detect.
            use_transfer_learning (bool): Whether to use transfer learning with MobileNetV2.
        """
        self.img_size = img_size
        self.num_joints = num_joints
        self.output_dim_joints = num_joints * 2 # Each joint has (x,y)
        self.output_dim_laterality = 2 # Left/Right
        self.output_dim_ra_class = 1 # RA/Non-RA (sigmoid)
        self.output_dim_svh_score = 1 # Continuous score
        
        self.model = None
        self.use_transfer_learning = use_transfer_learning

        self.laterality_class_names = ["Left Hand", "Right Hand"]
        self.ra_class_names = ["Non-RA", "RA"]
        self.joint_names = [ # Example names, match with hand_joint_detection
            "Wrist", "Thumb_MCP", "Thumb_PIP", "Thumb_DIP",
            "Index_MCP", "Index_PIP", "Index_DIP",
            "Middle_MCP", "Middle_PIP", "Middle_DIP",
            "Ring_MCP", "Ring_PIP", "Ring_DIP",
            "Pinky_MCP", "Pinky_PIP", "Pinky_DIP"
        ]
        self.max_svh_score = 220.0 # Match this with svh_scoring.py's max_svh_score

        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
            print(f"Loaded pre-trained multi-task model from {model_path}")
        else:
            print("No pre-trained multi-task model specified or found. A new one will be created.")
            if self.use_transfer_learning:
                print("Using MobileNetV2 for shared feature extraction.")
            else:
                print("Building a simple CNN from scratch for multi-task learning.")

    def _build_model(self):
        """
        Builds the multi-task model with a shared backbone and multiple output heads.
        """
        if self.use_transfer_learning:
            base_model = MobileNetV2(input_shape=(self.img_size[0], self.img_size[1], 3),
                                     include_top=False,
                                     weights='imagenet')
            base_model.trainable = False # Start with frozen backbone

            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1), name='image_input')
            x = layers.concatenate([inputs, inputs, inputs], axis=-1, name='replicate_channels')
            x = preprocess_input(x, name='mobilenet_preprocess')
            shared_features = base_model(x, training=False) # training=False for frozen base
            shared_features = layers.GlobalAveragePooling2D(name='global_average_pooling')(shared_features)
            shared_features = layers.Dense(512, activation='relu', name='shared_dense_features')(shared_features) # Shared dense layer
            shared_features = layers.Dropout(0.5, name='shared_dropout')(shared_features)

        else:
            # Simple CNN backbone (less ideal for multi-task than deeper models)
            inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 1), name='image_input')
            x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
            x = layers.MaxPooling2D((2, 2), name='pool1')(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
            x = layers.MaxPooling2D((2, 2), name='pool2')(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
            x = layers.MaxPooling2D((2, 2), name='pool3')(x)
            shared_features = layers.Flatten(name='flatten')(x)
            shared_features = layers.Dense(512, activation='relu', name='shared_dense_features')(shared_features)
            shared_features = layers.Dropout(0.5, name='shared_dropout')(shared_features)

        # --- Output Heads ---
        # Laterality Head (2 classes: Left/Right)
        laterality_output = layers.Dense(self.output_dim_laterality, activation='softmax', name='laterality_output')(shared_features)
        
        # RA Classification Head (Binary: RA/Non-RA)
        ra_output = layers.Dense(self.output_dim_ra_class, activation='sigmoid', name='ra_output')(shared_features)
        
        # Joint Detection Head (2*num_joints coordinates)
        joints_output = layers.Dense(self.output_dim_joints, activation='linear', name='joints_output')(shared_features)
        
        # SVH Scoring Head (Single continuous score)
        svh_output = layers.Dense(self.output_dim_svh_score, activation='linear', name='svh_output')(shared_features)

        model = models.Model(inputs=inputs, outputs=[laterality_output, ra_output, joints_output, svh_output], name='MultiTaskRAClassifier')

        # Compile the model with multiple losses and metrics
        # You'll need to define loss weights to balance training across tasks
        loss_weights = {
            'laterality_output': 1.0,
            'ra_output': 1.0,
            'joints_output': 0.1, # Joint detection often has higher loss values, so might need lower weight
            'svh_output': 0.1     # SVH scoring also regression, potentially needs lower weight
        }

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'laterality_output': 'sparse_categorical_crossentropy', # for integer labels
                'ra_output': 'binary_crossentropy',
                'joints_output': 'mse',
                'svh_output': 'mse'
            },
            loss_weights=loss_weights,
            metrics={
                'laterality_output': 'accuracy',
                'ra_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
                'joints_output': 'mae',
                'svh_output': 'mae'
            }
        )
        print("--- Built Multi-Task Model ---")
        return model

    def fine_tune_model(self, X_train, y_train_dict, X_val, y_val_dict, epochs=5, batch_size=32, fine_tune_from_layer=None):
        """
        Performs fine-tuning for the multi-task model.
        """
        if not self.use_transfer_learning or self.model is None:
            print("Fine-tuning is only applicable to models built with transfer learning.")
            return

        print("\n--- Starting Fine-tuning Phase for Multi-Task Model ---")
        base_model_layer = self.model.layers[3] # MobileNetV2 base
        base_model_layer.trainable = True

        if fine_tune_from_layer is not None:
            for layer in base_model_layer.layers[:fine_tune_from_layer]:
                layer.trainable = False
            print(f"Unfroze layers from index {fine_tune_from_layer} of the MobileNetV2 base model for fine-tuning.")
        else:
            print("Unfroze all layers of the MobileNetV2 base model for fine-tuning.")

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Very low learning rate for fine-tuning
            loss={
                'laterality_output': 'sparse_categorical_crossentropy',
                'ra_output': 'binary_crossentropy',
                'joints_output': 'mse',
                'svh_output': 'mse'
            },
            loss_weights={ # Ensure loss weights are kept consistent or re-tuned
                'laterality_output': 1.0,
                'ra_output': 1.0,
                'joints_output': 0.1,
                'svh_output': 0.1
            },
            metrics={
                'laterality_output': 'accuracy',
                'ra_output': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
                'joints_output': 'mae',
                'svh_output': 'mae'
            }
        )
        self.model.summary()
        
        history_fine_tune = self.model.fit(X_train, y_train_dict,
                                           epochs=epochs,
                                           batch_size=batch_size,
                                           validation_data=(X_val, y_val_dict),
                                           verbose=1)
        print("\nFine-tuning completed for Multi-Task Model.")
        return history_fine_tune

    def train(self, X_train, y_train_dict, X_val, y_val_dict, epochs=10, batch_size=32, model_save_path=None, fine_tune_epochs=0, fine_tune_from_layer=None):
        """
        Trains the multi-task model.
        Note: y_train_dict and y_val_dict must be dictionaries matching output names:
        {'laterality_output': y_laterality, 'ra_output': y_ra, ...}
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.summary()

        print("\nStarting multi-task model training (initial head training)...")
        history = self.model.fit(X_train, y_train_dict,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val_dict),
                                 verbose=1)

        print("\nInitial multi-task training completed.")
        
        if self.use_transfer_learning and fine_tune_epochs > 0:
            self.fine_tune_model(X_train, y_train_dict, X_val, y_val_dict, 
                                 epochs=fine_tune_epochs, 
                                 batch_size=batch_size,
                                 fine_tune_from_layer=fine_tune_from_layer)

        if model_save_path:
            self.save_model(model_save_path)
            print(f"Model saved to {model_save_path}")
        return history

    def predict(self, image_np, laterality_threshold=0.5, ra_threshold=0.5):
        """
        Predicts laterality, RA status, joint coordinates, and SVH score for a single image.

        Returns:
            dict: A dictionary containing all predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        if image_np.ndim == 2: # (H, W)
            input_image = np.expand_dims(np.expand_dims(image_np, axis=0), axis=-1) # -> (1, H, W, 1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 1: # (H, W, 1)
             input_image = np.expand_dims(image_np, axis=0) # -> (1, H, W, 1)
        else:
            raise ValueError(f"Unexpected image dimension: {image_np.shape}. Expected (H, W) or (H, W, 1).")

        predictions = self.model.predict(input_image)
        
        # Parse predictions from the multi-output model
        laterality_raw_proba = predictions[0][0]
        ra_raw_proba = predictions[1][0]
        joints_raw_coords = predictions[2][0]
        svh_raw_score = predictions[3][0]

        # Convert laterality
        laterality_predicted_idx = np.argmax(laterality_raw_proba)
        laterality_class = self.laterality_class_names[laterality_predicted_idx]
        laterality_proba = laterality_raw_proba[laterality_predicted_idx]

        # Convert RA class
        ra_class = self.ra_class_names[int(ra_raw_proba >= ra_threshold)]
        
        # Reshape joints
        joints_coords = joints_raw_coords.reshape((self.num_joints, 2))

        # Clamp SVH score
        svh_score = np.clip(svh_raw_score, 0, self.max_svh_score)

        return {
            'laterality_class': laterality_class,
            'laterality_proba': float(laterality_proba),
            'ra_class': ra_class,
            'ra_proba': float(ra_raw_proba),
            'joint_coords': joints_coords,
            'svh_score': float(svh_score)
        }

    def evaluate(self, X_test, y_test_dict, ra_threshold=0.5):
        """
        Evaluates the multi-task model on a test set.
        y_test_dict must be a dictionary matching output names.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Cannot evaluate.")
        
        print("\nEvaluating multi-task model on test set...")
        loss_dict = self.model.evaluate(X_test, y_test_dict, return_dict=True, verbose=0)
        
        print("\n--- Multi-Task Evaluation Results ---")
        for key, value in loss_dict.items():
            print(f"{key}: {value:.4f}")

        # Detailed classification reports and plots
        y_pred_laterality, y_pred_ra_proba, y_pred_joints, y_pred_svh = self.model.predict(X_test)

        # Laterality
        y_pred_laterality_classes = np.argmax(y_pred_laterality, axis=1)
        print("\n--- Laterality Classification Report ---")
        print(classification_report(np.argmax(y_test_dict['laterality_output'], axis=1), y_pred_laterality_classes, target_names=self.laterality_class_names))
        plot_confusion_matrix(np.argmax(y_test_dict['laterality_output'], axis=1), y_pred_laterality_classes, self.laterality_class_names, title='Laterality Confusion Matrix')

        # RA Classification
        y_pred_ra_classes = (y_pred_ra_proba >= ra_threshold).astype(int)
        print("\n--- RA Classification Report ---")
        print(classification_report(y_test_dict['ra_output'], y_pred_ra_classes, target_names=self.ra_class_names))
        plot_confusion_matrix(y_test_dict['ra_output'], y_pred_ra_classes, self.ra_class_names, title='RA Classification Confusion Matrix')
        plot_roc_curve(y_test_dict['ra_output'], y_pred_ra_proba, title='RA Classification ROC Curve')

        # Joint Detection
        mse_joints = mean_squared_error(y_test_dict['joints_output'], y_pred_joints)
        mae_joints = mean_absolute_error(y_test_dict['joints_output'], y_pred_joints)
        print(f"\n--- Joint Detection Metrics ---")
        print(f"Joints MSE: {mse_joints:.4f}")
        print(f"Joints MAE: {mae_joints:.4f}")

        # SVH Scoring
        mse_svh = mean_squared_error(y_test_dict['svh_output'], y_pred_svh)
        mae_svh = mean_absolute_error(y_test_dict['svh_output'], y_pred_svh)
        r2_svh = r2_score(y_test_dict['svh_output'], y_pred_svh)
        print(f"\n--- SVH Scoring Metrics ---")
        print(f"SVH MSE: {mse_svh:.4f}")
        print(f"SVH MAE: {mae_svh:.4f}")
        print(f"SVH R2 Score: {r2_svh:.4f}")
        
        # (Optional) Visualize some joint predictions on images similar to HandJointDetector

    def save_model(self, path):
        """Saves the trained Keras model to the specified path."""
        self.model.save(path)

    def load_model(self, path):
        """Loads a pre-trained Keras model from the specified path."""
        return keras.models.load_model(path)

# --- Command-line Interface (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with a multi-task X-ray hand analysis model.")
    parser.add_argument("--mode", type=str, choices=['train_multi', 'predict_multi'], default='train_multi',
                        help="Operation mode: 'train_multi' for training, 'predict_multi' for prediction.")
    parser.add_argument("--image_path", type=str,
                        help="Path to the RAW image for prediction (required if mode is 'predict_multi').")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a pre-trained multi-task model (.h5) to load.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for initial training.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--use_transfer_learning", action='store_true',
                        help="Use MobileNetV2 for shared feature extraction.")
    parser.add_argument("--fine_tune_epochs", type=int, default=0,
                        help="Number of epochs for fine-tuning phase.")
    parser.add_argument("--fine_tune_from_layer", type=int, default=None,
                        help="Index of the layer in MobileNetV2 to start unfreezing from.")
    parser.add_argument("--num_joints", type=int, default=16,
                        help="Number of hand joints to detect.")

    args = parser.parse_args()

    # Define common project paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml')
    multi_task_model_dir = os.path.join(base_dir, 'models', 'multi_task_model')
    os.makedirs(multi_task_model_dir, exist_ok=True)

    preprocessing_config = load_config(config_path)
    if not preprocessing_config:
        print("Failed to load preprocessing config. Using dummy config.")
        preprocessing_config = { # Fallback dummy config
            'normalize': {'target_range': [0, 255]},
            'clahe': {'apply': True, 'clip_limit': 2.0, 'tile_grid_size': [8, 8]},
            'denoising': {'apply': True, 'kernel_size': [3, 3]},
            'resize': {'target_size': [512, 512]}
        }

    preprocessor = Preprocessor(preprocessing_config)
    img_size = tuple(preprocessing_config['resize']['target_size'])
    multi_task_model_save_path = os.path.join(multi_task_model_dir, 'multi_task_ra_model.h5')

    if args.mode == 'train_multi':
        print("\n--- Running in TRAINING mode for Multi-Task Model ---")
        # --- SIMULATE DATA LOADING AND LABELING ---
        # For multi-task, you need ALL labels for each image.
        print("\n--- Simulating Data Loading for Multi-Task Model (REPLACE WITH REAL DATA) ---")
        num_samples = 200

        X = np.random.rand(num_samples, img_size[0], img_size[1], 1).astype(np.float32) * 255.0

        # Dummy Labels for each task
        # Laterality: 0 or 1, one-hot encoded for sparse_categorical_crossentropy if needed
        y_laterality = np.random.randint(0, 2, size=(num_samples,)).astype(np.float32) # 0:Left, 1:Right
        y_laterality_one_hot = keras.utils.to_categorical(y_laterality, num_classes=2) # Convert to one-hot for softmax
        
        # RA Classification: 0 or 1
        y_ra = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)

        # Joints: (num_joints * 2) coordinates
        y_joints = np.random.rand(num_samples, args.num_joints * 2).astype(np.float32)
        y_joints[:, ::2] = y_joints[:, ::2] * img_size[1]
        y_joints[:, 1::2] = y_joints[:, 1::2] * img_size[0]

        # SVH Score: single continuous value
        max_svh = 220.0 # Match this with class's max_svh_score if loaded
        y_svh = np.random.rand(num_samples, 1).astype(np.float32) * max_svh

        # Combine all labels into a dictionary for Keras multi-output training
        y_combined = {
            'laterality_output': y_laterality_one_hot,
            'ra_output': y_ra,
            'joints_output': y_joints,
            'svh_output': y_svh
        }

        # Split data (ensure splits are consistent across all labels)
        X_train, X_test, y_lat_train, y_lat_test, y_ra_train, y_ra_test, y_joint_train, y_joint_test, y_svh_train, y_svh_test = \
            train_test_split(X, y_laterality_one_hot, y_ra, y_joints, y_svh, test_size=0.2, random_state=42)
        
        X_train, X_val, y_lat_train, y_lat_val, y_ra_train, y_ra_val, y_joint_train, y_joint_val, y_svh_train, y_svh_val = \
            train_test_split(X_train, y_lat_train, y_ra_train, y_joint_train, y_svh_train, y_svh_val, test_size=0.25, random_state=42)

        y_train_dict = {
            'laterality_output': y_lat_train, 'ra_output': y_ra_train,
            'joints_output': y_joint_train, 'svh_output': y_svh_train
        }
        y_val_dict = {
            'laterality_output': y_lat_val, 'ra_output': y_ra_val,
            'joints_output': y_joint_val, 'svh_output': y_svh_val
        }
        y_test_dict = {
            'laterality_output': y_lat_test, 'ra_output': y_ra_test,
            'joints_output': y_joint_test, 'svh_output': y_svh_test
        }

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        multi_task_model = MultiTaskRAClassifier(
            model_path=args.model_path,
            img_size=img_size,
            num_joints=args.num_joints,
            use_transfer_learning=args.use_transfer_learning
        )

        history = multi_task_model.train(
            X_train, y_train_dict, X_val, y_val_dict,
            epochs=args.epochs, batch_size=args.batch_size,
            model_save_path=multi_task_model_save_path,
            fine_tune_epochs=args.fine_tune_epochs,
            fine_tune_from_layer=args.fine_tune_from_layer
        )
        
        multi_task_model.evaluate(X_test, y_test_dict)

    elif args.mode == 'predict_multi':
        print("\n--- Running in PREDICTION mode for Multi-Task Model ---")
        if not args.image_path:
            parser.error("--image_path is required when mode is 'predict_multi'.")
        if not args.model_path:
            if os.path.exists(multi_task_model_save_path):
                print(f"Using default model: {multi_task_model_save_path}")
                args.model_path = multi_task_model_save_path
            else:
                parser.error("--model_path is required for prediction if no default model exists or provided.")

        raw_image = preprocessor._load_image(args.image_path)
        if raw_image is None:
            print(f"Could not load image at {args.image_path}. Exiting prediction mode.")
            exit()
        
        preprocessed_image = preprocessor._normalize_image(raw_image, target_range=preprocessing_config['normalize']['target_range'])
        if preprocessing_config['clahe']['apply']:
            preprocessed_image = preprocessor._apply_clahe(
                preprocessed_image, clip_limit=preprocessing_config['clahe']['clip_limit'],
                tile_grid_size=tuple(preprocessing_config['clahe']['tile_grid_size'])
            )
        if preprocessing_config['denoising']['apply']:
            preprocessed_image = preprocessor._apply_gaussian_denoising(
                preprocessed_image, kernel_size=tuple(preprocessing_config['denoising']['kernel_size'])
            )
        preprocessed_image = preprocessor._resize_image(preprocessed_image, target_size=img_size)
        preprocessed_image, _ = preprocessor._segment_hand(preprocessed_image)
        preprocessed_image = preprocessed_image.astype(np.float32)

        multi_task_model = MultiTaskRAClassifier(
            model_path=args.model_path,
            img_size=img_size,
            num_joints=args.num_joints,
            use_transfer_learning=False # Loading pre-trained, so build is skipped
        )

        if multi_task_model.model is None:
             print("Error: Multi-task model could not be loaded for prediction. Please check --model_path.")
             exit()

        predictions = multi_task_model.predict(preprocessed_image)
        
        print("\n--- Multi-Task Prediction Results ---")
        print(f"Laterality: {predictions['laterality_class']} (Prob: {predictions['laterality_proba']:.2f})")
        print(f"RA Classification: {predictions['ra_class']} (Prob: {predictions['ra_proba']:.2f})")
        print(f"SVH Score: {predictions['svh_score']:.2f}")
        print(f"Joints Detected: {len(predictions['joint_coords'])} (e.g., first joint: {predictions['joint_coords'][0].round(2)})")

        # Basic visualization
        plt.figure(figsize=(10,10))
        plt.imshow(preprocessed_image, cmap='gray')
        plt.scatter(predictions['joint_coords'][:,0], predictions['joint_coords'][:,1], c='lime', marker='x', s=100, label='Predicted Joints')
        
        title_text = (f"Pred: {predictions['ra_class']} ({predictions['ra_proba']:.2f}), "
                      f"SVH: {predictions['svh_score']:.2f}, "
                      f"Laterality: {predictions['laterality_class']}")
        plt.title(title_text)
        plt.axis('off')
        plt.legend()
        plt.show()