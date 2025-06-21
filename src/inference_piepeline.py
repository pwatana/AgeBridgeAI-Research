import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import all necessary components from your pipeline
from src.data_preprocessing import Preprocessor
from src.left_right_hand_identification import HandLateralityIdentifier
from src.ra_classification import RAClassifier
from src.hand_joint_detection import HandJointDetector
from src.svh_scoring import SVHSccorer
from src.model_utils import load_config # For loading data_config.yaml

class XRayAnalysisPipeline:
    def __init__(self, config_path, models_dir):
        """
        Initializes the entire X-ray analysis pipeline by loading configurations
        and all necessary pre-trained models.

        Args:
            config_path (str): Path to the shared data preprocessing configuration file.
            models_dir (str): Base directory where all trained model .h5 files are stored.
        """
        self.config_path = config_path
        self.models_dir = models_dir

        # 1. Load Preprocessing Configuration
        self.preprocessing_config = load_config(self.config_path)
        if not self.preprocessing_config:
            raise FileNotFoundError(f"Preprocessing configuration not found or invalid at {config_path}")
        self.img_size = tuple(self.preprocessing_config['resize']['target_size'])
        self.preprocessor = Preprocessor(self.preprocessing_config)
        print("Preprocessing component initialized.")

        # 2. Load all Trained Models
        print("Loading trained models...")
        self.laterality_model = self._load_laterality_model()
        self.ra_classifier = self._load_ra_classifier()
        self.joint_detector = self._load_joint_detector()
        self.svh_scorer = self._load_svh_scorer()
        print("All models loaded.")

    def _load_laterality_model(self):
        model_path = os.path.join(self.models_dir, 'laterality_identification', 'hand_laterality_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Laterality model not found at {model_path}. Please train it first.")
        # use_transfer_learning=False as we are loading the saved model
        return HandLateralityIdentifier(model_path=model_path, img_size=self.img_size, use_transfer_learning=False)

    def _load_ra_classifier(self):
        model_path = os.path.join(self.models_dir, 'ra_classification', 'ra_classification_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RA Classification model not found at {model_path}. Please train it first.")
        # use_transfer_learning=False as we are loading the saved model
        return RAClassifier(model_path=model_path, img_size=self.img_size, use_transfer_learning=False)

    def _load_joint_detector(self):
        # We need num_joints when initializing HandJointDetector
        # It's not directly in config, so we assume a default or pass it via args.
        # For simplicity here, let's assume num_joints was 16 when trained.
        model_path = os.path.join(self.models_dir, 'joint_detection', 'hand_joint_detection_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Joint Detection model not found at {model_path}. Please train it first.")
        # You might need to make num_joints configurable or infer it from the loaded model
        # For this example, let's hardcode it to the default used in hand_joint_detection.py
        default_num_joints = 16 
        return HandJointDetector(model_path=model_path, img_size=self.img_size, num_joints=default_num_joints, use_transfer_learning=False)

    def _load_svh_scorer(self):
        model_path = os.path.join(self.models_dir, 'svh_scoring', 'svh_scoring_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SVH Scoring model not found at {model_path}. Please train it first.")
        # use_transfer_learning=False as we are loading the saved model
        return SVHSccorer(model_path=model_path, img_size=self.img_size, use_transfer_learning=False)

    def analyze_image(self, raw_image_path):
        """
        Runs the full analysis pipeline on a single raw X-ray image.

        Args:
            raw_image_path (str): Path to the input raw X-ray image file.

        Returns:
            dict: A dictionary containing all the analysis results.
        """
        print(f"\n--- Analyzing Image: {os.path.basename(raw_image_path)} ---")

        # 1. Preprocess the image
        preprocessed_image = self.preprocessor._load_image(raw_image_path)
        if preprocessed_image is None:
            print(f"Skipping {raw_image_path} due to loading error.")
            return None
        
        # Apply the rest of the preprocessing chain
        preprocessed_image = self.preprocessor._normalize_image(preprocessed_image, target_range=self.preprocessing_config['normalize']['target_range'])
        if self.preprocessing_config['clahe']['apply']:
            preprocessed_image = self.preprocessor._apply_clahe(
                preprocessed_image,
                clip_limit=self.preprocessing_config['clahe']['clip_limit'],
                tile_grid_size=tuple(self.preprocessing_config['clahe']['tile_grid_size'])
            )
        if self.preprocessing_config['denoising']['apply']:
            preprocessed_image = self.preprocessor._apply_gaussian_denoising(
                preprocessed_image,
                kernel_size=tuple(self.preprocessing_config['denoising']['kernel_size'])
            )
        preprocessed_image = self.preprocessor._resize_image(preprocessed_image, target_size=self.img_size)
        preprocessed_image, _ = self.preprocessor._segment_hand(preprocessed_image) # Placeholder

        preprocessed_image = preprocessed_image.astype(np.float32) # Ensure final type
        print("Image preprocessing complete.")

        results = {
            'image_path': raw_image_path,
            'preprocessed_image_data': preprocessed_image # Keep for visualization
        }

        # 2. Run predictions from all models on the SAME preprocessed image
        print("Running predictions...")
        
        # Laterality
        laterality_class, laterality_proba = self.laterality_model.predict(preprocessed_image)
        results['laterality'] = {'class': laterality_class, 'probability': laterality_proba}
        print(f"  - Laterality: {laterality_class} (Prob: {laterality_proba:.2f})")

        # RA Classification
        ra_class, ra_proba = self.ra_classifier.predict(preprocessed_image)
        results['ra_classification'] = {'class': ra_class, 'probability': ra_proba}
        print(f"  - RA Classification: {ra_class} (Prob: {ra_proba:.2f})")

        # Decision Logic: Only calculate SVH score if RA is classified as 'RA'
        if ra_class == 'RA':
            # SVH Scoring
            svh_score, _ = self.svh_scorer.predict(preprocessed_image)
            results['svh_score'] = svh_score
            print(f"  - SVH Score: {svh_score:.2f} (Calculated because RA was detected)")
        else:
            results['svh_score'] = None
            print(f"  - SVH Score: Not calculated (RA not detected)")

        # Joint Detection
        joint_coords, _ = self.joint_detector.predict(preprocessed_image)
        results['joint_coordinates'] = joint_coords
        print(f"  - Joint Detection: {self.joint_detector.num_joints} joints detected.")
        # print(f"    (First 3 joints: {joint_coords[:3].round(2)})") # Optional detailed print

        return results

    def visualize_results(self, results):
        """
        Visualizes the preprocessed image with predicted joints and text overlays of other results.
        """
        if not results or 'preprocessed_image_data' not in results:
            print("No results to visualize or image data missing.")
            return

        image = results['preprocessed_image_data']
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.title(f"Analysis Results for {os.path.basename(results['image_path'])}")
        plt.axis('off')

        # Add joint predictions
        if 'joint_coordinates' in results and results['joint_coordinates'] is not None:
            joint_coords = results['joint_coordinates']
            plt.scatter(joint_coords[:, 0], joint_coords[:, 1], c='lime', marker='x', s=80, linewidths=2, label='Predicted Joints')
            for i, (x, y) in enumerate(joint_coords):
                # Optionally label a few joints or all, adjust fontsize to prevent clutter
                if i % 3 == 0 or i < 4: # Label first 4 and every 3rd after
                    plt.text(x + 5, y + 5, self.joint_detector.joint_names[i], color='yellow', fontsize=8, weight='bold')
            plt.legend(loc='upper left')

        # Add text for other predictions
        text_y_start = image.shape[0] * 0.05
        text_x_start = image.shape[1] * 0.02
        line_height = image.shape[0] * 0.03
        text_color = 'cyan'

        plt.text(text_x_start, text_y_start,
                 f"Laterality: {results.get('laterality', {}).get('class', 'N/A')} (Prob: {results.get('laterality', {}).get('probability', 0):.2f})",
                 color=text_color, fontsize=12, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
        
        plt.text(text_x_start, text_y_start + line_height,
                 f"RA Class: {results.get('ra_classification', {}).get('class', 'N/A')} (Prob: {results.get('ra_classification', {}).get('probability', 0):.2f})",
                 color=text_color, fontsize=12, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

        svh_text = f"SVH Score: {results.get('svh_score', 'N/A'):.2f}" if results.get('svh_score') is not None else "SVH Score: N/A (RA not detected)"
        plt.text(text_x_start, text_y_start + 2*line_height,
                 svh_text,
                 color=text_color, fontsize=12, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

        plt.show()

# --- Command-line Interface (CLI) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full X-ray hand analysis pipeline.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the RAW X-ray image file to analyze.")
    
    args = parser.parse_args()

    # Define base paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml')
    models_base_dir = os.path.join(base_dir, 'models') # Base directory containing all model subfolders

    # --- IMPORTANT ---
    # Ensure all models are trained and saved to their respective directories within `models/`
    # before running this pipeline for real.
    # E.g., models/laterality_identification/hand_laterality_model.h5
    #       models/ra_classification/ra_classification_model.h5
    #       models/joint_detection/hand_joint_detection_model.h5
    #       models/svh_scoring/svh_scoring_model.h5
    # If a model is not found, the pipeline will raise a FileNotFoundError.

    try:
        pipeline = XRayAnalysisPipeline(config_path, models_base_dir)
        analysis_results = pipeline.analyze_image(args.image_path)
        
        if analysis_results:
            print("\n--- Pipeline Analysis Complete ---")
            print("Full Results:")
            print(analysis_results)
            pipeline.visualize_results(analysis_results)
        else:
            print("Pipeline analysis failed for the provided image.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required models are trained and saved in their respective subdirectories within 'models/'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")