# ra_svh_prototype/src/data_preprocessing.py

import os
import cv2
import numpy as np
from skimage.util import img_as_ubyte
from PIL import Image 
import yaml 

class Preprocessor:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _load_image(file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in ['.jpg', '.jpeg', '.png']:
            try:
                image = Image.open(file_path).convert('L') # Convert to grayscale
                image = np.array(image).astype(np.float32)
                return image
            except Exception as e:
                print(f"Error loading image file {file_path}: {e}")
                return None
        else:
            print(f"Unsupported file format: {ext} for {file_path}. Please use .jpg, .jpeg, or .png.")
            return None

    @staticmethod
    def _normalize_image(image, target_range=(0, 255)):
        if image is None:
            return None

        current_min = np.min(image)
        current_max = np.max(image)

        if current_min == current_max:
            return np.full_like(image, target_range[0], dtype=np.float32)

        normalized_image = (image - current_min) / (current_max - current_min)
        normalized_image = normalized_image * (target_range[1] - target_range[0]) + target_range[0]
        return normalized_image.astype(np.float32)

    @staticmethod
    def _apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)): # Contrast Limited Adaptive Histogram Equalization
        # Input image 8-bit (0-255)
        if image is None:
            return None
        
        image_8bit = img_as_ubyte(Preprocessor._normalize_image(image, target_range=(0, 255))) 

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_image = clahe.apply(image_8bit)
        return clahe_image.astype(np.float32)

    @staticmethod
    def _apply_gaussian_denoising(image, kernel_size=(3, 3)): # Gaussian blurring, kernel_size should be odd (3,3), (5,5)
        if image is None:
            return None
        return cv2.GaussianBlur(image, kernel_size, 0).astype(np.float32)

    @staticmethod
    def _resize_image(image, target_size=(512, 512)): # cubic interpolation
        if image is None:
            return None
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC).astype(np.float32)

    @staticmethod
    def _segment_hand(image): # Placeholder for hand region segmentation
        print("--- INFO: Placeholder for Hand Region Segmentation. ---")
        print("   A dedicated U-Net model trained on X-ray hand masks would go here.")
        print("   Returning original image for now, and a dummy full mask.")
        
        return image, np.ones_like(image, dtype=np.uint8) * 255

    def process_image(self, file_path, output_dir): # pipeline preprocessing
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(file_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_preprocessed.png")

        print(f"Processing: {file_path}")

        image = self._load_image(file_path)
        if image is None:
            print(f"Skipping {file_path} due to loading error.")
            return None, None

        image = self._normalize_image(image, target_range=self.config['normalize']['target_range'])

        if self.config['clahe']['apply']:
            image = self._apply_clahe(
                image,
                clip_limit=self.config['clahe']['clip_limit'],
                tile_grid_size=tuple(self.config['clahe']['tile_grid_size'])
            )

        if self.config['denoising']['apply']:
            image = self._apply_gaussian_denoising(
                image,
                kernel_size=tuple(self.config['denoising']['kernel_size'])
            )

        image = self._resize_image(image, target_size=tuple(self.config['resize']['target_size']))

        segmented_image, _ = self._segment_hand(image)

        processed_image_uint8 = cv2.normalize(segmented_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(output_path, processed_image_uint8)
        print(f"Processed image saved to: {output_path}")

        return processed_image_uint8, output_path

if __name__ == "__main__":
    # --- Example Usage ---

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy_raw_dir = os.path.join(base_dir, 'data', 'raw', 'test_images')
    dummy_processed_dir = os.path.join(base_dir, 'data', 'processed')
    config_path = os.path.join(base_dir, 'configs', 'data_config.yaml')

    os.makedirs(dummy_raw_dir, exist_ok=True)
    os.makedirs(dummy_processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    dummy_image_png = np.random.randint(0, 256, size=(300, 400), dtype=np.uint8)
    filename_png = os.path.join(dummy_raw_dir, 'test_xray_01.png')
    cv2.imwrite(filename_png, dummy_image_png)
    print(f"Created dummy PNG: {filename_png}")

    if not os.path.exists(config_path):
        example_config = {
            'normalize': {
                'target_range': [0, 255]
            },
            'clahe': {
                'apply': True,
                'clip_limit': 2.0,
                'tile_grid_size': [8, 8]
            },
            'denoising': {
                'apply': True,
                'kernel_size': [3, 3]
            },
            'resize': {
                'target_size': [512, 512]
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
        print(f"Created example config file: {config_path}")
    else:
        print(f"Using existing config file: {config_path}")

    with open(config_path, 'r') as f:
        preprocessing_config = yaml.safe_load(f)

    preprocessor = Preprocessor(preprocessing_config)

    processed_img, processed_path = preprocessor.process_image(filename_png, dummy_processed_dir)

    if processed_img is not None:
        print(f"Processed image shape: {processed_img.shape}")
    print("\nPreprocessing pipeline completed for dummy image.")
    print(f"Check '{dummy_processed_dir}' directory for output.")