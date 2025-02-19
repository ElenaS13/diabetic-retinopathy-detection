import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

def load_labels(labels_file_path):
    """Load and process the labels file."""
    labels_df = pd.read_csv(labels_file_path)
    
    # Checkng for file extension if not present
    if not labels_df['image'].str.contains('.jpeg').any():
        labels_df['filename'] = labels_df['image'] + '.jpeg'
    
    severity_map = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }
    
    labels_df['severity'] = labels_df['level'].map(severity_map)
    return labels_df

class ImagePreprocessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def resize_image(self, image):
        return cv2.resize(image, self.target_size)

    def normalize_color(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def reduce_noise(self, image):
        return cv2.GaussianBlur(image, (3, 3), 0)

    def preprocess_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = self.resize_image(image)
        image = self.normalize_color(image)
        image = self.reduce_noise(image)
        return image

def process_all_images():
    """Process all images with progress tracking"""
    print("Starting full dataset preprocessing...")
    
    # Add file directories
    input_dir = "data/raw"
    output_dir = "data/processed/all"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get labels
    try:
        labels = pd.read_csv("data/raw/trainLabels.csv")
        print(f"Found {len(labels)} labels")
    except Exception as e:
        print(f"Error loading labels: {str(e)}")
        return

    # Get all images
    image_files = list(Path(input_dir).glob('*.jpeg'))
    total_images = len(image_files)
    print(f"Found {total_images} images to process")

    processed = 0
    errors = 0
    start_time = time.time()

    preprocessor = ImagePreprocessor()

    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        try:
            processed_img = preprocessor.preprocess_image(img_path)
            output_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(output_path), processed_img)
            processed += 1
            
        except Exception as e:
            errors += 1
            print(f"\nError processing {img_path.name}: {str(e)}")
            
        if processed % 1000 == 0:
            elapsed_time = time.time() - start_time
            images_per_second = processed / elapsed_time
            print(f"\nProgress update:")
            print(f"Processed: {processed}/{total_images} images")
            print(f"Errors: {errors}")
            print(f"Speed: {images_per_second:.2f} images/second")
            print(f"Estimated time remaining: {(total_images - processed) / images_per_second / 60:.2f} minutes")

    total_time = time.time() - start_time
    print("\nProcessing complete!")
    print(f"Total images processed: {processed}")
    print(f"Total errors: {errors}")
    print(f"Total time: {total_time / 60:.2f} minutes")
    print(f"Average speed: {processed / total_time:.2f} images/second")

if __name__ == "__main__":
    process_all_images()