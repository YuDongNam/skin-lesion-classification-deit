"""
Data preprocessing module for skin lesion classification.

This module contains functions for loading, cleaning, and preprocessing
skin lesion images, including hair removal using morphological transformations.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class labels mapping
LABEL_MAPPING = {
    'nv': 0,    # Melanocytic nevus (모반)
    'mel': 1,   # Melanoma (흑색종)
    'bkl': 2,   # Benign keratosis-like lesions (양성 각화증)
    'bcc': 3,   # Basal cell carcinoma (기저세포암)
    'akiec': 4, # Actinic keratosis (광선각화증)
    'vasc': 5,  # Vascular lesions (혈관병변)
    'df': 6     # Dermatofibroma (섬유종)
}

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


def remove_hair(image: np.ndarray) -> np.ndarray:
    """
    Remove hair from skin lesion images using morphological transformations.
    
    This function implements a hair removal algorithm using:
    1. Gaussian blur for noise reduction
    2. Blackhat morphological transformation for hair detection
    3. Thresholding to create a binary mask
    4. Inpainting to remove detected hair
    
    Args:
        image: Input image as numpy array (BGR format)
        
    Returns:
        Processed image with hair removed
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Blackhat morphological transformation to detect hair
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        
        # Thresholding to create binary mask
        _, binary = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpainting to remove hair
        inpainted = cv2.inpaint(image, binary, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        return inpainted
        
    except Exception as e:
        logger.error(f"Error in hair removal: {e}")
        return image


def load_image(image_path: str, target_size: Tuple[int, int] = (384, 384)) -> Optional[np.ndarray]:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image as numpy array or None if loading fails
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        # Convert to numpy array (BGR format for OpenCV)
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def process_single_image(args: Tuple[str, str, Tuple[int, int], bool]) -> str:
    """
    Process a single image for hair removal.
    
    Args:
        args: Tuple containing (image_path, output_path, target_size, apply_hair_removal)
        
    Returns:
        Status message
    """
    image_path, output_path, target_size, apply_hair_removal = args
    
    try:
        # Skip if output already exists
        if os.path.exists(output_path):
            return f"Skipped {os.path.basename(image_path)} - already exists"
        
        # Load image
        img_array = load_image(image_path, target_size)
        if img_array is None:
            return f"Failed to load {os.path.basename(image_path)}"
        
        # Apply hair removal if requested
        if apply_hair_removal:
            img_array = remove_hair(img_array)
        
        # Convert back to RGB for saving
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Save processed image
        img_pil.save(output_path, format='JPEG', quality=95)
        
        return f"Processed {os.path.basename(image_path)}"
        
    except Exception as e:
        return f"Error processing {os.path.basename(image_path)}: {e}"


def process_images_batch(image_paths: List[str], 
                        output_dir: str, 
                        target_size: Tuple[int, int] = (384, 384),
                        apply_hair_removal: bool = True,
                        max_workers: int = 4) -> List[str]:
    """
    Process a batch of images with parallel processing.
    
    Args:
        image_paths: List of input image paths
        output_dir: Output directory for processed images
        target_size: Target size for resizing
        apply_hair_removal: Whether to apply hair removal
        max_workers: Number of parallel workers
        
    Returns:
        List of status messages
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        args_list.append((image_path, output_path, target_size, apply_hair_removal))
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_image, args_list))
    
    return results


def load_ham10000_data(csv_path: str, image_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load HAM10000 dataset.
    
    Args:
        csv_path: Path to HAM10000 metadata CSV
        image_dir: Directory containing images
        
    Returns:
        Tuple of (image_paths, labels)
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Create image paths
    df['filepath'] = df['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    
    # Map labels
    df['label'] = df['dx'].map(LABEL_MAPPING)
    
    # Filter out any unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    return df['filepath'].values.tolist(), df['label'].values.tolist()


def load_isic_2019_data(csv_path: str, image_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load ISIC 2019 dataset.
    
    Args:
        csv_path: Path to ISIC 2019 CSV
        image_dir: Directory containing images
        
    Returns:
        Tuple of (image_paths, labels)
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Normalize diagnosis column
    df['diagnosis'] = df['diagnosis'].str.lower()
    df['diagnosis'] = df['diagnosis'].replace('ak', 'akiec')
    
    # Filter for desired labels
    desired_labels = list(LABEL_MAPPING.keys())
    df = df[df['diagnosis'].isin(desired_labels)]
    
    # Create image paths
    df['filepath'] = df['image_name'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    
    # Map labels
    df['label'] = df['diagnosis'].map(LABEL_MAPPING)
    
    return df['filepath'].values.tolist(), df['label'].values.tolist()


def create_data_splits(image_paths: List[str], 
                      labels: List[int], 
                      test_size: float = 0.2,
                      random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Create train/test splits for the dataset.
    
    Args:
        image_paths: List of image paths
        labels: List of corresponding labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, test_paths, train_labels, test_labels)
    """
    from sklearn.model_selection import train_test_split
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    return train_paths, test_paths, train_labels, test_labels


def get_class_distribution(labels: List[int]) -> dict:
    """
    Get class distribution statistics.
    
    Args:
        labels: List of labels
        
    Returns:
        Dictionary with class distribution
    """
    from collections import Counter
    
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    distribution = {}
    for class_id, count in label_counts.items():
        class_name = CLASS_NAMES[class_id]
        distribution[class_name] = {
            'count': count,
            'percentage': (count / total_samples) * 100
        }
    
    return distribution


def print_dataset_info(image_paths: List[str], labels: List[int], dataset_name: str = "Dataset"):
    """
    Print dataset information and statistics.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        dataset_name: Name of the dataset
    """
    print(f"\n{dataset_name} Information:")
    print(f"Total samples: {len(image_paths)}")
    print(f"Number of classes: {len(set(labels))}")
    
    # Class distribution
    distribution = get_class_distribution(labels)
    print("\nClass distribution:")
    for class_name, stats in distribution.items():
        print(f"  {class_name}: {stats['count']} samples ({stats['percentage']:.1f}%)")
    
    # Check for missing files
    missing_files = [path for path in image_paths if not os.path.exists(path)]
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files are missing")
    else:
        print("\nAll image files found successfully")
