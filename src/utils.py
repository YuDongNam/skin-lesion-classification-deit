"""
Utility functions for skin lesion classification project.

This module contains helper functions for visualization, metrics calculation,
and other common utilities used throughout the project.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Training Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1].plot(history['train_acc'], label='Training Accuracy', marker='o')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def plot_class_distribution(labels: List[int], 
                           class_names: List[str],
                           title: str = "Class Distribution",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot class distribution in the dataset.
    
    Args:
        labels: List of class labels
        class_names: List of class names
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Count labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    bars = ax1.bar(range(len(unique_labels)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))))
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title} - Counts')
    ax1.set_xticks(range(len(unique_labels)))
    ax1.set_xticklabels([class_names[i] for i in unique_labels], rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=[class_names[i] for i in unique_labels], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} - Percentages')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    return fig


def visualize_predictions(images: List[np.ndarray], 
                         true_labels: List[int],
                         predicted_labels: List[int],
                         class_names: List[str],
                         num_samples: int = 16,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize model predictions on sample images.
    
    Args:
        images: List of images as numpy arrays
        true_labels: True class labels
        predicted_labels: Predicted class labels
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        # Display image
        axes[i].imshow(images[idx])
        axes[i].axis('off')
        
        # Set title with true and predicted labels
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[predicted_labels[idx]]
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_labels[idx] == predicted_labels[idx] else 'red'
        
        axes[i].set_title(f'True: {true_name}\nPred: {pred_name}', 
                         color=color, fontsize=10)
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions visualization saved to {save_path}")
    
    return fig


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of calculated metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        try:
            y_true_binary = np.eye(y_prob.shape[1])[y_true]
            metrics['roc_auc_macro'] = roc_auc_score(y_true_binary, y_prob, average='macro', multi_class='ovr')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true_binary, y_prob, average='weighted', multi_class='ovr')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float]):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    # Basic metrics
    print(f"Accuracy:           {metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}")
    
    # Macro averages
    print(f"\nMacro Averages:")
    print(f"  Precision:        {metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall:           {metrics.get('recall_macro', 0):.4f}")
    print(f"  F1-Score:         {metrics.get('f1_macro', 0):.4f}")
    
    # Weighted averages
    print(f"\nWeighted Averages:")
    print(f"  Precision:        {metrics.get('precision_weighted', 0):.4f}")
    print(f"  Recall:           {metrics.get('recall_weighted', 0):.4f}")
    print(f"  F1-Score:         {metrics.get('f1_weighted', 0):.4f}")
    
    # ROC AUC if available
    if 'roc_auc_macro' in metrics:
        print(f"\nROC AUC:")
        print(f"  Macro:            {metrics.get('roc_auc_macro', 0):.4f}")
        print(f"  Weighted:         {metrics.get('roc_auc_weighted', 0):.4f}")
    
    print("="*50)


def save_predictions(predictions: np.ndarray, 
                    true_labels: np.ndarray,
                    image_paths: List[str],
                    class_names: List[str],
                    output_path: str):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        image_paths: List of image file paths
        class_names: List of class names
        output_path: Path to save the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': true_labels,
        'predicted_label': predictions,
        'true_class': [class_names[i] for i in true_labels],
        'predicted_class': [class_names[i] for i in predictions],
        'correct': true_labels == predictions
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def create_experiment_summary(experiment_name: str,
                             model_config: Dict[str, Any],
                             training_config: Dict[str, Any],
                             results: Dict[str, float],
                             output_path: str):
    """
    Create a comprehensive experiment summary.
    
    Args:
        experiment_name: Name of the experiment
        model_config: Model configuration
        training_config: Training configuration
        results: Results dictionary
        output_path: Path to save the summary
    """
    with open(output_path, 'w') as f:
        f.write(f"EXPERIMENT SUMMARY: {experiment_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Model configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        for key, value in model_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 25 + "\n")
        for key, value in training_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Results
        f.write("RESULTS:\n")
        f.write("-" * 10 + "\n")
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Experiment summary saved to {output_path}")


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return system information.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    return gpu_info


def print_system_info():
    """
    Print system information including GPU availability.
    """
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    # GPU information
    gpu_info = check_gpu_availability()
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"GPU Count: {gpu_info['cuda_device_count']}")
        print(f"Current Device: {gpu_info['current_device']}")
        print(f"Device Name: {gpu_info['device_name']}")
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Python version
    import sys
    print(f"Python Version: {sys.version}")
    
    print("="*50)


def ensure_dir(path: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print model information including parameter count.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Trainable Percentage: {(trainable_params / total_params) * 100:.2f}%")


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file and return dict. Returns empty dict if not found.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML가 필요합니다. requirements.txt에 포함되어 있습니다.") from exc
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}
