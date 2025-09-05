"""
Evaluation module for skin lesion classification.

This module contains functions for evaluating trained models and generating
comprehensive evaluation reports including confusion matrices and ROC curves.
"""

import os
import argparse
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_curve
)
from data_preprocessing import CLASS_NAMES, LABEL_MAPPING
from train import SkinLesionDataset, load_images_from_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_model_and_processor(model_path: str):
    """
    Load a trained model and processor.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Tuple of (model, processor)
    """
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(model_path)
        return model, processor
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def predict_on_dataset(model, dataset, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, true_labels)
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Move to device
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            labels = sample["labels"].to(device)
            
            # Get predictions
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Get predicted class
            predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
            true_label = labels.cpu().numpy()
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            all_predictions.append(predicted_class)
            all_labels.append(true_label)
            all_probabilities.append(probabilities)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: List[str],
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(y_true: np.ndarray, 
                   y_prob: np.ndarray, 
                   class_names: List[str],
                   save_path: str = None) -> plt.Figure:
    """
    Plot ROC curves for all classes.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # Convert to binary classification
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_prob_binary)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    y_true_binary = np.eye(n_classes)[y_true]
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binary.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot individual class ROC curves
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro and macro averages
    ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    ax.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Skin Lesion Classification')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    return fig


def plot_precision_recall_curves(y_true: np.ndarray, 
                                y_prob: np.ndarray, 
                                class_names: List[str],
                                save_path: str = None) -> plt.Figure:
    """
    Plot Precision-Recall curves for all classes.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    n_classes = len(class_names)
    
    # Compute Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_prob_binary)
        pr_auc[i] = auc(recall[i], precision[i])
    
    # Plot Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {pr_auc[i]:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves for Skin Lesion Classification')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curves saved to {save_path}")
    
    return fig


def generate_classification_report(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 class_names: List[str]) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)


def analyze_class_performance(y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_prob: np.ndarray,
                            class_names: List[str]) -> Dict[str, Any]:
    """
    Analyze performance for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        class_names: List of class names
        
    Returns:
        Dictionary with class-wise performance metrics
    """
    n_classes = len(class_names)
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        # Binary classification metrics for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        y_prob_binary = y_prob[:, i]
        
        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC AUC for this class
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true_binary, y_prob_binary) if len(np.unique(y_true_binary)) > 1 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'support': np.sum(y_true_binary),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    return class_metrics


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """
    Save evaluation results to a text file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the results
    """
    with open(output_path, 'w') as f:
        f.write("SKIN LESION CLASSIFICATION - EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"Macro ROC AUC: {results['macro_roc_auc']:.4f}\n")
        f.write(f"Micro ROC AUC: {results['micro_roc_auc']:.4f}\n\n")
        
        # Classification report
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 20 + "\n")
        f.write(results['classification_report'])
        f.write("\n")
        
        # Class-wise performance
        f.write("CLASS-WISE PERFORMANCE:\n")
        f.write("-" * 25 + "\n")
        for class_name, metrics in results['class_metrics'].items():
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n")
    
    logger.info(f"Evaluation results saved to {output_path}")


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description="Evaluate skin lesion classification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model and processor
        logger.info("Loading model and processor...")
        model, processor = load_model_and_processor(args.model_path)
        
        # Load test data (assuming HAM10000 format for now)
        logger.info("Loading test data...")
        from data_preprocessing import load_ham10000_data
        
        test_csv = os.path.join(args.data_dir, "HAM10000_metadata.csv")
        test_images_dir = os.path.join(args.data_dir, "images")
        
        if not os.path.exists(test_csv):
            logger.error(f"Test data not found at {test_csv}")
            return
        
        test_paths, test_labels = load_ham10000_data(test_csv, test_images_dir)
        
        # Load images
        test_images = load_images_from_paths(test_paths)
        
        # Filter out failed loads
        valid_indices = [i for i, img in enumerate(test_images) if img is not None]
        test_images = [test_images[i] for i in valid_indices]
        test_labels = [test_labels[i] for i in valid_indices]
        
        logger.info(f"Loaded {len(test_images)} test images")
        
        # Create dataset
        test_dataset = SkinLesionDataset(test_images, test_labels, processor)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions, true_labels, probabilities = predict_on_dataset(model, test_dataset, args.device)
        
        # Calculate overall metrics
        accuracy = accuracy_score(true_labels, predictions)
        balanced_accuracy = balanced_accuracy_score(true_labels, predictions)
        
        # Calculate ROC AUC
        from sklearn.metrics import roc_auc_score
        y_true_binary = np.eye(len(CLASS_NAMES))[true_labels]
        macro_roc_auc = roc_auc_score(y_true_binary, probabilities, average='macro', multi_class='ovr')
        micro_roc_auc = roc_auc_score(y_true_binary, probabilities, average='micro', multi_class='ovr')
        
        # Generate classification report
        classification_report_str = generate_classification_report(true_labels, predictions, CLASS_NAMES)
        
        # Analyze class-wise performance
        class_metrics = analyze_class_performance(true_labels, predictions, probabilities, CLASS_NAMES)
        
        # Create plots
        logger.info("Generating evaluation plots...")
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(true_labels, predictions, CLASS_NAMES, cm_path)
        
        # ROC curves
        roc_path = os.path.join(args.output_dir, "roc_curves.png")
        plot_roc_curves(true_labels, probabilities, CLASS_NAMES, roc_path)
        
        # Precision-Recall curves
        pr_path = os.path.join(args.output_dir, "precision_recall_curves.png")
        plot_precision_recall_curves(true_labels, probabilities, CLASS_NAMES, pr_path)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'macro_roc_auc': macro_roc_auc,
            'micro_roc_auc': micro_roc_auc,
            'classification_report': classification_report_str,
            'class_metrics': class_metrics
        }
        
        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.txt")
        save_evaluation_results(results, results_path)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Macro ROC AUC: {macro_roc_auc:.4f}")
        print(f"Micro ROC AUC: {micro_roc_auc:.4f}")
        print("\nClass-wise Performance:")
        print("-" * 30)
        for class_name, metrics in class_metrics.items():
            print(f"{class_name:>8}: F1={metrics['f1_score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")
        print("="*60)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
