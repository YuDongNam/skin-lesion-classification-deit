"""
Training module for skin lesion classification using DeiT (Data-efficient Image Transformer).

This module contains the main training script for the DeiT model, which achieved
the best performance in the project with 45% balanced accuracy.
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from PIL import Image
import evaluate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
import wandb
from data_preprocessing import load_ham10000_data, load_isic_2019_data, CLASS_NAMES, LABEL_MAPPING
from utils import print_system_info, setup_logging, load_yaml_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionDataset(Dataset):
    """
    Custom dataset class for skin lesion images.
    """
    
    def __init__(self, images: list, labels: list, processor: AutoImageProcessor):
        """
        Initialize the dataset.
        
        Args:
            images: List of PIL Images
            labels: List of corresponding labels
            processor: Image processor for the model
        """
        self.images = images
        self.labels = labels
        self.processor = processor
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Process image with the processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_images_from_paths(image_paths: list, max_workers: int = 8) -> list:
    """
    Load images from file paths using parallel processing.
    
    Args:
        image_paths: List of image file paths
        max_workers: Number of parallel workers
        
    Returns:
        List of PIL Images
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def load_single_image(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_single_image, image_paths))
    
    # Filter out None values (failed loads)
    images = [img for img in images if img is not None]
    
    return images


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Evaluation predictions tuple (logits, labels)
        
    Returns:
        Dictionary of computed metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    
    # ROC AUC (macro average)
    num_labels = len(CLASS_NAMES)
    labels_onehot = np.zeros((len(labels), num_labels))
    labels_onehot[np.arange(len(labels)), labels] = 1
    
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    roc_auc = roc_auc_score(labels_onehot, probs, average='macro', multi_class='ovr')
    
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
    }


def setup_model_and_processor(model_name: str = "facebook/deit-base-patch16-384", 
                             num_labels: int = 7) -> tuple:
    """
    Setup the model and processor.
    
    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of output classes
        
    Returns:
        Tuple of (model, processor)
    """
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=True
    )
    
    return model, processor


def train_transfer_learning(model, 
                           train_dataset, 
                           eval_dataset, 
                           output_dir: str,
                           run_name: str = "deit_transfer",
                           num_epochs: int = 10,
                           batch_size: int = 32,
                           use_wandb: bool = True) -> Trainer:
    """
    Train the model using transfer learning (classifier only).
    
    Args:
        model: Pre-trained model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        run_name: Name for the training run
        num_epochs: Number of training epochs
        batch_size: Batch size
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Trained trainer object
    """
    # Freeze all parameters except classifier
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "transfer_learning"),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="wandb" if use_wandb else None,
        fp16=True,  # Mixed precision training
        dataloader_num_workers=4,
        save_total_limit=2,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train the model
    logger.info("Starting transfer learning phase...")
    trainer.train()
    
    return trainer


def train_fine_tuning(model, 
                     train_dataset, 
                     eval_dataset, 
                     output_dir: str,
                     run_name: str = "deit_finetuning",
                     num_epochs: int = 20,
                     batch_size: int = 32,
                     use_wandb: bool = True) -> Trainer:
    """
    Fine-tune the model (unfreeze all parameters).
    
    Args:
        model: Pre-trained model
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        run_name: Name for the training run
        num_epochs: Number of training epochs
        batch_size: Batch size
        use_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Trained trainer object
    """
    # Unfreeze all parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Setup training arguments for fine-tuning
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "fine_tuning"),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="wandb" if use_wandb else None,
        fp16=True,  # Mixed precision training
        dataloader_num_workers=4,
        save_total_limit=2,
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    # Train the model
    logger.info("Starting fine-tuning phase...")
    trainer.train()
    
    return trainer


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description="Train DeiT model for skin lesion classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory for models")
    parser.add_argument("--model_name", type=str, default="facebook/deit-base-patch16-384", 
                       help="Pre-trained model name")
    parser.add_argument("--image_size", type=int, default=384, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--transfer_epochs", type=int, default=10, help="Transfer learning epochs")
    parser.add_argument("--finetune_epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="skin-lesion-classification", 
                       help="W&B project name")
    parser.add_argument("--config", type=str, default=None, help="YAML config path for experiment")
    
    args = parser.parse_args()

    # Load YAML config if provided and override args
    if args.config:
        cfg = load_yaml_config(args.config)
        if cfg:
            # Map known keys
            for k, v in cfg.get('train', {}).items():
                if hasattr(args, k):
                    setattr(args, k, v)
            # Top-level shortcuts
            for key in [
                'data_dir','output_dir','model_name','image_size','batch_size',
                'transfer_epochs','finetune_epochs','use_wandb','wandb_project'
            ]:
                if key in cfg:
                    setattr(args, key, cfg[key])
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    try:
        # Load datasets
        logger.info("Loading datasets...")
        
        # Load HAM10000 data (training)
        ham10000_csv = os.path.join(args.data_dir, "HAM10000_metadata.csv")
        ham10000_images_dir = os.path.join(args.data_dir, "images")
        
        if os.path.exists(ham10000_csv):
            train_paths, train_labels = load_ham10000_data(ham10000_csv, ham10000_images_dir)
            logger.info(f"Loaded HAM10000 dataset: {len(train_paths)} samples")
        else:
            logger.error(f"HAM10000 metadata not found at {ham10000_csv}")
            return
        
        # Load images
        logger.info("Loading images...")
        train_images = load_images_from_paths(train_paths)
        
        # Filter out failed loads
        valid_indices = [i for i, img in enumerate(train_images) if img is not None]
        train_images = [train_images[i] for i in valid_indices]
        train_labels = [train_labels[i] for i in valid_indices]
        
        logger.info(f"Successfully loaded {len(train_images)} training images")
        
        # Setup model and processor
        logger.info("Setting up model and processor...")
        model, processor = setup_model_and_processor(args.model_name)
        
        # Create datasets
        train_dataset = SkinLesionDataset(train_images, train_labels, processor)
        
        # For evaluation, we'll use a subset of training data
        # In a real scenario, you would have separate validation data
        eval_size = min(1000, len(train_images) // 10)
        eval_indices = np.random.choice(len(train_images), eval_size, replace=False)
        eval_images = [train_images[i] for i in eval_indices]
        eval_labels = [train_labels[i] for i in eval_indices]
        eval_dataset = SkinLesionDataset(eval_images, eval_labels, processor)
        
        # Train transfer learning phase
        logger.info("Starting transfer learning...")
        transfer_trainer = train_transfer_learning(
            model, train_dataset, eval_dataset, args.output_dir,
            run_name="deit_transfer", num_epochs=args.transfer_epochs,
            batch_size=args.batch_size, use_wandb=args.use_wandb
        )
        
        # Evaluate transfer learning
        transfer_results = transfer_trainer.evaluate()
        logger.info(f"Transfer learning results: {transfer_results}")
        
        # Train fine-tuning phase
        logger.info("Starting fine-tuning...")
        finetune_trainer = train_fine_tuning(
            model, train_dataset, eval_dataset, args.output_dir,
            run_name="deit_finetuning", num_epochs=args.finetune_epochs,
            batch_size=args.batch_size, use_wandb=args.use_wandb
        )
        
        # Final evaluation
        final_results = finetune_trainer.evaluate()
        logger.info(f"Final results: {final_results}")
        
        # Save the final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        finetune_trainer.save_model(final_model_path)
        processor.save_pretrained(final_model_path)
        
        logger.info(f"Model saved to {final_model_path}")
        
        # Print final metrics
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Final Accuracy: {final_results['eval_accuracy']:.4f}")
        print(f"Final Balanced Accuracy: {final_results['eval_balanced_accuracy']:.4f}")
        print(f"Final ROC AUC: {final_results['eval_roc_auc']:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
