"""
Configuration file for skin lesion classification project.

This file contains all the configuration parameters used throughout the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset configuration
DATASET_CONFIG = {
    "ham10000": {
        "csv_path": DATA_DIR / "HAM10000_metadata.csv",
        "image_dir": DATA_DIR / "images",
        "classes": 7
    },
    "isic2019": {
        "csv_path": DATA_DIR / "train.csv",
        "image_dir": DATA_DIR / "images",
        "classes": 7
    }
}

# Class labels mapping
CLASS_LABELS = {
    0: "nv",      # Melanocytic nevus (모반)
    1: "mel",     # Melanoma (흑색종)
    2: "bkl",     # Benign keratosis-like lesions (양성 각화증)
    3: "bcc",     # Basal cell carcinoma (기저세포암)
    4: "akiec",   # Actinic keratosis (광선각화증)
    5: "vasc",    # Vascular lesions (혈관병변)
    6: "df"       # Dermatofibroma (섬유종)
}

# Reverse mapping
LABEL_TO_ID = {v: k for k, v in CLASS_LABELS.items()}

# Model configuration
MODEL_CONFIG = {
    "deit_base_384": {
        "name": "facebook/deit-base-patch16-384",
        "input_size": (384, 384),
        "patch_size": 16,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12
    },
    "deit_base_224": {
        "name": "facebook/deit-base-patch16-224",
        "input_size": (224, 224),
        "patch_size": 16,
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12
    }
}

# Training configuration
TRAINING_CONFIG = {
    "default": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs_transfer": 10,
        "num_epochs_finetune": 20,
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        "fp16": True,
        "dataloader_num_workers": 4,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 50,
        "early_stopping_patience": 10
    },
    "fast_demo": {
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "num_epochs_transfer": 2,
        "num_epochs_finetune": 3,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "fp16": True,
        "dataloader_num_workers": 2,
        "save_steps": 100,
        "eval_steps": 100,
        "logging_steps": 10,
        "early_stopping_patience": 3
    }
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "train": {
        "rotation_range": 30,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "vertical_flip": False,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2]
    },
    "validation": {
        "rotation_range": 0,
        "width_shift_range": 0,
        "height_shift_range": 0,
        "shear_range": 0,
        "zoom_range": 0,
        "horizontal_flip": False,
        "vertical_flip": False,
        "brightness_range": None,
        "contrast_range": None
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "metrics": [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro"
    ],
    "plot_config": {
        "figsize": (12, 8),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "husl"
    },
    "confusion_matrix": {
        "normalize": True,
        "figsize": (10, 8)
    },
    "roc_curves": {
        "figsize": (10, 8),
        "include_micro": True,
        "include_macro": True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # Set to a file path to enable file logging
    "console": True
}

# Weights & Biases configuration
WANDB_CONFIG = {
    "project": "skin-lesion-classification",
    "entity": None,  # Set to your W&B entity
    "tags": ["medical-imaging", "vision-transformer", "skin-lesion"],
    "notes": "DeiT-based skin lesion classification with transfer learning"
}

# Hardware configuration
HARDWARE_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto",
    "mixed_precision": True,
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 4
}

# File paths
PATHS = {
    "data": DATA_DIR,
    "models": MODELS_DIR,
    "results": RESULTS_DIR,
    "notebooks": NOTEBOOKS_DIR,
    "logs": PROJECT_ROOT / "logs",
    "checkpoints": MODELS_DIR / "checkpoints",
    "final_models": MODELS_DIR / "final_models"
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "excellent": {
        "balanced_accuracy": 0.80,
        "roc_auc": 0.90
    },
    "good": {
        "balanced_accuracy": 0.70,
        "roc_auc": 0.80
    },
    "acceptable": {
        "balanced_accuracy": 0.60,
        "roc_auc": 0.70
    },
    "needs_improvement": {
        "balanced_accuracy": 0.50,
        "roc_auc": 0.60
    }
}

# Clinical relevance thresholds
CLINICAL_THRESHOLDS = {
    "melanoma_recall": 0.80,  # High recall for cancer detection
    "melanoma_precision": 0.40,  # Lower precision acceptable
    "overall_accuracy": 0.60  # Minimum acceptable accuracy
}

def get_config(config_name: str = "default"):
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "default": {
            "dataset": DATASET_CONFIG,
            "model": MODEL_CONFIG["deit_base_384"],
            "training": TRAINING_CONFIG["default"],
            "augmentation": AUGMENTATION_CONFIG,
            "evaluation": EVALUATION_CONFIG,
            "logging": LOGGING_CONFIG,
            "wandb": WANDB_CONFIG,
            "hardware": HARDWARE_CONFIG,
            "paths": PATHS
        },
        "fast_demo": {
            "dataset": DATASET_CONFIG,
            "model": MODEL_CONFIG["deit_base_224"],
            "training": TRAINING_CONFIG["fast_demo"],
            "augmentation": AUGMENTATION_CONFIG,
            "evaluation": EVALUATION_CONFIG,
            "logging": LOGGING_CONFIG,
            "wandb": WANDB_CONFIG,
            "hardware": HARDWARE_CONFIG,
            "paths": PATHS
        }
    }
    
    return configs.get(config_name, configs["default"])

def print_config_summary(config_name: str = "default"):
    """
    Print a summary of the current configuration.
    
    Args:
        config_name: Name of the configuration to summarize
    """
    config = get_config(config_name)
    
    print(f"\n{'='*50}")
    print(f"CONFIGURATION SUMMARY: {config_name.upper()}")
    print(f"{'='*50}")
    
    print(f"\nModel: {config['model']['name']}")
    print(f"Input Size: {config['model']['input_size']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Transfer Epochs: {config['training']['num_epochs_transfer']}")
    print(f"Fine-tune Epochs: {config['training']['num_epochs_finetune']}")
    print(f"Device: {config['hardware']['device']}")
    print(f"Mixed Precision: {config['hardware']['mixed_precision']}")
    
    print(f"\nData Directory: {config['paths']['data']}")
    print(f"Models Directory: {config['paths']['models']}")
    print(f"Results Directory: {config['paths']['results']}")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    # Print configuration summary
    print_config_summary("default")
    print_config_summary("fast_demo")
