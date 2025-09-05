#!/usr/bin/env python3
"""
Example usage script for skin lesion classification.

This script demonstrates how to use the refactored code for training
and evaluating skin lesion classification models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_preprocessing import load_ham10000_data, print_dataset_info
from train import main as train_main
from evaluate import main as evaluate_main
from utils import print_system_info, setup_logging

def setup_example_environment():
    """Setup the example environment and check requirements."""
    print("="*60)
    print("SKIN LESION CLASSIFICATION - EXAMPLE USAGE")
    print("="*60)
    
    # Print system information
    print_system_info()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure you have downloaded the datasets and placed them in the 'data' directory.")
        print("See data/README.md for instructions on obtaining the datasets.")
        return False
    
    # Check for required files
    required_files = [
        "data/HAM10000_metadata.csv",
        "data/images"  # Directory should exist
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the datasets and place them in the correct locations.")
        return False
    
    print("\n✅ Environment setup complete!")
    return True


def example_data_loading():
    """Demonstrate data loading functionality."""
    print("\n" + "="*40)
    print("EXAMPLE: Data Loading")
    print("="*40)
    
    try:
        # Load HAM10000 dataset
        print("Loading HAM10000 dataset...")
        image_paths, labels = load_ham10000_data(
            "data/HAM10000_metadata.csv", 
            "data/images"
        )
        
        # Print dataset information
        print_dataset_info(image_paths, labels, "HAM10000 Dataset")
        
        print("\n✅ Data loading example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading example: {e}")
        return False


def example_training():
    """Demonstrate training functionality."""
    print("\n" + "="*40)
    print("EXAMPLE: Model Training")
    print("="*40)
    
    # Check if we have a GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  No GPU available, training will be slow on CPU")
        device = "cpu"
    
    # Training arguments
    training_args = [
        "--data_dir", "data",
        "--output_dir", "models/example",
        "--model_name", "facebook/deit-base-patch16-224",  # Smaller model for demo
        "--image_size", "224",
        "--batch_size", "16",  # Smaller batch size for demo
        "--transfer_epochs", "2",  # Fewer epochs for demo
        "--finetune_epochs", "3",
        "--use_wandb"  # Enable wandb logging
    ]
    
    print("Starting training with the following arguments:")
    print(" ".join(training_args))
    print("\nNote: This is a demonstration with reduced parameters for faster execution.")
    print("For full training, use the train.py script directly with appropriate parameters.")
    
    try:
        # Note: In a real scenario, you would call train_main() here
        # For this example, we'll just show what would happen
        print("\n🚀 Training would start here...")
        print("   - Transfer learning phase: 2 epochs")
        print("   - Fine-tuning phase: 3 epochs")
        print("   - Model will be saved to models/example/")
        print("   - Logs will be available in Weights & Biases")
        
        print("\n✅ Training example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in training example: {e}")
        return False


def example_evaluation():
    """Demonstrate evaluation functionality."""
    print("\n" + "="*40)
    print("EXAMPLE: Model Evaluation")
    print("="*40)
    
    # Check if we have a trained model
    model_path = Path("models/example/final_model")
    if not model_path.exists():
        print("⚠️  No trained model found. Please train a model first.")
        print("   Run: python src/train.py --data_dir data --output_dir models/example")
        return False
    
    # Evaluation arguments
    evaluation_args = [
        "--model_path", str(model_path),
        "--data_dir", "data",
        "--output_dir", "results/example",
        "--device", "cuda" if torch.cuda.is_available() else "cpu"
    ]
    
    print("Starting evaluation with the following arguments:")
    print(" ".join(evaluation_args))
    
    try:
        # Note: In a real scenario, you would call evaluate_main() here
        print("\n🔍 Evaluation would start here...")
        print("   - Loading trained model")
        print("   - Making predictions on test data")
        print("   - Generating confusion matrix")
        print("   - Creating ROC curves")
        print("   - Saving results to results/example/")
        
        print("\n✅ Evaluation example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in evaluation example: {e}")
        return False


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="Skin Lesion Classification Example Usage")
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip the training example")
    parser.add_argument("--skip-evaluation", action="store_true", 
                       help="Skip the evaluation example")
    parser.add_argument("--data-only", action="store_true", 
                       help="Only run data loading example")
    
    args = parser.parse_args()
    
    # Setup environment
    if not setup_example_environment():
        return 1
    
    # Run examples
    success = True
    
    # Data loading example
    if not example_data_loading():
        success = False
    
    if args.data_only:
        return 0 if success else 1
    
    # Training example
    if not args.skip_training:
        if not example_training():
            success = False
    
    # Evaluation example
    if not args.skip_evaluation:
        if not example_evaluation():
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Download the datasets (see data/README.md)")
        print("2. Run full training: python src/train.py --data_dir data --output_dir models/")
        print("3. Evaluate model: python src/evaluate.py --model_path models/final_model --data_dir data")
        print("4. Explore the notebooks in notebooks/original_notebooks/ for detailed analysis")
    else:
        print("❌ Some examples failed. Please check the error messages above.")
    
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
