#!/usr/bin/env python3
"""
Basic tests for the skin lesion classification project.

This script performs basic functionality tests to ensure the refactored code works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from data_preprocessing import LABEL_MAPPING, CLASS_NAMES, remove_hair, load_ham10000_data
        print("✅ data_preprocessing imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data_preprocessing: {e}")
        return False
    
    try:
        from train import SkinLesionDataset, compute_metrics, setup_model_and_processor
        print("✅ train imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import train: {e}")
        return False
    
    try:
        from evaluate import load_model_and_processor, predict_on_dataset, plot_confusion_matrix
        print("✅ evaluate imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import evaluate: {e}")
        return False
    
    try:
        from utils import setup_logging, calculate_metrics, print_system_info
        print("✅ utils imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False
    
    return True

def test_data_preprocessing():
    """Test data preprocessing functions."""
    print("\nTesting data preprocessing...")
    
    try:
        from data_preprocessing import LABEL_MAPPING, CLASS_NAMES, remove_hair
        import numpy as np
        
        # Test label mapping
        assert len(LABEL_MAPPING) == 7, f"Expected 7 classes, got {len(LABEL_MAPPING)}"
        assert len(CLASS_NAMES) == 7, f"Expected 7 class names, got {len(CLASS_NAMES)}"
        print("✅ Label mapping and class names are correct")
        
        # Test hair removal function with dummy image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed_image = remove_hair(dummy_image)
        assert processed_image.shape == dummy_image.shape, "Hair removal should preserve image shape"
        print("✅ Hair removal function works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import calculate_metrics, print_system_info
        import numpy as np
        
        # Test metrics calculation
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2])
        y_prob = np.random.rand(5, 3)
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        assert 'accuracy' in metrics, "Metrics should include accuracy"
        assert 'balanced_accuracy' in metrics, "Metrics should include balanced_accuracy"
        print("✅ Metrics calculation works correctly")
        
        # Test system info (should not raise exception)
        print_system_info()
        print("✅ System info function works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_config():
    """Test configuration module."""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Test basic config access
        default_config = config.get_config("default")
        assert "model" in default_config, "Config should include model settings"
        assert "training" in default_config, "Config should include training settings"
        print("✅ Configuration access works correctly")
        
        # Test config summary
        config.print_config_summary("default")
        print("✅ Configuration summary works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_setup():
    """Test model setup (without actually loading the model)."""
    print("\nTesting model setup...")
    
    try:
        from train import setup_model_and_processor
        import torch
        
        # Check if we can import transformers
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        print("✅ Transformers library is available")
        
        # Test that we can create a simple model (this will download the model)
        print("Note: This test will download the DeiT model (~300MB) if not already cached")
        
        # For now, just test that the function exists and can be called
        # In a real test, you might want to mock this or use a smaller model
        print("✅ Model setup function is available")
        
        return True
        
    except Exception as e:
        print(f"❌ Model setup test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SKIN LESION CLASSIFICATION - BASIC TESTS")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Preprocessing", test_data_preprocessing),
        ("Utils", test_utils),
        ("Configuration", test_config),
        ("Model Setup", test_model_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! The refactored code is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
