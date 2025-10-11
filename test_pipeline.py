#!/usr/bin/env python3
"""
Test script for the PyTorch-based pipeline

This is a basic smoke test to verify imports and basic functionality.
For full pipeline testing, run the individual scripts manually.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[:200])  # First 200 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stderr:
            print(f"Error: {e.stderr[:500]}")  # First 500 chars
        return False


def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing PyTorch Pipeline Imports")
    print("=" * 60)
    
    try:
        import torch
        import torchvision
        import optuna
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ Torchvision version: {torchvision.__version__}")
        print(f"✅ Optuna version: {optuna.__version__}")
        print(f"✅ MPS available: {torch.backends.mps.is_available()}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_creation():
    """Test that models can be created"""
    print("\n🏗️  Testing Model Creation")
    print("=" * 60)
    
    try:
        import torch
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from models.mobilenetv2_model import MobileNetV2Model
        from models.resnet18_model import ResNet18Model
        
        config = {
            'activation_function': 'ReLU',
            'dropout_rate': 0.2
        }
        
        input_shape = (128, 128, 1)
        
        print("Creating MobileNetV2...")
        model1 = MobileNetV2Model(config, training=False, input_shape=input_shape)
        print("✅ MobileNetV2 created successfully")
        
        print("Creating ResNet18...")
        model2 = ResNet18Model(config, training=False, input_shape=input_shape)
        print("✅ ResNet18 created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 128, 128)
        output1 = model1(dummy_input)
        output2 = model2(dummy_input)
        
        print(f"✅ MobileNetV2 output shape: {output1.shape}")
        print(f"✅ ResNet18 output shape: {output2.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test that data loading works"""
    print("\n📊 Testing Data Loading")
    print("=" * 60)
    
    # Check if dummy data exists
    data_path = "data/spectrogram_cherrypicked/train.tfrecord"
    if not os.path.exists(data_path):
        print("⚠️  Dummy dataset not found")
        print("   Run: python create_dummy_dataset.py")
        return True  # Not a failure, just skip
    
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from utils import read_tfrecords, get_dataset_length
        
        print(f"Loading {data_path}...")
        dataset = read_tfrecords(data_path)
        size = get_dataset_length(dataset)
        
        print(f"✅ Dataset loaded successfully")
        print(f"✅ Dataset size: {size}")
        
        # Get first sample
        sample, label = dataset[0]
        print(f"✅ Sample shape: {sample.shape}")
        print(f"✅ Label: {label}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PyTorch Migration - Smoke Test")
    print("="*60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Model Creation
    results.append(("Model Creation", test_model_creation()))
    
    # Test 3: Data Loading
    results.append(("Data Loading", test_data_loading()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\n📚 Next steps:")
        print("1. Create dummy data: python create_dummy_dataset.py")
        print("2. Run cross-validation: python scripts/cross_validation.py --model mobilenetv2 --config config_test.yaml")
        print("3. Train a model: python scripts/train_regular.py --model mobilenetv2 --config config_test.yaml")
        print("\n📖 See QUICKSTART.md for more details")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
