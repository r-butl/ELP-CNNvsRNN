#!/usr/bin/env python3
"""
Test script for the new Snakemake pipeline with dummy data
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
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def test_pipeline():
    """Test the complete pipeline with dummy data"""
    
    print("🧪 Testing Snakemake Pipeline with Dummy Data")
    print("=" * 50)
    
    # Step 1: Create dummy dataset
    if not run_command("python create_dummy_dataset.py", "Creating dummy dataset"):
        return False
    
    # Step 2: Test cross-validation for one model
    print("\n📊 Testing cross-validation...")
    if not run_command("python scripts/cross_validation.py --model mobilenetv2 --config config_test.yaml", 
                      "Cross-validation for MobileNetV2"):
        return False
    
    # Step 3: Test regular training
    print("\n🏋️ Testing regular training...")
    if not run_command("python scripts/train_regular.py --model mobilenetv2 --config config_test.yaml --best_config cross_validation_results/mobilenetv2_best_config/best_config.json", 
                      "Regular training for MobileNetV2"):
        return False
    
    # Step 4: Test QAT training
    print("\n⚡ Testing QAT training...")
    if not run_command("python scripts/train_qat.py --model mobilenetv2 --config config_test.yaml --best_config cross_validation_results/mobilenetv2_best_config/best_config.json", 
                      "QAT training for MobileNetV2"):
        return False
    
    # Step 5: Test PTQ application
    print("\n🔧 Testing PTQ application...")
    if not run_command("python scripts/apply_ptq.py --model mobilenetv2 --model_path training_results/mobilenetv2_regular --config config_test.yaml", 
                      "PTQ application for MobileNetV2"):
        return False
    
    # Step 6: Test model evaluation
    print("\n📈 Testing model evaluation...")
    if not run_command("python scripts/test_model.py --model mobilenetv2 --model_path training_results/mobilenetv2_regular --test_type regular --config config_test.yaml", 
                      "Testing regular MobileNetV2"):
        return False
    
    if not run_command("python scripts/test_model.py --model mobilenetv2 --model_path training_results/mobilenetv2_qat --test_type qat --config config_test.yaml", 
                      "Testing QAT MobileNetV2"):
        return False
    
    if not run_command("python scripts/test_model.py --model mobilenetv2 --model_path ptq_results/mobilenetv2_regular_ptq --test_type ptq --config config_test.yaml", 
                      "Testing PTQ MobileNetV2"):
        return False
    
    # Step 7: Test Snakemake workflow
    print("\n🐍 Testing Snakemake workflow...")
    if not run_command("snakemake --configfile config_test.yaml --cores 2 --dry-run", 
                      "Snakemake dry run"):
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("\n📁 Generated files:")
    print("  - data/spectrogram_cherrypicked/ (dummy dataset)")
    print("  - cross_validation_results/ (CV results)")
    print("  - training_results/ (trained models)")
    print("  - ptq_results/ (quantized models)")
    print("  - test_results/ (evaluation results)")
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
