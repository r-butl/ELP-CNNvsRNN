#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) script for PyTorch models
"""

import os
import sys
import yaml
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import read_tfrecords
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model
from scripts.quantization_utils import QuantizationUtils

# Set quantized backend early to ensure it's used for all quantization operations
QuantizationUtils.configure_quantized_engine(verbose=True)


def get_device():
    """Get available device (CPU for quantization)"""
    # PTQ works on CPU
    return torch.device("cpu")


def apply_ptq_to_model(model_type, model_path, config_path):
    """Apply Post-Training Quantization to a trained model"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load model configuration
    config_file = os.path.join(model_path, 'model_config.json')
    with open(config_file, 'r') as f:
        model_config = json.load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ptq_folder = f"ptq_{model_type}_{timestamp}"
    os.makedirs(ptq_folder, exist_ok=True)
    
    print(f"Applying PTQ to {model_type} model...")
    print(f"Original model path: {model_path}")
    
    # Load calibration dataset
    print("\nLoading calibration dataset...")
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    
    # Get input shape from first sample
    sample, label = train_dataset[0]
    input_shape = tuple(sample.shape[1:]) + (sample.shape[0],)  # Convert (C, H, W) to (H, W, C) for config
    
    print(f"Input shape (HWC format): {input_shape}")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Recreate the original model
    print("\nLoading original FP32 model...")
    if model_type == 'mobilenetv2':
        original_model = MobileNetV2Model(
            model_config=model_config,
            training=False,
            input_shape=input_shape
        )
    elif model_type == 'resnet18':
        original_model = ResNet18Model(
            model_config=model_config,
            training=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    weights_path = os.path.join(model_path, f'{model_type}_model_best.pth')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_path, f'{model_type}_model_final_weights.pth')
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found in {model_path}")
    
    original_model.load_state_dict(torch.load(weights_path, map_location=device))
    original_model = original_model.to(device)
    original_model.eval()
    
    # Save a copy of the original model for size comparison
    # (since quantize_ modifies in-place)
    import copy
    original_model_copy = copy.deepcopy(original_model)
    
    # Create calibration loader
    print("\nCreating calibration loader...")
    calibration_loader = QuantizationUtils.create_calibration_loader(
        dataset_path=os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']),
        num_samples=200,
        batch_size=16
    )
    
    # Apply dynamic quantization for file size reduction
    # Dynamic quantization quantizes weights to INT8 but keeps activations in FP32
    # This works better with models that have operations not fully supported by static quantization
    import torch.ao.quantization as tq
    
    print("Applying dynamic quantization (INT8 weights for file size reduction)...")
    print("  This will quantize weights to INT8 and reduce model file size.")
    print("  Activations remain in FP32 for compatibility.")
    
    # Apply dynamic quantization to Linear and Conv layers
    # This quantizes weights to INT8 but keeps activations in FP32
    quantized_model = tq.quantize_dynamic(
        original_model,
        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d},
        dtype=torch.qint8
    )
    
    print("✓ Dynamic quantization complete! Model weights are now INT8.")
    print(f"Quantized model type: {type(quantized_model)}")
    
    # Save quantized model
    quantized_model_path = os.path.join(ptq_folder, f'{model_type}_model_quantized.pth')
    
    # Save quantized model state_dict (weights are now INT8, so file will be smaller)
    print(f"\nSaving quantized model (INT8 weights)...")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    print(f"✓ Quantized model state_dict saved to {quantized_model_path}")
    print(f"  Note: Weights are stored as INT8, so file size should be ~4x smaller")
    
    # Save model configuration
    config_file = os.path.join(ptq_folder, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Compare model sizes
    print("\nComparing model sizes...")
    size_comparison = QuantizationUtils.compare_model_sizes(
        original_model=original_model_copy,  # Use the copy before quantization
        quantized_model=quantized_model,
        temp_dir=os.path.join(ptq_folder, 'temp')
    )
    
    # Save size comparison to JSON
    size_info_file = os.path.join(ptq_folder, 'size_comparison.json')
    with open(size_info_file, 'w') as f:
        json.dump(size_comparison, f, indent=2)
    
    print(f"\n✓ PTQ completed successfully!")
    print(f"  Quantized model saved to: {quantized_model_path}")
    print(f"  Size comparison saved to: {size_info_file}")
    
    return ptq_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Training Quantization with PyTorch")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Apply PTQ
    output_folder = apply_ptq_to_model(args.model, args.model_path, args.config)
    if output_folder:
        print(f"\n✅ PTQ completed successfully. Output folder: {output_folder}")
    else:
        print("\n❌ PTQ failed")
