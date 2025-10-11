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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model
from quantization_utils import QuantizationUtils


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
    
    original_model.load_state_dict(torch.load(weights_path, map_location=device))
    original_model = original_model.to(device)
    original_model.eval()
    
    print(f"Loaded weights from: {weights_path}")
    
    # Create calibration loader
    print("\nCreating calibration data loader...")
    calibration_loader = QuantizationUtils.create_calibration_loader(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']),
        num_samples=cfg['quantization']['ptq_calibration_samples'],
        batch_size=16
    )
    
    # Apply PTQ
    print("\n" + "="*60)
    print("Applying Post-Training Quantization...")
    print("="*60)
    quantized_model = QuantizationUtils.apply_ptq_to_model(
        original_model,
        calibration_loader,
        device=device
    )
    
    # Save quantized model
    quantized_model_path = os.path.join(ptq_folder, f'{model_type}_quantized.pth')
    QuantizationUtils.save_quantized_model(quantized_model, quantized_model_path)
    
    # Save complete quantized model (architecture + weights)
    quantized_complete_path = os.path.join(ptq_folder, f'{model_type}_quantized_complete.pth')
    torch.save(quantized_model, quantized_complete_path)
    
    # Save original model for comparison
    original_model_path = os.path.join(ptq_folder, f'{model_type}_original.pth')
    torch.save(original_model.state_dict(), original_model_path)
    
    # Compare model sizes
    print("\n" + "="*60)
    print("Comparing Model Sizes")
    print("="*60)
    size_comparison = QuantizationUtils.compare_model_sizes(
        original_model,
        quantized_model,
        temp_dir=ptq_folder
    )
    
    # Save model configuration
    config_output = os.path.join(ptq_folder, 'model_config.json')
    with open(config_output, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Save results
    results = {
        'model_type': model_type,
        'original_model_path': model_path,
        'quantized_model_path': quantized_model_path,
        'quantized_complete_path': quantized_complete_path,
        'size_comparison': size_comparison,
        'ptq_config': {
            'calibration_samples': cfg['quantization']['ptq_calibration_samples'],
            'representative_dataset_size': cfg['quantization'].get('ptq_representative_dataset_size', 200)
        }
    }
    
    results_file = os.path.join(ptq_folder, 'ptq_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PTQ Summary")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Original model size: {size_comparison['original_size_mb']:.2f} MB")
    print(f"Quantized model size: {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"Size reduction: {size_comparison['size_reduction_percent']:.1f}%")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    print(f"{'='*60}")
    print(f"Results saved to: {ptq_folder}")
    
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
