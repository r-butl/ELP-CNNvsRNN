#!/usr/bin/env python3
"""
Compare in-memory model sizes for regular, QAT, and PTQ models
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

from scripts.utils import read_tfrecords
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model
from scripts.quantization_utils import QuantizationUtils
import torch.ao.quantization as tq


def load_model(model_type, model_path, model_config, input_shape, model_variant="regular"):
    """Load a model based on variant (regular, qat, ptq)"""
    
    # Create base model
    if model_type == 'mobilenetv2':
        model = MobileNetV2Model(
            model_config=model_config,
            training=False,
            input_shape=input_shape
        )
    elif model_type == 'resnet18':
        model = ResNet18Model(
            model_config=model_config,
            training=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Determine weights path based on variant
    if model_variant == "regular":
        weights_path = os.path.join(model_path, f'{model_type}_model_best.pth')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, f'{model_type}_model_final_weights.pth')
    elif model_variant == "qat":
        weights_path = os.path.join(model_path, f'{model_type}_qat_model_best.pth')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, f'{model_type}_qat_model_final_weights.pth')
    elif model_variant == "ptq":
        weights_path = os.path.join(model_path, f'{model_type}_model_quantized.pth')
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    print(f"  Loading {model_variant} model from: {weights_path}")
    
    # Load weights
    if model_variant == "ptq":
        # For PTQ, apply dynamic quantization first, then load weights
        quantized_model = tq.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d},
            dtype=torch.qint8
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        quantized_model.load_state_dict(state_dict)
        model = quantized_model
    elif model_variant == "qat":
        # For QAT, load state_dict and check if it needs conversion
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        # Check if model is already quantized or needs conversion
        # If it has qconfig, it might need conversion
        if hasattr(model, 'qconfig') and model.qconfig is not None:
            try:
                # Try to convert QAT model to quantized form
                from torch.ao.quantization import convert
                model = convert(model, inplace=False)
                print(f"    Converted QAT model to quantized form")
            except Exception as e:
                print(f"    Note: QAT model not converted (may already be quantized): {e}")
    else:
        # For regular, just load the state_dict
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    
    model = model.to('cpu')
    model.eval()
    
    return model


def compare_all_models(model_type, regular_model_path, qat_model_path, ptq_model_path, config_path):
    """Compare in-memory sizes of regular, QAT, and PTQ models"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get input shape from dataset
    print("Loading dataset to determine input shape...")
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    sample, label = train_dataset[0]
    input_shape = tuple(sample.shape[1:]) + (sample.shape[0],)  # Convert (C, H, W) to (H, W, C)
    
    print(f"Input shape (HWC format): {input_shape}")
    
    # Load model configs
    def load_model_config(model_path):
        config_file = os.path.join(model_path, 'model_config.json')
        if not os.path.exists(config_file):
            config_file = os.path.join(model_path, 'qat_model_config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        return {
            'activation_function': 'ReLU',
            'dropout_rate': 0.2,
            'batch_size': 32
        }
    
    regular_config = load_model_config(regular_model_path)
    qat_config = load_model_config(qat_model_path) if qat_model_path else regular_config
    ptq_config = load_model_config(ptq_model_path) if ptq_model_path else regular_config
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"size_comparison_{model_type}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    # Load and measure regular model
    print(f"\n{'='*60}")
    print("Loading Regular Model")
    print(f"{'='*60}")
    try:
        regular_model = load_model(model_type, regular_model_path, regular_config, input_shape, "regular")
        regular_memory = QuantizationUtils.get_model_memory_footprint(regular_model, device='cpu')
        results['regular'] = {
            'memory_mb': regular_memory['total_memory_mb'],
            'param_memory_mb': regular_memory['parameter_memory_mb'],
            'buffer_memory_mb': regular_memory['buffer_memory_mb'],
            'quantized_param_mb': regular_memory['quantized_param_memory_mb'],
            'fp32_param_mb': regular_memory['fp32_param_memory_mb'],
            'param_info': {
                'total_params': regular_memory['parameter_info']['total_params'],
                'quantized_params': regular_memory['parameter_info']['quantized_params'],
                'fp32_params': regular_memory['parameter_info']['fp32_params']
            }
        }
        print(f"✓ Regular model loaded")
        print(f"  Total memory: {regular_memory['total_memory_mb']:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to load regular model: {e}")
        results['regular'] = None
    
    # Load and measure QAT model
    if qat_model_path and os.path.exists(qat_model_path):
        print(f"\n{'='*60}")
        print("Loading QAT Model")
        print(f"{'='*60}")
        try:
            qat_model = load_model(model_type, qat_model_path, qat_config, input_shape, "qat")
            qat_memory = QuantizationUtils.get_model_memory_footprint(qat_model, device='cpu')
            results['qat'] = {
                'memory_mb': qat_memory['total_memory_mb'],
                'param_memory_mb': qat_memory['parameter_memory_mb'],
                'buffer_memory_mb': qat_memory['buffer_memory_mb'],
                'quantized_param_mb': qat_memory['quantized_param_memory_mb'],
                'fp32_param_mb': qat_memory['fp32_param_memory_mb'],
                'param_info': {
                    'total_params': qat_memory['parameter_info']['total_params'],
                    'quantized_params': qat_memory['parameter_info']['quantized_params'],
                    'fp32_params': qat_memory['parameter_info']['fp32_params']
                }
            }
            print(f"✓ QAT model loaded")
            print(f"  Total memory: {qat_memory['total_memory_mb']:.2f} MB")
        except Exception as e:
            print(f"✗ Failed to load QAT model: {e}")
            results['qat'] = None
    else:
        print(f"\n⚠️  QAT model path not provided or doesn't exist")
        results['qat'] = None
    
    # Load and measure PTQ model
    if ptq_model_path and os.path.exists(ptq_model_path):
        print(f"\n{'='*60}")
        print("Loading PTQ Model")
        print(f"{'='*60}")
        try:
            ptq_model = load_model(model_type, ptq_model_path, ptq_config, input_shape, "ptq")
            ptq_memory = QuantizationUtils.get_model_memory_footprint(ptq_model, device='cpu')
            results['ptq'] = {
                'memory_mb': ptq_memory['total_memory_mb'],
                'param_memory_mb': ptq_memory['parameter_memory_mb'],
                'buffer_memory_mb': ptq_memory['buffer_memory_mb'],
                'quantized_param_mb': ptq_memory['quantized_param_memory_mb'],
                'fp32_param_mb': ptq_memory['fp32_param_memory_mb'],
                'param_info': {
                    'total_params': ptq_memory['parameter_info']['total_params'],
                    'quantized_params': ptq_memory['parameter_info']['quantized_params'],
                    'fp32_params': ptq_memory['parameter_info']['fp32_params']
                }
            }
            print(f"✓ PTQ model loaded")
            print(f"  Total memory: {ptq_memory['total_memory_mb']:.2f} MB")
        except Exception as e:
            print(f"✗ Failed to load PTQ model: {e}")
            results['ptq'] = None
    else:
        print(f"\n⚠️  PTQ model path not provided or doesn't exist")
        results['ptq'] = None
    
    # Calculate comparisons
    print(f"\n{'='*60}")
    print("In-Memory Size Comparison")
    print(f"{'='*60}")
    
    if results['regular']:
        regular_mem = results['regular']['memory_mb']
        print(f"\n📊 Regular Model (Baseline):")
        print(f"  Total Memory: {regular_mem:.2f} MB")
        print(f"  Parameter Memory: {results['regular']['param_memory_mb']:.2f} MB")
        print(f"  Buffer Memory: {results['regular']['buffer_memory_mb']:.2f} MB")
        print(f"  Total Parameters: {results['regular']['param_info']['total_params']:,}")
        
        # Compare QAT
        if results['qat']:
            qat_mem = results['qat']['memory_mb']
            qat_reduction = ((regular_mem - qat_mem) / regular_mem) * 100
            qat_ratio = regular_mem / qat_mem if qat_mem > 0 else 0
            
            print(f"\n📊 QAT Model:")
            print(f"  Total Memory: {qat_mem:.2f} MB")
            print(f"  Parameter Memory: {results['qat']['param_memory_mb']:.2f} MB")
            print(f"  Buffer Memory: {results['qat']['buffer_memory_mb']:.2f} MB")
            print(f"  Quantized Parameters: {results['qat']['param_info']['quantized_params']:,}")
            print(f"  FP32 Parameters: {results['qat']['param_info']['fp32_params']:,}")
            print(f"\n  vs Regular Model:")
            print(f"    Memory Reduction: {qat_reduction:.1f}%")
            print(f"    Compression Ratio: {qat_ratio:.2f}x")
            
            results['qat']['reduction_percent'] = qat_reduction
            results['qat']['compression_ratio'] = qat_ratio
        
        # Compare PTQ
        if results['ptq']:
            ptq_mem = results['ptq']['memory_mb']
            ptq_reduction = ((regular_mem - ptq_mem) / regular_mem) * 100
            ptq_ratio = regular_mem / ptq_mem if ptq_mem > 0 else 0
            
            print(f"\n📊 PTQ Model:")
            print(f"  Total Memory: {ptq_mem:.2f} MB")
            print(f"  Parameter Memory: {results['ptq']['param_memory_mb']:.2f} MB")
            print(f"  Buffer Memory: {results['ptq']['buffer_memory_mb']:.2f} MB")
            print(f"  Quantized Parameters: {results['ptq']['param_info']['quantized_params']:,}")
            print(f"  FP32 Parameters: {results['ptq']['param_info']['fp32_params']:,}")
            print(f"\n  vs Regular Model:")
            print(f"    Memory Reduction: {ptq_reduction:.1f}%")
            print(f"    Compression Ratio: {ptq_ratio:.2f}x")
            
            results['ptq']['reduction_percent'] = ptq_reduction
            results['ptq']['compression_ratio'] = ptq_ratio
        
        # Summary table
        print(f"\n{'='*60}")
        print("Summary Table")
        print(f"{'='*60}")
        print(f"{'Model Type':<15} {'Memory (MB)':<15} {'Reduction %':<15} {'Compression':<15}")
        print(f"{'-'*60}")
        print(f"{'Regular':<15} {regular_mem:<15.2f} {'0.0%':<15} {'1.00x':<15}")
        if results['qat']:
            print(f"{'QAT':<15} {results['qat']['memory_mb']:<15.2f} {results['qat']['reduction_percent']:<15.1f} {results['qat']['compression_ratio']:<15.2f}x")
        if results['ptq']:
            print(f"{'PTQ':<15} {results['ptq']['memory_mb']:<15.2f} {results['ptq']['reduction_percent']:<15.1f} {results['ptq']['compression_ratio']:<15.2f}x")
        print(f"{'='*60}")
    
    # Save results to JSON
    results_file = os.path.join(output_folder, 'size_comparison.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Output folder: {output_folder}")
    
    return output_folder, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare in-memory model sizes")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--regular_path", required=True, help="Path to regular model directory")
    parser.add_argument("--qat_path", help="Path to QAT model directory (optional)")
    parser.add_argument("--ptq_path", help="Path to PTQ model directory (optional)")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Run comparison
    output_folder, results = compare_all_models(
        args.model,
        args.regular_path,
        args.qat_path,
        args.ptq_path,
        args.config
    )
    
    print(f"\n✅ Size comparison completed. Output folder: {output_folder}")

