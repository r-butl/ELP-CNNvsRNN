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
from utils import read_tfrecords, get_dataset_length, create_dataloader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import read_tfrecords, create_dataloader
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model
print(f"PyTorch version: {torch.__version__}")

def get_device():
    """Get available device (CPU for quantization)"""
    # PTQ works on CPU
    return torch.device("cpu")


def apply_ptq_to_model(model_type, model_path, config_path, num_calibration_samples=500):
    """Apply Post-Training Quantization to a trained model"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load model configuration
    config_file = os.path.join(model_path, 'model_config.json')
    with open(config_file, 'r') as f:
        model_config = json.load(f)
    
    # # Create output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ptq_folder = f"ptq_{model_type}_{timestamp}"
    # os.makedirs(ptq_folder, exist_ok=True)
    
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

    # Load the calibration data
    indices = list(range(num_calibration_samples))
    dataset = torch.utils.data.Subset(train_dataset, indices)

    # Create calibration loader
    print("\nCreating calibration loader...")
    calibration_loader = create_dataloader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
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
    
    #  ###### Begin quantization
    example_inputs = (sample.unsqueeze(0).to(device),)
    exported_model = torch.export.export(original_model, example_inputs).module()

    from torchao.quantization.pt2e.quantize_pt2e import (
        prepare_pt2e,
        convert_pt2e
    )

    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            get_symmetric_quantization_config,
            XNNPACKQuantizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "XNNPACK quantizer is required for PTQ. Install ExecuTorch: "
            "`pip install executorch`."
        ) from exc
        exit()

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared_model = prepare_pt2e(exported_model, quantizer)

    print("\nCalibrating quantizer statistics...")
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            prepared_model(inputs.to(device))

    print("Converting calibrated model to INT8...")
    quantized_model = convert_pt2e(prepared_model)

    # print("\nExporting quantized model via torch.export...")

    # quantized_program = torch.export.export(quantized_model, example_inputs, strict=False)  # Basic python interpreter tracing, not using TorchDynamo
    # quantized_model_path = os.path.join(ptq_folder, f'{model_type}_model_quantized.pt2')
    # torch.export.save(quantized_program, quantized_model_path)

    return quantized_model



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


    print(f"✓ PT2E quantized program saved to {quantized_model_path}")
    
    import io
    # Quick size comparison
    def serialized_size_bytes(module):
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        return buffer.tell()

    float_bytes = serialized_size_bytes(original_model)
    int8_bytes = serialized_size_bytes(quantized_model)

    print(f"FP32 size: {float_bytes/1024/1024:.2f} MB")
    print(f"INT8 size: {int8_bytes/1024/1024:.2f} MB")
    print(f"Compression: {float_bytes/int8_bytes:.2f}×")

    # Save model configuration
    config_file = os.path.join(ptq_folder, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
        
