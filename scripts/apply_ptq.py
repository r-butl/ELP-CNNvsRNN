#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) script
"""

import os
import sys
import yaml
import json
import tensorflow as tf
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model
from quantization_utils import QuantizationUtils

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
    
    # Load datasets for calibration
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    test_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['test_file']), 
        buffer_size=64000
    )
    
    # Get input shape
    for sample, label in train_dataset.take(1):
        shape = [None] + sample.shape
    
    # Recreate the original model
    if model_type == 'mobilenetv2':
        original_model = MobileNetV2Model(
            model_config=model_config, 
            training=False, 
            input_shape=shape[1:]
        )
    elif model_type == 'resnet18':
        original_model = ResNet18Model(
            model_config=model_config, 
            training=False, 
            input_shape=shape[1:]
        )
    
    original_model.build(shape)
    
    # Load weights
    weights_path = os.path.join(model_path, f'{model_type}_model_weights')
    original_model.load_weights(weights_path)
    
    # Create representative dataset for calibration
    representative_dataset = QuantizationUtils.create_representative_dataset(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']),
        num_samples=cfg['quantization']['ptq_calibration_samples']
    )
    
    # Apply PTQ
    print("Applying Post-Training Quantization...")
    quantized_tflite_model = QuantizationUtils.apply_ptq_to_model(
        original_model, 
        representative_dataset, 
        cfg
    )
    
    if quantized_tflite_model is None:
        print("Failed to apply PTQ")
        return None
    
    # Save quantized model
    quantized_model_path = os.path.join(ptq_folder, f'{model_type}_quantized.tflite')
    QuantizationUtils.save_quantized_model(quantized_tflite_model, quantized_model_path)
    
    # Evaluate original model
    print("Evaluating original model...")
    original_interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    original_interpreter.allocate_tensors()
    
    # For original model evaluation, we need to convert to TFLite first
    converter = tf.lite.TFLiteConverter.from_keras_model(original_model)
    original_tflite_model = converter.convert()
    original_tflite_path = os.path.join(ptq_folder, f'{model_type}_original.tflite')
    with open(original_tflite_path, 'wb') as f:
        f.write(original_tflite_model)
    
    # Evaluate both models
    print("Evaluating quantized model...")
    quantized_interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
    quantized_interpreter.allocate_tensors()
    
    # Compare model sizes
    size_comparison = QuantizationUtils.compare_model_sizes(
        original_tflite_path, 
        quantized_model_path
    )
    
    # Save results
    results = {
        'model_type': model_type,
        'original_model_path': model_path,
        'quantized_model_path': quantized_model_path,
        'size_comparison': size_comparison,
        'ptq_config': {
            'calibration_samples': cfg['quantization']['ptq_calibration_samples'],
            'representative_dataset_size': cfg['quantization']['ptq_representative_dataset_size']
        }
    }
    
    results_file = os.path.join(ptq_folder, 'ptq_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"PTQ completed. Results saved to {ptq_folder}")
    print(f"Model size reduction: {size_comparison['size_reduction_percent']:.2f}%")
    print(f"Compression ratio: {size_comparison['compression_ratio']:.2f}x")
    
    return ptq_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Training Quantization")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    
    args = parser.parse_args()
    
    # Apply PTQ
    output_folder = apply_ptq_to_model(args.model, args.model_path, args.config)
    if output_folder:
        print(f"PTQ completed. Output folder: {output_folder}")
    else:
        print("PTQ failed")
