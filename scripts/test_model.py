#!/usr/bin/env python3
"""
Model testing and evaluation script for PyTorch models
Supports regular, QAT, and PTQ models
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Suppress verbose PyTorch output
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords, create_dataloader
from models.mobilenetv2_model import MobileNetV2Model, QuantizedMobileNetV2Model
from models.resnet18_model import ResNet18Model, QuantizedResNet18Model


def get_device(test_type="regular"):
    """Get available device"""
    # Quantized models must run on CPU due to CUDA limitations
    if test_type == "ptq":
        return torch.device("cpu")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def evaluate_model(model_type, model_path, config_path, test_type="regular"):
    """Evaluate a trained model on test data
    
    Args:
        model_type: Type of model ('mobilenetv2' or 'resnet18')
        model_path: Path to model file (.pth) or directory containing model files
        config_path: Path to main configuration file (for data paths, etc.)
        test_type: Type of model to test ('regular', 'qat', 'ptq')
    """
    
    # Load main configuration (for data paths, etc.)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Determine if model_path is a file or directory
    if os.path.isfile(model_path):
        # Direct model file provided
        model_file = model_path
        model_dir = os.path.dirname(model_path)
        print(f"Testing direct model file: {model_file}")
    else:
        # Directory provided (original behavior)
        model_dir = model_path
        print(f"Testing model from directory: {model_dir}")
    
    # Load model configuration from the model directory
    def load_config_file(file_path):
        """Load configuration file, handling both JSON and YAML formats"""
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    # Look for model config files in the model directory
    model_config_file = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(model_config_file):
        model_config_file = os.path.join(model_dir, 'qat_model_config.json')
    
    if not os.path.exists(model_config_file):
        print(f"⚠️  Warning: No model config found in {model_dir}")
        print("   Using default configuration...")
        # Use a default config if none found
        model_config = {
            'activation_function': 'ReLU',
            'dropout_rate': 0.2,
            'batch_size': 32
        }
    else:
        print(f"Using model config file: {model_config_file}")
        try:
            model_config = load_config_file(model_config_file)
        except Exception as e:
            print(f"⚠️  Error loading model config file {model_config_file}: {e}")
            print("   Using default configuration...")
            model_config = {
                'activation_function': 'ReLU',
                'dropout_rate': 0.2,
                'batch_size': 32
            }
                
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"test_results_{model_type}_{test_type}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Get device first (needed for data loader settings)
    device = get_device(test_type)
    print(f"Using device: {device}")
    if test_type == "ptq":
        print("  Note: Quantized models must run on CPU due to CUDA limitations")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['test_file']), 
        buffer_size=64000
    )
    
    # Get input shape from first sample
    sample, label = test_dataset[0]
    input_shape = tuple(sample.shape[1:]) + (sample.shape[0],)  # Convert (C, H, W) to (H, W, C) for config
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Input shape (HWC format): {input_shape}")
    
    # Create data loader with appropriate settings based on device
    use_pin_memory = device.type == 'cuda'
    test_loader = create_dataloader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    # Load model based on test type
    print(f"\nLoading {test_type} model...")
    
    if test_type == "regular":
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
        
        # Determine weights path
        if 'model_file' in locals():
            # Direct model file provided
            weights_path = model_file
        else:
            # Directory provided - look for weights
            weights_path = os.path.join(model_dir, f'{model_type}_model_best.pth')
            if not os.path.exists(weights_path):
                weights_path = os.path.join(model_dir, f'{model_type}_model_final_weights.pth')
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        print(f"Loading weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        
    elif test_type == "qat":
        # Load QAT model (before final quantization)
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
        
        # Determine weights path
        if 'model_file' in locals():
            # Direct model file provided
            weights_path = model_file
        else:
            # Directory provided - look for QAT weights
            weights_path = os.path.join(model_dir, f'{model_type}_qat_model_best.pth')
            if not os.path.exists(weights_path):
                weights_path = os.path.join(model_dir, f'{model_type}_qat_model_final_weights.pth')
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"QAT model weights not found: {weights_path}")
        
        print(f"Loading QAT weights from: {weights_path}")
        # Load state dict with strict=False to handle quantization parameters gracefully
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
        # Only show summary of missing/unexpected keys if there are any
        if missing_keys:
            print(f"  ⚠️  {len(missing_keys)} missing keys (quantization parameters)")
        if unexpected_keys:
            print(f"  ⚠️  {len(unexpected_keys)} unexpected keys (quantization parameters)")
        
        model = model.to(device)
        
    elif test_type == "ptq":
        # Load fully quantized model
        if 'model_file' in locals():
            # Direct model file provided
            quantized_path = model_file
        else:
            # Directory provided - look for quantized model
            quantized_path = os.path.join(model_dir, f'{model_type}_quantized_complete.pth')
            if not os.path.exists(quantized_path):
                quantized_path = os.path.join(model_dir, f'{model_type}_quantized_final.pth')
        
        if not os.path.exists(quantized_path):
            raise FileNotFoundError(f"Quantized model not found: {quantized_path}")
        
        print(f"Loading quantized model from: {quantized_path}")
        
        # Set quantized backend to qnnpack to avoid cuDNN grouped convolution issues
        # cuDNN quantized operations don't support grouped convolutions (groups > 1)
        # which MobileNetV2 uses for depthwise separable convolutions
        original_quantized_engine = None
        original_quantized_engine_env = None
        
        # Try to set backend via PyTorch API
        try:
            original_quantized_engine = torch.backends.quantized.engine
            torch.backends.quantized.engine = 'qnnpack'
            print("  Using quantized backend: qnnpack (to avoid cuDNN grouped conv limitations)")
        except AttributeError:
            # Older PyTorch versions might not have this attribute
            pass
        
        # Also try setting via environment variable as a fallback
        if 'PYTORCH_QUANTIZED_ENGINE' not in os.environ:
            os.environ['PYTORCH_QUANTIZED_ENGINE'] = 'qnnpack'
            original_quantized_engine_env = 'qnnpack'  # Mark that we set it
        else:
            original_quantized_engine_env = os.environ.get('PYTORCH_QUANTIZED_ENGINE')
            os.environ['PYTORCH_QUANTIZED_ENGINE'] = 'qnnpack'
        
        try:
            # Try loading with weights_only=False for complete models
            model = torch.load(quantized_path, map_location=device, weights_only=False)
            model = model.to(device)
            
            # Restore original backend after successful load
            if original_quantized_engine is not None:
                try:
                    torch.backends.quantized.engine = original_quantized_engine
                except:
                    pass
        except Exception as e:
            # Restore original backend before handling errors
            if original_quantized_engine is not None:
                try:
                    torch.backends.quantized.engine = original_quantized_engine
                except:
                    pass
            if original_quantized_engine_env is not None:
                if original_quantized_engine_env == 'qnnpack':
                    # We set it, so remove it
                    os.environ.pop('PYTORCH_QUANTIZED_ENGINE', None)
                else:
                    # Restore original value
                    os.environ['PYTORCH_QUANTIZED_ENGINE'] = original_quantized_engine_env
            
            # Check if this is the grouped convolution error
            if "groups" in str(e) and "cudnn" in str(e).lower():
                print(f"  ⚠️  Error: {e}")
                print("  This error occurs because cuDNN quantized operations don't support grouped convolutions.")
                print("  The quantized model may have been saved with cuDNN backend.")
                print("  Attempting workaround: loading with qnnpack backend forced...")
                
                # Try to force qnnpack by setting environment variable approach
                # Re-quantize the model on-the-fly if possible, or provide guidance
                raise RuntimeError(
                    f"Unable to load quantized model: {e}\n"
                    "Solution: The quantized model was likely created with cuDNN backend which doesn't support "
                    "grouped convolutions used in MobileNetV2. Please re-run PTQ quantization with qnnpack backend, "
                    "or use the state_dict loading method if available."
                )
            elif "weights_only" in str(e) or "Unsupported global" in str(e):
                print("  ⚠️  Complete model loading failed, trying to load as state_dict...")
                # Fallback: try to load as state_dict and create model architecture
                try:
                    # Load the state dict
                    state_dict = torch.load(quantized_path, map_location=device, weights_only=True)
                    
                    # Create model architecture
                    if model_type == 'mobilenetv2':
                        model = QuantizedMobileNetV2Model(
                            model_config=model_config,
                            training=False,
                            input_shape=input_shape
                        )
                    elif model_type == 'resnet18':
                        model = QuantizedResNet18Model(
                            model_config=model_config,
                            training=False,
                            input_shape=input_shape
                        )
                    
                    # Load the state dict
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    print("  ✅ Successfully loaded as state_dict")
                except Exception as e2:
                    print(f"  ❌ Failed to load as state_dict: {e2}")
                    raise e  # Re-raise original error
            else:
                raise e
    
    model.eval()
    
    # Collect predictions and true labels
    predictions = []
    true_labels = []
    
    print(f"\nEvaluating {model_type} {test_type} model...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).squeeze()
            
            # Handle single sample case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)
            
            predictions.extend(outputs.cpu().numpy().tolist())
            true_labels.extend(labels.cpu().numpy().tolist())
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(test_loader)}")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions, zero_division=0)
    recall = recall_score(true_labels, binary_predictions, zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, zero_division=0)
    
    # Check if we have both classes for AUC calculation
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, predictions)
    else:
        auc = 0.0
        print("Warning: Only one class present in test data, AUC set to 0.0")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, binary_predictions)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_type} {test_type}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # ROC Curve
    plt.subplot(1, 3, 2)
    if len(np.unique(true_labels)) > 1:
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type} {test_type}')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Insufficient data for ROC', ha='center', va='center')
        plt.title(f'ROC Curve - {model_type} {test_type}')
    
    # Precision-Recall Curve
    plt.subplot(1, 3, 3)
    if len(np.unique(true_labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, predictions)
        plt.plot(recall_curve, precision_curve)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_type} {test_type}')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for PR curve', ha='center', va='center')
        plt.title(f'Precision-Recall Curve - {model_type} {test_type}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f'{model_type}_{test_type}_evaluation.png'))
    plt.close()
    
    # Save detailed results
    results = {
        'model_type': model_type,
        'test_type': test_type,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc)
        },
        'confusion_matrix': cm.tolist(),
        'num_samples': len(true_labels)
    }
    
    # Save results to JSON
    with open(os.path.join(results_folder, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions to CSV
    predictions_data = list(zip(true_labels, predictions, binary_predictions))
    with open(os.path.join(results_folder, 'predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_label', 'prediction_prob', 'prediction_binary'])
        writer.writerows(predictions_data)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {model_type} {test_type}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"{'='*50}")
    print(f"Results saved to: {results_folder}")
    
    return results_folder, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing and evaluation with PyTorch")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--model_path", required=True, help="Path to trained model file (.pth) or directory containing model files")
    parser.add_argument("--test_type", choices=["regular", "qat", "ptq"], required=True, help="Type of model to test")
    parser.add_argument("--config", default="config.yaml", help="Main configuration file (for data paths, etc.)")
    
    args = parser.parse_args()
    
    # Run evaluation
    results_folder, results = evaluate_model(args.model, args.model_path, args.config, args.test_type)
    print(f"\n✅ Evaluation completed. Results folder: {results_folder}")
