"""
PyTorch Quantization Utilities
Supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
"""

import torch
import torch.quantization
try:
    from torch.ao.quantization import (
        get_default_qat_qconfig, 
        get_default_qconfig,
        prepare_qat, 
        prepare,
        convert
    )
except ImportError:
    from torch.quantization import (
        get_default_qat_qconfig, 
        get_default_qconfig,
        prepare_qat, 
        prepare,
        convert
    )
import os
import numpy as np
from utils import read_tfrecords, create_dataloader


class QuantizationUtils:
    """Utilities for quantization aware training and post-training quantization"""
    
    @staticmethod
    def prepare_model_for_qat(model):
        """
        Prepare model for Quantization-Aware Training
        
        Args:
            model: PyTorch model with QuantStub and DeQuantStub
        
        Returns:
            Model prepared for QAT
        """
        model.train()
        
        # Set QAT config (qnnpack is good for mobile/ARM)
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        
        # Prepare model for QAT
        model_prepared = prepare_qat(model)
        
        return model_prepared
    
    @staticmethod
    def convert_qat_model(model):
        """
        Convert QAT model to quantized model
        
        Args:
            model: QAT-trained model
        
        Returns:
            Quantized model
        """
        model.eval()
        model_quantized = convert(model)
        return model_quantized
    
    @staticmethod
    def apply_ptq_to_model(model, calibration_loader, device='cpu'):
        """
        Apply Post-Training Quantization to a model using calibration data
        
        Args:
            model: PyTorch model with QuantStub and DeQuantStub
            calibration_loader: DataLoader with calibration data
            device: Device to run calibration on (default: 'cpu')
        
        Returns:
            Quantized model
        """
        model.eval()
        model.to(device)
        
        # Set quantization config for PTQ
        # Use 'qnnpack' for ARM CPUs (like Apple Silicon), 'fbgemm' for x86 CPUs
        # Default to 'qnnpack' as it's more commonly used and works well on ARM
        backend = 'qnnpack'  # Change to 'fbgemm' for x86 CPUs if needed
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # Prepare model for quantization (inplace operation)
        print("Preparing model for quantization...")
        torch.quantization.prepare(model, inplace=True)
        
        # Run calibration data through the model
        print("Running calibration data through model...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(calibration_loader):
                inputs = inputs.to(device)
                _ = model(inputs)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
        
        # Convert to quantized model (inplace operation)
        print("Converting to quantized model...")
        torch.quantization.convert(model, inplace=True)
        
        print("✓ Quantization complete!")
        return model
    
    @staticmethod
    def create_calibration_loader(dataset_path, num_samples=200, batch_size=16):
        """
        Create a calibration DataLoader for PTQ
        
        Args:
            dataset_path: Path to TFRecord file
            num_samples: Number of samples to use for calibration
            batch_size: Batch size for calibration
        
        Returns:
            DataLoader for calibration
        """
        dataset = read_tfrecords(dataset_path, buffer_size=64000)
        
        # Limit to num_samples
        dataset_size = len(dataset)
        if dataset_size > num_samples:
            indices = list(range(num_samples))
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"Using {num_samples} samples from {dataset_size} total for calibration")
        else:
            print(f"Using all {dataset_size} samples for calibration")
        
        loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return loader
    
    @staticmethod
    def save_quantized_model(model, save_path):
        """
        Save quantized PyTorch model
        
        For quantized models, we save both state_dict and complete model
        as quantized models can be tricky to reconstruct
        """
        # Save state_dict
        torch.save(model.state_dict(), save_path)
        
        # Also save complete model for easier loading
        complete_model_path = save_path.replace('.pth', '_complete.pth')
        torch.save(model, complete_model_path)
        
        print(f"Quantized model saved to {save_path}")
        print(f"Complete quantized model saved to {complete_model_path}")
    
    @staticmethod
    def load_quantized_model(model_class, model_config, model_path, input_shape=None):
        """
        Load quantized PyTorch model
        
        Args:
            model_class: Model class to instantiate
            model_config: Model configuration dict
            model_path: Path to saved model weights
            input_shape: Input shape for model
        
        Returns:
            Loaded quantized model
        """
        model = model_class(model_config=model_config, training=False, input_shape=input_shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    @staticmethod
    def evaluate_quantized_model(model, test_loader, device='cpu', threshold=0.5):
        """
        Evaluate quantized model performance
        
        Args:
            model: Quantized PyTorch model
            test_loader: DataLoader with test data
            device: Device to run evaluation on
            threshold: Classification threshold
        
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs).squeeze()
                predictions = (outputs > threshold).float()
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    @staticmethod
    def compare_model_sizes(original_model, quantized_model, temp_dir='./temp'):
        """
        Compare sizes of FP32 vs quantized models
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            temp_dir: Temporary directory for saving models
        
        Returns:
            Dictionary with size comparison
        """
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save models temporarily
        fp32_path = os.path.join(temp_dir, 'model_fp32.pth')
        int8_path = os.path.join(temp_dir, 'model_int8.pth')
        
        torch.save(original_model.state_dict(), fp32_path)
        torch.save(quantized_model.state_dict(), int8_path)
        
        # Get file sizes
        fp32_size = os.path.getsize(fp32_path)
        int8_size = os.path.getsize(int8_path)
        
        # Clean up
        os.remove(fp32_path)
        os.remove(int8_path)
        
        compression_ratio = fp32_size / int8_size if int8_size > 0 else 0
        
        result = {
            'original_size_mb': fp32_size / (1024 * 1024),
            'quantized_size_mb': int8_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0
        }
        
        print(f"\nModel Size Comparison:")
        print(f"  FP32 Model: {result['original_size_mb']:.2f} MB")
        print(f"  INT8 Model: {result['quantized_size_mb']:.2f} MB")
        print(f"  Compression Ratio: {result['compression_ratio']:.2f}x")
        print(f"  Size Reduction: {result['size_reduction_percent']:.1f}%")
        
        return result
