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
        convert,
    )
except ImportError:
    from torch.quantization import (
        get_default_qat_qconfig,
        get_default_qconfig,
        prepare_qat,
        prepare,
        convert,
    )
import os
import numpy as np

import torchao  # noqa: F401
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from torchao.quantization.pt2e import allow_exported_model_train_eval
from torchao.quantization.utils import recommended_inductor_config_setter

from utils import read_tfrecords, create_dataloader


class QuantizationUtils:
    """Utilities for quantization aware training and post-training quantization"""
    
    @staticmethod
    def select_quantized_backend(preferred=None):
        """
        Determine the best available quantized backend.
        Prioritizes fbgemm on x86 and qnnpack otherwise.
        """
        engines = getattr(torch.backends.quantized, "supported_engines", [])
        if preferred and preferred in engines:
            return preferred
        if "fbgemm" in engines:
            return "fbgemm"
        if "qnnpack" in engines:
            return "qnnpack"
        if engines:
            return engines[0]
        raise RuntimeError("No quantized backends are available in this PyTorch build.")
    
    @staticmethod
    def configure_quantized_engine(preferred=None, verbose=True):
        """
        Select and activate the quantized backend (both torch backend and env var).
        Returns the backend name that was set.
        """
        backend = QuantizationUtils.select_quantized_backend(preferred=preferred)
        try:
            torch.backends.quantized.engine = backend
        except AttributeError:
            pass
        os.environ["PYTORCH_QUANTIZED_ENGINE"] = backend
        if verbose:
            print(f"✓ Quantized backend set to: {backend}")
        return backend
    
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
    def apply_ptq_to_model(model, calibration_loader, example_inputs, device='cpu'):
        """
        Apply Post-Training Quantization to a model using calibration data
        
        Args:
            model: PyTorch model
            calibration_loader: DataLoader with calibration data
            example_inputs: Tensor batch (or tuple of tensors) representing example inputs
            device: Device to run calibration on (default: 'cpu')
        
        Returns:
            Quantized model
        """
        if example_inputs is None:
            raise ValueError("example_inputs must be provided for torchao PTQ export.")
        
        model.to(device)
        
        if isinstance(example_inputs, (list, tuple)):
            example_args = tuple(inp.to(device) for inp in example_inputs)
        else:
            example_args = (example_inputs.to(device),)
        
        # TorchAO recommends configuring inductor flags prior to prepare/convert
        recommended_inductor_config_setter()
        
        print("Exporting model graph for torchao PTQ...")
        exported = torch.export.export(model, example_args)
        graph_module = exported.module()
        
        quant_config = get_default_x86_inductor_quantization_config()
        quantizer = X86InductorQuantizer().set_global(quant_config)
        
        print("Preparing model with torchao PT2E workflow...")
        prepared_module = prepare_pt2e(graph_module, quantizer)
        allow_exported_model_train_eval(prepared_module)
        
        print("Running calibration data through prepared model...")
        with torch.no_grad():
            prepared_module(*example_args)
            for batch_idx, (inputs, _) in enumerate(calibration_loader):
                if isinstance(inputs, (list, tuple)):
                    prepared_module(*[inp.to(device) for inp in inputs])
                else:
                    prepared_module(inputs.to(device))
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} calibration batches...")
        
        print("Converting prepared model to quantized form...")
        quantized_module = convert_pt2e(prepared_module)
        allow_exported_model_train_eval(quantized_module)
        print("✓ torchao PTQ complete!")
        return quantized_module, example_args
    
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
            pin_memory=False,
            drop_last=True
        )
        
        return loader
    
    @staticmethod
    def save_quantized_model(model, save_path, example_args=None):
        """
        Save quantized PyTorch model
        
        For torchao quantized models we save both the state_dict and an exported
        program that can be reloaded later.
        """
        if example_args is None:
            raise ValueError("example_args must be provided to save quantized model export.")
        
        # Save state_dict
        torch.save(model.state_dict(), save_path)
        
        # Export the quantized graph for reliable reloading
        if not isinstance(example_args, (tuple, list)):
            example_args = (example_args,)
        example_args = tuple(example_args)
        exported_program = torch.export.export(model, example_args)
        complete_model_path = save_path.replace('.pth', '_complete.pt2')
        torch.export.save(exported_program, complete_model_path)
        
        print(f"Quantized model state_dict saved to {save_path}")
        print(f"Exported quantized program saved to {complete_model_path}")
    
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
