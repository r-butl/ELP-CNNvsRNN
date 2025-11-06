"""
PyTorch Quantization Utilities
Supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
"""

import torch
import torch.quantization
try:
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
except ImportError:
    from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
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
        Apply Post-Training Quantization to a model
        Fast and simple, no retraining needed
        
        Args:
            model: PyTorch model
            calibration_loader: DataLoader with calibration data
            device: Device to run calibration on
        
        Returns:
            Quantized model
        """
        model.eval()
        model.to(device)
        
        # Explicitly set quantized backend to qnnpack to avoid cuDNN grouped conv issues
        # cuDNN quantized operations don't support grouped convolutions (groups > 1)
        # which MobileNetV2 uses for depthwise separable convolutions
        try:
            torch.backends.quantized.engine = 'qnnpack'
        except AttributeError:
            # Older PyTorch versions might not have this attribute
            pass
        
        # Also set environment variable as a fallback
        os.environ['PYTORCH_QUANTIZED_ENGINE'] = 'qnnpack'
        
        # Set quantization config
        qconfig = torch.quantization.get_default_qconfig('qnnpack')
        model.qconfig = qconfig
        
        # IMPORTANT: Explicitly set qconfig for all modules, especially rgb_conv
        # This ensures that all layers (including rgb_conv) are properly quantized
        # We need to set qconfig for all quantizable operations, not just the top-level model
        
        # First, set qconfig for direct children (like rgb_conv) before traversing nested modules
        # This is critical because direct children might be missed by automatic propagation
        direct_children_set = []
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Conv2d):
                if not hasattr(child, 'qconfig') or child.qconfig is None:
                    child.qconfig = qconfig
                    direct_children_set.append(name)
                    print(f"  ✓ Set qconfig for direct child: {name}")
        
        # Then set qconfig for all nested modules
        modules_set = []
        for name, module in model.named_modules():
            # Skip QuantStub and DeQuantStub - they have their own handling
            if isinstance(module, (torch.quantization.QuantStub, torch.quantization.DeQuantStub)):
                continue
            # Set qconfig for all Conv2d, Linear, and activation layers
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, 
                                 torch.nn.LeakyReLU, torch.nn.Sigmoid)):
                # Only set if not already set
                if not hasattr(module, 'qconfig') or module.qconfig is None:
                    module.qconfig = qconfig
                    # Only print for important modules to avoid too much output
                    if 'rgb_conv' in name or name.count('.') <= 1:  # Top-level or first-level modules
                        modules_set.append(name)
        
        if modules_set:
            print(f"  ✓ Set qconfig for {len(modules_set)} additional modules")
            if 'rgb_conv' in str(modules_set):
                print("  ✓ rgb_conv qconfig confirmed")
        
        # Verify qconfig is set before prepare
        if hasattr(model, 'rgb_conv') and model.rgb_conv is not None:
            if hasattr(model.rgb_conv, 'qconfig'):
                print(f"  rgb_conv qconfig BEFORE prepare: {model.rgb_conv.qconfig}")
            else:
                print(f"  ⚠️  WARNING: rgb_conv has no qconfig before prepare!")
                # Try to set it again
                model.rgb_conv.qconfig = qconfig
                print(f"  ✓ Manually set rgb_conv qconfig before prepare")
        
        # Use propagate_qconfig_ to ensure qconfig propagates correctly
        # This is important for layers that might not get qconfig automatically
        try:
            torch.quantization.propagate_qconfig_(model, qconfig_dict=None)
            print("  ✓ Propagated qconfig to all modules")
        except AttributeError:
            # propagate_qconfig_ might not be available in all PyTorch versions
            print("  ⚠️  propagate_qconfig_ not available, using default propagation")
        
        # Prepare model for quantization
        model_prepared = torch.quantization.prepare(model)
        
        # Verify that rgb_conv was prepared (if it exists)
        if hasattr(model_prepared, 'rgb_conv') and model_prepared.rgb_conv is not None:
            rgb_conv_prepared = model_prepared.rgb_conv
            print(f"  rgb_conv after prepare: {type(rgb_conv_prepared).__name__} (module: {type(rgb_conv_prepared).__module__})")
            if hasattr(rgb_conv_prepared, 'qconfig'):
                print(f"  rgb_conv qconfig after prepare: {rgb_conv_prepared.qconfig}")
            # Check if it's an observer wrapper (which is what prepare should create)
            if hasattr(rgb_conv_prepared, 'activation_post_process'):
                print(f"  ✓ rgb_conv has activation_post_process (properly prepared)")
            else:
                print(f"  ⚠️  WARNING: rgb_conv does not have activation_post_process after prepare!")
        
        # Calibrate with representative data
        print("Calibrating model for PTQ...")
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(calibration_loader):
                inputs = inputs.to(device)
                model_prepared(inputs)
                if batch_idx % 10 == 0:
                    print(f"  Calibration batch {batch_idx}/{len(calibration_loader)}")
        
        # Convert to quantized model (with error handling)
        try:
            model_quantized = torch.quantization.convert(model_prepared)
            
            # Verify that all layers were properly quantized
            # Check for any remaining non-quantized Conv2d/Linear layers (except if they're wrapped)
            def check_quantization_status(module, path=""):
                """Recursively check if modules are properly quantized"""
                issues = []
                for name, child in module.named_children():
                    full_path = f"{path}.{name}" if path else name
                    # Check if this is a Conv2d or Linear that should be quantized
                    if isinstance(child, (torch.nn.Conv2d, torch.nn.Linear)):
                        # If it's not a quantized version, that's a problem
                        if not isinstance(child, (torch.nn.quantized.modules.conv.Conv2d, 
                                                torch.nn.quantized.modules.linear.Linear)):
                            # But it might be wrapped in a fused module, so check parent
                            if not hasattr(module, '_modules') or name not in module._modules:
                                issues.append(f"{full_path}: {type(child).__name__} not quantized")
                    # Recursively check children
                    issues.extend(check_quantization_status(child, full_path))
                return issues
            
            quantization_issues = check_quantization_status(model_quantized)
            if quantization_issues:
                print("⚠️  Warning: Some layers may not be fully quantized:")
                for issue in quantization_issues[:5]:  # Show first 5 issues
                    print(f"   - {issue}")
                if len(quantization_issues) > 5:
                    print(f"   ... and {len(quantization_issues) - 5} more")
            
            # Specifically check for rgb_conv if it exists (for grayscale models)
            # This is critical because rgb_conv sits between QuantStub and base_model
            # and might not get quantized automatically
            rgb_conv_paths = []
            def find_rgb_conv(module, path=""):
                """Recursively find rgb_conv in the model"""
                if hasattr(module, 'rgb_conv') and module.rgb_conv is not None:
                    rgb_conv_paths.append((path, module.rgb_conv))
                for name, child in module.named_children():
                    child_path = f"{path}.{name}" if path else name
                    find_rgb_conv(child, child_path)
            
            find_rgb_conv(model_quantized)
            
            if rgb_conv_paths:
                for path, rgb_conv in rgb_conv_paths:
                    rgb_conv_type = type(rgb_conv).__name__
                    print(f"  Found rgb_conv at '{path}': {rgb_conv_type}")
                    
                    # Check if it's quantized by looking at the type name and module path
                    # QuantizedConv2d is in torch.nn.quantized.modules.conv
                    is_quantized = (
                        'Quantized' in rgb_conv_type or 
                        'quantized' in rgb_conv_type.lower() or
                        hasattr(rgb_conv, '_packed_params') or  # Quantized modules have _packed_params
                        'quantized' in str(type(rgb_conv).__module__).lower()
                    )
                    
                    # Check if it's a regular Conv2d (not quantized)
                    is_regular_conv2d = (
                        isinstance(rgb_conv, torch.nn.Conv2d) and 
                        not is_quantized and
                        'quantized' not in str(type(rgb_conv).__module__).lower()
                    )
                    
                    if is_regular_conv2d:
                        print(f"  ⚠️  ERROR: rgb_conv at '{path}' is still a regular Conv2d!")
                        print(f"     Type: {rgb_conv_type}")
                        print(f"     Module: {type(rgb_conv).__module__}")
                        print(f"     This will cause errors during inference.")
                        
                        # Try to get more info about why it wasn't quantized
                        if hasattr(rgb_conv, 'qconfig'):
                            print(f"     qconfig: {rgb_conv.qconfig}")
                        else:
                            print(f"     No qconfig attribute found!")
                        
                        raise RuntimeError(
                            f"rgb_conv layer at '{path}' was not properly quantized (type: {rgb_conv_type}). "
                            "This is a critical error that will prevent inference. "
                            "The layer sits between QuantStub and base_model and requires special handling. "
                            "Please check that qconfig was properly set for this layer before quantization."
                        )
                    elif is_quantized:
                        print(f"  ✅ rgb_conv at '{path}' is properly quantized: {rgb_conv_type}")
                    else:
                        print(f"  ⚠️  rgb_conv at '{path}' has unexpected type: {rgb_conv_type} (module: {type(rgb_conv).__module__})")
            
            print("PTQ completed!")
            return model_quantized
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized::" in str(e):
                print("⚠️  Warning: Quantization engine not available on this platform.")
                print("   Returning calibrated model without final quantization.")
                print("   Model has been calibrated with quantization-aware statistics.")
                return model_prepared
            else:
                raise
    
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
        if len(dataset) > num_samples:
            indices = list(range(num_samples))
            dataset = torch.utils.data.Subset(dataset, indices)
        
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
        """Save quantized PyTorch model"""
        torch.save(model.state_dict(), save_path)
        print(f"Quantized model saved to {save_path}")
    
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
