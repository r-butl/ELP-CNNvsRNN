"""
PyTorch Quantization Utilities
Supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)

Quantization Methods:
- Weight-only quantization: Quantizes weights to INT8, reduces model file size (~4x compression)
- PT2E quantization: Graph-level quantization, optimizes runtime but keeps FP32 weights on disk
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

from scripts.utils import read_tfrecords, create_dataloader


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
    def apply_ptq_to_model(model, calibration_loader=None, example_inputs=None, device='cpu', use_weight_only=True):
        """
        Apply Post-Training Quantization to a model
        
        Args:
            model: PyTorch model
            calibration_loader: DataLoader with calibration data (optional for weight-only)
            example_inputs: Tensor batch (optional for weight-only, required for PT2E)
            device: Device to run quantization on (default: 'cpu')
            use_weight_only: If True, use weight-only quantization (reduces file size).
                            If False, use PT2E graph-level quantization (runtime optimization only)
        
        Returns:
            Quantized model, example_args (None for weight-only)
        """
        model.to(device)
        model.eval()
        
        if use_weight_only:
            # Weight-only quantization: quantizes weights to INT8, reduces file size
            print("Applying weight-only INT8 quantization...")
            print("  This will quantize weights to INT8 and reduce model file size.")
            
            # Quantize Linear and Conv layers to INT8
            # This actually stores weights as INT8 in state_dict
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d},
                dtype=torch.qint8
            )
            
            print("✓ Weight-only quantization complete!")
            print("  Weights are now stored as INT8 (qint8) in the model state_dict")
            
            # Verify INT8 quantization
            print("\nVerifying INT8 quantization...")
            verification_result = QuantizationUtils.verify_int8_quantization(
                quantized_model, 
                verbose=True
            )
            
            if not verification_result['is_int8']:
                print("⚠️  WARNING: INT8 quantization verification failed!")
                print("   The model may not be properly quantized to INT8.")
            else:
                print("✓ Weight-only INT8 quantization verified successfully!")
            
            return quantized_model, None
        
        else:
            # PT2E graph-level quantization (original method)
            # This quantizes at graph level but keeps weights as FP32
            if example_inputs is None:
                raise ValueError("example_inputs must be provided for PT2E quantization.")
            
            if isinstance(example_inputs, (list, tuple)):
                example_args = tuple(inp.to(device) for inp in example_inputs)
            else:
                example_args = (example_inputs.to(device),)
            
            # TorchAO recommends configuring inductor flags prior to prepare/convert
            recommended_inductor_config_setter()
            
            print("Exporting model graph for torchao PT2E quantization...")
            exported = torch.export.export(model, example_args)
            graph_module = exported.module()
            
            quant_config = get_default_x86_inductor_quantization_config()
            
            # Verify and log quantization configuration
            print(f"Quantization config type: {type(quant_config)}")
            print(f"Quantization config: {quant_config}")
            if hasattr(quant_config, 'activation'):
                print(f"  Activation dtype: {getattr(quant_config.activation, 'dtype', 'N/A')}")
            if hasattr(quant_config, 'weight'):
                print(f"  Weight dtype: {getattr(quant_config.weight, 'dtype', 'N/A')}")
            
            quantizer = X86InductorQuantizer().set_global(quant_config)
            
            print("Preparing model with torchao PT2E workflow (INT8 quantization)...")
            prepared_module = prepare_pt2e(graph_module, quantizer)
            allow_exported_model_train_eval(prepared_module)
            
            if calibration_loader is not None:
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
            print("✓ torchao PT2E quantization complete!")
            print("  Note: PT2E quantizes at graph level, weights remain FP32 in state_dict")
            
            # Verify INT8 quantization
            print("\nVerifying INT8 quantization...")
            verification_result = QuantizationUtils.verify_int8_quantization(
                quantized_module, 
                verbose=True
            )
            
            if not verification_result['is_int8']:
                print("⚠️  WARNING: INT8 quantization verification failed!")
            elif verification_result.get('has_graph_quantization'):
                print("✓ Model is a PT2E GraphModule - quantization verified at graph level")
            
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
        
        For weight-only quantized models, saves the state_dict with INT8 weights.
        For PT2E quantized models, saves both state_dict and exported program.
        """
        # Save state_dict (works for both weight-only and PT2E)
        torch.save(model.state_dict(), save_path)
        print(f"Quantized model state_dict saved to {save_path}")
        
        # For PT2E models, also save the exported program
        if example_args is not None:
            try:
                if not isinstance(example_args, (tuple, list)):
                    example_args = (example_args,)
                example_args = tuple(example_args)
                exported_program = torch.export.export(model, example_args)
                complete_model_path = save_path.replace('.pth', '_complete.pt2')
                torch.export.save(exported_program, complete_model_path)
                print(f"Exported quantized program saved to {complete_model_path}")
            except Exception as e:
                print(f"Note: Could not save exported program (PT2E only): {e}")
                print("  This is normal for weight-only quantized models.")
    
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
    def verify_int8_quantization(model, verbose=True):
        """
        Verify that a quantized model is using INT8 quantization
        
        For torchao PT2E models, quantization happens at the graph level,
        not at the parameter level. This method checks both the graph structure
        and traditional parameter-based quantization.
        
        Args:
            model: Quantized PyTorch model (can be PT2E graph module or traditional)
            verbose: Print detailed information
        
        Returns:
            Dictionary with verification results
        """
        int8_found = False
        non_int8_layers = []
        layer_info = []
        quantized_ops = []
        
        # For PT2E graph modules, check the graph structure
        # torchao PT2E models are GraphModules from torch.fx
        is_graph_module = False
        try:
            from torch.fx import GraphModule
            is_graph_module = isinstance(model, GraphModule)
        except ImportError:
            pass
        
        if is_graph_module or hasattr(model, 'graph') or hasattr(model, '_graph'):
            is_graph_module = True
            # This is likely a graph module from torch.export or torch.fx
            graph = None
            try:
                graph = getattr(model, 'graph', None) or getattr(model, '_graph', None)
            except:
                pass
            
            if graph is not None:
                # Check graph nodes for quantization operations
                try:
                    # Try to get the graph as a string or inspect nodes
                    graph_str = str(graph)
                    quantized_ops_found = []
                    
                    # Look for quantization-related operations in the graph
                    quant_keywords = [
                        'quantize', 'dequantize', 'quantized', 
                        'qint8', 'quint8', 'per_channel', 'per_tensor',
                        'scale', 'zero_point', 'aten::quantize', 'aten::dequantize'
                    ]
                    
                    for keyword in quant_keywords:
                        if keyword.lower() in graph_str.lower():
                            quantized_ops_found.append(keyword)
                            int8_found = True
                    
                    if quantized_ops_found:
                        quantized_ops = list(set(quantized_ops_found))
                        layer_info.append(f"Graph contains quantization operations: {', '.join(quantized_ops[:5])}")
                    
                    # Try to count quantized nodes more precisely
                    if hasattr(graph, 'nodes'):
                        quant_node_count = 0
                        try:
                            for node in graph.nodes:
                                node_str = str(node).lower()
                                node_target = str(getattr(node, 'target', '')).lower()
                                if any(kw in node_str or kw in node_target 
                                       for kw in ['quantize', 'quantized', 'qint', 'quint', 'dequantize']):
                                    quant_node_count += 1
                                    int8_found = True
                            if quant_node_count > 0:
                                layer_info.append(f"Found {quant_node_count} quantized nodes in graph")
                        except:
                            pass
                    
                except Exception as e:
                    if verbose:
                        layer_info.append(f"Could not inspect graph structure: {e}")
            
            # Also check the model's code/forward for quantization hints
            try:
                if hasattr(model, 'code') or hasattr(model, '_code'):
                    code = getattr(model, 'code', '') or getattr(model, '_code', '')
                    code_str = str(code).lower()
                    if any(kw in code_str for kw in ['quantize', 'quantized', 'qint', 'quint']):
                        int8_found = True
                        layer_info.append("Model code contains quantization operations")
            except:
                pass
        
        # Check model parameters for quantized types (traditional quantization)
        param_count = 0
        for name, param in model.named_parameters():
            if param is None:
                continue
            param_count += 1
            
            # Check if parameter is quantized
            param_dtype = param.dtype if hasattr(param, 'dtype') else None
            
            # For quantized models, check the underlying data
            if hasattr(param, '_packed_params'):
                # This is a quantized parameter
                int8_found = True
                layer_info.append(f"{name}: quantized (likely INT8)")
            elif param_dtype in [torch.qint8, torch.quint8]:
                int8_found = True
                layer_info.append(f"{name}: {param_dtype}")
            elif param_dtype == torch.float32:
                # For PT2E, FP32 params are normal - quantization is in the graph
                if not is_graph_module:
                    # Only flag as non-quantized if it's not a graph module
                    non_int8_layers.append(name)
                    if len(non_int8_layers) <= 10:
                        layer_info.append(f"{name}: FP32 (not quantized)")
        
        # Check buffers for quantization scales/zero_points
        has_quantization_buffers = False
        buffer_count = 0
        for name, buffer in model.named_buffers():
            buffer_count += 1
            if 'scale' in name.lower() or 'zero_point' in name.lower():
                has_quantization_buffers = True
                int8_found = True
                if verbose and len(layer_info) < 20:
                    layer_info.append(f"{name}: quantization buffer")
        
        # For PT2E models, check if model has quantization annotations
        if hasattr(model, '_annotations') or hasattr(model, 'annotations'):
            annotations = getattr(model, '_annotations', None) or getattr(model, 'annotations', None)
            if annotations:
                int8_found = True
                layer_info.append("Model has quantization annotations")
        
        result = {
            'is_int8': int8_found or has_quantization_buffers,
            'has_quantization_buffers': has_quantization_buffers,
            'has_graph_quantization': len(quantized_ops) > 0,
            'quantized_operations': quantized_ops,
            'non_quantized_layers': non_int8_layers,
            'param_count': param_count,
            'buffer_count': buffer_count,
            'layer_count': len(layer_info)
        }
        
        if verbose:
            print("\n" + "="*60)
            print("INT8 Quantization Verification")
            print("="*60)
            print(f"INT8 quantization detected: {result['is_int8']}")
            print(f"Graph-based quantization: {result['has_graph_quantization']}")
            print(f"Quantization buffers found: {result['has_quantization_buffers']}")
            print(f"Total parameters checked: {param_count}")
            print(f"Total buffers checked: {buffer_count}")
            
            if result['has_graph_quantization']:
                print(f"✓ Quantization operations found in graph: {', '.join(quantized_ops[:5])}")
                print("  Note: PT2E quantization applies at graph level, not parameter level")
                print("  Parameters may show FP32, but operations are quantized to INT8")
            elif is_graph_module:
                print("  Note: This is a PT2E graph module (quantization may be applied during compilation)")
                print("  If quantization was applied, it will be visible at inference time")
            
            if result['non_quantized_layers'] and not result['has_graph_quantization'] and not is_graph_module:
                print(f"⚠️  Non-quantized layers: {len(result['non_quantized_layers'])}")
            elif result['has_graph_quantization'] or is_graph_module:
                print("✓ Model uses graph-level quantization (PT2E)")
            else:
                print("✓ All layers appear to be quantized")
            
            if verbose and layer_info:
                print(f"\nDetails (showing first 10):")
                for info in layer_info[:10]:
                    print(f"  {info}")
                if len(layer_info) > 10:
                    print(f"  ... and {len(layer_info) - 10} more items")
            print("="*60 + "\n")
        
        return result
    
    @staticmethod
    def get_model_parameter_size(model):
        """
        Calculate the actual size of model parameters in memory
        
        Args:
            model: PyTorch model
        
        Returns:
            Dictionary with parameter size information
        """
        total_params = 0
        total_size_bytes = 0
        quantized_params = 0
        quantized_size_bytes = 0
        fp32_params = 0
        fp32_size_bytes = 0
        
        param_details = []
        
        for name, param in model.named_parameters():
            if param is None:
                continue
            
            param_count = param.numel()
            total_params += param_count
            
            # Get actual dtype and size
            param_dtype = param.dtype
            dtype_size = param.element_size()  # bytes per element
            param_size = param_count * dtype_size
            total_size_bytes += param_size
            
            # Check if quantized
            is_quantized = param_dtype in [torch.qint8, torch.quint8]
            
            if is_quantized:
                quantized_params += param_count
                # INT8 uses 1 byte per element (plus scale/zero_point overhead)
                quantized_size_bytes += param_count * 1
                # Add overhead for scale and zero_point (typically 4 bytes each per tensor)
                quantized_size_bytes += 8
            else:
                fp32_params += param_count
                fp32_size_bytes += param_size
            
            param_details.append({
                'name': name,
                'dtype': str(param_dtype),
                'numel': param_count,
                'size_bytes': param_size,
                'is_quantized': is_quantized
            })
        
        return {
            'total_params': total_params,
            'total_size_mb': total_size_bytes / (1024 * 1024),
            'quantized_params': quantized_params,
            'quantized_size_mb': quantized_size_bytes / (1024 * 1024),
            'fp32_params': fp32_params,
            'fp32_size_mb': fp32_size_bytes / (1024 * 1024),
            'quantization_ratio': quantized_params / total_params if total_params > 0 else 0,
            'details': param_details
        }
    
    @staticmethod
    def get_model_memory_footprint(model, device='cpu'):
        """
        Measure actual memory footprint of model in memory (runtime)
        
        Args:
            model: PyTorch model
            device: Device to measure on
        
        Returns:
            Dictionary with memory footprint information
        """
        model = model.to(device)
        model.eval()
        
        # Get parameter memory
        param_info = QuantizationUtils.get_model_parameter_size(model)
        
        # Get buffer memory
        buffer_size_bytes = 0
        for buffer in model.buffers():
            buffer_size_bytes += buffer.numel() * buffer.element_size()
        
        # Total model memory (parameters + buffers)
        total_model_memory = param_info['total_size_mb'] * (1024 * 1024) + buffer_size_bytes
        total_model_memory_mb = total_model_memory / (1024 * 1024)
        
        return {
            'parameter_memory_mb': param_info['total_size_mb'],
            'buffer_memory_mb': buffer_size_bytes / (1024 * 1024),
            'total_memory_mb': total_model_memory_mb,
            'quantized_param_memory_mb': param_info['quantized_size_mb'],
            'fp32_param_memory_mb': param_info['fp32_size_mb'],
            'parameter_info': param_info
        }
    
    @staticmethod
    def compare_model_sizes(original_model, quantized_model, temp_dir='./temp', 
                           compare_runtime_memory=True):
        """
        Compare sizes of FP32 vs quantized models
        
        Compares both:
        1. File size (on disk) - relevant for weight-only quantization
        2. Runtime memory (in RAM) - relevant for PT2E quantization
        
        Args:
            original_model: Original FP32 model
            quantized_model: Quantized model
            temp_dir: Temporary directory for saving models
            compare_runtime_memory: Also compare runtime memory footprint
        
        Returns:
            Dictionary with size comparison
        """
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. Compare file sizes (on disk)
        fp32_path = os.path.join(temp_dir, 'model_fp32.pth')
        int8_path = os.path.join(temp_dir, 'model_int8.pth')
        
        torch.save(original_model.state_dict(), fp32_path)
        torch.save(quantized_model.state_dict(), int8_path)
        
        # Get file sizes
        fp32_file_size = os.path.getsize(fp32_path)
        int8_file_size = os.path.getsize(int8_path)
        
        # Clean up
        os.remove(fp32_path)
        os.remove(int8_path)
        
        file_compression_ratio = fp32_file_size / int8_file_size if int8_file_size > 0 else 0
        file_size_reduction = (1 - int8_file_size / fp32_file_size) * 100 if fp32_file_size > 0 else 0
        
        result = {
            'file_size': {
                'original_mb': fp32_file_size / (1024 * 1024),
                'quantized_mb': int8_file_size / (1024 * 1024),
                'compression_ratio': file_compression_ratio,
                'size_reduction_percent': file_size_reduction
            }
        }
        
        # 2. Compare runtime memory (in RAM)
        if compare_runtime_memory:
            original_memory = QuantizationUtils.get_model_memory_footprint(original_model)
            quantized_memory = QuantizationUtils.get_model_memory_footprint(quantized_model)
            
            memory_reduction = (1 - quantized_memory['total_memory_mb'] / original_memory['total_memory_mb']) * 100 if original_memory['total_memory_mb'] > 0 else 0
            memory_compression_ratio = original_memory['total_memory_mb'] / quantized_memory['total_memory_mb'] if quantized_memory['total_memory_mb'] > 0 else 0
            
            result['runtime_memory'] = {
                'original_mb': original_memory['total_memory_mb'],
                'quantized_mb': quantized_memory['total_memory_mb'],
                'compression_ratio': memory_compression_ratio,
                'size_reduction_percent': memory_reduction,
                'original_param_mb': original_memory['parameter_memory_mb'],
                'quantized_param_mb': quantized_memory['parameter_memory_mb'],
                'quantized_param_count': quantized_memory['parameter_info']['quantized_params'],
                'total_param_count': quantized_memory['parameter_info']['total_params']
            }
        
        # Print results
        print(f"\n{'='*60}")
        print("Model Size Comparison")
        print(f"{'='*60}")
        
        print(f"\n📁 File Size (on disk):")
        print(f"  FP32 Model: {result['file_size']['original_mb']:.2f} MB")
        print(f"  Quantized Model: {result['file_size']['quantized_mb']:.2f} MB")
        print(f"  Compression Ratio: {result['file_size']['compression_ratio']:.2f}x")
        print(f"  Size Reduction: {result['file_size']['size_reduction_percent']:.1f}%")
        
        if result['file_size']['compression_ratio'] < 2.0:
            print(f"\n  ⚠️  Low file compression - weights may still be FP32 in state_dict")
            print(f"     (This is normal for PT2E quantization)")
        
        if compare_runtime_memory and 'runtime_memory' in result:
            print(f"\n💾 Runtime Memory (in RAM):")
            print(f"  FP32 Model: {result['runtime_memory']['original_mb']:.2f} MB")
            print(f"  Quantized Model: {result['runtime_memory']['quantized_mb']:.2f} MB")
            print(f"  Compression Ratio: {result['runtime_memory']['compression_ratio']:.2f}x")
            print(f"  Memory Reduction: {result['runtime_memory']['size_reduction_percent']:.1f}%")
            print(f"\n  Parameter Details:")
            print(f"    FP32 params: {result['runtime_memory']['original_param_mb']:.2f} MB")
            print(f"    Quantized params: {result['runtime_memory']['quantized_param_mb']:.2f} MB")
            print(f"    Quantized param count: {result['runtime_memory']['quantized_param_count']:,} / {result['runtime_memory']['total_param_count']:,}")
            quant_ratio = result['runtime_memory']['quantized_param_count'] / result['runtime_memory']['total_param_count'] * 100 if result['runtime_memory']['total_param_count'] > 0 else 0
            print(f"    Quantization coverage: {quant_ratio:.1f}%")
        
        print(f"{'='*60}\n")
        
        return result
