import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from utils import read_tfrecords

class QuantizationUtils:
    """Utilities for quantization aware training and post-training quantization"""
    
    @staticmethod
    def apply_qat_to_model(model, config):
        """Apply quantization aware training to a model"""
        # Apply quantization to the entire model
        quantize_model = tfmot.quantization.keras.quantize_model
        
        # Create a quantized version of the model
        qat_model = quantize_model(model)
        
        return qat_model
    
    @staticmethod
    def create_representative_dataset(dataset_path, num_samples=200):
        """Create a representative dataset for PTQ calibration"""
        dataset = read_tfrecords(dataset_path, buffer_size=64000)
        
        def representative_data_gen():
            count = 0
            for sample, _ in dataset:
                if count >= num_samples:
                    break
                # Ensure the sample has the right shape for the model
                yield [np.expand_dims(sample.numpy(), axis=0)]
                count += 1
        
        return representative_data_gen
    
    @staticmethod
    def apply_ptq_to_model(model, representative_dataset, config):
        """Apply post-training quantization to a model"""
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            quantized_tflite_model = converter.convert()
            return quantized_tflite_model
        except Exception as e:
            print(f"Error during PTQ: {e}")
            return None
    
    @staticmethod
    def save_quantized_model(quantized_model, save_path):
        """Save quantized model to file"""
        with open(save_path, 'wb') as f:
            f.write(quantized_model)
    
    @staticmethod
    def load_quantized_model(model_path):
        """Load quantized TFLite model"""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    @staticmethod
    def evaluate_quantized_model(interpreter, test_dataset, config):
        """Evaluate quantized model performance"""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        total_accuracy = 0.0
        total_samples = 0
        
        for sample, label in test_dataset:
            # Prepare input
            input_data = np.expand_dims(sample.numpy(), axis=0).astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]
            
            # Calculate accuracy
            pred_class = 1 if prediction > config['PROB_THRESHOLD'] else 0
            true_class = int(label.numpy())
            
            if pred_class == true_class:
                total_accuracy += 1.0
            
            total_samples += 1
        
        return total_accuracy / total_samples if total_samples > 0 else 0.0
    
    @staticmethod
    def compare_model_sizes(original_model_path, quantized_model_path):
        """Compare file sizes of original and quantized models"""
        import os
        
        original_size = os.path.getsize(original_model_path)
        quantized_size = os.path.getsize(quantized_model_path)
        
        compression_ratio = original_size / quantized_size
        
        return {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100
        }
