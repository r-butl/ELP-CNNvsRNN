#!/usr/bin/env python3
"""
Create dummy dataset for testing the new Snakemake pipeline
"""

import os
import numpy as np
import tensorflow as tf
from data_creation.utils import write_tfrecords

def create_dummy_spectrogram_data(num_samples, height=563, width=98):
    """Create dummy 2D spectrogram data for both CNN models"""
    
    samples = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:  # Positive samples (label=1)
            # Create spectrogram-like patterns with more structure
            # Higher energy in certain frequency bands
            spectrogram = np.random.randn(height, width) * 0.5
            # Add some structured patterns
            spectrogram[height//3:2*height//3, :] += 1.0  # Middle frequency band
            spectrogram[:, width//4:3*width//4] += 0.5   # Middle time range
            label = 1
        else:  # Negative samples (label=0)
            # More random, less structured patterns
            spectrogram = np.random.randn(height, width) * 0.8
            label = 0
        
        # Normalize
        spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)
        
        # Add channel dimension for CNN models (height, width) -> (height, width, 1)
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        
        samples.append(spectrogram.astype(np.float32))
        labels.append(label)
    
    return samples, labels

def create_dummy_dataset():
    """Create dummy dataset for testing"""
    
    print("🎵 Creating dummy dataset for testing...")
    
    # Create directories
    spec_dir = "data/spectrogram_cherrypicked"
    os.makedirs(spec_dir, exist_ok=True)
    
    # Dataset sizes
    train_size = 200
    val_size = 50
    test_size = 50
    
    print(f"📊 Creating {train_size} training samples, {val_size} validation samples, {test_size} test samples")
    
    # Create spectrogram data (for both CNN models)
    print("📈 Creating spectrogram data...")
    train_spec, train_spec_labels = create_dummy_spectrogram_data(train_size)
    val_spec, val_spec_labels = create_dummy_spectrogram_data(val_size)
    test_spec, test_spec_labels = create_dummy_spectrogram_data(test_size)
    
    # Convert to TensorFlow datasets
    def create_tf_dataset(samples, labels):
        return tf.data.Dataset.from_tensor_slices((samples, labels))
    
    # Spectrogram datasets
    train_spec_dataset = create_tf_dataset(train_spec, train_spec_labels)
    val_spec_dataset = create_tf_dataset(val_spec, val_spec_labels)
    test_spec_dataset = create_tf_dataset(test_spec, test_spec_labels)
    
    # Write spectrogram TFRecords
    print("💾 Writing spectrogram TFRecords...")
    write_tfrecords(train_spec_dataset, os.path.join(spec_dir, "train"))
    write_tfrecords(val_spec_dataset, os.path.join(spec_dir, "validate"))
    write_tfrecords(test_spec_dataset, os.path.join(spec_dir, "toughset_test"))
    
    print("✅ Dummy dataset created successfully!")
    print(f"📁 Spectrogram data saved to: {spec_dir}")
    
    # Print dataset statistics
    print("\n📊 Dataset Statistics:")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples: {test_size}")
    print(f"Spectrogram shape: {train_spec[0].shape}")
    print(f"Positive samples: {sum(train_spec_labels)}/{len(train_spec_labels)}")

if __name__ == "__main__":
    create_dummy_dataset()