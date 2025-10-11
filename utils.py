"""
Utility functions for PyTorch-based training pipeline
Includes data loading from TFRecords and conversion to PyTorch datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorflow as tf


def format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))


class TFRecordDataset(Dataset):
    """PyTorch Dataset wrapper for TFRecord files"""
    
    def __init__(self, tfrecord_path, buffer_size=1000):
        self.tfrecord_path = tfrecord_path
        self.buffer_size = buffer_size
        
        # Load all data into memory (since TFRecords are TF-specific)
        self.samples = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load TFRecord data using TensorFlow and convert to NumPy"""
        feature_description = {
            'sample': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        
        def _parse_function(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            sample = tf.io.parse_tensor(parsed['sample'], out_type=tf.float32)
            label = parsed['label']
            return sample, label
        
        # Load TFRecord dataset
        dataset = tf.data.TFRecordDataset(self.tfrecord_path, buffer_size=self.buffer_size)
        dataset = dataset.map(_parse_function)
        
        # Convert to lists
        for sample, label in dataset:
            # Convert TF tensors to numpy, then will convert to PyTorch tensors in __getitem__
            self.samples.append(sample.numpy())
            self.labels.append(int(label.numpy()))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        # TensorFlow uses (H, W, C) format, PyTorch uses (C, H, W)
        sample_tensor = torch.from_numpy(sample).permute(2, 0, 1).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return sample_tensor, label_tensor


def read_tfrecords(file_name, buffer_size=1000):
    """
    Read TFRecord file and return PyTorch Dataset
    
    Args:
        file_name: Path to TFRecord file
        buffer_size: Buffer size for reading
    
    Returns:
        TFRecordDataset: PyTorch Dataset object
    """
    return TFRecordDataset(file_name, buffer_size=buffer_size)


def get_dataset_length(dataset):
    """Get length of a PyTorch dataset"""
    return len(dataset)


def create_dataloader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    """
    Create PyTorch DataLoader from dataset
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (useful for GPU training)
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
