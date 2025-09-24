#!/usr/bin/env python3
"""
Minimal test to isolate TensorFlow import issues
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

try:
    print("Testing TensorFlow import...")
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    print("Testing basic operations...")
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = a + b
    print(f"✅ Basic operation result: {c.numpy()}")
    
    print("Testing dataset creation...")
    import numpy as np
    data = np.random.randn(10, 5)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    print(f"✅ Dataset created with {len(list(dataset))} samples")
    
    print("All tests passed! 🎉")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
