#!/usr/bin/env python3
"""
Regular training script for models (Simplified)
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

from utils import read_tfrecords, get_tfrecord_length
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model

def train_model(model_type, config_path, best_config_path=None):
    """Train a model with the best configuration from cross-validation"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set default path if not provided
    if best_config_path is None:
        best_config_path = os.path.join(
            cfg['output']['cross_validation_dir'], 
            f'{model_type}_cv_results',
            f'{model_type}_best_config',
            'best_config.json'
        )
    
    # Load best configuration with helpful error message
    if not os.path.exists(best_config_path):
        print(f"ERROR: Best config file not found at: {best_config_path}")
        print(f"Please run cross-validation first using:")
        print(f"  python scripts/cross_validation.py --model {model_type}")
        raise FileNotFoundError(f"Best config not found: {best_config_path}")
    
    with open(best_config_path, 'r') as f:
        best_config = json.load(f)
    
    print(f"Training {model_type} with configuration:")
    print(json.dumps(best_config, indent=2))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"training_run_{model_type}_regular_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Load datasets
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    val_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['validate_file']), 
        buffer_size=64000
    )
    
    # Get input shape and dataset sizes
    for sample, label in train_dataset.take(1):
        input_shape = tuple(sample.shape.as_list())  # Convert to tuple
    
    train_size = get_tfrecord_length(train_dataset)
    val_size = get_tfrecord_length(val_dataset)
    print(f"\nDataset Info:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Input shape: {input_shape}")
    
    # Prepare datasets
    shuffle_buffer = min(10000, train_size)
    train_batches = train_dataset.shuffle(shuffle_buffer).batch(best_config['batch_size']).repeat()
    val_batches = val_dataset.batch(best_config['batch_size'])
    
    # Calculate steps per epoch
    steps_per_epoch = (train_size + best_config['batch_size'] - 1) // best_config['batch_size']
    
    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == 'mobilenetv2':
        model = MobileNetV2Model(model_config=best_config, training=True, input_shape=input_shape)
    elif model_type == 'resnet18':
        model = ResNet18Model(model_config=best_config, training=True, input_shape=input_shape)
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=best_config['learning_rate'],
        decay_steps=best_config['learning_rate_decay_steps'],
        decay_rate=best_config['learning_rate_decay'],
        staircase=True
    )
    
    # Optimizer
    if best_config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif best_config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=best_config["momentum"])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cfg['training']['patience'],
            min_delta=cfg['training']['min_delta'],
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_folder, f'{model_type}_model_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        # CSV Logger
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(run_folder, 'training_results.csv'),
            separator=',',
            append=False
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_folder, 'logs'),
            histogram_freq=0,
            write_graph=True
        )
    ]
    
    # Train model
    print(f"\nStarting training...")
    print(f"  Max epochs: {cfg['training']['max_epochs']}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Early stopping patience: {cfg['training']['patience']}")
    
    history = model.fit(
        train_batches,
        epochs=cfg['training']['max_epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_batches,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model weights
    model.save_weights(os.path.join(run_folder, f'{model_type}_model_final_weights'))
    
    # Save model configuration
    config_file = os.path.join(run_folder, 'model_config.json')
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Print summary
    print(f"\nTraining completed!")
    print(f"  Final train accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Best val loss: {min(history.history['val_loss']):.4f}")
    print(f"  Results saved to: {run_folder}")
    print(f"  View TensorBoard: tensorboard --logdir {os.path.join(run_folder, 'logs')}")
    
    return run_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regular training")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--best_config", default=None, help="Path to best configuration from CV (defaults to cross_validation_results/{model}_best_config/best_config.json)")
    
    args = parser.parse_args()
    
    # Run training
    output_folder = train_model(args.model, args.config, args.best_config)
    print(f"Training completed. Output folder: {output_folder}")
