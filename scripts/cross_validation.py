#!/usr/bin/env python3
"""
Cross-validation script for hyperparameter selection using Keras Tuner
"""

import os
import sys
import yaml
import tensorflow as tf
import keras_tuner as kt
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords, get_tfrecord_length
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model

def k_fold_split(dataset, num_folds, fold_idx, dataset_size):
    """Split dataset into k folds for cross-validation"""
    fold_size = dataset_size // num_folds
    
    # Create validation dataset for the current fold
    val_dataset = dataset.skip(fold_idx * fold_size).take(fold_size)
    
    # Create training dataset by skipping the validation fold
    train_dataset = dataset.take(fold_idx * fold_size).concatenate(
        dataset.skip((fold_idx + 1) * fold_size)
    )
    
    return train_dataset, val_dataset

class ModelHyperModel(kt.HyperModel):
    """Hypermodel for Keras Tuner with cross-validation"""
    
    def __init__(self, model_type, config_path, dataset_path, input_shape, dataset_size, k_folds):
        self.model_type = model_type
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.dataset_size = dataset_size
        self.k_folds = k_folds
        
    def build(self, hp):
        """Build model with hyperparameters"""
        # Define hyperparameters
        config = {
            'model_type': self.model_type,
            'learning_rate': hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001]),
            'learning_rate_decay_steps': hp.Choice('learning_rate_decay_steps', values=[100, 200, 500]),
            'learning_rate_decay': hp.Choice('learning_rate_decay', values=[0.9, 0.95, 0.97]),
            'momentum': hp.Choice('momentum', values=[0.9, 0.95]),
            'batch_size': hp.Choice('batch_size', values=[16, 32, 64]),
            'dropout_rate': hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3]),
            'activation_function': hp.Choice('activation_function', values=['ReLU', 'LeakyReLU']),
            'optimizer': hp.Choice('optimizer', values=['adam', 'sgd'])
        }
        
        # Create model
        if self.model_type == 'mobilenetv2':
            model = MobileNetV2Model(model_config=config, training=True, input_shape=self.input_shape)
        elif self.model_type == 'resnet18':
            model = ResNet18Model(model_config=config, training=True, input_shape=self.input_shape)
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['learning_rate'],
            decay_steps=config['learning_rate_decay_steps'],
            decay_rate=config['learning_rate_decay'],
            staircase=True
        )
        
        # Optimizer
        if config['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config['momentum'])
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        """Custom fit with k-fold cross-validation"""
        batch_size = hp.get('batch_size')
        
        fold_scores = []
        for fold_idx in range(self.k_folds):
            print(f"\nFold {fold_idx + 1}/{self.k_folds}")
            
            # Load and split dataset
            training_dataset = read_tfrecords(self.dataset_path, buffer_size=64000)
            train_dataset, val_dataset = k_fold_split(training_dataset, self.k_folds, fold_idx, self.dataset_size)
            
            # Prepare datasets
            shuffle_buffer = min(1000, self.dataset_size)
            train_batches = train_dataset.shuffle(shuffle_buffer).batch(batch_size).repeat()
            val_batches = val_dataset.batch(batch_size)
            
            # Calculate steps
            train_fold_size = self.dataset_size * (self.k_folds - 1) // self.k_folds
            steps_per_epoch = (train_fold_size + batch_size - 1) // batch_size
            
            # Train
            history = model.fit(
                train_batches,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_batches,
                verbose=1,
                **kwargs
            )
            
            # Store validation accuracy
            fold_scores.append(history.history['val_accuracy'][-1])
            
            # Clean up
            del training_dataset, train_dataset, val_dataset
            tf.keras.backend.clear_session()
        
        # Return average validation accuracy
        return {'val_accuracy': sum(fold_scores) / len(fold_scores)}

def run_cross_validation(model_type, config_path='config.yaml'):
    """Run cross-validation for a specific model type using Keras Tuner"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Prepare dataset info
    dataset_path = os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file'])
    
    # Get input shape and dataset size
    temp_dataset = read_tfrecords(dataset_path, buffer_size=64000)
    for sample, label in temp_dataset.take(1):
        input_shape = tuple(sample.shape.as_list())  # Convert to tuple
    
    temp_dataset = read_tfrecords(dataset_path, buffer_size=64000)
    dataset_size = sum(1 for _ in temp_dataset)
    del temp_dataset
    
    print(f"Dataset size: {dataset_size}, Input shape: {input_shape}")
    
    # Create hypermodel
    hypermodel = ModelHyperModel(
        model_type=model_type,
        config_path=config_path,
        dataset_path=dataset_path,
        input_shape=input_shape,
        dataset_size=dataset_size,
        k_folds=cfg['cross_validation']['k_folds']
    )
    
    # Create output directory
    output_dir = os.path.join(cfg['output']['cross_validation_dir'], f'{model_type}_cv_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create tuner (using Bayesian Optimization)
    tuner = kt.BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=cfg['cross_validation']['num_trials'],
        directory=output_dir,
        project_name=f'{model_type}_tuning',
        overwrite=False
    )
    
    # Print search space summary
    print("\nStarting hyperparameter search...")
    tuner.search_space_summary()
    
    # Run search
    tuner.search(epochs=cfg['hyperparameters']['epochs'][0])  # Use first epoch value
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Convert to config dict
    best_config = {
        'model_type': model_type,
        'learning_rate': best_hps.get('learning_rate'),
        'learning_rate_decay_steps': best_hps.get('learning_rate_decay_steps'),
        'learning_rate_decay': best_hps.get('learning_rate_decay'),
        'momentum': best_hps.get('momentum'),
        'batch_size': best_hps.get('batch_size'),
        'epochs': cfg['hyperparameters']['epochs'][0],
        'dropout_rate': best_hps.get('dropout_rate'),
        'activation_function': best_hps.get('activation_function'),
        'optimizer': best_hps.get('optimizer')
    }
    
    # Save best configuration
    best_config_dir = os.path.join(output_dir, f'{model_type}_best_config')
    os.makedirs(best_config_dir, exist_ok=True)
    
    with open(os.path.join(best_config_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Print results
    print(f"\nBest configuration for {model_type}:")
    print(f"Best validation accuracy: {tuner.get_best_models(num_models=1)[0].evaluate(None)[1]:.4f}")
    print(f"Config: {json.dumps(best_config, indent=2)}")
    
    return best_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation for model selection")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Run cross-validation
    best_config = run_cross_validation(args.model, args.config)
