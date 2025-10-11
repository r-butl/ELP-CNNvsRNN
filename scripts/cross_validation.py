#!/usr/bin/env python3
"""
Cross-validation script for hyperparameter selection
"""

import os
import sys
import yaml
import tensorflow as tf
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
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

@tf.function
def train_step(net, optimizer, loss_fn, samples, labels):
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = net(samples, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(gradients, net.trainable_weights))
    
    return loss

def trainable_cv(config):
    """Cross-validation training function"""
    with tf.device('/GPU:0'):
        # Load configuration
        with open('config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Load dataset - reload for each fold to avoid consumption issues
        dataset_path = os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file'])
        
        # Get input shape
        temp_dataset = read_tfrecords(dataset_path, buffer_size=64000)
        for sample, label in temp_dataset.take(1):
            shape = [None] + sample.shape
        
        # Get dataset size
        temp_dataset = read_tfrecords(dataset_path, buffer_size=64000)
        dataset_size = sum(1 for _ in temp_dataset)
        
        del temp_dataset  # Free memory
        
        # Define learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['learning_rate'],
            decay_steps=config['learning_rate_decay_steps'],
            decay_rate=config['learning_rate_decay'],
            staircase=True
        )
        
        # Loss function
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        # Store results
        fold_loss_results = []
        fold_accuracy_results = []
        
        # Iterate through folds
        for fold_idx in range(cfg['cross_validation']['k_folds']):
            tf.print(f"Starting fold {fold_idx + 1}")
            
            # Create fresh model for each fold
            if config['model_type'] == 'mobilenetv2':
                net = MobileNetV2Model(model_config=config, training=True, input_shape=shape[1:])
            elif config['model_type'] == 'resnet18':
                net = ResNet18Model(model_config=config, training=True, input_shape=shape[1:])
            
            net.build(shape)
            
            # Optimizer
            if config["optimizer"] == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            elif config["optimizer"] == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config["momentum"])
            
            optimizer.build(net.trainable_weights)
            
            # Reload dataset for this fold to avoid consumption issues
            training_dataset = read_tfrecords(dataset_path, buffer_size=64000)
            
            # Split data for this fold
            train_dataset, val_dataset = k_fold_split(training_dataset, cfg['cross_validation']['k_folds'], fold_idx, dataset_size)
            
            # Training loop (use smaller shuffle buffer to reduce memory usage)
            shuffle_buffer = min(1000, dataset_size)
            for epoch in range(config['epochs']):
                tf.print(f"  Epoch {epoch + 1}/{config['epochs']}")
                epoch_loss = 0.0
                num_batches = 0
                for step, (samples, labels) in enumerate(train_dataset.batch(config['batch_size']).shuffle(buffer_size=shuffle_buffer)):
                    loss = train_step(net, optimizer, loss_fn, samples, labels)
                    epoch_loss += loss
                    num_batches += 1
                    # Print progress every 10 batches
                    if (step + 1) % 10 == 0:
                        tf.print(f"    Batch {step + 1}, Loss: {loss:.4f}")
                
                avg_epoch_loss = epoch_loss / num_batches
                tf.print(f"  Epoch {epoch + 1} complete - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Validation
            tf.print(f"  Running validation...")
            total_loss = 0.0
            total_accuracy = 0.0
            batches = 0
            
            for samples, labels in val_dataset.batch(config['batch_size']):
                predictions = net(samples, training=False)
                loss = loss_fn(labels, predictions)
                
                # Calculate accuracy
                pred_classes = tf.cast(predictions > 0.5, dtype=tf.int32)
                correct_predictions = tf.equal(pred_classes, tf.cast(labels, tf.int32))
                
                total_accuracy += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                total_loss += loss.numpy()
                batches += 1
            
            # Compute validation metrics
            validation_loss = total_loss / batches
            validation_accuracy = total_accuracy / batches
            
            tf.print(f"Fold {fold_idx + 1} - Validation loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}")
            
            fold_loss_results.append(validation_loss)
            fold_accuracy_results.append(validation_accuracy)
            
            # Clean up memory after each fold
            del net, optimizer, train_dataset, val_dataset, training_dataset
            tf.keras.backend.clear_session()
        
        # Average results across folds
        avg_loss = sum(fold_loss_results) / len(fold_loss_results)
        avg_acc = sum(fold_accuracy_results) / len(fold_accuracy_results)
        
        tf.print(f"Model: {config['model_type']} - Avg. Val. Loss: {avg_loss:.4f}, Avg. Val. Acc: {avg_acc:.4f}")
        
        # Report results to Ray Tune (using dictionary for compatibility)
        tune.report({"avg_loss": float(avg_loss), "avg_acc": float(avg_acc)})

def run_cross_validation(model_type, config_path='config.yaml'):
    """Run cross-validation for a specific model type"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        "learning_rate": tune.choice(cfg['hyperparameters']['learning_rate']),
        "learning_rate_decay_steps": tune.choice(cfg['hyperparameters']['learning_rate_decay_steps']),
        "learning_rate_decay": tune.choice(cfg['hyperparameters']['learning_rate_decay']),
        "momentum": tune.choice(cfg['hyperparameters']['momentum']),
        "batch_size": tune.choice(cfg['hyperparameters']['batch_size']),
        "epochs": tune.choice(cfg['hyperparameters']['epochs']),
        "activation_function": tune.choice(cfg['hyperparameters']['activation_function']),
        "dropout_rate": tune.choice(cfg['hyperparameters']['dropout_rate']),
        "optimizer": tune.choice(cfg['hyperparameters']['optimizer']),
        "model_type": model_type
    }
    
    # Resources
    resources = {"cpu": 1, "gpu": 1}
    
    # Search algorithm
    search_alg = OptunaSearch()
    
    # Create tuner
    tuner = tune.Tuner(
        tune.with_resources(trainable_cv, resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=cfg['cross_validation']['num_trials'],
            max_concurrent_trials=cfg['cross_validation']['max_concurrent_trials'],
            search_alg=search_alg
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.join(cfg['output']['cross_validation_dir'], f'{model_type}_cv_results')
        )
    )
    
    # Run optimization
    results = tuner.fit()
    
    # Get best result with error handling
    try:
        best_result = results.get_best_result(metric="avg_acc", mode="max")
        
        # Check if metrics are available
        if not best_result.metrics or 'avg_acc' not in best_result.metrics:
            print(f"ERROR: No valid trials completed for {model_type}. All trials failed.")
            print("Check for memory issues (OOM) or dataset problems.")
            print(f"Best result metrics: {best_result.metrics}")
            return None
        
        best_config = best_result.config
        
        # Save best configuration
        output_dir = os.path.join(cfg['output']['cross_validation_dir'], f'{model_type}_best_config')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"Best configuration for {model_type}:")
        print(f"Accuracy: {best_result.metrics['avg_acc']:.4f}")
        print(f"Loss: {best_result.metrics['avg_loss']:.4f}")
        print(f"Config: {best_config}")
        
        return best_config
    except Exception as e:
        print(f"ERROR: Failed to get best result for {model_type}: {str(e)}")
        print("This likely means all trials failed. Check logs for OOM or other errors.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation for model selection")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Run cross-validation
    best_config = run_cross_validation(args.model, args.config)
