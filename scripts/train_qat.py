#!/usr/bin/env python3
"""
Quantization Aware Training (QAT) script
"""

import os
import sys
import yaml
import json
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import argparse
import csv
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords, get_tfrecord_length
from models.mobilenetv2_model import QuantizedMobileNetV2Model
from models.resnet18_model import QuantizedResNet18Model

@tf.function
def train_step(net, optimizer, loss_fn, samples, labels):
    """Single training step for QAT"""
    with tf.GradientTape() as tape:
        predictions = net(samples, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(gradients, net.trainable_weights))
    
    predicted_labels = tf.cast(predictions >= 0.5, tf.int64)
    correct = tf.equal(predicted_labels, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return loss, accuracy

def train_qat_model(model_type, config_path, best_config_path):
    """Train a model with Quantization Aware Training"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load best configuration
    with open(best_config_path, 'r') as f:
        best_config = json.load(f)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"training_run_{model_type}_qat_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Initialize results tracking
    results_dict = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # Load datasets
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    val_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['validate_file']), 
        buffer_size=64000
    )
    
    # Get input shape and dataset size
    for sample, label in train_dataset.take(1):
        shape = [None] + sample.shape
    
    dataset_size = get_tfrecord_length(train_dataset)
    print(f"Training samples: {dataset_size}")
    print(f"Validation samples: {get_tfrecord_length(val_dataset)}")
    
    # Learning rate schedule (typically lower for QAT)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg['quantization']['qat_learning_rate'],
        decay_steps=best_config['learning_rate_decay_steps'],
        decay_rate=best_config['learning_rate_decay'],
        staircase=True
    )
    
    # Loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # Create quantized model
    if model_type == 'mobilenetv2':
        net = QuantizedMobileNetV2Model(
            model_config=best_config, 
            training=True, 
            input_shape=shape[1:], 
            quantize=True
        )
    elif model_type == 'resnet18':
        net = QuantizedResNet18Model(
            model_config=best_config, 
            training=True, 
            input_shape=shape[1:], 
            quantize=True
        )
    
    net.build(shape)
    
    # Apply quantization to the entire model
    qat_model = tfmot.quantization.keras.quantize_model(net)
    
    # Optimizer
    if best_config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif best_config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=best_config["momentum"])
    
    optimizer.build(qat_model.trainable_weights)
    
    # Training loop
    patience_counter = 0
    best_loss = float('inf')
    
    for epoch in range(cfg['quantization']['qat_epochs']):
        # Training
        train_loss = 0.0
        train_accuracy = 0.0
        batches = 0
        
        for step, (samples, labels) in enumerate(train_dataset.batch(best_config['batch_size']).shuffle(buffer_size=dataset_size)):
            loss, acc = train_step(qat_model, optimizer, loss_fn, samples, labels)
            train_loss += loss
            train_accuracy += acc
            batches += 1
        
        avg_train_loss = train_loss / batches
        avg_train_acc = train_accuracy / batches
        
        # Validation
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        for samples, labels in val_dataset.batch(best_config['batch_size']):
            predictions = qat_model(samples, training=False)
            loss = loss_fn(labels, predictions)
            val_loss += loss.numpy()
            
            # Calculate accuracy
            pred_classes = tf.cast(predictions > 0.5, dtype=tf.int32)
            correct_predictions = tf.equal(pred_classes, tf.cast(labels, tf.int32))
            val_accuracy += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_acc = val_accuracy / val_batches
        
        # Store results
        results_dict["epoch"].append(epoch + 1)
        results_dict["train_loss"].append(float(avg_train_loss))
        results_dict["train_acc"].append(float(avg_train_acc))
        results_dict["val_loss"].append(float(avg_val_loss))
        results_dict["val_acc"].append(float(avg_val_acc))
        
        print(f"QAT Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Early stopping
        if best_loss - cfg['training']['min_delta'] > avg_val_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            qat_model.save_weights(os.path.join(run_folder, f'{model_type}_qat_model_weights'))
            print(f"Best QAT model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter > cfg['training']['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Save results
    results_file = os.path.join(run_folder, 'qat_training_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(results_dict.keys())
        for i in range(len(results_dict["epoch"])):
            writer.writerow([results_dict[key][i] for key in results_dict.keys()])
    
    # Save model configuration
    config_file = os.path.join(run_folder, 'qat_model_config.json')
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"QAT training completed. Results saved to {run_folder}")
    return run_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Aware Training")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--best_config", required=True, help="Path to best configuration from CV")
    
    args = parser.parse_args()
    
    # Run QAT training
    output_folder = train_qat_model(args.model, args.config, args.best_config)
    print(f"QAT training completed. Output folder: {output_folder}")
