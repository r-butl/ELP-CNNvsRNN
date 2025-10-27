#!/usr/bin/env python3
"""
Cross-validation script for hyperparameter selection using Optuna
PyTorch implementation with gradient accumulation and TensorFlow-like memory management
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import optuna
from torch.utils.data import DataLoader, Subset
import numpy as np

# Configure PyTorch to use TensorFlow-like memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords, get_dataset_length, create_dataloader
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet18_model import ResNet18Model

import torch

def check_mem():
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        # Total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        print(f"Total GPU memory: {total_memory:.2f} GB")
    
        # Currently allocated memory
        allocated = torch.cuda.memory_allocated(0) / (1024**3)  # in GB
        print(f"Currently allocated: {allocated:.2f} GB")
    
        # Reserved (cached) memory
        reserved = torch.cuda.memory_reserved(0) / (1024**3)  # in GB
        print(f"Reserved (cached): {reserved:.2f} GB")
    
        # Available memory (approximate)
        available = total_memory - (reserved / 1024**3)
        print(f"Approximately available: {available:.2f} GB")


def get_device():
    """Get available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def k_fold_split(dataset, num_folds, fold_idx):
    """Split dataset into k folds for cross-validation"""
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds
    
    # Get indices for validation fold
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size
    
    # Create indices
    all_indices = list(range(dataset_size))
    val_indices = all_indices[val_start:val_end]
    train_indices = all_indices[:val_start] + all_indices[val_end:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def train_and_evaluate_fold(model, train_loader, val_loader, optimizer, criterion, device, epochs=15, accumulation_steps=8, patience=5):
    """
    Train model on one fold and return validation accuracy
    Uses gradient accumulation to simulate larger batch sizes
    Implements early stopping based on validation accuracy
    
    Args:
        accumulation_steps: Number of batches to accumulate (default 8)
                          effective_batch = actual_batch * accumulation_steps
                          e.g., batch=4 * accum=8 = effective batch of 32
        patience: Number of epochs to wait for improvement before stopping (default 5)
    """
    best_val_accuracy = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        print(f"Epoch: {epoch}")
        optimizer.zero_grad()  # Zero gradients at start
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * accumulation_steps
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs).squeeze(-1)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  ✓ New best accuracy!")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"  Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"  Restored best model (accuracy: {best_val_accuracy:.4f})")
    
    return best_val_accuracy


def objective(trial, model_type, dataset, input_shape, cfg, device):
    """Optuna objective function for hyperparameter optimization"""
    
    # Clear any cached memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
   
    config = {
        'learning_rate': trial.suggest_categorical('learning_rate', cfg['hyperparameters']['learning_rate']),
        'learning_rate_decay_steps': trial.suggest_categorical('learning_rate_decay_steps', cfg['hyperparameters']['learning_rate_decay_steps']),
        'learning_rate_decay': trial.suggest_categorical('learning_rate_decay', cfg['hyperparameters']['learning_rate_decay']),
        'momentum': trial.suggest_categorical('momentum', cfg['hyperparameters']['momentum']),
        'batch_size': trial.suggest_categorical('batch_size', cfg['hyperparameters']['batch_size']),
        'dropout_rate': trial.suggest_categorical('dropout_rate', cfg['hyperparameters']['dropout_rate']),
        'activation_function': trial.suggest_categorical('activation_function', cfg['hyperparameters']['activation_function']),
        'optimizer': trial.suggest_categorical('optimizer', cfg['hyperparameters']['optimizer']),
        'model_type': model_type
    }

    try:
        # K-fold cross-validation
        k_folds = cfg['cross_validation']['k_folds']
        fold_scores = []
        
        for fold_idx in range(k_folds):
            print(f"\n  Trial {trial.number}, Fold {fold_idx + 1}/{k_folds}")
            
            # Split data
            train_dataset, val_dataset = k_fold_split(dataset, k_folds, fold_idx)
            
            # Check class distribution in this fold
            train_labels = [dataset[i][1].item() for i in train_dataset.indices]
            val_labels = [dataset[i][1].item() for i in val_dataset.indices]
            train_pos = sum(train_labels)
            val_pos = sum(val_labels)
            print(f"    Train: {train_pos}/{len(train_labels)} positive ({train_pos/len(train_labels)*100:.1f}%)")
            print(f"    Val: {val_pos}/{len(val_labels)} positive ({val_pos/len(val_labels)*100:.1f}%)")
            
            # Create data loaders
            # Disable pin_memory for CUDA to reduce memory pressure
            use_pin = device.type != 'cuda'
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=use_pin
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=use_pin
            )
            
            # Create model
            if model_type == 'mobilenetv2':
                model = MobileNetV2Model(model_config=config, training=True, input_shape=input_shape)
            elif model_type == 'resnet18':
                model = ResNet18Model(model_config=config, training=True, input_shape=input_shape)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model.to(device)
            
            # Loss function
            criterion = nn.BCELoss()
            
            # Optimizer
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            elif config['optimizer'] == 'sgd':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config['learning_rate'],
                    momentum=config['momentum']
                )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['learning_rate_decay_steps'],
                gamma=config['learning_rate_decay']
            )
            
            # Clear cache before training to avoid fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Train and evaluate with gradient accumulation and early stopping
            # Effective batch = config['batch_size'] * 8
            # e.g., batch=4 * 8 = effective batch of 32 (like TensorFlow!)
            patience = cfg['cross_validation'].get('early_stopping_patience', 5)
            val_accuracy = train_and_evaluate_fold(
                model, train_loader, val_loader, optimizer, criterion, device,
                epochs=cfg['cross_validation']['max_epochs'],
                accumulation_steps=8,
                patience=patience
            )
            
            fold_scores.append(val_accuracy)
            print(f"    Fold {fold_idx + 1} Validation Accuracy: {val_accuracy:.4f}")
            
            # Clean up GPU memory aggressively
            del model, train_loader, val_loader, optimizer, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Return average validation accuracy
        avg_accuracy = np.mean(fold_scores)
        print(f"  Trial {trial.number} Average Accuracy: {avg_accuracy:.4f}")
        
        return avg_accuracy
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ⚠️  Trial {trial.number} failed with OOM error")
        print(f"     Batch size was: {config['batch_size']}")
        print(f"     Suggest reducing batch_size in config.yaml")
        
        # Clean up and retry with smaller batch if possible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Raise to let Optuna handle it
        raise optuna.exceptions.TrialPruned("OOM - suggest smaller batch_size")
    
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "CUBLAS" in error_msg:
            print(f"  ⚠️  Trial {trial.number} failed with CUDA error")
            print(f"     Error: {error_msg[:100]}")
            print(f"     GPU state may be corrupted after OOM errors")
            print(f"     Suggestion: Restart the script or reduce batch_size further")
            
            # Try to reset GPU state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Prune this trial and continue
            raise optuna.exceptions.TrialPruned(f"CUDA error: {error_msg[:50]}")
        else:
            # Other runtime errors - let them fail
            print(f"  ⚠️  Trial {trial.number} failed with error: {e}")
            raise
    
    except Exception as e:
        print(f"  ⚠️  Trial {trial.number} failed with unexpected error: {e}")
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def save_best_config(study, model_type, output_dir, cfg):
    """Helper function to save the best configuration from the study"""
    # Get best trial
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        return None
    
    best_params = study.best_params
    best_value = study.best_value
    
    # Convert to config dict
    best_config = {
        'model_type': model_type,
        'learning_rate': best_params['learning_rate'],
        'learning_rate_decay_steps': best_params['learning_rate_decay_steps'],
        'learning_rate_decay': best_params['learning_rate_decay'],
        'momentum': best_params['momentum'],
        'batch_size': best_params['batch_size'],
        'epochs': cfg['cross_validation']['max_epochs'],
        'dropout_rate': best_params['dropout_rate'],
        'activation_function': best_params['activation_function'],
        'optimizer': best_params['optimizer']
    }
    
    # Save best configuration
    best_config_dir = os.path.join(output_dir, f'{model_type}_best_config')
    os.makedirs(best_config_dir, exist_ok=True)
    
    with open(os.path.join(best_config_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save study results
    results = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'n_completed': len(completed_trials)
    }
    
    with open(os.path.join(output_dir, 'study_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_config


def run_cross_validation(model_type, config_path='config.yaml'):
    """Run cross-validation for a specific model type using Optuna"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Configure memory management
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using TensorFlow-like memory management")
        print(f"Gradient accumulation: enabled (effective batch = actual_batch × 8)")
    
    # Prepare dataset
    dataset_path = os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file'])
    print(f"\nLoading dataset: {dataset_path}")
    
    dataset = read_tfrecords(dataset_path, buffer_size=64000)
    
    # Get input shape from first sample
    sample, label = dataset[0]
    input_shape = tuple(sample.shape[1:]) + (sample.shape[0],)  # Convert (C, H, W) to (H, W, C) for config
    
    dataset_size = len(dataset)
    
    # Check overall class distribution
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    total_pos = sum(all_labels)
    print(f"Overall class distribution: {total_pos}/{len(all_labels)} positive ({total_pos/len(all_labels)*100:.1f}%)")
    
    # Create output directory
    output_dir = os.path.join(cfg['output']['cross_validation_dir'], f'{model_type}_cv_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Optuna study
    study_name = f'{model_type}_tuning'
    storage_path = f"sqlite:///{os.path.join(output_dir, 'optuna_study.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='maximize',
        load_if_exists=True
    )
    
    # Define callback to save best config after each trial
    def save_callback(study, trial):
        """Save best configuration after each successful trial"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            try:
                save_best_config(study, model_type, output_dir, cfg)
                print(f"  💾 Best config saved (current best accuracy: {study.best_value:.4f})")
            except Exception as e:
                # If saving fails (e.g., due to database locks), just log it
                print(f"  ⚠️  Warning: Could not save config immediately: {str(e)[:50]}")
                print(f"     (Config will be saved at the end)")
    
    print("\nStarting hyperparameter search with Optuna...")
    print(f"Number of trials: {cfg['cross_validation']['num_trials']}")
    print(f"K-folds: {cfg['cross_validation']['k_folds']}")
    print(f"Max epochs per fold: {cfg['cross_validation']['max_epochs']}")
    print(f"Early stopping patience: {cfg['cross_validation'].get('early_stopping_patience', 5)} epochs")
    print(f"Note: Best config will be saved after each successful trial")
    
    # Run optimization with exception handling
    try:
        study.optimize(
            lambda trial: objective(trial, model_type, dataset, input_shape, cfg, device),
            n_trials=cfg['cross_validation']['num_trials'],
            show_progress_bar=True,
            callbacks=[save_callback],
            catch=(torch.cuda.OutOfMemoryError, RuntimeError)  # Continue on OOM and CUDA errors
        )
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        print("   Saving best results found so far...")
    
    # Check if we have any successful trials
    if len(study.trials) == 0 or len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
        print("\n❌ No trials completed successfully!")
        print("   Possible issues:")
        print("   1. Batch size too large - reduce in config.yaml")
        print("   2. GPU memory too small - try CPU or smaller model")
        print("   3. Input data too large - check data dimensions")
        raise RuntimeError("Cross-validation failed - no successful trials")
    
    # Save final best configuration (in case callback didn't run on last trial)
    best_config = save_best_config(study, model_type, output_dir, cfg)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results for {model_type}")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"\nBest configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"{'='*60}")
    print(f"Results saved to: {os.path.join(output_dir, f'{model_type}_best_config')}")
    
    return best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validation with Optuna for PyTorch models")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Run cross-validation
    best_config = run_cross_validation(args.model, args.config)
    print(f"\n✅ Cross-validation completed!")
