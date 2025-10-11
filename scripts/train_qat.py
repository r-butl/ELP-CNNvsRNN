#!/usr/bin/env python3
"""
Quantization Aware Training (QAT) script for PyTorch models
Supports MobileNetV2 and ResNet18 with QAT
With gradient accumulation and TensorFlow-like memory management
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datetime import datetime
import csv
from torch.utils.tensorboard import SummaryWriter

# Configure PyTorch to use TensorFlow-like memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

try:
    from torch.ao.quantization import prepare_qat, convert
except ImportError:
    from torch.quantization import prepare_qat, convert

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_tfrecords, get_dataset_length, create_dataloader
from models.mobilenetv2_model import QuantizedMobileNetV2Model
from models.resnet18_model import QuantizedResNet18Model


def get_device():
    """Get available device (CPU for QAT - quantization not fully supported on MPS/CUDA)"""
    # QAT works best on CPU for now
    return torch.device("cpu")


def train_qat_model(model_type, config_path, best_config_path=None):
    """Train a model with Quantization Aware Training"""
    
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
    
    print(f"Training {model_type} with QAT using configuration:")
    print(json.dumps(best_config, indent=2))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"training_run_{model_type}_qat_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['train_file']), 
        buffer_size=64000
    )
    val_dataset = read_tfrecords(
        os.path.join(cfg['data']['dataset_folder'], cfg['data']['validate_file']), 
        buffer_size=64000
    )
    
    # Get input shape from first sample
    sample, label = train_dataset[0]
    input_shape = tuple(sample.shape[1:]) + (sample.shape[0],)  # Convert (C, H, W) to (H, W, C) for config
    
    train_size = get_dataset_length(train_dataset)
    val_size = get_dataset_length(val_dataset)
    print(f"\nDataset Info:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Input shape (HWC format): {input_shape}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=best_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False  # CPU only for QAT
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=best_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Get device (CPU for QAT)
    device = get_device()
    print(f"\nUsing device: {device} (QAT requires CPU)")
    
    # Create quantized model
    print(f"\nCreating quantized {model_type} model...")
    if model_type == 'mobilenetv2':
        qat_model = QuantizedMobileNetV2Model(model_config=best_config, training=True, input_shape=input_shape)
    elif model_type == 'resnet18':
        qat_model = QuantizedResNet18Model(model_config=best_config, training=True, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Prepare model for QAT
    qat_model = prepare_qat(qat_model.model)
    qat_model = qat_model.to(device)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizer (typically use lower learning rate for QAT)
    qat_lr = cfg['quantization']['qat_learning_rate']
    if best_config["optimizer"] == "adam":
        optimizer = optim.Adam(qat_model.parameters(), lr=qat_lr)
    elif best_config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            qat_model.parameters(),
            lr=qat_lr,
            momentum=best_config.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {best_config['optimizer']}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=best_config['learning_rate_decay_steps'],
        gamma=best_config['learning_rate_decay']
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(run_folder, 'logs'))
    
    # Training loop
    print(f"\nStarting QAT training...")
    print(f"  Max epochs: {cfg['quantization']['qat_epochs']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Learning rate: {qat_lr} (lower for QAT)")
    print(f"  Early stopping patience: {cfg['training']['patience']}")
    
    # Clear GPU cache before training (QAT is memory intensive!)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Using TensorFlow-like memory management")
        print(f"  Gradient accumulation: 8× (effective batch = {best_config['batch_size']} × 8 = {best_config['batch_size'] * 8})")
        print(f"  ⚠️  QAT requires more memory than regular training")
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    accumulation_steps = 8  # Simulate larger batches
    
    for epoch in range(cfg['quantization']['qat_epochs']):
        try:
            # Training phase
            qat_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            optimizer.zero_grad()  # Zero gradients at start
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = qat_model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Update weights every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Statistics (unscale loss for logging)
                train_loss += loss.item() * inputs.size(0) * accumulation_steps
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * accumulation_steps:.4f}")
                
                # Clear GPU memory more frequently for QAT
                if batch_idx % 25 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate training metrics
            avg_train_loss = train_loss / train_total
            train_accuracy = train_correct / train_total
            
            # Validation phase
            qat_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs = qat_model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            # Calculate validation metrics
            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            
            # Step scheduler
            scheduler.step()
            
            # Log metrics
            print(f"Epoch {epoch+1}/{cfg['quantization']['qat_epochs']} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Check for improvement
            if avg_val_loss < best_val_loss - cfg['training']['min_delta']:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best QAT model
                torch.save(qat_model.state_dict(), os.path.join(run_folder, f'{model_type}_qat_model_best.pth'))
                print(f"  ✓ Saved best QAT model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epoch(s)")
            
            # Early stopping
            if patience_counter >= cfg['training']['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
        except torch.cuda.OutOfMemoryError:
            print(f"\n❌ Out of memory error at epoch {epoch+1}")
            print(f"   QAT requires ~2x more memory than regular training")
            print(f"   Try reducing batch_size in config (current: {best_config['batch_size']})")
            print(f"   Current batch_size: {best_config['batch_size']} → Try: {best_config['batch_size']//2}")
            
            # Save what we have so far
            if training_history:
                csv_file = os.path.join(run_folder, 'qat_training_results_partial.csv')
                with open(csv_file, 'w', newline='') as f:
                    writer_csv = csv.DictWriter(f, fieldnames=training_history[0].keys())
                    writer_csv.writeheader()
                    writer_csv.writerows(training_history)
                print(f"   Partial results saved to: {run_folder}")
            
            raise
    
    # Convert to quantized model (with error handling)
    print("\n" + "="*60)
    print("Converting QAT model to fully quantized model...")
    print("="*60)
    qat_model.eval()
    
    quantization_successful = False
    try:
        quantized_model = convert(qat_model)
        print("✅ Successfully converted to INT8 quantized model!")
        quantization_successful = True
    except RuntimeError as e:
        if "NoQEngine" in str(e) or "quantized::conv2d_prepack" in str(e):
            print("⚠️  Quantization engine not available on this platform")
            print("   → Saving QAT-trained model without final INT8 conversion")
            print("   → Model was trained with quantization awareness (still beneficial!)")
            print("   → Consider using dynamic quantization instead:")
            print("      python scripts/apply_dynamic_quantization.py --model {} --model_path {}".format(
                model_type, run_folder))
            quantized_model = qat_model
        else:
            raise
    
    print("="*60)
    
    # Save final models
    torch.save(qat_model.state_dict(), os.path.join(run_folder, f'{model_type}_qat_model_final_weights.pth'))
    torch.save(quantized_model.state_dict(), os.path.join(run_folder, f'{model_type}_quantized_model_final.pth'))
    
    # Try to save complete model, but only state_dict if it fails
    try:
        torch.save(quantized_model, os.path.join(run_folder, f'{model_type}_quantized_model_complete.pth'))
    except (AttributeError, RuntimeError) as e:
        print(f"⚠️  Could not save complete model (pickle error): {e}")
        print("   Saved state_dict only - use model architecture to load")
        # Save a note about how to load
        load_instructions = {
            "note": "Complete model could not be saved due to quantization pickle issues",
            "how_to_load": "Create model instance and use model.load_state_dict(torch.load('weights.pth'))",
            "weights_file": f"{model_type}_quantized_model_final.pth"
        }
        with open(os.path.join(run_folder, 'load_instructions.json'), 'w') as f:
            json.dump(load_instructions, f, indent=2)
    
    # Save model configuration
    config_file = os.path.join(run_folder, 'qat_model_config.json')
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Save training history to CSV
    csv_file = os.path.join(run_folder, 'qat_training_results.csv')
    with open(csv_file, 'w', newline='') as f:
        if training_history:
            writer_csv = csv.DictWriter(f, fieldnames=training_history[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(training_history)
    
    # Close TensorBoard writer
    writer.close()
    
    # Compare model sizes (only if quantization was successful)
    if quantization_successful:
        print("\nComparing model sizes...")
        from quantization_utils import QuantizationUtils
        
        # Create a temporary FP32 model for comparison
        if model_type == 'mobilenetv2':
            from models.mobilenetv2_model import MobileNetV2Model
            fp32_model = MobileNetV2Model(model_config=best_config, training=False, input_shape=input_shape)
        else:
            from models.resnet18_model import ResNet18Model
            fp32_model = ResNet18Model(model_config=best_config, training=False, input_shape=input_shape)
        
        QuantizationUtils.compare_model_sizes(fp32_model, quantized_model, temp_dir=run_folder)
    else:
        print("\n⚠️  Skipping size comparison (quantization not fully completed)")
    
    # Print summary
    print(f"\nQAT training completed!")
    if training_history:
        print(f"  Final train accuracy: {training_history[-1]['train_accuracy']:.4f}")
        print(f"  Final val accuracy: {training_history[-1]['val_accuracy']:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Results saved to: {run_folder}")
    print(f"  View TensorBoard: tensorboard --logdir {os.path.join(run_folder, 'logs')}")
    
    return run_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Aware Training with PyTorch")
    parser.add_argument("--model", choices=["mobilenetv2", "resnet18"], required=True, help="Model type")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--best_config", default=None, help="Path to best configuration from CV")
    
    args = parser.parse_args()
    
    # Run QAT training
    output_folder = train_qat_model(args.model, args.config, args.best_config)
    print(f"\n✅ QAT training completed. Output folder: {output_folder}")
