# Quick Start Guide - PyTorch Version

Get up and running with the PyTorch implementation in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- macOS (with Apple Silicon for MPS acceleration) or Linux/Windows

## Installation

### 1. Clone and setup environment

```bash
# Navigate to project
cd ELP-CNNvsRNN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify installation

```bash
# Quick test
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

Expected output on Apple Silicon:
```
PyTorch: 2.x.x
MPS Available: True
```

## Running the Pipeline

### Option 1: Quick Test with Dummy Data

```bash
# Create dummy dataset (if not already present)
python create_dummy_dataset.py

# Run cross-validation
python scripts/cross_validation.py --model mobilenetv2 --config config_test.yaml

# Train model
python scripts/train_regular.py --model mobilenetv2 --config config_test.yaml

# View training progress
tensorboard --logdir training_run_mobilenetv2_regular_*/logs
```

### Option 2: Full Pipeline

```bash
# 1. Cross-validation (find best hyperparameters)
python scripts/cross_validation.py --model mobilenetv2

# 2. Train regular model
python scripts/train_regular.py --model mobilenetv2 --best_config <path> --config config.yaml

# 3. Train with Quantization-Aware Training
python scripts/train_qat.py --model mobilenetv2

# 4. Apply Post-Training Quantization
python scripts/apply_ptq.py \
    --model mobilenetv2 \
    --model_path training_run_mobilenetv2_regular_YYYYMMDD_HHMMSS

# 5. Evaluate models
python scripts/test_model.py \
    --model mobilenetv2 \
    --model_path training_run_mobilenetv2_regular_YYYYMMDD_HHMMSS \
    --test_type regular
```

## Understanding the Output

### Training Output
```
training_run_mobilenetv2_regular_YYYYMMDD_HHMMSS/
├── logs/                               # TensorBoard logs
├── mobilenetv2_model_best.pth          # Best model weights
├── mobilenetv2_model_final_weights.pth # Final epoch weights
├── mobilenetv2_model_complete.pth      # Complete model (arch + weights)
├── model_config.json                   # Model configuration
└── training_results.csv                # Training history
```

### Quantized Output
```
training_run_mobilenetv2_qat_YYYYMMDD_HHMMSS/
├── mobilenetv2_qat_model_best.pth           # Best QAT weights
├── mobilenetv2_quantized_model_final.pth    # Quantized weights
├── mobilenetv2_quantized_model_complete.pth # Complete quantized model
└── qat_training_results.csv                 # Training history
```

### Test Results
```
test_results_mobilenetv2_regular_YYYYMMDD_HHMMSS/
├── evaluation_results.json                  # Metrics (accuracy, F1, etc.)
├── predictions.csv                          # All predictions
└── mobilenetv2_regular_evaluation.png       # Confusion matrix, ROC, PR curves
```

## Common Commands

### View TensorBoard
```bash
tensorboard --logdir training_run_mobilenetv2_*/logs
# Open browser to http://localhost:6006
```

### View Optuna Dashboard (Cross-Validation Results)
```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///cross_validation_results/mobilenetv2_cv_results/optuna_study.db
# Open browser to http://localhost:8080
```

### Check Model Size
```bash
ls -lh training_run_mobilenetv2_*/mobilenetv2_model_best.pth
ls -lh training_run_mobilenetv2_qat_*/mobilenetv2_quantized_model_final.pth
```

### Resume Training
PyTorch models can be loaded and continue training:
```python
import torch
model = torch.load('path/to/model_complete.pth')
# Continue training...
```

## Troubleshooting

### Issue: "MPS backend out of memory"
**Solution**: Reduce batch size in config.yaml or use CPU:
```python
device = torch.device("cpu")
```

### Issue: "TFRecord not found"
**Solution**: Create dummy dataset or check path:
```bash
python create_dummy_dataset.py
# Or check config.yaml data paths
```

### Issue: "Best config not found"
**Solution**: Run cross-validation first:
```bash
python scripts/cross_validation.py --model mobilenetv2
```

### Issue: Quantization fails
**Solution**: Quantization only works on CPU:
```bash
# This is expected - QAT/PTQ automatically use CPU
```

## Performance Tips

### 1. Use MPS on Apple Silicon
```python
# Automatic - no action needed
# Check in logs: "Using device: mps"
```

### 2. Adjust Batch Size
```yaml
# config.yaml
hyperparameters:
  batch_size: [32]  # Increase for faster training (if memory allows)
```

### 3. Enable Pin Memory
For GPU training, pin memory is automatically enabled in data loaders.

### 4. Use Mixed Precision (Future)
```python
# Not yet implemented, but coming soon
from torch.cuda.amp import autocast, GradScaler
```

## Next Steps

1. **Read the full documentation**: [PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md)
2. **Understand the models**: Check `models/mobilenetv2_model.py` and `models/resnet18_model.py`
3. **Customize hyperparameters**: Edit `config.yaml`
4. **Try both models**: Compare MobileNetV2 vs ResNet18
5. **Experiment with quantization**: Compare QAT vs PTQ

## Quick Reference

| Task | Command |
|------|---------|
| Cross-validation | `python scripts/cross_validation.py --model MODEL` |
| Train regular | `python scripts/train_regular.py --model MODEL` |
| Train QAT | `python scripts/train_qat.py --model MODEL` |
| Apply PTQ | `python scripts/apply_ptq.py --model MODEL --model_path PATH` |
| Test model | `python scripts/test_model.py --model MODEL --model_path PATH --test_type TYPE` |
| TensorBoard | `tensorboard --logdir PATH/logs` |

Where:
- `MODEL` = `mobilenetv2` or `resnet18`
- `TYPE` = `regular`, `qat`, or `ptq`
- `PATH` = training output directory

## Support

- 📚 Documentation: See [PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md)
- 🐛 Issues: Check [CONVERSION_SUMMARY.md](CONVERSION_SUMMARY.md)
- 💬 Questions: Open an issue on GitHub

---

Happy training! 🚀

