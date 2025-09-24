# Snakemake Workflow for MobileNetV2 vs ResNet18 Experiment

This workflow compares MobileNetV2 and ResNet18 models using three different training approaches:
1. **Regular Training**: Standard training without quantization
2. **QAT (Quantization Aware Training)**: Training with quantization simulation
3. **PTQ (Post-Training Quantization)**: Applying quantization after regular training

## Experiment Overview

The workflow will test **6 model combinations**:
- MobileNetV2 + Regular Training
- MobileNetV2 + QAT
- MobileNetV2 + PTQ (applied to regular model)
- ResNet18 + Regular Training  
- ResNet18 + QAT
- ResNet18 + PTQ (applied to regular model)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Experiment

Edit `config.yaml` to set your data paths and experiment parameters:

```yaml
# Update these paths to match your setup
data:
  dataset_folder: "data/spectrogram_cherrypicked"  # Update this path
  train_file: "train.tfrecord"
  validate_file: "validate.tfrecord"
  test_file: "toughset_test.tfrecord"
```

### 3. Run the Complete Workflow

```bash
# Run the entire experiment
snakemake --cores 4

# Or run with more cores for parallel execution
snakemake --cores 8
```

### 4. Run Specific Parts

```bash
# Run only cross-validation for both models
snakemake cross_validation --cores 4

# Run only regular training
snakemake train_regular --cores 4

# Run only QAT training  
snakemake train_qat --cores 4

# Run only PTQ application
snakemake apply_ptq --cores 4

# Run only testing
snakemake test_regular test_qat test_ptq --cores 4
```

## Workflow Steps

### 1. Cross-Validation (Hyperparameter Selection)
- **Input**: Configuration file, training data
- **Output**: Best hyperparameters for each model
- **Duration**: ~30-60 minutes per model (depending on hardware)

### 2. Regular Training
- **Input**: Best hyperparameters from CV, training data
- **Output**: Trained regular models
- **Duration**: ~2-4 hours per model

### 3. QAT Training
- **Input**: Best hyperparameters from CV, training data
- **Output**: Quantization-aware trained models
- **Duration**: ~1-2 hours per model

### 4. PTQ Application
- **Input**: Trained regular models, calibration data
- **Output**: Quantized TFLite models
- **Duration**: ~10-30 minutes per model

### 5. Model Testing
- **Input**: All trained models, test data
- **Output**: Performance metrics, visualizations
- **Duration**: ~30 minutes per model

### 6. Summary Report
- **Input**: All test results
- **Output**: Comprehensive comparison report
- **Duration**: ~5 minutes

## Output Structure

```
results/
├── cross_validation_results/
│   ├── mobilenetv2_best_config/
│   └── resnet18_best_config/
├── training_results/
│   ├── mobilenetv2_regular/
│   ├── mobilenetv2_qat/
│   ├── resnet18_regular/
│   └── resnet18_qat/
├── ptq_results/
│   ├── mobilenetv2_regular_ptq/
│   └── resnet18_regular_ptq/
├── test_results/
│   ├── mobilenetv2_regular/
│   ├── mobilenetv2_qat/
│   ├── mobilenetv2_ptq/
│   ├── resnet18_regular/
│   ├── resnet18_qat/
│   └── resnet18_ptq/
└── summary_report.json
```

## Key Features

### Cross-Validation
- Uses Ray Tune with Optuna for hyperparameter optimization
- 5-fold cross-validation for robust parameter selection
- Configurable search space and number of trials

### Quantization Support
- **QAT**: Full quantization-aware training pipeline
- **PTQ**: Post-training quantization with calibration dataset
- Model size comparison and compression metrics

### Comprehensive Evaluation
- Multiple metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion matrices and ROC curves
- Model size and performance comparisons

### Reproducible Workflow
- Snakemake ensures dependency management
- All intermediate results are preserved
- Easy to resume from any point

## Monitoring Progress

```bash
# Check workflow status
snakemake --list

# View detailed execution
snakemake --cores 4 --printshellcmds

# Resume from failure
snakemake --cores 4 --rerun-incomplete
```

## Troubleshooting

### GPU Memory Issues
- Reduce batch size in `config.yaml`
- Use `tf.config.experimental.set_memory_growth(gpu, True)`

### Out of Memory
- Reduce `num_trials` in cross-validation
- Use smaller representative dataset for PTQ

### Ray Tune Issues
- Check Ray cluster status: `ray status`
- Restart Ray: `ray stop && ray start --head`

## Customization

### Adding New Models
1. Create model class in `models/`
2. Add to `MODELS` list in `Snakefile`
3. Update cross-validation and training scripts

### Modifying Hyperparameter Search
Edit the search space in `config.yaml`:
```yaml
hyperparameters:
  learning_rate: [0.01, 0.001, 0.0001]  # Add more values
  batch_size: [8, 16, 32, 64]           # Add more values
```

### Changing Quantization Parameters
```yaml
quantization:
  qat_epochs: 20              # Increase QAT training epochs
  ptq_calibration_samples: 500  # Increase calibration samples
```

## Expected Results

The workflow will generate:
- **Performance comparison** across all 6 model combinations
- **Quantization impact analysis** (accuracy vs. model size)
- **Best model recommendations** for different use cases
- **Detailed metrics** for each model combination

## Next Steps

After running the workflow:
1. Review `results/summary_report.json` for overall comparison
2. Check individual model results in `test_results/`
3. Analyze quantization trade-offs in `ptq_results/`
4. Use best performing model for your specific use case
