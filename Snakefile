# Snakemake workflow for MobileNetV2 vs ResNet18 experiment
# Comparing Regular Training, QAT, and PTQ methods

import yaml
import os

# Load configuration
configfile: "config.yaml"

# Define all possible combinations
MODELS = config["models"]
TRAINING_METHODS = config["training_methods"]
QUANTIZATION_METHODS = config["quantization_methods"]

# Define all possible outputs
rule all:
    input:
        # Cross-validation results for each model
        expand("cross_validation_results/{model}_best_config/best_config.json", model=MODELS),
        
        # Regular training results
        expand("training_results/{model}_regular/training_results.csv", model=MODELS),
        
        # QAT training results  
        expand("training_results/{model}_qat/qat_training_results.csv", model=MODELS),
        
        # PTQ results for regular models
        expand("ptq_results/{model}_regular_ptq/ptq_results.json", model=MODELS),
        
        # Test results for all combinations
        expand("test_results/{model}_{method}/evaluation_results.json", 
               model=MODELS, method=["regular", "qat"]),
        expand("test_results/{model}_ptq/ptq_evaluation_results.json", model=MODELS)

# Cross-validation for hyperparameter selection
rule cross_validation:
    input:
        config="config.yaml"
    output:
        config="cross_validation_results/{model}_best_config/best_config.json"
    params:
        model="{model}"
    shell:
        "python scripts/cross_validation.py --model {params.model} --config {input.config}"

# Regular training
rule train_regular:
    input:
        config="config.yaml",
        best_config="cross_validation_results/{model}_best_config/best_config.json"
    output:
        results="training_results/{model}_regular/training_results.csv",
        weights="training_results/{model}_regular/{model}_model_weights.index",
        model_config="training_results/{model}_regular/model_config.json"
    params:
        model="{model}"
    shell:
        "python scripts/train_regular.py --model {params.model} --config {input.config} --best_config {input.best_config}"

# QAT training
rule train_qat:
    input:
        config="config.yaml",
        best_config="cross_validation_results/{model}_best_config/best_config.json"
    output:
        results="training_results/{model}_qat/qat_training_results.csv",
        weights="training_results/{model}_qat/{model}_qat_model_weights.index",
        model_config="training_results/{model}_qat/qat_model_config.json"
    params:
        model="{model}"
    shell:
        "python scripts/train_qat.py --model {params.model} --config {input.config} --best_config {input.best_config}"

# Apply PTQ to regular models
rule apply_ptq:
    input:
        model_path="training_results/{model}_regular",
        config="config.yaml"
    output:
        results="ptq_results/{model}_regular_ptq/ptq_results.json",
        quantized_model="ptq_results/{model}_regular_ptq/{model}_quantized.tflite"
    params:
        model="{model}"
    shell:
        "python scripts/apply_ptq.py --model {params.model} --model_path {input.model_path} --config {input.config}"

# Test regular models
rule test_regular:
    input:
        model_path="training_results/{model}_regular",
        config="config.yaml"
    output:
        results="test_results/{model}_regular/evaluation_results.json",
        predictions="test_results/{model}_regular/predictions.csv",
        plots="test_results/{model}_regular/{model}_regular_evaluation.png"
    params:
        model="{model}"
    shell:
        "python scripts/test_model.py --model {params.model} --model_path {input.model_path} --test_type regular --config {input.config}"

# Test QAT models
rule test_qat:
    input:
        model_path="training_results/{model}_qat",
        config="config.yaml"
    output:
        results="test_results/{model}_qat/evaluation_results.json",
        predictions="test_results/{model}_qat/predictions.csv",
        plots="test_results/{model}_qat/{model}_qat_evaluation.png"
    params:
        model="{model}"
    shell:
        "python scripts/test_model.py --model {params.model} --model_path {input.model_path} --test_type qat --config {input.config}"

# Test PTQ models
rule test_ptq:
    input:
        model_path="ptq_results/{model}_regular_ptq",
        config="config.yaml"
    output:
        results="test_results/{model}_ptq/ptq_evaluation_results.json",
        predictions="test_results/{model}_ptq/predictions.csv",
        plots="test_results/{model}_ptq/{model}_ptq_evaluation.png"
    params:
        model="{model}"
    shell:
        "python scripts/test_model.py --model {params.model} --model_path {input.model_path} --test_type ptq --config {input.config}"

# Create summary report
rule create_summary:
    input:
        expand("test_results/{model}_{method}/evaluation_results.json", 
               model=MODELS, method=["regular", "qat"]),
        expand("test_results/{model}_ptq/ptq_evaluation_results.json", model=MODELS)
    output:
        summary="results/summary_report.json",
        comparison="results/model_comparison.csv"
    shell:
        "python scripts/create_summary.py --output results/"

# Clean up intermediate files (optional)
rule clean:
    shell:
        "rm -rf cross_validation_results/*/trainable_*"
