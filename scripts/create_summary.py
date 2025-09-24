#!/usr/bin/env python3
"""
Create summary report comparing all model combinations
"""

import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path

def create_summary_report(output_dir):
    """Create a comprehensive summary report of all experiments"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all results
    results = []
    
    # Models and methods to check
    models = ["mobilenetv2", "resnet18"]
    methods = ["regular", "qat", "ptq"]
    
    for model in models:
        for method in methods:
            if method == "ptq":
                results_file = f"test_results/{model}_ptq/ptq_evaluation_results.json"
            else:
                results_file = f"test_results/{model}_{method}/evaluation_results.json"
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'model': model,
                        'method': method,
                        'accuracy': data['metrics']['accuracy'],
                        'precision': data['metrics']['precision'],
                        'recall': data['metrics']['recall'],
                        'f1_score': data['metrics']['f1_score'],
                        'auc': data['metrics']['auc'],
                        'num_samples': data['num_samples']
                    })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save comparison CSV
    comparison_file = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(comparison_file, index=False)
    
    # Create summary JSON
    summary = {
        'experiment_overview': {
            'models_tested': models,
            'methods_tested': methods,
            'total_combinations': len(results)
        },
        'best_performers': {
            'highest_accuracy': df.loc[df['accuracy'].idxmax()].to_dict() if not df.empty else None,
            'highest_f1_score': df.loc[df['f1_score'].idxmax()].to_dict() if not df.empty else None,
            'highest_auc': df.loc[df['auc'].idxmax()].to_dict() if not df.empty else None
        },
        'model_comparison': df.to_dict('records'),
        'statistics': {
            'accuracy_stats': df['accuracy'].describe().to_dict() if not df.empty else None,
            'f1_stats': df['f1_score'].describe().to_dict() if not df.empty else None,
            'auc_stats': df['auc'].describe().to_dict() if not df.empty else None
        }
    }
    
    # Save summary JSON
    summary_file = os.path.join(output_dir, 'summary_report.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Models tested: {', '.join(models)}")
    print(f"Methods tested: {', '.join(methods)}")
    print(f"Total combinations: {len(results)}")
    print()
    
    if not df.empty:
        print("BEST PERFORMERS:")
        print("-" * 30)
        
        best_acc = df.loc[df['accuracy'].idxmax()]
        print(f"Highest Accuracy: {best_acc['model']} {best_acc['method']} - {best_acc['accuracy']:.4f}")
        
        best_f1 = df.loc[df['f1_score'].idxmax()]
        print(f"Highest F1 Score: {best_f1['model']} {best_f1['method']} - {best_f1['f1_score']:.4f}")
        
        best_auc = df.loc[df['auc'].idxmax()]
        print(f"Highest AUC: {best_auc['model']} {best_auc['method']} - {best_auc['auc']:.4f}")
        
        print()
        print("DETAILED RESULTS:")
        print("-" * 30)
        print(df.to_string(index=False))
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Comparison CSV saved to: {comparison_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiment summary")
    parser.add_argument("--output", required=True, help="Output directory for summary")
    
    args = parser.parse_args()
    
    create_summary_report(args.output)
