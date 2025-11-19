#!/usr/bin/env python3
"""
Professional Results Saver Module for STEMS-GNN Research

This module handles all result persistence including:
- JSON exports of comprehensive results
- CSV summaries for easy analysis
- Visualization generation
- Model checkpoint saving
- Training history logs
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import torch


class ResultsSaver:
    """
    Professional results management for research experiments.

    Handles saving of metrics, visualizations, model checkpoints, and logs
    in a structured, reproducible manner.
    """

    def __init__(self, results_dir: str = "results", run_id: Optional[str] = None):
        """
        Initialize results saver.

        Args:
            results_dir: Base directory for saving results
            run_id: Optional unique identifier for this run (default: timestamp)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.visualizations_dir = self.results_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)

        print(f"Results will be saved to: {self.results_dir}")
        print(f"Run ID: {self.run_id}")

    def save_comprehensive_results(
        self,
        roberta_results: Dict[str, Any],
        gnn_results: Dict[str, Any],
        ablation_results: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        filename: str = "comprehensive_results.json"
    ) -> None:
        """
        Save comprehensive results to JSON file.

        Args:
            roberta_results: RoBERTa baseline results dictionary
            gnn_results: Semantic Ego-GNN results dictionary
            ablation_results: Optional ablation study results
            dataset_info: Optional dataset characteristics
            filename: Output filename
        """
        print(f"\nSaving comprehensive results to {filename}...")

        results = {
            "metadata": {
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "methodology": "Semantic Ego-Network GNN vs RoBERTa baseline"
            }
        }

        if dataset_info:
            results["dataset_characteristics"] = dataset_info

        model_performance = {}

        if roberta_results and 'memory_efficient_transformer' in roberta_results:
            rb_metrics = roberta_results['memory_efficient_transformer']['metrics']
            model_performance['roberta_baseline'] = {
                'accuracy': rb_metrics.get('accuracy', 0.0),
                'precision': rb_metrics.get('precision', 0.0),
                'recall': rb_metrics.get('recall', 0.0),
                'f1': rb_metrics.get('f1', 0.0),
                'auc': rb_metrics.get('auc', 0.0)
            }

        if gnn_results and 'correct_semantic_gnn' in gnn_results:
            gnn_metrics = gnn_results['correct_semantic_gnn']['metrics']
            model_performance['semantic_ego_gnn'] = {
                'accuracy': gnn_metrics.get('accuracy', 0.0),
                'precision': gnn_metrics.get('precision', 0.0),
                'recall': gnn_metrics.get('recall', 0.0),
                'f1': gnn_metrics.get('f1', 0.0),
                'auc': gnn_metrics.get('auc', 0.0)
            }

        results['model_performance'] = model_performance

        results['model_configurations'] = {
            'roberta_baseline': roberta_results.get('memory_efficient_transformer', {}),
            'semantic_ego_gnn': gnn_results.get('correct_semantic_gnn', {})
        }

        if ablation_results:
            results['ablation_study'] = ablation_results

        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)

        print(f"✓ Comprehensive results saved to: {output_path}")

    def save_performance_comparison_csv(
        self,
        roberta_results: Dict[str, Any],
        gnn_results: Dict[str, Any],
        filename: str = "model_performance_comparison.csv"
    ) -> None:
        """
        Save model performance comparison to CSV.

        Args:
            roberta_results: RoBERTa baseline results
            gnn_results: Semantic Ego-GNN results
            filename: Output filename
        """
        print(f"\nSaving performance comparison to {filename}...")

        data = []

        if roberta_results and 'memory_efficient_transformer' in roberta_results:
            rb_metrics = roberta_results['memory_efficient_transformer']['metrics']
            data.append({
                'model': 'RoBERTa Baseline',
                'accuracy': rb_metrics.get('accuracy', 0.0),
                'precision': rb_metrics.get('precision', 0.0),
                'recall': rb_metrics.get('recall', 0.0),
                'f1': rb_metrics.get('f1', 0.0),
                'auc': rb_metrics.get('auc', 0.0)
            })

        if gnn_results and 'correct_semantic_gnn' in gnn_results:
            gnn_metrics = gnn_results['correct_semantic_gnn']['metrics']
            data.append({
                'model': 'Semantic Ego-GNN',
                'accuracy': gnn_metrics.get('accuracy', 0.0),
                'precision': gnn_metrics.get('precision', 0.0),
                'recall': gnn_metrics.get('recall', 0.0),
                'f1': gnn_metrics.get('f1', 0.0),
                'auc': gnn_metrics.get('auc', 0.0)
            })

        df = pd.DataFrame(data)
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        print(f"✓ Performance comparison saved to: {output_path}")

    def save_ablation_results_csv(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        filename: str = "ablation_study_results.csv"
    ) -> None:
        """
        Save ablation study results to CSV.

        Args:
            ablation_results: Dictionary mapping config names to metrics
            filename: Output filename
        """
        print(f"\nSaving ablation study results to {filename}...")

        data = []
        for config_name, metrics in ablation_results.items():
            data.append({
                'configuration': config_name,
                'accuracy': metrics.get('accuracy', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1': metrics.get('f1', 0.0),
                'auc': metrics.get('auc', 0.0)
            })

        df = pd.DataFrame(data)
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        print(f"✓ Ablation study results saved to: {output_path}")

    def save_dataset_characteristics_csv(
        self,
        dataset_info: Dict[str, Any],
        filename: str = "dataset_characteristics.csv"
    ) -> None:
        """
        Save dataset characteristics to CSV.

        Args:
            dataset_info: Dataset information dictionary
            filename: Output filename
        """
        print(f"\nSaving dataset characteristics to {filename}...")

        df = pd.DataFrame([dataset_info])
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        print(f"✓ Dataset characteristics saved to: {output_path}")

    def save_performance_metrics_csv(
        self,
        performance_data: Dict[str, Dict[str, float]],
        filename: str = "performance_metrics.csv"
    ) -> None:
        """
        Save execution time and memory usage metrics to CSV.

        Args:
            performance_data: Performance metrics (execution_time, memory, etc.)
            filename: Output filename
        """
        print(f"\nSaving performance metrics to {filename}...")

        df = pd.DataFrame(performance_data)
        output_path = self.results_dir / filename
        df.to_csv(output_path)

        print(f"✓ Performance metrics saved to: {output_path}")

    def plot_model_comparison(
        self,
        roberta_results: Dict[str, Any],
        gnn_results: Dict[str, Any],
        filename: str = "performance_comparison.png"
    ) -> None:
        """
        Create and save model performance comparison visualization.

        Args:
            roberta_results: RoBERTa baseline results
            gnn_results: Semantic Ego-GNN results
            filename: Output filename
        """
        print(f"\nGenerating performance comparison visualization...")

        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        roberta_values = []
        gnn_values = []

        if roberta_results and 'memory_efficient_transformer' in roberta_results:
            rb_metrics = roberta_results['memory_efficient_transformer']['metrics']
            roberta_values = [rb_metrics.get(m, 0.0) for m in metrics_list]

        if gnn_results and 'correct_semantic_gnn' in gnn_results:
            gnn_metrics = gnn_results['correct_semantic_gnn']['metrics']
            gnn_values = [gnn_metrics.get(m, 0.0) for m in metrics_list]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(metrics_list))
        width = 0.35

        if roberta_values:
            bars1 = ax.bar(x - width/2, roberta_values, width, label='RoBERTa Baseline', alpha=0.8)
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        if gnn_values:
            bars2 = ax.bar(x + width/2, gnn_values, width, label='Semantic Ego-GNN', alpha=0.8)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics_list])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Performance comparison plot saved to: {output_path}")

    def plot_ablation_study(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        filename: str = "ablation_visualization.png"
    ) -> None:
        """
        Create and save ablation study visualization.

        Args:
            ablation_results: Ablation study results
            filename: Output filename
        """
        print(f"\nGenerating ablation study visualization...")

        configs = list(ablation_results.keys())
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']

        data = []
        for config in configs:
            for metric in metrics_list:
                data.append({
                    'Configuration': config.replace('_', ' ').title(),
                    'Metric': metric.upper(),
                    'Score': ablation_results[config].get(metric, 0.0)
                })

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(data=df, x='Metric', y='Score', hue='Configuration', ax=ax)

        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Ablation Study: Individual Similarity Components', fontsize=14, fontweight='bold')
        ax.legend(title='Configuration', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Ablation study plot saved to: {output_path}")

    def plot_roc_curves(
        self,
        predictions_dict: Dict[str, Tuple[List[int], List[float]]],
        filename: str = "roc_curves.png"
    ) -> None:
        """
        Create and save ROC curves for multiple models.

        Args:
            predictions_dict: Dictionary mapping model names to (y_true, y_prob) tuples
            filename: Output filename
        """
        print(f"\nGenerating ROC curves...")

        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, (y_true, y_prob) in predictions_dict.items():
            if len(y_true) > 0 and len(y_prob) > 0 and len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ ROC curves saved to: {output_path}")

    def plot_performance_metrics(
        self,
        performance_data: Dict[str, Dict[str, float]],
        filename: str = "performance_metrics.png"
    ) -> None:
        """
        Create visualization for execution time and memory usage.

        Args:
            performance_data: Dictionary with execution_time_sec and memory metrics
            filename: Output filename
        """
        print(f"\nGenerating performance metrics visualization...")

        models = list(performance_data.get('execution_time_sec', {}).keys())

        if not models:
            print("No performance data available")
            return

        exec_times = [performance_data['execution_time_sec'][m] for m in models]
        memory_usage = [performance_data.get('peak_memory_mb', {}).get(m, 0) for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        bars1 = ax1.bar(models, exec_times, alpha=0.8, color='skyblue')
        ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(models, memory_usage, alpha=0.8, color='lightcoral')
        ax2.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Memory Usage', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f} MB', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Performance metrics plot saved to: {output_path}")

    def save_model_checkpoint(
        self,
        model: torch.nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save PyTorch model checkpoint with metadata.

        Args:
            model: PyTorch model to save
            model_name: Name for the checkpoint file
            metadata: Optional metadata (hyperparameters, training stats, etc.)
        """
        print(f"\nSaving model checkpoint for {model_name}...")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id
        }

        if metadata:
            checkpoint['metadata'] = metadata

        output_path = self.checkpoints_dir / f"{model_name}_{self.run_id}.pt"
        torch.save(checkpoint, output_path)

        print(f"✓ Model checkpoint saved to: {output_path}")

    def save_training_history(
        self,
        history: Dict[str, List[float]],
        model_name: str,
        filename_prefix: str = "training_history"
    ) -> None:
        """
        Save training history to CSV.

        Args:
            history: Dictionary mapping metric names to lists of values per epoch
            model_name: Model name for filename
            filename_prefix: Prefix for the output filename
        """
        print(f"\nSaving training history for {model_name}...")

        df = pd.DataFrame(history)
        output_path = self.results_dir / f"{filename_prefix}_{model_name}.csv"
        df.to_csv(output_path, index=False)

        print(f"✓ Training history saved to: {output_path}")

    def save_all_results(
        self,
        roberta_results: Dict[str, Any],
        gnn_results: Dict[str, Any],
        ablation_results: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, Tuple]] = None,
        performance_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """
        Convenience method to save all results at once.

        Args:
            roberta_results: RoBERTa baseline results
            gnn_results: Semantic Ego-GNN results
            ablation_results: Optional ablation study results
            dataset_info: Optional dataset characteristics
            predictions: Optional predictions for ROC curves
            performance_metrics: Optional execution time and memory metrics
        """
        print("\n" + "="*60)
        print("SAVING ALL RESULTS")
        print("="*60)

        self.save_comprehensive_results(
            roberta_results, gnn_results, ablation_results, dataset_info
        )

        self.save_performance_comparison_csv(roberta_results, gnn_results)

        if dataset_info:
            self.save_dataset_characteristics_csv(dataset_info)

        if ablation_results:
            self.save_ablation_results_csv(ablation_results)

        if performance_metrics:
            self.save_performance_metrics_csv(performance_metrics)

        self.plot_model_comparison(roberta_results, gnn_results)

        if ablation_results:
            self.plot_ablation_study(ablation_results)

        if predictions:
            self.plot_roc_curves(predictions)

        if performance_metrics:
            self.plot_performance_metrics(performance_metrics)

        print("\n" + "="*60)
        print(f"ALL RESULTS SAVED TO: {self.results_dir}")
        print("="*60)

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for NumPy types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
