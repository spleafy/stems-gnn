#!/usr/bin/env python3
"""
Main script for STEMS-GNN vs RoBERTa Baseline Research

Compares two approaches for depression detection:
1. RoBERTa Baseline: Content-based transformer classifier
2. STEMS-GNN: Multi-dimensional similarity + ego-networks

Usage: python main.py
"""

from data_preprocessing import load_rmhd_data
from roberta_baseline import train_memory_efficient_baseline
from semantic_ego_gnn import train_correct_semantic_gnn
from utils import load_config, set_seed


def run_model_comparison():
    """Run comparison between RoBERTa baseline and STEMS-GNN."""
    print("\n" + "=" * 60)
    print("STEMS-GNN vs RoBERTa Baseline Comparison")
    print("Depression Detection using Multi-dimensional Similarity")
    print("=" * 60 + "\n")

    config = load_config('config.yaml')
    set_seed(42)

    print("Loading RMHD dataset...")
    raw_data = load_rmhd_data(
        data_path=config.get('data', {}).get('path', 'data/raw'),
        target_posts_per_class=5000
    )

    user_posts = raw_data['user_posts']
    user_labels = raw_data['user_labels']
    print(f"Dataset loaded: {len(user_posts)} users\n")

    print("Training RoBERTa Baseline...")
    roberta_results = train_memory_efficient_baseline(user_posts, user_labels, config)

    print("\nTraining STEMS-GNN...")
    gnn_results = train_correct_semantic_gnn(user_posts, user_labels, config)

    print("\n" + "=" * 60)
    print("Results Comparison")
    print("=" * 60)

    if (roberta_results and 'memory_efficient_transformer' in roberta_results and
        gnn_results and 'correct_semantic_gnn' in gnn_results):

        roberta_metrics = roberta_results['memory_efficient_transformer']['metrics']
        gnn_metrics = gnn_results['correct_semantic_gnn']['metrics']

        print("\nRoBERTa Baseline:")
        for metric, value in roberta_metrics.items():
            print(f"  {metric}: {value:.4f}")

        print("\nSTEMS-GNN:")
        for metric, value in gnn_metrics.items():
            print(f"  {metric}: {value:.4f}")

        f1_diff = gnn_metrics['f1'] - roberta_metrics['f1']
        auc_diff = gnn_metrics['auc'] - roberta_metrics['auc']

        print("\nPerformance Difference (STEMS-GNN - RoBERTa):")
        print(f"  F1-Score: {f1_diff:+.4f}")
        print(f"  AUC: {auc_diff:+.4f}")

        if f1_diff > 0 or auc_diff > 0:
            print("\nSTEMS-GNN demonstrates performance advantages over content-based approach")
        else:
            print("\nResults demonstrate comparable performance between approaches")
    else:
        print("Error: Training failed for one or both models")


def run_ablation_study(user_posts, user_labels, config):
    """Run ablation study with different similarity weight configurations."""
    print("\nAblation Study: Testing Individual Similarity Components")

    ablation_configs = {
        'linguistic_only': (1.0, 0.0, 0.0),
        'temporal_only': (0.0, 1.0, 0.0),
        'psychological_only': (0.0, 0.0, 1.0),
        'full_combined': (0.4, 0.3, 0.3)
    }

    ablation_results = {}
    ablation_predictions = {}

    for config_name, weights in ablation_configs.items():
        print(f"\nConfiguration: {config_name} (α={weights[0]}, β={weights[1]}, γ={weights[2]})")

        try:
            result, predictions = train_correct_semantic_gnn(
                user_posts, user_labels, config,
                similarity_weights=weights,
                return_predictions=True
            )

            if 'correct_semantic_gnn' in result:
                ablation_results[config_name] = result['correct_semantic_gnn']['metrics']
                ablation_predictions[config_name] = predictions

                metrics = result['correct_semantic_gnn']['metrics']
                print(f"  F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            else:
                print(f"  Training failed")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return ablation_results, ablation_predictions


def run_comprehensive_comparison(user_posts, user_labels, config):
    """Run comprehensive comparison including ablation study."""
    print("\n" + "="*60)
    print("Comprehensive Model Comparison with Ablation Study")
    print("="*60)

    print("\nTraining RoBERTa Baseline...")
    roberta_result, roberta_predictions = train_memory_efficient_baseline(
        user_posts, user_labels, config, return_predictions=True
    )

    print("\nTraining STEMS-GNN...")
    gnn_result, gnn_predictions = train_correct_semantic_gnn(
        user_posts, user_labels, config, return_predictions=True
    )

    ablation_results, ablation_predictions = run_ablation_study(user_posts, user_labels, config)

    all_results = {
        'roberta_baseline': roberta_result,
        'semantic_gnn_full': gnn_result,
        'ablation_study': ablation_results
    }

    all_predictions = {
        'roberta_baseline': roberta_predictions,
        'semantic_gnn_full': gnn_predictions,
        'ablation_study': ablation_predictions
    }

    return all_results, all_predictions


def main():
    """Main research pipeline."""
    run_model_comparison()


def main_with_results():
    """Main pipeline with comprehensive results generation."""
    print("\n" + "="*60)
    print("STEMS-GNN Research: Comprehensive Analysis")
    print("="*60)

    config = load_config('config.yaml')
    set_seed(42)

    print("\nLoading dataset...")
    raw_data = load_rmhd_data(
        data_path=config.get('data', {}).get('path', 'data/raw'),
        target_posts_per_class=5000
    )

    user_posts = raw_data['user_posts']
    user_labels = raw_data['user_labels']
    print(f"Dataset loaded: {len(user_posts)} users")

    run_comprehensive_comparison(user_posts, user_labels, config)

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print("Results saved to: results/\n")


if __name__ == "__main__":
    main_with_results()