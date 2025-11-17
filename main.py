#!/usr/bin/env python3
"""
Main Execution Pipeline for STEMS-GNN Research.

Implements comprehensive comparison of depression detection approaches:
    - RoBERTa Baseline: Content-based transformer classifier
    - STEMS-GNN: Multi-dimensional semantic similarity with ego-networks
    - Ablation Studies: Individual component validation

Usage:
    python main.py
"""

import time
import psutil
import os
from data_preprocessing import load_rmhd_data
from roberta_baseline import train_memory_efficient_baseline
from semantic_ego_gnn import train_correct_semantic_gnn
from utils import load_config, set_seed
from results_saver import ResultsSaver
from feature_extractor import UnifiedFeatureExtractor


def run_model_comparison():
    """
    Execute comparative evaluation of RoBERTa baseline and STEMS-GNN.

    Trains both models on RMHD dataset and reports performance metrics.
    """
    print("\n" + "=" * 60)
    print("STEMS-GNN vs RoBERTa Baseline Comparison")
    print("Depression Detection using Multi-dimensional Similarity")
    print("=" * 60 + "\n")

    config = load_config('config.yaml')
    set_seed(42)

    print("Loading RMHD dataset...")
    target_posts = config.get('data', {}).get('target_posts_per_class', 5000)
    print(f"Configuration: {target_posts} posts per class (total: {target_posts * 2} posts)")

    raw_data = load_rmhd_data(
        data_path=config.get('data', {}).get('path', 'data/raw'),
        target_posts_per_class=target_posts,
        config=config
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


def run_ablation_study(user_posts, user_labels, config, cached_features=None):
    """
    Execute ablation study to validate individual similarity components.

    Tests linguistic-only, temporal-only, psychological-only, and full combined
    similarity configurations to assess individual component contributions.

    Args:
        user_posts: Dictionary mapping user IDs to post lists
        user_labels: Dictionary mapping user IDs to binary labels
        config: Configuration dictionary from config.yaml
        cached_features: Pre-extracted features to avoid redundant computation

    Returns:
        Tuple of (ablation_results, ablation_predictions) dictionaries
    """
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
                return_predictions=True,
                cached_features=cached_features
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


def run_comprehensive_comparison(user_posts, user_labels, config, results_saver=None, save_models=False):
    """
    Execute comprehensive comparison with baseline, proposed method, and ablation studies.

    Uses centralized feature extraction with caching to avoid redundant computations.

    Args:
        user_posts: Dictionary mapping user IDs to post lists
        user_labels: Dictionary mapping user IDs to binary labels
        config: Configuration dictionary from config.yaml
        results_saver: Optional ResultsSaver instance for checkpoint persistence
        save_models: If True, saves model checkpoints

    Returns:
        Tuple of (all_results, all_predictions, performance_metrics)
    """
    print("\n" + "="*60)
    print("Comprehensive Model Comparison with Ablation Study")
    print("="*60)

    performance_metrics = {
        'execution_time_sec': {},
        'memory_delta_mb': {},
        'peak_memory_mb': {}
    }

    process = psutil.Process(os.getpid())

    # Extract features ONCE for all models
    print("\n" + "="*60)
    print("Extracting Features (shared across all models)")
    print("="*60)

    feature_extractor = UnifiedFeatureExtractor(config)
    start_feature_extraction = time.time()

    cached_features = feature_extractor.extract_all_features(
        user_posts, user_labels, max_users=config.get('semantic_ego_gnn', {}).get('max_users', 150)
    )

    end_feature_extraction = time.time()
    print(f"\nFeature extraction complete in {end_feature_extraction - start_feature_extraction:.1f} seconds")
    print(f"Features will be reused for baseline, STEMS-GNN, and all ablation studies\n")

    # Training models using cached features
    print("\nTraining RoBERTa Baseline...")
    mem_before_rb = process.memory_info().rss / 1024 / 1024
    start_time_rb = time.time()

    roberta_result, roberta_predictions = train_memory_efficient_baseline(
        user_posts, user_labels, config, return_predictions=True,
        save_model=save_models, results_saver=results_saver,
        cached_features=cached_features
    )

    end_time_rb = time.time()
    mem_after_rb = process.memory_info().rss / 1024 / 1024

    performance_metrics['execution_time_sec']['RoBERTa-memory_efficient_transformer'] = end_time_rb - start_time_rb
    performance_metrics['memory_delta_mb']['RoBERTa-memory_efficient_transformer'] = mem_after_rb - mem_before_rb
    performance_metrics['peak_memory_mb']['RoBERTa-memory_efficient_transformer'] = mem_after_rb

    print("\nTraining STEMS-GNN...")
    mem_before_gnn = process.memory_info().rss / 1024 / 1024
    start_time_gnn = time.time()

    gnn_result, gnn_predictions = train_correct_semantic_gnn(
        user_posts, user_labels, config, return_predictions=True,
        save_model=save_models, results_saver=results_saver,
        cached_features=cached_features
    )

    end_time_gnn = time.time()
    mem_after_gnn = process.memory_info().rss / 1024 / 1024

    performance_metrics['execution_time_sec']['Semantic Ego-GNN'] = end_time_gnn - start_time_gnn
    performance_metrics['memory_delta_mb']['Semantic Ego-GNN'] = mem_after_gnn - mem_before_gnn
    performance_metrics['peak_memory_mb']['Semantic Ego-GNN'] = mem_after_gnn

    ablation_results, ablation_predictions = run_ablation_study(
        user_posts, user_labels, config, cached_features=cached_features
    )

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

    return all_results, all_predictions, performance_metrics


def main():
    """Execute standard model comparison pipeline."""
    run_model_comparison()


def main_with_results():
    """
    Execute comprehensive analysis pipeline with full result persistence.

    Trains RoBERTa baseline and STEMS-GNN, runs ablation studies, and saves
    all results (metrics, visualizations, model checkpoints) to results directory.
    """
    print("\n" + "="*60)
    print("STEMS-GNN Research: Comprehensive Analysis")
    print("="*60)

    config = load_config('config.yaml')
    set_seed(42)

    saver = ResultsSaver(results_dir='results')

    print("\nLoading dataset...")
    target_posts = config.get('data', {}).get('target_posts_per_class', 5000)
    print(f"Configuration: {target_posts} posts per class (total: {target_posts * 2} posts)")

    raw_data = load_rmhd_data(
        data_path=config.get('data', {}).get('path', 'data/raw'),
        target_posts_per_class=target_posts,
        config=config
    )

    user_posts = raw_data['user_posts']
    user_labels = raw_data['user_labels']
    print(f"Dataset loaded: {len(user_posts)} users")

    dataset_info = {
        'total_users': len(user_posts),
        'depression_users': sum(user_labels.values()),
        'control_users': len(user_labels) - sum(user_labels.values()),
        'class_ratio': sum(user_labels.values()) / len(user_labels) if len(user_labels) > 0 else 0,
        'avg_posts_per_user': sum(len(posts) for posts in user_posts.values()) / len(user_posts) if len(user_posts) > 0 else 0,
    }

    all_results, all_predictions, performance_metrics = run_comprehensive_comparison(
        user_posts, user_labels, config, results_saver=saver, save_models=True
    )

    roberta_result = all_results.get('roberta_baseline', {})
    gnn_result = all_results.get('semantic_gnn_full', {})
    ablation_results = all_results.get('ablation_study', {})

    predictions_dict = {}
    if 'roberta_baseline' in all_predictions:
        predictions_dict['RoBERTa Baseline'] = all_predictions['roberta_baseline']
    if 'semantic_gnn_full' in all_predictions:
        predictions_dict['Semantic Ego-GNN'] = all_predictions['semantic_gnn_full']

    if 'ablation_study' in all_predictions:
        for config_name, pred_tuple in all_predictions['ablation_study'].items():
            predictions_dict[config_name.replace('_', ' ').title()] = pred_tuple

    saver.save_all_results(
        roberta_results=roberta_result,
        gnn_results=gnn_result,
        ablation_results=ablation_results,
        dataset_info=dataset_info,
        predictions=predictions_dict,
        performance_metrics=performance_metrics
    )

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print(f"Results saved to: {saver.results_dir}/\n")


if __name__ == "__main__":
    main_with_results()