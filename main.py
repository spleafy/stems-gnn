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
import numpy as np
from data_preprocessing import load_rmhd_data
from roberta_baseline import train_memory_efficient_baseline
from semantic_ego_gnn import train_correct_semantic_gnn
from utils import load_config, set_seed
from results_saver import ResultsSaver
from feature_extractor import UnifiedFeatureExtractor
from statistical_significance import mcnemar_test, bootstrap_confidence_interval, plot_confusion_matrix, analyze_misclassifications
from statistical_significance import mcnemar_test, bootstrap_confidence_interval, plot_confusion_matrix, analyze_misclassifications
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


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
    similarity_weights_config = config.get('ego_network', {}).get('similarity_weights', {})
    similarity_weights = (
        similarity_weights_config.get('alpha', 0.6),
        similarity_weights_config.get('beta', 0.1),
        similarity_weights_config.get('gamma', 0.3)
    )
    gnn_results = train_correct_semantic_gnn(user_posts, user_labels, config, similarity_weights=similarity_weights)

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
            
        print("\n" + "=" * 60)
        print("Statistical Significance Analysis")
        print("=" * 60)        
    else:
        print("Error: Training failed for one or both models")


def run_ablation_study(user_posts, user_labels, config, cached_features=None, data_splits=None):
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
    }

    similarity_weights_config = config.get('ego_network', {}).get('similarity_weights', {})
    full_combined_weights = (
        similarity_weights_config.get('alpha', 0.6),
        similarity_weights_config.get('beta', 0.1),
        similarity_weights_config.get('gamma', 0.3)
    )
    ablation_configs['full_combined'] = full_combined_weights

    ablation_results = {}
    ablation_predictions = {}

    for config_name, weights in ablation_configs.items():
        print(f"\nConfiguration: {config_name} (α={weights[0]}, β={weights[1]}, γ={weights[2]})")

        try:
            result, predictions = train_correct_semantic_gnn(
                user_posts, user_labels, config,
                similarity_weights=weights,
                return_predictions=True,
                cached_features=cached_features,
                data_splits=data_splits
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


def run_comprehensive_comparison(seed=42):
    """
    Run a comprehensive comparison between RoBERTa baseline and STEMS-GNN.
    
    Args:
        seed (int): Random seed for reproducibility
    """
    print(f"\n{'='*80}")
    print(f"STEMS-GNN EXPERIMENTAL PIPELINE | SEED: {seed}")
    print(f"{'='*80}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objective: Comparative evaluation of STEMS-GNN vs. RoBERTa Baseline")
    print(f"{'-'*80}")
    
    config = load_config('config.yaml')
    set_seed(seed)

    saver = ResultsSaver(results_dir=f'results_seed_{seed}')

    performance_metrics = {
        'execution_time_sec': {},
        'memory_delta_mb': {},
        'peak_memory_mb': {}
    }

    process = psutil.Process(os.getpid())

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

    print("\n" + "="*60)
    print("Extracting Features (shared across all models)")
    print("="*60)

    feature_extractor = UnifiedFeatureExtractor(config)
    start_feature_extraction = time.time()

    combined_df = raw_data.get('combined_df', None)

    max_users_config = config.get('data', {}).get('max_users', 10000)
    cached_features = feature_extractor.extract_all_features(
        user_posts, user_labels, combined_df=combined_df,
        max_users=max_users_config
    )

    end_feature_extraction = time.time()
    print(f"\nFeature extraction complete in {end_feature_extraction - start_feature_extraction:.1f} seconds")
    print(f"Features will be reused for baseline, STEMS-GNN, and all ablation studies\n")

    print("\n" + "="*60)
    print("Creating Shared Train/Val/Test Splits")
    print("="*60)

    all_user_ids = cached_features['user_ids']
    all_labels = [cached_features['labels'][uid] for uid in all_user_ids]

    train_ids, test_ids = train_test_split(all_user_ids, test_size=0.2, random_state=seed, stratify=all_labels)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=seed, stratify=[cached_features['labels'][uid] for uid in train_ids])

    print(f"Data splits created:")
    print(f"  Train: {len(train_ids)} users ({len(train_ids)/len(all_user_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} users ({len(val_ids)/len(all_user_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} users ({len(test_ids)/len(all_user_ids)*100:.1f}%)")

    data_splits = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }

    depression_users = sum(1 for uid in all_user_ids if cached_features['labels'][uid] == 1)
    control_users = len(all_user_ids) - depression_users
    total_posts = sum(len(user_posts[uid]) for uid in all_user_ids)

    dataset_info = {
        'total_users': len(all_user_ids),
        'depression_users': depression_users,
        'control_users': control_users,
        'class_ratio': depression_users / len(all_user_ids) if len(all_user_ids) > 0 else 0,
        'avg_posts_per_user': total_posts / len(all_user_ids) if len(all_user_ids) > 0 else 0
    }

    print("\nTraining RoBERTa Baseline...")
    mem_before_rb = process.memory_info().rss / 1024 / 1024
    start_time_rb = time.time()

    roberta_result, roberta_predictions = train_memory_efficient_baseline(
        user_posts, user_labels, config, return_predictions=True,
        save_model=True, results_saver=saver,
        cached_features=cached_features,
        data_splits=data_splits
    )

    end_time_rb = time.time()
    mem_after_rb = process.memory_info().rss / 1024 / 1024

    performance_metrics['execution_time_sec']['RoBERTa-memory_efficient_transformer'] = end_time_rb - start_time_rb
    performance_metrics['memory_delta_mb']['RoBERTa-memory_efficient_transformer'] = mem_after_rb - mem_before_rb
    performance_metrics['peak_memory_mb']['RoBERTa-memory_efficient_transformer'] = mem_after_rb

    print("\nTraining STEMS-GNN...")
    mem_before_gnn = process.memory_info().rss / 1024 / 1024
    start_time_gnn = time.time()

    similarity_weights_config = config.get('ego_network', {}).get('similarity_weights', {})
    similarity_weights = (
        similarity_weights_config.get('alpha', 0.6),
        similarity_weights_config.get('beta', 0.1),
        similarity_weights_config.get('gamma', 0.3)
    )
    gnn_result, gnn_predictions = train_correct_semantic_gnn(
        user_posts, user_labels, config, return_predictions=True,
        save_model=True, results_saver=saver,
        cached_features=cached_features,
        data_splits=data_splits,
        similarity_weights=similarity_weights
    )

    end_time_gnn = time.time()
    mem_after_gnn = process.memory_info().rss / 1024 / 1024

    performance_metrics['execution_time_sec']['Semantic Ego-GNN'] = end_time_gnn - start_time_gnn
    performance_metrics['memory_delta_mb']['Semantic Ego-GNN'] = mem_after_gnn - mem_before_gnn
    performance_metrics['peak_memory_mb']['Semantic Ego-GNN'] = mem_after_gnn

    ablation_results, ablation_predictions = run_ablation_study(
        user_posts, user_labels, config, cached_features=cached_features,
        data_splits=data_splits
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
    print("Generating Research-Grade Evaluation Artifacts")
    print("="*60)
    
    if 'roberta_baseline' in all_predictions and 'semantic_gnn_full' in all_predictions:
        y_true_rob, y_pred_rob_prob = all_predictions['roberta_baseline']
        y_true_gnn, y_pred_gnn_prob = all_predictions['semantic_gnn_full']
                
        y_pred_rob = [1 if p > 0.5 else 0 for p in y_pred_rob_prob]
        y_pred_gnn = [1 if p > 0.5 else 0 for p in y_pred_gnn_prob]
        
        if len(y_true_rob) == len(y_true_gnn):
             chi2, p_value = mcnemar_test(y_true_gnn, y_pred_rob, y_pred_gnn)
             print(f"\nMcNemar's Test (STEMS-GNN vs RoBERTa):")
             print(f"  chi2 statistic: {chi2:.4f}")
             print(f"  p-value: {p_value:.4e}")
             if p_value < 0.05:
                 print("  RESULT: Statistically significant difference (p < 0.05)")
             else:
                 print("  RESULT: No statistically significant difference")
                 
             print("\nBootstrap Confidence Intervals (95%):")
             low_r, high_r = bootstrap_confidence_interval(y_true_rob, y_pred_rob, f1_score)
             print(f"  RoBERTa F1: {f1_score(y_true_rob, y_pred_rob):.4f} ({low_r:.4f}, {high_r:.4f})")
             
             low_g, high_g = bootstrap_confidence_interval(y_true_gnn, y_pred_gnn, f1_score)
             print(f"  STEMS-GNN F1: {f1_score(y_true_gnn, y_pred_gnn):.4f} ({low_g:.4f}, {high_g:.4f})")
             
             plot_confusion_matrix(y_true_gnn, y_pred_gnn, save_path=os.path.join(saver.results_dir, 'cm_stems_gnn.png'))
             plot_confusion_matrix(y_true_rob, y_pred_rob, save_path=os.path.join(saver.results_dir, 'cm_roberta.png'))
             print(f"\nConfusion matrices saved to {saver.results_dir}")
             
             analyze_misclassifications(y_true_gnn, y_pred_gnn, range(len(y_true_gnn)), {}, save_path=os.path.join(saver.results_dir, 'misclassified_gnn.csv'))
             print(f"Misclassified examples saved to {saver.results_dir}/misclassified_gnn.csv")
             
        else:
            print("Warning: Label mismatch between models, skipping statistical tests.")

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)
    print(f"Results saved to: {saver.results_dir}/\n")

    return all_results, all_predictions, performance_metrics


def run_multi_seed_evaluation(num_seeds=5):
    """
    Run the evaluation across multiple random seeds and generate aggregated results.

    This function:
    1. Runs comprehensive comparison for each seed (training both models + ablation)
    2. Prints aggregated statistics to console
    3. Automatically generates aggregated visualizations and saves to results/
    4. Creates publication-quality figures and JSON summary

    Args:
        num_seeds (int): Number of seeds to evaluate (default: 5)

    Outputs:
        - results_seed_XX/ directories for each seed
        - results/ directory with aggregated visualizations
        - Console summary of mean ± std metrics
    """
    seeds = [42 + i for i in range(num_seeds)]
    all_results = []

    print(f"\n{'='*80}")
    print(f"MULTI-SEED ROBUSTNESS EVALUATION")
    print(f"{'='*80}")
    print(f"Total Seeds: {num_seeds}")
    print(f"Seeds: {seeds}")
    print(f"Objective: Validate model stability and statistical significance of results.")
    print(f"{'-'*80}")

    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        try:
            results, predictions, metrics = run_comprehensive_comparison(seed=seed)
            all_results.append(results)
        except Exception as e:
            print(f"Error running seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("All runs failed.")
        return

    metrics_to_agg = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    available_models = {}
    if all_results:
        first_result = all_results[0]
        for key in ['roberta_baseline', 'memory_efficient_transformer', 'tfidf_logistic']:
            if key in first_result:
                available_models['RoBERTa Baseline'] = key
                break
        for key in ['semantic_gnn_full', 'correct_semantic_gnn', 'semantic_ego_gnn']:
            if key in first_result:
                available_models['STEMS-GNN'] = key
                break

    print(f"\n{'='*80}")
    print(f"Multi-Seed Evaluation Results ({len(all_results)} runs)")
    print(f"{'='*80}")

    for display_name, model_key in available_models.items():
        print(f"\nModel: {display_name}")
        print("-" * 40)

        for metric in metrics_to_agg:
            try:
                values = []
                for r in all_results:
                    if model_key in r:
                        if 'metrics' in r[model_key]:
                            values.append(r[model_key]['metrics'][metric])
                        elif metric in r[model_key]:
                            values.append(r[model_key][metric])

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"{metric.capitalize():<15}: {mean_val:.4f} ± {std_val:.4f} (CI: {mean_val-1.96*std_val:.4f} - {mean_val+1.96*std_val:.4f})")
            except (KeyError, TypeError) as e:
                print(f"{metric.capitalize():<15}: N/A (error: {e})")

    print(f"\n{'='*80}")
    print("Evaluation Complete.")
    print(f"{'='*80}")

    # Generate aggregated results and visualizations
    print(f"\n{'='*80}")
    print("GENERATING AGGREGATED RESULTS AND VISUALIZATIONS")
    print(f"{'='*80}\n")

    try:
        from aggregate_results import (
            load_seed_results,
            aggregate_metrics,
            save_aggregated_results,
            plot_model_comparison,
            plot_ablation_study,
            plot_metrics_heatmap,
            plot_error_comparison
        )

        # Load results from disk
        all_results_loaded, all_ablations, result_seeds = load_seed_results()

        if all_results_loaded:
            # Aggregate metrics
            aggregated = aggregate_metrics(all_results_loaded)

            # Save aggregated results
            output_dir = "results"
            save_aggregated_results(aggregated, result_seeds, output_dir=output_dir)

            # Generate visualizations
            print(f"\nGenerating publication-quality visualizations...")
            plot_model_comparison(aggregated, output_dir=output_dir)
            plot_metrics_heatmap(aggregated, output_dir=output_dir)
            plot_error_comparison(aggregated, output_dir=output_dir)
            plot_ablation_study(all_ablations, output_dir=output_dir)

            print(f"\n{'='*80}")
            print("AGGREGATED RESULTS COMPLETE")
            print(f"{'='*80}")
            print(f"\nAll aggregated results saved to: {output_dir}/")
            print("\nGenerated files:")
            print("  - multi_seed_aggregated.json")
            print("  - aggregated_model_comparison.png")
            print("  - aggregated_metrics_heatmap.png")
            print("  - aggregated_error_distribution.png")
            print("  - aggregated_ablation_study.png")
            print("  - aggregated_summary.txt")
        else:
            print("No results found for aggregation.")

    except Exception as e:
        print(f"Error generating aggregated results: {e}")
        import traceback
        traceback.print_exc()


def main_with_results():
    """
    Main execution function that saves results.
    """
    run_multi_seed_evaluation(num_seeds=5)


if __name__ == "__main__":
    main_with_results()