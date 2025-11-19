#!/usr/bin/env python3
"""
Aggregate results from multiple seed runs and generate comprehensive visualizations.
"""

import json
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_seed_results(results_dir_pattern="results_seed_"):
    """Load all available seed results from disk including ablation studies."""
    all_results = []
    all_ablations = []
    seeds = []

    base_dir = Path(".")
    seed_dirs = sorted([d for d in base_dir.glob(f"{results_dir_pattern}*") if d.is_dir()])

    for seed_dir in seed_dirs:
        comprehensive_results = seed_dir / "comprehensive_results.json"

        if comprehensive_results.exists():
            try:
                with open(comprehensive_results, 'r') as f:
                    data = json.load(f)

                seed_num = int(seed_dir.name.replace(results_dir_pattern, ""))
                seeds.append(seed_num)

                model_performance = data.get('model_performance', {})
                all_results.append(model_performance)

                ablation_study = data.get('ablation_study', {})
                all_ablations.append(ablation_study)

                print(f"✓ Loaded results from {seed_dir.name} (seed {seed_num})")
            except Exception as e:
                print(f"✗ Failed to load {seed_dir.name}: {e}")
        else:
            print(f"✗ No comprehensive_results.json in {seed_dir.name}")

    return all_results, all_ablations, seeds


def aggregate_metrics(all_results):
    """Aggregate metrics across seeds."""
    metrics_to_agg = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    available_models = {}
    if all_results:
        first_result = all_results[0]
        for key in ['roberta_baseline', 'memory_efficient_transformer', 'tfidf_logistic']:
            if key in first_result:
                available_models['RoBERTa Baseline'] = key
                break
        for key in ['semantic_ego_gnn', 'semantic_gnn_full', 'correct_semantic_gnn']:
            if key in first_result:
                available_models['STEMS-GNN'] = key
                break

    print(f"\n{'='*80}")
    print(f"Multi-Seed Evaluation Results ({len(all_results)} runs)")
    print(f"{'='*80}")

    aggregated_results = {}

    for display_name, model_key in available_models.items():
        print(f"\nModel: {display_name}")
        print("-" * 40)

        aggregated_results[display_name] = {}

        for metric in metrics_to_agg:
            try:
                values = []
                for r in all_results:
                    if model_key in r:
                        if 'metrics' in r[model_key]:
                            if metric in r[model_key]['metrics']:
                                values.append(r[model_key]['metrics'][metric])
                        elif metric in r[model_key]:
                            values.append(r[model_key][metric])

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ci_lower = mean_val - 1.96 * std_val
                    ci_upper = mean_val + 1.96 * std_val

                    aggregated_results[display_name][metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'ci_95': (ci_lower, ci_upper),
                        'values': values
                    }

                    print(f"{metric.capitalize():<15}: {mean_val:.4f} ± {std_val:.4f} "
                          f"(95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
                else:
                    print(f"{metric.capitalize():<15}: N/A (no data)")

            except Exception as e:
                print(f"{metric.capitalize():<15}: N/A (error: {e})")

    print(f"\n{'='*80}")
    print("Aggregation Complete.")
    print(f"{'='*80}\n")

    return aggregated_results


def save_aggregated_results(aggregated_results, seeds, output_dir="results", output_file="multi_seed_aggregated.json"):
    """Save aggregated results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    output = {
        'metadata': {
            'num_seeds': len(seeds),
            'seeds': seeds,
        },
        'aggregated_metrics': {}
    }

    for model_name, metrics in aggregated_results.items():
        output['aggregated_metrics'][model_name] = {}
        for metric_name, metric_data in metrics.items():
            output['aggregated_metrics'][model_name][metric_name] = {
                'mean': float(metric_data['mean']),
                'std': float(metric_data['std']),
                'ci_95_lower': float(metric_data['ci_95'][0]),
                'ci_95_upper': float(metric_data['ci_95'][1]),
                'individual_values': [float(v) for v in metric_data['values']]
            }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Aggregated results saved to: {output_path}")


def plot_model_comparison(aggregated_results, output_dir="results"):
    """Create bar chart comparing models across all metrics with error bars."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(aggregated_results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.35

    for i, model_name in enumerate(model_names):
        means = []
        stds = []

        for metric in metrics:
            if metric in aggregated_results[model_name]:
                means.append(aggregated_results[model_name][metric]['mean'])
                stds.append(aggregated_results[model_name][metric]['std'])
            else:
                means.append(0)
                stds.append(0)

        offset = width * (i - len(model_names)/2 + 0.5)
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=model_name, capsize=5, alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Multi-Seed Model Performance Comparison (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aggregated_model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Model comparison plot saved to: {output_path}")


def plot_ablation_study(all_ablations, output_dir="results"):
    """Create visualization of ablation study results across seeds."""
    if not all_ablations or not all_ablations[0]:
        print("No ablation data available for visualization")
        return

    ablation_configs = list(all_ablations[0].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    ablation_agg = {}
    for config in ablation_configs:
        ablation_agg[config] = {}
        for metric in metrics:
            values = []
            for ablation in all_ablations:
                if config in ablation and metric in ablation[config]:
                    values.append(ablation[config][metric])

            if values:
                ablation_agg[config][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        configs = []
        means = []
        stds = []

        for config in ablation_configs:
            if metric in ablation_agg[config]:
                configs.append(config.replace('_', '\n'))
                means.append(ablation_agg[config][metric]['mean'])
                stds.append(ablation_agg[config][metric]['std'])

        x = np.arange(len(configs))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color='steelblue')

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study: Multi-Seed Results (Mean ± Std)', y=1.02, fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aggregated_ablation_study.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Ablation study plot saved to: {output_path}")


def plot_metrics_heatmap(aggregated_results, output_dir="results"):
    """Create heatmap of mean performance metrics across models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(aggregated_results.keys())

    data = []
    for model_name in model_names:
        row = []
        for metric in metrics:
            if metric in aggregated_results[model_name]:
                row.append(aggregated_results[model_name][metric]['mean'])
            else:
                row.append(0)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_yticklabels(model_names)

    for i in range(len(model_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Multi-Seed Performance Heatmap (Mean Values)')
    fig.colorbar(im, ax=ax, label='Score')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aggregated_metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Metrics heatmap saved to: {output_path}")


def plot_error_comparison(aggregated_results, output_dir="results"):
    """Create visualization showing variation across seeds for each model."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(aggregated_results.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for j, model_name in enumerate(model_names):
            if metric in aggregated_results[model_name]:
                values = aggregated_results[model_name][metric]['values']
                mean_val = aggregated_results[model_name][metric]['mean']

                positions = [j]
                bp = ax.boxplot([values], positions=positions, widths=0.6,
                               patch_artist=True, showmeans=True)

                colors = ['lightblue', 'lightcoral', 'lightgreen']
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[j % len(colors)])

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m.split()[0] for m in model_names], fontsize=9)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Performance Variation Across Seeds', y=1.02, fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aggregated_error_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Error distribution plot saved to: {output_path}")


def main():
    print("\n" + "="*80)
    print("MULTI-SEED RESULTS AGGREGATION AND VISUALIZATION")
    print("="*80)
    print("Loading existing results from disk...\n")

    all_results, all_ablations, seeds = load_seed_results()

    if not all_results:
        print("\n✗ No results found. Please run experiments first.")
        return

    print(f"\nFound {len(all_results)} valid result sets")
    print(f"Seeds: {seeds}\n")

    aggregated = aggregate_metrics(all_results)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    save_aggregated_results(aggregated, seeds, output_dir=output_dir)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    plot_model_comparison(aggregated, output_dir=output_dir)
    plot_metrics_heatmap(aggregated, output_dir=output_dir)
    plot_error_comparison(aggregated, output_dir=output_dir)
    plot_ablation_study(all_ablations, output_dir=output_dir)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nAll results and figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - multi_seed_aggregated.json")
    print("  - aggregated_model_comparison.png")
    print("  - aggregated_metrics_heatmap.png")
    print("  - aggregated_error_distribution.png")
    print("  - aggregated_ablation_study.png")
    print()


if __name__ == "__main__":
    main()
