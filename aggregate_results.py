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
    """Load all available seed results from disk including ablation studies and confusion matrices."""
    all_results = []
    all_ablations = []
    all_full_data = []
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

                all_full_data.append(data)

                print(f"✓ Loaded results from {seed_dir.name} (seed {seed_num})")
            except Exception as e:
                print(f"✗ Failed to load {seed_dir.name}: {e}")
        else:
            print(f"✗ No comprehensive_results.json in {seed_dir.name}")

    return all_results, all_ablations, seeds, all_full_data


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
    """Create visualization of ablation study results across seeds with color-coded configurations."""
    if not all_ablations or not all_ablations[0]:
        print("No ablation data available for visualization")
        return

    ablation_configs = list(all_ablations[0].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    color_map = {
        'linguistic_only': '#3498db',      # Blue - linguistic features
        'temporal_only': '#e74c3c',        # Red - temporal features
        'psychological_only': '#f39c12',   # Orange - psychological features
        'full_combined': '#2ecc71'         # Green - full model (best performance)
    }

    fallback_colors = ['#9b59b6', '#1abc9c', '#34495e', '#e67e22']

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
        colors = []

        for config in ablation_configs:
            if metric in ablation_agg[config]:
                configs.append(config.replace('_', '\n'))
                means.append(ablation_agg[config][metric]['mean'])
                stds.append(ablation_agg[config][metric]['std'])

                if config in color_map:
                    colors.append(color_map[config])
                else:
                    fallback_idx = len(colors) % len(fallback_colors)
                    colors.append(fallback_colors[fallback_idx])

        x = np.arange(len(configs))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.85, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Configuration', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(f'{metric.capitalize()}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=0, ha='center', fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Ablation Study: Multi-Seed Results (Mean ± Std)', y=1.02, fontsize=14, fontweight='bold')
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


def plot_aggregated_confusion_matrices(all_results_with_cm, output_dir="results"):
    """
    Create aggregated confusion matrices for both models across all seeds.

    Args:
        all_results_with_cm: List of result dictionaries with confusion_matrix data
        output_dir: Directory to save the plots
    """
    print("\nGenerating aggregated confusion matrices...")

    models_cm = {}

    for results in all_results_with_cm:
        model_performance = results.get('model_performance', {})

        for model_name, model_data in model_performance.items():
            if 'confusion_matrix' in model_data:
                if model_name not in models_cm:
                    models_cm[model_name] = []
                models_cm[model_name].append(np.array(model_data['confusion_matrix']))

    if not models_cm:
        print("No confusion matrix data found in results.")
        return

    mean_cms = {}
    for model_name, cm_list in models_cm.items():
        mean_cms[model_name] = np.mean(cm_list, axis=0).astype(int)

    num_models = len(mean_cms)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    class_names = ['Control', 'Depression']

    for idx, (model_name, cm) in enumerate(mean_cms.items()):
        ax = axes[idx]

        display_name = model_name.replace('_', ' ').title()
        if 'Roberta' in display_name:
            display_name = 'RoBERTa Baseline'
        elif 'Semantic' in display_name or 'Gnn' in display_name:
            display_name = 'STEMS-GNN'

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14, 'weight': 'bold'})

        ax.set_title(f'{display_name}\n(Averaged across all seeds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=9, color='gray')

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = f'Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}'
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Aggregated Confusion Matrices (Mean across all seeds)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'aggregated_confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Aggregated confusion matrices saved to: {output_path}")

    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))

    if num_models == 1:
        axes = [axes]

    for idx, (model_name, cm) in enumerate(mean_cms.items()):
        ax = axes[idx]

        display_name = model_name.replace('_', ' ').title()
        if 'Roberta' in display_name:
            display_name = 'RoBERTa Baseline'
        elif 'Semantic' in display_name or 'Gnn' in display_name:
            display_name = 'STEMS-GNN'

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Percentage'},
                   vmin=0, vmax=1, annot_kws={'size': 14, 'weight': 'bold'})

        ax.set_title(f'{display_name} (Normalized)\n(Averaged across all seeds)',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)

    plt.suptitle('Normalized Confusion Matrices (Row-wise percentages)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'aggregated_confusion_matrices_normalized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Normalized confusion matrices saved to: {output_path}")


def plot_aggregated_roc_curves(all_full_data, output_dir="results"):
    """
    Generate aggregated ROC curves from saved prediction probabilities.

    Args:
        all_full_data: List of full comprehensive results from each seed
        output_dir: Directory to save output figure
    """
    from sklearn.metrics import roc_curve, auc

    os.makedirs(output_dir, exist_ok=True)

    # Check if prediction probabilities are available
    has_probs = False
    for data in all_full_data:
        perf = data.get('model_performance', {})
        if 'roberta_baseline' in perf and 'y_pred_prob' in perf['roberta_baseline']:
            has_probs = True
            break

    if not has_probs:
        print("⚠ Skipping ROC curves: prediction probabilities not found in saved results")
        print("  To generate ROC curves, re-run experiments (prediction probabilities are now saved)")
        return None

    print("Generating aggregated ROC curves...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Define common FPR points for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    # Collect ROC data for each seed
    roberta_tprs = []
    roberta_aucs = []
    gnn_tprs = []
    gnn_aucs = []

    for data in all_full_data:
        perf = data.get('model_performance', {})

        # RoBERTa
        if 'roberta_baseline' in perf and 'y_pred_prob' in perf['roberta_baseline']:
            y_true = perf['roberta_baseline']['y_true']
            y_prob = perf['roberta_baseline']['y_pred_prob']

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            roberta_aucs.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            roberta_tprs.append(interp_tpr)

            # Plot individual seed curve
            ax.plot(fpr, tpr, color='#1f77b4', alpha=0.15, linewidth=1)

        # STEMS-GNN
        if 'semantic_ego_gnn' in perf and 'y_pred_prob' in perf['semantic_ego_gnn']:
            y_true = perf['semantic_ego_gnn']['y_true']
            y_prob = perf['semantic_ego_gnn']['y_pred_prob']

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            gnn_aucs.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            gnn_tprs.append(interp_tpr)

            # Plot individual seed curve
            ax.plot(fpr, tpr, color='#ff7f0e', alpha=0.15, linewidth=1)

    # Plot RoBERTa mean curve
    if roberta_tprs:
        roberta_mean_tpr = np.mean(roberta_tprs, axis=0)
        roberta_mean_tpr[-1] = 1.0
        roberta_std_tpr = np.std(roberta_tprs, axis=0)
        roberta_mean_auc = np.mean(roberta_aucs)
        roberta_std_auc = np.std(roberta_aucs)

        ax.plot(
            mean_fpr, roberta_mean_tpr,
            color='#1f77b4',
            label=f'RoBERTa (AUC = {roberta_mean_auc:.3f} ± {roberta_std_auc:.3f})',
            linewidth=2.5
        )

        roberta_tpr_upper = np.minimum(roberta_mean_tpr + roberta_std_tpr, 1)
        roberta_tpr_lower = np.maximum(roberta_mean_tpr - roberta_std_tpr, 0)
        ax.fill_between(
            mean_fpr, roberta_tpr_lower, roberta_tpr_upper,
            color='#1f77b4', alpha=0.2, label='RoBERTa ± 1 std'
        )

    # Plot STEMS-GNN mean curve
    if gnn_tprs:
        gnn_mean_tpr = np.mean(gnn_tprs, axis=0)
        gnn_mean_tpr[-1] = 1.0
        gnn_std_tpr = np.std(gnn_tprs, axis=0)
        gnn_mean_auc = np.mean(gnn_aucs)
        gnn_std_auc = np.std(gnn_aucs)

        ax.plot(
            mean_fpr, gnn_mean_tpr,
            color='#ff7f0e',
            label=f'STEMS-GNN (AUC = {gnn_mean_auc:.3f} ± {gnn_std_auc:.3f})',
            linewidth=2.5
        )

        gnn_tpr_upper = np.minimum(gnn_mean_tpr + gnn_std_tpr, 1)
        gnn_tpr_lower = np.maximum(gnn_mean_tpr - gnn_std_tpr, 0)
        ax.fill_between(
            mean_fpr, gnn_tpr_lower, gnn_tpr_upper,
            color='#ff7f0e', alpha=0.2, label='STEMS-GNN ± 1 std'
        )

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Aggregated ROC Curves (n={len(roberta_aucs)} seeds)\nDepression Detection Performance',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'aggregated_roc_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Aggregated ROC curves saved: {output_path}")
    return output_path


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

    all_results, all_ablations, seeds, all_full_data = load_seed_results()

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
    plot_aggregated_confusion_matrices(all_full_data, output_dir=output_dir)

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
    print("  - aggregated_confusion_matrices.png")
    print("  - aggregated_confusion_matrices_normalized.png")
    print()


if __name__ == "__main__":
    main()
