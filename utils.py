"""
Utility functions for the research project.
"""

import random
import numpy as np
import torch
import yaml
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        return get_default_config()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def get_default_config():
    """
    Get default configuration for the project.

    Returns:
        dict: Default configuration
    """
    config = {
        'data': {
            'path': 'data/raw/rmhd_dataset',
            'temporal_windows': {
                'pre_pandemic': ['2018-01-01', '2019-12-31'],
                'pandemic_onset': ['2020-01-01', '2020-06-30'],
                'pandemic_peak': ['2020-07-01', '2020-12-31']
            }
        },
        'similarity': {
            'linguistic_weight': 0.4,
            'temporal_weight': 0.3,
            'psychological_weight': 0.3,
            'threshold': 0.5
        },
        'network': {
            'threshold': 0.5,
            'k_hop': 2
        },
        'model': {
            'input_dim': 70,
            'hidden_dim': 128,
            'output_dim': 2,
            'dropout': 0.5,
            'num_heads': 4,
            'use_temporal': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.001,
            'patience': 20
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }
    }

    return config

def save_config(config, config_path):
    """
    Save configuration to YAML file.

    Args:
        config (dict): Configuration to save
        config_path (str): Path to save config
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def save_pickle(obj, filepath):
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath (str): Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """
    Load object from pickle file.

    Args:
        filepath (str): Path to pickle file

    Returns:
        Object loaded from file
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def setup_logging(log_level=logging.INFO):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )

def plot_similarity_distribution(similarities, save_path=None):
    """
    Plot distribution of similarity scores.

    Args:
        similarities (dict): Similarity matrices
        save_path (str): Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sim_types = ['linguistic', 'temporal', 'psychological', 'composite']

    for i, sim_type in enumerate(sim_types):
        ax = axes[i // 2, i % 2]

        sim_matrix = similarities[sim_type]
        upper_triangle = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        ax.hist(upper_triangle, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{sim_type.capitalize()} Similarity Distribution')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_network_statistics(ego_networks, save_path=None):
    """
    Plot ego-network statistics.

    Args:
        ego_networks (dict): Ego-networks
        save_path (str): Optional path to save plot
    """
    degrees = [net['stats']['degree'] for net in ego_networks.values()]
    clustering = [net['stats']['clustering'] for net in ego_networks.values()]
    network_sizes = [net['stats']['network_size'] for net in ego_networks.values()]
    avg_similarities = [net['stats']['avg_similarity'] for net in ego_networks.values()]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(degrees, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Degree Distribution')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(clustering, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Clustering Coefficient Distribution')
    axes[0, 1].set_xlabel('Clustering Coefficient')
    axes[0, 1].set_ylabel('Frequency')

    axes[1, 0].hist(network_sizes, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Ego-Network Size Distribution')
    axes[1, 0].set_xlabel('Network Size')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].hist(avg_similarities, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Average Similarity Distribution')
    axes[1, 1].set_xlabel('Average Similarity')
    axes[1, 1].set_ylabel('Frequency')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_model_comparison(results, save_path=None):
    """
    Plot model comparison results.

    Args:
        results (dict): Results from different models
        save_path (str): Optional path to save plot
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize(), alpha=0.8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def create_results_summary(results, ablation_results=None, save_path=None):
    """
    Create a comprehensive results summary.

    Args:
        results (dict): Main model results
        ablation_results (dict): Optional ablation study results
        save_path (str): Optional path to save summary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = f"""
    =====================================
    SEMANTIC EGO-NETWORK GNN RESEARCH
    =====================================
    Generated: {timestamp}

    MAIN RESULTS:
    """

    for model_name, metrics in results.items():
        summary += f"\n    {model_name}:\n"
        for metric, value in metrics.items():
            if metric not in ['predictions', 'labels', 'probabilities']:
                summary += f"        {metric.capitalize()}: {value:.4f}\n"

    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    summary += f"\n    Best Model: {best_model} (F1: {results[best_model]['f1']:.4f})\n"

    if ablation_results:
        summary += "\n    ABLATION STUDY:\n"
        for sim_type, metrics in ablation_results.items():
            summary += f"        {sim_type.capitalize()} only: F1 = {metrics['f1']:.4f}\n"

        full_f1 = results.get('Semantic_Ego_GNN', {}).get('f1', 0)
        summary += f"        Combined (Full Model): F1 = {full_f1:.4f}\n"

    print(summary)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)

    return summary

def check_cuda_availability():
    """
    Check CUDA availability and print device information.
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"CUDA is available!")
        print(f"Number of CUDA devices: {device_count}")
        print(f"Current device: {current_device} ({device_name})")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.")

def memory_usage_report():
    """
    Report current memory usage.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    else:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB)")