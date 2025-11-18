#!/usr/bin/env python3
"""
STEMS-GNN Implementation

Implementation of Graph Neural Network for depression detection using semantic
ego-networks constructed from multi-dimensional user similarity.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import gc

def clear_memory():
    gc.collect()

class CorrectSemanticProcessor:
    """
    Semantic similarity processor and ego-network constructor for depression detection.

    NOTE: Feature extraction is handled by UnifiedFeatureExtractor. This class only handles:
    - Multi-dimensional similarity calculation
    - Ego-network construction
    """

    def __init__(self, max_users=150, similarity_weights=(0.4, 0.3, 0.3)):
        self.max_users = max_users
        self.alpha, self.beta, self.gamma = similarity_weights

    def calculate_multi_similarity(self, user_features):
        """Calculate multi-dimensional similarity following research methodology."""
        print(f"Computing multi-dimensional similarity matrices (α={self.alpha}, β={self.beta}, γ={self.gamma})...")

        users = list(user_features.keys())
        features_matrix = np.array([user_features[user] for user in users])

        semantic_features = features_matrix[:, :64]

        liwc_start = 64
        liwc_end = liwc_start + 62
        liwc_features = features_matrix[:, liwc_start:liwc_end]

        temporal_features = features_matrix[:, -9:]

        linguistic_cols = [i for i in range(liwc_features.shape[1])
                          if i < 30]
        liwc_linguistic = liwc_features[:, linguistic_cols] if linguistic_cols else liwc_features[:, :10]

        linguistic_features_combined = np.concatenate([semantic_features, liwc_linguistic], axis=1)
        linguistic_sim = cosine_similarity(linguistic_features_combined)

        temporal_sim = cosine_similarity(temporal_features)

        psychological_cols = [i for i in range(liwc_features.shape[1])
                             if i >= 30]
        liwc_psychological = liwc_features[:, psychological_cols] if psychological_cols else liwc_features[:, 10:]

        psychological_sim = cosine_similarity(liwc_psychological)

        combined_similarity = (
            self.alpha * linguistic_sim +
            self.beta * temporal_sim +
            self.gamma * psychological_sim
        )

        return combined_similarity, users

    def build_ego_networks(self, user_features, k_neighbors=50, threshold=0.6, k_hops=2, min_neighbors=5):
        """
        Build semantic ego-networks with k-hop expansion.

        Constructs ego-networks by finding semantically similar neighbors and expanding
        to k-hop neighborhoods. Ensures minimum connectivity through adaptive neighbor selection.

        Args:
            user_features: Dictionary mapping user IDs to feature vectors
            k_neighbors: Maximum number of neighbors per user
            threshold: Similarity threshold for edge creation (τ)
            k_hops: Number of hops for neighborhood expansion
            min_neighbors: Minimum neighbors required to include a user

        Returns:
            Dictionary mapping user IDs to PyTorch Geometric Data objects
        """
        print("Constructing semantic ego-networks...")

        similarity_matrix, users = self.calculate_multi_similarity(user_features)
        features_matrix = np.array([user_features[user] for user in users])

        # Debug: Report similarity statistics
        sim_mean = np.mean(similarity_matrix)
        sim_std = np.std(similarity_matrix)
        sim_max = np.max(similarity_matrix)
        above_threshold = np.sum(similarity_matrix >= threshold) - len(users)  # Exclude diagonal
        total_pairs = len(users) * (len(users) - 1)
        sparsity = above_threshold / total_pairs if total_pairs > 0 else 0

        print(f"  Similarity statistics: mean={sim_mean:.3f}, std={sim_std:.3f}, max={sim_max:.3f}")
        print(f"  Edges above threshold (τ={threshold}): {above_threshold}/{total_pairs} ({sparsity*100:.2f}%)")

        ego_networks = {}

        for i, ego_user in enumerate(users):
            ego_similarities = similarity_matrix[i]

            neighbor_indices = []
            for j, sim in enumerate(ego_similarities):
                if i != j and sim >= threshold:
                    neighbor_indices.append((j, sim))

            neighbor_indices.sort(key=lambda x: x[1], reverse=True)

            if len(neighbor_indices) > k_neighbors:
                neighbor_indices = neighbor_indices[:k_neighbors]
            elif len(neighbor_indices) < min_neighbors:
                all_neighbors = [(j, similarity_matrix[i][j]) for j in range(len(users)) if j != i]
                all_neighbors.sort(key=lambda x: x[1], reverse=True)
                neighbor_indices = all_neighbors[:min_neighbors]

            if len(neighbor_indices) < min_neighbors:
                continue

            all_network_nodes = set([i])
            hop_1_neighbors = set([idx for idx, _ in neighbor_indices])
            all_network_nodes.update(hop_1_neighbors)

            if k_hops >= 2:
                for hop_1_neighbor in hop_1_neighbors:
                    hop_2_candidates = []
                    for j, sim in enumerate(similarity_matrix[hop_1_neighbor]):
                        if j not in all_network_nodes and sim >= threshold:
                            hop_2_candidates.append((j, sim))

                    hop_2_candidates.sort(key=lambda x: x[1], reverse=True)
                    for neighbor_idx, _ in hop_2_candidates[:5]:
                        all_network_nodes.add(neighbor_idx)

            network_indices = list(all_network_nodes)
            ego_idx_in_network = network_indices.index(i)

            network_indices = [i] + [idx for idx in network_indices if idx != i]
            network_features = features_matrix[network_indices]

            edge_list = []
            edge_weights = []

            for k, (neighbor_idx, weight) in enumerate(neighbor_indices, 1):
                if neighbor_idx in network_indices:
                    local_idx = network_indices.index(neighbor_idx)
                    edge_list.extend([[0, local_idx], [local_idx, 0]])
                    edge_weights.extend([weight, weight])

            for idx1 in range(1, len(network_indices)):
                for idx2 in range(idx1 + 1, len(network_indices)):
                    global_idx1 = network_indices[idx1]
                    global_idx2 = network_indices[idx2]
                    sim = similarity_matrix[global_idx1][global_idx2]

                    if sim >= threshold:
                        edge_list.extend([[idx1, idx2], [idx2, idx1]])
                        edge_weights.extend([sim, sim])

            node_features = torch.FloatTensor(network_features)
            edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.FloatTensor(edge_weights) if edge_weights else torch.empty(0)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(network_indices)
            )

            ego_networks[ego_user] = data

        # Report network statistics
        if len(ego_networks) > 0:
            network_sizes = [net.num_nodes for net in ego_networks.values()]
            edge_counts = [net.edge_index.shape[1] for net in ego_networks.values()]
            avg_size = np.mean(network_sizes)
            avg_edges = np.mean(edge_counts)
            print(f"Built {len(ego_networks)} ego-networks with k={k_hops} hop expansion")
            print(f"  Avg network size: {avg_size:.1f} nodes, {avg_edges:.1f} edges")
        else:
            print(f"WARNING: No ego-networks built! Check threshold and similarity values.")

        return ego_networks

class CorrectSemanticGNN(nn.Module):
    """
    STEMS-GNN: Graph Attention Network with temporal attention mechanism.

    Architecture:
        - 3 GAT layers with 4 attention heads each
        - Hidden dimension: 128, Output dimension: 64
        - Temporal attention weighting of structural attention
        - Residual connections and layer normalization for training stability
    """

    def __init__(self, input_dim=135, hidden_dim=128, dropout=0.3):
        super(CorrectSemanticGNN, self).__init__()

        self.semantic_liwc_dim = 126
        self.temporal_dim = 9

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False)
        self.gat3 = GATConv(hidden_dim, 64, heads=4, dropout=dropout, concat=False)

        self.temporal_weight = nn.Parameter(torch.tensor(0.3))
        self.temporal_proj = nn.Linear(self.temporal_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(64)

        self.residual1 = nn.Linear(hidden_dim, hidden_dim)
        self.residual2 = nn.Linear(hidden_dim, hidden_dim)
        self.residual3 = nn.Linear(hidden_dim, 64)

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(64),
            nn.Linear(64, 2)
        )

        self.dropout = dropout
        self.hidden_dim = hidden_dim

    def compute_temporal_attention(self, data):
        """
        Compute temporal attention weights using cosine similarity of temporal features.

        Returns temporal multiplier: 1 + β · temporal_similarity(t_i, t_j)
        """
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            return None

        temporal_features = data.x[:, -self.temporal_dim:]
        edge_index = data.edge_index
        temporal_sim = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            t_i = temporal_features[src]
            t_j = temporal_features[dst]

            sim = F.cosine_similarity(t_i.unsqueeze(0), t_j.unsqueeze(0))
            temporal_sim.append(sim)

        temporal_sim = torch.stack(temporal_sim)
        temporal_multiplier = 1.0 + self.temporal_weight * temporal_sim

        return temporal_multiplier

    def forward(self, data):
        """
        Forward pass with temporal-aware graph attention.

        Args:
            data: PyTorch Geometric Data object containing node features and edge structure

        Returns:
            Binary classification logits (depression vs control)
        """
        if data.num_nodes == 0:
            return torch.zeros(1, 2)

        x, edge_index = data.x, data.edge_index

        x = self.input_proj(x)
        temporal_attention = self.compute_temporal_attention(data)

        identity1 = x
        x = self.gat1(x, edge_index)
        if temporal_attention is not None and hasattr(data, 'edge_attr'):
            x = x * (1.0 + 0.1 * torch.mean(temporal_attention))
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm1(x + self.residual1(identity1))

        identity2 = x
        x = self.gat2(x, edge_index)
        if temporal_attention is not None:
            x = x * (1.0 + 0.1 * torch.mean(temporal_attention))
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm2(x + self.residual2(identity2))

        identity3 = x
        x = self.gat3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm3(x + self.residual3(identity3))

        ego_features = x[0:1]
        logits = self.classifier(ego_features)

        return logits

def train_correct_semantic_gnn(user_posts, user_labels, config, similarity_weights=(0.4, 0.3, 0.3), return_predictions=False, save_model=False, results_saver=None, cached_features=None, data_splits=None):
    """
    Train STEMS-GNN with multi-dimensional semantic similarity networks using shared data splits.

    Args:
        user_posts: Dictionary mapping user IDs to post lists
        user_labels: Dictionary mapping user IDs to binary labels (0=control, 1=depression)
        config: Configuration dictionary from config.yaml
        similarity_weights: Tuple of (α, β, γ) for linguistic, temporal, psychological weights
        return_predictions: If True, returns (y_true, y_prob) tuple for ROC analysis
        save_model: If True, saves model checkpoint via results_saver
        results_saver: ResultsSaver instance for checkpoint persistence
        cached_features: Pre-extracted features from UnifiedFeatureExtractor (required)
        data_splits: Shared train/val/test splits (required for fair comparison)

    Returns:
        Dictionary containing metrics, architecture details, and training statistics
    """
    print("=== Training STEMS-GNN ===")
    print(f"Similarity weights: α={similarity_weights[0]}, β={similarity_weights[1]}, γ={similarity_weights[2]}")

    # Require cached features and data splits
    if cached_features is None:
        raise ValueError("cached_features is required. Feature extraction must be done by UnifiedFeatureExtractor before calling this function.")

    if data_splits is None:
        raise ValueError("data_splits is required for fair comparison with baseline")

    processor = CorrectSemanticProcessor(similarity_weights=similarity_weights)

    print("Using pre-extracted cached features")
    user_features_raw = cached_features['combined_features']

    if len(user_features_raw) < 30:
        return {'error': 'Insufficient users for training'}

    # CRITICAL FIX: Normalize features using ONLY training data statistics
    from sklearn.preprocessing import StandardScaler

    # Get train/val/test splits first
    train_ids_temp = data_splits['train_ids']
    val_ids_temp = data_splits['val_ids']
    test_ids_temp = data_splits['test_ids']

    # Extract feature arrays for each split
    train_features_array = np.array([user_features_raw[uid] for uid in train_ids_temp if uid in user_features_raw])
    val_features_array = np.array([user_features_raw[uid] for uid in val_ids_temp if uid in user_features_raw])
    test_features_array = np.array([user_features_raw[uid] for uid in test_ids_temp if uid in user_features_raw])

    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    scaler.fit(train_features_array)
    print(f"Feature normalization: fitted on {len(train_features_array)} training samples")

    # Transform all splits using training statistics
    train_features_scaled = scaler.transform(train_features_array)
    val_features_scaled = scaler.transform(val_features_array)
    test_features_scaled = scaler.transform(test_features_array)

    # Reconstruct dictionary with normalized features
    user_features = {}
    for i, uid in enumerate([u for u in train_ids_temp if u in user_features_raw]):
        user_features[uid] = train_features_scaled[i]
    for i, uid in enumerate([u for u in val_ids_temp if u in user_features_raw]):
        user_features[uid] = val_features_scaled[i]
    for i, uid in enumerate([u for u in test_ids_temp if u in user_features_raw]):
        user_features[uid] = test_features_scaled[i]

    print(f"Feature normalization complete: {len(user_features)} users")

    # Get ego-network parameters from config
    k_neighbors = config.get('ego_network', {}).get('k_neighbors', 8)
    threshold = config.get('semantic_similarity', {}).get('similarity_threshold', 0.6)
    k_hops = config.get('ego_network', {}).get('k_hops', 2)

    # CRITICAL FIX: Build ego-networks using ONLY training data to prevent leakage
    # First, extract train/val/test splits
    train_ids = data_splits['train_ids']
    val_ids = data_splits['val_ids']
    test_ids = data_splits['test_ids']

    # Build networks using ONLY training data
    train_features = {uid: user_features[uid] for uid in train_ids if uid in user_features}

    print(f"Building ego-networks from training data only ({len(train_features)} users)...")
    train_ego_networks = processor.build_ego_networks(
        train_features,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=5
    )

    if len(train_ego_networks) < 15:
        return {'error': 'Insufficient ego-networks in training data'}

    # For validation and test: build networks independently
    # (They don't see training data, preventing leakage)
    val_features = {uid: user_features[uid] for uid in val_ids if uid in user_features}
    val_ego_networks = processor.build_ego_networks(
        val_features,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=3  # Lower threshold for smaller val set
    )

    test_features = {uid: user_features[uid] for uid in test_ids if uid in user_features}
    test_ego_networks = processor.build_ego_networks(
        test_features,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=3  # Lower threshold for smaller test set
    )

    # Combine all networks (but they were built separately, no leakage)
    ego_networks = {**train_ego_networks, **val_ego_networks, **test_ego_networks}

    # Use SHARED splits (same as RoBERTa)
    train_users = [u for u in train_ids if u in train_ego_networks]
    val_users = [u for u in val_ids if u in val_ego_networks]
    test_users = [u for u in test_ids if u in test_ego_networks]

    print(f"Using shared data splits:")
    print(f"  Train: {len(train_users)} users")
    print(f"  Val:   {len(val_users)} users")
    print(f"  Test:  {len(test_users)} users")

    train_labels = [user_labels[u] for u in train_users]
    print(f"Training balance: {sum(train_labels)}/{len(train_labels)} = {np.mean(train_labels):.2f}")

    device = torch.device('cpu')
    sample_user = list(user_features.keys())[0]
    input_dim = len(user_features[sample_user])
    print(f"Input dimension: {input_dim} (64 semantic + 62 LIWC + 9 temporal)")

    # Get model hyperparameters from config
    hidden_dim = config.get('models', {}).get('semantic_gnn', {}).get('hidden_dim', 128)
    dropout = config.get('models', {}).get('semantic_gnn', {}).get('dropout', 0.4)

    model = CorrectSemanticGNN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)

    # Get training hyperparameters from config
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    weight_decay = config.get('training', {}).get('weight_decay', 0.0001)
    epochs = config.get('training', {}).get('epochs', 100)
    patience = config.get('training', {}).get('patience', 20)
    min_delta = config.get('training', {}).get('min_delta', 0.001)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    batch_size = config.get('training', {}).get('batch_size', 32)
    num_epochs = epochs

    print(f"Hyperparameters: lr={learning_rate}, weight_decay={weight_decay}, hidden_dim={hidden_dim}, dropout={dropout}, batch_size={batch_size}, epochs={num_epochs}, patience={patience}, min_delta={min_delta}")
    print("Starting training...")

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        np.random.shuffle(train_users)
        for i in range(0, len(train_users), batch_size):
            batch_users = train_users[i:i+batch_size]

            optimizer.zero_grad()
            batch_loss = 0
            batch_count = 0

            for user in batch_users:
                if user in ego_networks:
                    try:
                        data = ego_networks[user].to(device)
                        logits = model(data)

                        target = torch.LongTensor([user_labels[user]]).to(device)
                        loss = criterion(logits, target)

                        batch_loss += loss
                        batch_count += 1

                    except Exception as e:
                        continue

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1

        model.eval()
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for user in val_users:
                if user in ego_networks and user in user_labels:
                    try:
                        data = ego_networks[user].to(device)
                        logits = model(data)
                        pred = torch.argmax(logits, dim=1).cpu().item()

                        val_predictions.append(pred)
                        val_true_labels.append(user_labels[user])

                    except Exception as e:
                        continue

        if len(val_predictions) > 0 and len(set(val_true_labels)) > 1:
            val_f1 = f1_score(val_true_labels, val_predictions, zero_division=0)

            # Use min_delta for early stopping (consistent with RoBERTa)
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
                print(f"Epoch {epoch}: Loss={total_loss/max(num_batches,1):.4f}, Val F1={val_f1:.4f} ✓ (best)")
            else:
                epochs_without_improvement += 1
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={total_loss/max(num_batches,1):.4f}, Val F1={val_f1:.4f} (no improvement: {epochs_without_improvement}/{patience})")

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        else:
            if epoch % 10 == 0 and num_batches > 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation F1={best_val_f1:.4f}")

    print("Evaluating...")
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    print(f"Test users: {len(test_users)}")
    print(f"Users in ego_networks: {len([u for u in test_users if u in ego_networks])}")
    print(f"Users in user_labels: {len([u for u in test_users if u in user_labels])}")

    with torch.no_grad():
        for user in test_users:
            if user in ego_networks and user in user_labels:
                try:
                    data = ego_networks[user].to(device)
                    logits = model(data)
                    probs = F.softmax(logits, dim=1)

                    pred = torch.argmax(logits, dim=1).cpu().item()
                    prob = probs[0, 1].cpu().item()

                    predictions.append(pred)
                    probabilities.append(prob)
                    true_labels.append(user_labels[user])

                except Exception as e:
                    print(f"Error evaluating user {user}: {e}")
                    continue

    print(f"Evaluation results: {len(predictions)} predictions, {len(true_labels)} true labels")
    if len(predictions) > 0:
        print(f"Label distribution: {set(true_labels)}")

    if len(predictions) > 0 and len(set(true_labels)) > 1:
        base_metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'auc': roc_auc_score(true_labels, probabilities)
        }

        metrics = base_metrics

        results = {
            'correct_semantic_gnn': {
                'metrics': metrics,
                'feature_dimensions': input_dim,
                'architecture': {
                    'num_gat_layers': 3,
                    'hidden_dim': hidden_dim,
                    'output_dim': 64,
                    'attention_heads': 4,
                    'temporal_attention': True,
                    'temporal_beta': 0.3,
                    'dropout': dropout,
                    'gnn_type': '3-layer GAT with temporal attention',
                    'network_construction': f'k={k_hops} hop expansion with τ={threshold}'
                },
                'network_stats': {
                    'total_networks': len(ego_networks),
                    'avg_network_size': np.mean([net.num_nodes for net in ego_networks.values()]),
                    'similarity_threshold': threshold,
                    'max_neighbors': k_neighbors,
                    'min_neighbors': 5,
                    'k_hops': k_hops,
                    'similarity_components': ['linguistic (64-dim SBERT+LIWC)', 'temporal (9-dim CSV)', 'psychological (LIWC)'],
                    'weights': {'alpha': similarity_weights[0], 'beta': similarity_weights[1], 'gamma': similarity_weights[2]}
                },
                'training_stats': {
                    'best_val_f1': best_val_f1,
                    'test_f1': metrics['f1'],
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'epochs_trained': epoch + 1,
                    'max_epochs': num_epochs,
                    'early_stopping_patience': patience,
                    'min_delta': min_delta,
                    'train_users': len(train_users),
                    'val_users': len(val_users),
                    'test_users': len(test_users)
                },
                'data_source': 'RMHD with precomputed LIWC and temporal features',
                'training_samples': len(train_users)
            }
        }

        print(f"\n=== STEMS-GNN Results ===")
        print(f"Feature Dimensions: {input_dim} (64 semantic + 62 LIWC + 9 temporal)")
        print(f"\nArchitecture:")
        print(f"• 3 GAT layers with 4 attention heads")
        print(f"• Hidden dim: {hidden_dim}, Output dim: 64")
        print(f"• Temporal attention mechanism: β=0.3")
        print(f"• Dropout: {dropout}")
        print(f"\nNetwork Construction:")
        print(f"• Similarity threshold τ={threshold}")
        print(f"• Max neighbors: {k_neighbors}, Min neighbors: 5")
        print(f"• k={k_hops} hop neighborhood expansion")
        print(f"\nSimilarity Weights:")
        print(f"• α={similarity_weights[0]} (linguistic)")
        print(f"• β={similarity_weights[1]} (temporal)")
        print(f"• γ={similarity_weights[2]} (psychological)")
        print(f"\nTraining:")
        print(f"• Learning rate: {learning_rate}")
        print(f"• Weight decay (L2): {weight_decay}")
        print(f"• Batch size: {batch_size}")
        print(f"• Epochs: {epoch + 1}/{num_epochs}")
        print(f"• Early stopping patience: {patience}")
        print(f"\nResults:")
        print(f"• Accuracy:  {metrics['accuracy']:.4f}")
        print(f"• Precision: {metrics['precision']:.4f}")
        print(f"• Recall:    {metrics['recall']:.4f}")
        print(f"• F1-Score:  {metrics['f1']:.4f}")
        print(f"• AUC-ROC:   {metrics['auc']:.4f}")
        print(f"\nDataset:")
        print(f"• Ego-networks built: {len(ego_networks)}")
        print(f"• Avg network size: {np.mean([net.num_nodes for net in ego_networks.values()]):.1f} nodes")
        print(f"• Train/Val/Test: {len(train_users)}/{len(val_users)}/{len(test_users)}")

        # Save model checkpoint if requested
        if save_model and results_saver is not None:
            model_metadata = {
                'similarity_weights': {'alpha': similarity_weights[0], 'beta': similarity_weights[1], 'gamma': similarity_weights[2]},
                'architecture': results['correct_semantic_gnn']['architecture'],
                'training_stats': results['correct_semantic_gnn']['training_stats'],
                'metrics': metrics
            }
            results_saver.save_model_checkpoint(model, 'semantic_ego_gnn', model_metadata)

        if return_predictions:
            return results, (true_labels, probabilities)
        else:
            return results
    else:
        if return_predictions:
            return {'error': 'No valid predictions'}, ([], [])
        else:
            return {'error': 'No valid predictions'}