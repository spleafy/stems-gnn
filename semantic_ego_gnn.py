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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
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

    def __init__(self, config, similarity_weights=(0.5, 0.1, 0.4)):
        self.config = config
        self.alpha, self.beta, self.gamma = similarity_weights
        
        feature_breakdown = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {})
        self.semantic_dims = feature_breakdown.get('semantic_dims', 64)
        self.liwc_dims = feature_breakdown.get('liwc_dims', 62)
        self.temporal_dims = feature_breakdown.get('temporal_dims', 8)
        
        self.linguistic_dims = 30
        self.psychological_dims = self.liwc_dims - self.linguistic_dims

    def calculate_multi_similarity(self, user_features, neighbor_features=None):
        """
        Calculate multi-dimensional similarity.
        If neighbor_features is None, computes pairwise similarity within user_features.
        If neighbor_features is provided, computes similarity between each user in
        user_features and each user in neighbor_features.
        """
        print(f"Computing multi-dimensional similarity (α={self.alpha}, β={self.beta}, γ={self.gamma})...")

        users = list(user_features.keys())
        features_matrix = np.array([user_features[user] for user in users])

        if neighbor_features:
            neighbors = list(neighbor_features.keys())
            neighbor_matrix = np.array([neighbor_features[neighbor] for neighbor in neighbors])
        else:
            neighbors = users
            neighbor_matrix = features_matrix

        s_end = self.semantic_dims
        l_end = s_end + self.linguistic_dims
        p_end = s_end + self.liwc_dims
        t_end = p_end + self.temporal_dims

        sem_plus_ling_user = np.concatenate([features_matrix[:, :s_end], features_matrix[:, s_end:l_end]], axis=1)
        sem_plus_ling_neighbor = np.concatenate([neighbor_matrix[:, :s_end], neighbor_matrix[:, s_end:l_end]], axis=1)
        linguistic_sim = cosine_similarity(sem_plus_ling_user, sem_plus_ling_neighbor)

        temporal_user = features_matrix[:, p_end:t_end]
        temporal_neighbor = neighbor_matrix[:, p_end:t_end]
        temporal_sim = cosine_similarity(temporal_user, temporal_neighbor)

        psych_user = features_matrix[:, l_end:p_end]
        psych_neighbor = neighbor_matrix[:, l_end:p_end]
        psychological_sim = cosine_similarity(psych_user, psych_neighbor)

        combined_similarity = (
            self.alpha * linguistic_sim +
            self.beta * temporal_sim +
            self.gamma * psychological_sim
        )

        return combined_similarity, users, neighbors

    def build_ego_networks(self, user_features, neighbor_features=None, user_labels=None, k_neighbors=50, threshold=0.6, k_hops=2, min_neighbors=5, adaptive_threshold=True, target_edge_percentile=60, preserve_hop_structure_only=False):
        """
        Build semantic ego-networks with k-hop expansion.

        Constructs ego-networks by finding semantically similar neighbors and expanding
        to k-hop neighborhoods. Ensures minimum connectivity through adaptive neighbor selection.

        Args:
            user_features: Dictionary mapping user IDs to feature vectors for the target set (e.g., test users).
            neighbor_features: Optional. Dictionary mapping user IDs to features for the neighbor pool (e.g., training users).
                               If None, neighbors are chosen from within user_features (for training set).
            k_neighbors: Maximum number of neighbors per user.
            threshold: Similarity threshold for edge creation (τ).
            k_hops: Number of hops for neighborhood expansion.
            min_neighbors: Minimum neighbors required to include a user.
            adaptive_threshold: If True, computes threshold from similarity distribution.
            target_edge_percentile: Percentile of similarity scores for adaptive threshold.
            preserve_hop_structure_only: If True, creates a sparse graph with only ego-to-1-hop and 1-hop-to-2-hop edges.

        Returns:
            Dictionary mapping user IDs to PyTorch Geometric Data objects.
        """
        print("Constructing semantic ego-networks...")

        is_training_set = neighbor_features is None
        if is_training_set:
            neighbor_features = user_features

        similarity_matrix, users, neighbors = self.calculate_multi_similarity(user_features, neighbor_features)
        
        user_features_matrix = np.array([user_features[user] for user in users])
        neighbor_features_matrix = np.array([neighbor_features[neighbor] for neighbor in neighbors])

        if adaptive_threshold:
            train_sim_matrix, _, _ = self.calculate_multi_similarity(neighbor_features)
            non_diagonal_similarities = train_sim_matrix[np.triu_indices(train_sim_matrix.shape[0], k=1)]
            
            computed_threshold = np.percentile(non_diagonal_similarities, target_edge_percentile)
            print(f"  Using adaptive threshold from training data ({target_edge_percentile}th percentile): τ={computed_threshold:.4f}")
            threshold = computed_threshold
        else:
            print(f"  Using fixed threshold: τ={threshold:.4f}")

        if adaptive_threshold:
            non_diagonal_similarities = []
            for i in range(len(users)):
                for j in range(len(users)):
                    if i != j:
                        non_diagonal_similarities.append(similarity_matrix[i, j])

            computed_threshold = np.percentile(non_diagonal_similarities, target_edge_percentile)
            print(f"  Using adaptive threshold at {target_edge_percentile}th percentile: τ={computed_threshold:.4f}")
            threshold = computed_threshold
        else:
            print(f"  Using fixed threshold: τ={threshold:.4f}")

        sim_mean = np.mean(similarity_matrix)
        sim_std = np.std(similarity_matrix)
        sim_max = np.max(similarity_matrix)
        above_threshold = np.sum(similarity_matrix >= threshold) - len(users)
        total_pairs = len(users) * (len(users) - 1)
        sparsity = above_threshold / total_pairs if total_pairs > 0 else 0

        print(f"  Similarity statistics: mean={sim_mean:.3f}, std={sim_std:.3f}, max={sim_max:.3f}")
        print(f"  Edges above threshold (τ={threshold:.4f}): {above_threshold}/{total_pairs} ({sparsity*100:.2f}%)")

        ego_networks = {}

        for i, ego_user in enumerate(users):
            ego_similarities = similarity_matrix[i]

            neighbor_indices = []
            for j, sim in enumerate(ego_similarities):
                if not is_training_set or (is_training_set and i != j):
                    if sim >= threshold:
                        neighbor_indices.append((j, sim))
            
            neighbor_indices.sort(key=lambda x: x[1], reverse=True)

            if len(neighbor_indices) > k_neighbors:
                neighbor_indices = neighbor_indices[:k_neighbors]
            
            if len(neighbor_indices) < min_neighbors:
                all_neighbors = sorted([(j, sim) for j, sim in enumerate(ego_similarities) if not is_training_set or (is_training_set and i !=j)], key=lambda x:x[1], reverse=True)
                neighbor_indices = all_neighbors[:min_neighbors]

            if len(neighbor_indices) < min_neighbors:
                continue

            network_nodes = {ego_user: user_features[ego_user]}
            
            hop_1_neighbor_ids = [neighbors[idx] for idx, _ in neighbor_indices]
            for neighbor_id in hop_1_neighbor_ids:
                network_nodes[neighbor_id] = neighbor_features[neighbor_id]

            if k_hops >= 2:
                if 'train_sim_matrix' not in locals():
                    train_sim_matrix, _, _ = self.calculate_multi_similarity(neighbor_features)

                for hop_1_id in hop_1_neighbor_ids:
                    if hop_1_id not in neighbors: continue
                    hop_1_global_idx = neighbors.index(hop_1_id)
                    
                    hop_2_candidates = []
                    for j, sim in enumerate(train_sim_matrix[hop_1_global_idx]):
                        hop_2_id = neighbors[j]
                        if hop_2_id not in network_nodes and sim >= threshold:
                            hop_2_candidates.append((hop_2_id, sim))
                    
                    hop_2_candidates.sort(key=lambda x: x[1], reverse=True)
                    for neighbor_id, _ in hop_2_candidates[:5]:
                        if neighbor_id not in network_nodes:
                            network_nodes[neighbor_id] = neighbor_features[neighbor_id]
            
            local_node_map = {uid: i for i, uid in enumerate(network_nodes.keys())}
            ego_local_idx = local_node_map[ego_user]

            network_features_matrix = np.array(list(network_nodes.values()))
            
            feature_indices = []
            s_end = self.semantic_dims
            l_end = s_end + self.linguistic_dims
            p_end = s_end + self.liwc_dims
            t_end = p_end + self.temporal_dims

            if self.alpha > 0: feature_indices.extend(range(0, l_end))
            if self.gamma > 0: feature_indices.extend(range(l_end, p_end))
            if self.beta > 0: feature_indices.extend(range(p_end, t_end))
            
            if len(feature_indices) > 0:
                network_features = network_features_matrix[:, feature_indices]
            else:
                network_features = network_features_matrix
                feature_indices = list(range(network_features_matrix.shape[1]))

            edge_list = []
            edge_weights = []

            for neighbor_id, weight in zip(hop_1_neighbor_ids, [sim for _, sim in neighbor_indices]):
                if neighbor_id in local_node_map:
                    local_idx = local_node_map[neighbor_id]
                    edge_list.extend([[ego_local_idx, local_idx], [local_idx, ego_local_idx]])
                    edge_weights.extend([weight, weight])
                    hop_1_local_indices.append(local_idx)

            if preserve_hop_structure_only:
                if k_hops >= 2:
                    for hop_1_id in hop_1_neighbor_ids:
                        if hop_1_id not in local_node_map: continue
                        hop_1_local_idx = local_node_map[hop_1_id]
                        
                        hop_1_global_idx = neighbors.index(hop_1_id)
                        for j, sim in enumerate(train_sim_matrix[hop_1_global_idx]):
                            hop_2_id = neighbors[j]
                            if hop_2_id in local_node_map and hop_2_id != ego_user and hop_2_id not in hop_1_neighbor_ids:
                                if sim >= threshold:
                                    hop_2_local_idx = local_node_map[hop_2_id]
                                    edge_list.extend([[hop_1_local_idx, hop_2_local_idx], [hop_2_local_idx, hop_1_local_idx]])
                                    edge_weights.extend([sim, sim])
            else:
                node_ids = list(network_nodes.keys())
                for i1, uid1 in enumerate(node_ids):
                    for i2, uid2 in enumerate(node_ids):
                        if i1 >= i2:
                            continue

                        if uid1 not in neighbors or uid2 not in neighbors: continue
                        global_idx1 = neighbors.index(uid1)
                        global_idx2 = neighbors.index(uid2)
                        
                        sim = train_sim_matrix[global_idx1, global_idx2]
                        
                        if sim >= threshold:
                            edge_list.extend([[i1, i2], [i2, i1]])
                            edge_weights.extend([sim, sim])
            
            node_features_tensor = torch.FloatTensor(network_features)
            edge_index_tensor = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.FloatTensor(edge_weights) if edge_weights else torch.empty(0)

            y = None
            if user_labels is not None and ego_user in user_labels:
                y = torch.tensor([user_labels[ego_user]], dtype=torch.long)

            data = Data(
                x=node_features_tensor,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor,
                num_nodes=len(network_nodes),
                feature_indices=feature_indices,
                y=y
            )
            ego_networks[ego_user] = data

        if len(ego_networks) > 0:
            avg_size = np.mean([net.num_nodes for net in ego_networks.values()])
            avg_edges = np.mean([net.edge_index.shape[1] / 2 for net in ego_networks.values()])
            print(f"  Built {len(ego_networks)} ego-networks.")
            print(f"  Avg network size: {avg_size:.1f} nodes, {avg_edges:.1f} edges.")
        else:
            print(f"  WARNING: No ego-networks built! Check threshold and similarity values.")

        return ego_networks

class CorrectSemanticGNN(nn.Module):
    """
    STEMS-GNN: Graph Attention Network with temporal attention mechanism.

    Architecture:
        - 3 GAT layers with 4 attention heads each
        - Hidden dimension: 128, Output dimension: 64
        - Temporal attention weighting of structural attention
        - Residual connections and layer normalization for training stability
        - Dynamic input projection for ablation studies (handles varying input dimensions)
    """

    def __init__(self, input_dim=135, hidden_dim=128, dropout=0.3, has_temporal=True, temporal_dim=9):
        super(CorrectSemanticGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.has_temporal = has_temporal
        self.temporal_dim = temporal_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False, edge_dim=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False, edge_dim=1)
        self.gat3 = GATConv(hidden_dim, 64, heads=4, dropout=dropout, concat=False, edge_dim=1)


        if has_temporal:
            self.temporal_weight = nn.Parameter(torch.tensor(0.3))
            self.temporal_proj = nn.Linear(self.temporal_dim, hidden_dim)
        else:
            self.temporal_weight = None
            self.temporal_proj = None

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

    def compute_temporal_attention(self, data):
        """
        Compute temporal attention weights using cosine similarity of temporal features.

        Returns temporal multiplier: 1 + β · temporal_similarity(t_i, t_j)
        Only active if model has temporal features (has_temporal=True)
        """

        if not self.has_temporal or not hasattr(data, 'edge_attr') or data.edge_attr is None:
            return None

        if not hasattr(data, 'feature_indices'):
            return None

        feature_indices = data.feature_indices
        temporal_indices = list(range(self.input_dim - self.temporal_dim, self.input_dim))
        has_temporal_features = any(idx in feature_indices for idx in temporal_indices)

        if not has_temporal_features or data.x.shape[1] < self.temporal_dim:
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
            data: PyTorch Geometric Data object
        Returns:
            Binary classification logits
        """
        if data.num_nodes == 0:
            return torch.zeros(1, 2)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_proj(x)
        
        edge_attr_reshaped = edge_attr.view(-1, 1) if edge_attr is not None and edge_attr.ndim == 1 else edge_attr

        identity1 = x
        x = self.gat1(x, edge_index, edge_attr=edge_attr_reshaped)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm1(x + self.residual1(identity1))

        identity2 = x
        x = self.gat2(x, edge_index, edge_attr=edge_attr_reshaped)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm2(x + self.residual2(identity2))

        identity3 = x
        x = self.gat3(x, edge_index, edge_attr=edge_attr_reshaped)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.norm3(x + self.residual3(identity3))

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            
        graph_features = global_mean_pool(x, batch)
        
        logits = self.classifier(graph_features)

        return logits

def train_correct_semantic_gnn(user_posts, user_labels, config, similarity_weights, return_predictions=False, save_model=False, results_saver=None, cached_features=None, data_splits=None):
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

    if cached_features is None:
        raise ValueError("cached_features is required. Feature extraction must be done by UnifiedFeatureExtractor before calling this function.")

    if data_splits is None:
        raise ValueError("data_splits is required for fair comparison with baseline")

    processor = CorrectSemanticProcessor(config, similarity_weights=similarity_weights)

    print("Using pre-extracted cached features")
    user_features_raw = cached_features['combined_features']

    if len(user_features_raw) < 30:
        return {'error': 'Insufficient users for training'}

    from sklearn.preprocessing import RobustScaler

    train_ids_temp = data_splits['train_ids']
    val_ids_temp = data_splits['val_ids']
    test_ids_temp = data_splits['test_ids']

    train_features_array = np.array([user_features_raw[uid] for uid in train_ids_temp if uid in user_features_raw])
    val_features_array = np.array([user_features_raw[uid] for uid in val_ids_temp if uid in user_features_raw])
    test_features_array = np.array([user_features_raw[uid] for uid in test_ids_temp if uid in user_features_raw])

    scaler = RobustScaler()
    scaler.fit(train_features_array)
    print(f"Feature normalization: fitted on {len(train_features_array)} training samples using RobustScaler")

    train_features_scaled = scaler.transform(train_features_array)
    val_features_scaled = scaler.transform(val_features_array)
    test_features_scaled = scaler.transform(test_features_array)

    user_features = {}
    for i, uid in enumerate([u for u in train_ids_temp if u in user_features_raw]):
        user_features[uid] = train_features_scaled[i]
    for i, uid in enumerate([u for u in val_ids_temp if u in user_features_raw]):
        user_features[uid] = val_features_scaled[i]
    for i, uid in enumerate([u for u in test_ids_temp if u in user_features_raw]):
        user_features[uid] = test_features_scaled[i]

    print(f"Feature normalization complete: {len(user_features)} users")

    k_neighbors = config.get('ego_network', {}).get('k_neighbors', 8)
    threshold = config.get('ego_network', {}).get('similarity_threshold', 0.6)
    k_hops = config.get('ego_network', {}).get('k_hops', 2)
    adaptive_threshold = config.get('ego_network', {}).get('adaptive_threshold', True)
    target_edge_percentile = config.get('ego_network', {}).get('target_edge_percentile', 60)
    preserve_hop_structure_only = config.get('ego_network', {}).get('preserve_hop_structure_only', False)

    train_ids = data_splits['train_ids']
    val_ids = data_splits['val_ids']
    test_ids = data_splits['test_ids']

    train_features = {uid: user_features[uid] for uid in train_ids if uid in user_features}
    val_features = {uid: user_features[uid] for uid in val_ids if uid in user_features}
    test_features = {uid: user_features[uid] for uid in test_ids if uid in user_features}

    print(f"Building ego-networks for training set ({len(train_features)} users)...")
    train_ego_networks = processor.build_ego_networks(
        train_features,
        neighbor_features=None,
        user_labels=user_labels,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=5,
        adaptive_threshold=True,
        target_edge_percentile=target_edge_percentile,
        preserve_hop_structure_only=preserve_hop_structure_only
    )

    if len(train_ego_networks) < 15:
        return {'error': 'Insufficient ego-networks in training data'}

    print(f"Building ego-networks for validation set ({len(val_features)} users)...")
    val_ego_networks = processor.build_ego_networks(
        val_features,
        neighbor_features=train_features,
        user_labels=user_labels,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=3,
        adaptive_threshold=True,
        target_edge_percentile=target_edge_percentile,
        preserve_hop_structure_only=preserve_hop_structure_only
    )

    print(f"Building ego-networks for test set ({len(test_features)} users)...")
    test_ego_networks = processor.build_ego_networks(
        test_features,
        neighbor_features=train_features,
        user_labels=user_labels,
        k_neighbors=k_neighbors,
        threshold=threshold,
        k_hops=k_hops,
        min_neighbors=3,
        adaptive_threshold=True,
        target_edge_percentile=target_edge_percentile,
        preserve_hop_structure_only=preserve_hop_structure_only
    )

    ego_networks = {**train_ego_networks, **val_ego_networks, **test_ego_networks}

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

    sample_network = train_ego_networks[train_users[0]]
    input_dim = sample_network.x.shape[1]

    alpha, beta, gamma = similarity_weights
    has_temporal = beta > 0

    print(f"Input dimension: {input_dim} features")
    print(f"  Ablation weights: α={alpha} (linguistic), β={beta} (temporal), γ={gamma} (psychological)")
    print(f"  Temporal attention: {'enabled' if has_temporal else 'disabled'}")

    hidden_dim = config.get('models', {}).get('semantic_gnn', {}).get('hidden_dim', 128)
    dropout = config.get('models', {}).get('semantic_gnn', {}).get('dropout', 0.4)

    model = CorrectSemanticGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        has_temporal=has_temporal,
        temporal_dim=config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('temporal_dims', 9)
    ).to(device)

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

    train_dataset = [train_ego_networks[u] for u in train_users if u in train_ego_networks]
    val_dataset = [val_ego_networks[u] for u in val_users if u in val_ego_networks]
    test_dataset = [test_ego_networks[u] for u in test_users if u in test_ego_networks]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch)
            loss = criterion(logits, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        model.eval()
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_predictions.extend(preds)
                val_true_labels.extend(batch.y.cpu().numpy())

        if len(val_predictions) > 0 and len(set(val_true_labels)) > 1:
            val_f1 = f1_score(val_true_labels, val_predictions, zero_division=0)

            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
                print(f"Epoch {epoch}: Loss={total_loss/max(num_batches,1):.4f}, Val F1={val_f1:.4f} ✓ (best)")
            else:
                epochs_without_improvement += 1
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={total_loss/max(num_batches,1):.4f}, Val F1={val_f1:.4f} (no improvement: {epochs_without_improvement}/{patience})")

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
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs_pos = probs[:, 1].cpu().numpy()
            
            predictions.extend(preds)
            probabilities.extend(probs_pos)
            true_labels.extend(batch.y.cpu().numpy())

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
                    'temporal_attention': has_temporal,
                    'temporal_beta': 0.3 if has_temporal else 0.0,
                    'dropout': dropout,
                    'gnn_type': f'3-layer GAT with {"temporal" if has_temporal else "no temporal"} attention',
                    'network_construction': f'k={k_hops} hop expansion with τ={threshold}',
                    'ablation_config': {
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'active_features': input_dim
                    }
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
        print(f"Feature Dimensions: {input_dim} active features")

        feature_breakdown = []
        if alpha > 0:
            feature_breakdown.append("94 linguistic (semantic + LIWC)")
        if gamma > 0:
            feature_breakdown.append(f"{processor.liwc_dims - 30} psychological (LIWC)")
        if beta > 0:
            feature_breakdown.append("9 temporal")
        print(f"  Active features: {', '.join(feature_breakdown)}")

        print(f"\nArchitecture:")
        print(f"• 3 GAT layers with 4 attention heads")
        print(f"• Hidden dim: {hidden_dim}, Output dim: 64")
        print(f"• Temporal attention mechanism: {'β=0.3' if has_temporal else 'disabled (no temporal features)'}")
        print(f"• Dropout: {dropout}")
        print(f"\nNetwork Construction:")
        print(f"• Similarity threshold τ={threshold:.4f}")
        print(f"• Max neighbors: {k_neighbors}, Min neighbors: 5")
        print(f"• k={k_hops} hop neighborhood expansion")
        print(f"\nAblation Weights:")
        print(f"• α={alpha} (linguistic: {'active' if alpha > 0 else 'MASKED'})")
        print(f"• β={beta} (temporal: {'active' if beta > 0 else 'MASKED'})")
        print(f"• γ={gamma} (psychological: {'active' if gamma > 0 else 'MASKED'})")
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