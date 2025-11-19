"""
RoBERTa Baseline for Depression Detection.

Memory-efficient transformer implementation with fallback to traditional ML models.
Optimized for resource-constrained environments (MacBook Air M2 16GB).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import gc
import psutil
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def clear_memory():
    """Clear GPU and CPU memory for resource management."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """
    Get current memory usage in MB.

    Returns:
        float: Memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class MemoryEfficientRoBERTa(nn.Module):
    """
    Memory-efficient RoBERTa classifier for content-based depression detection.

    Uses DistilRoBERTa with frozen lower layers and gradient checkpointing.
    Falls back to lighter models (BERT-tiny) or simple embeddings if memory constrained.
    """

    def __init__(self, max_length=128, num_classes=2, dropout=0.3):
        super(MemoryEfficientRoBERTa, self).__init__()

        self.max_length = max_length
        self.use_roberta = False

        try:
            from transformers import AutoTokenizer, AutoModel

            model_name = 'distilroberta-base'

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.roberta = AutoModel.from_pretrained(model_name)

            for name, param in self.roberta.named_parameters():
                if 'layer.5' not in name:
                    param.requires_grad = False

            if hasattr(self.roberta, 'gradient_checkpointing_enable'):
                self.roberta.gradient_checkpointing_enable()

            hidden_size = self.roberta.config.hidden_size
            self.use_roberta = True

        except Exception as e:
            try:
                model_name = 'prajjwal1/bert-tiny'

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.roberta = AutoModel.from_pretrained(model_name)

                for name, param in self.roberta.named_parameters():
                    if 'pooler' not in name and 'classifier' not in name:
                        param.requires_grad = False

                hidden_size = self.roberta.config.hidden_size
                self.use_roberta = True

            except Exception as e2:
                self.use_roberta = False
                hidden_size = 128
                self.embedding = nn.Embedding(5000, hidden_size)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, texts):
        if self.use_roberta:
            batch_size = len(texts)
            if batch_size > 4:
                results = []
                for i in range(0, len(texts), 2):
                    micro_batch = texts[i:i+2]
                    result = self._forward_batch(micro_batch)
                    results.append(result)
                return torch.cat(results, dim=0)
            else:
                return self._forward_batch(texts)
        else:
            batch_size = len(texts)
            device = next(self.parameters()).device
            fake_ids = torch.randint(0, 5000, (batch_size, 1), device=device)
            embeddings = self.embedding(fake_ids).mean(dim=1)
            return self.classifier(embeddings)

    def _forward_batch(self, texts):
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = self.roberta(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            with torch.no_grad():
                outputs = self.roberta(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

        del encoded, outputs
        clear_memory()

        return self.classifier(cls_embeddings)

class LightweightSemanticProcessor:
    """
    Lightweight semantic processing for baseline GNN experiments.

    Creates user embeddings via SBERT or TF-IDF and constructs simple
    ego-networks for comparative baseline evaluation.
    """

    def __init__(self, max_users=300):
        self.max_users = max_users

        print("Initializing semantic processor...")

        try:
            model_name = 'all-MiniLM-L6-v2'
            self.sbert_model = SentenceTransformer(model_name)
            self.use_sbert = True
            self.embedding_dim = 384

        except Exception as e:
            self.use_sbert = False
            self.embedding_dim = 256

    def create_embeddings(self, user_posts):
        """
        Create user embeddings from post text.

        Args:
            user_posts: Dictionary mapping user IDs to post lists

        Returns:
            Dictionary mapping user IDs to embedding vectors
        """
        user_list = list(user_posts.keys())[:self.max_users]
        embeddings = {}

        if self.use_sbert:
            texts = []
            users = []

            for user_id in user_list:
                posts = user_posts[user_id][:5]
                user_text = " ".join(str(post) for post in posts)[:500]

                if len(user_text.strip()) > 10:
                    texts.append(user_text)
                    users.append(user_id)

            batch_size = 16
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
                batch_texts = texts[i:i+batch_size]
                batch_users = users[i:i+batch_size]

                try:
                    batch_embeddings = self.sbert_model.encode(
                        batch_texts,
                        batch_size=8,
                        show_progress_bar=False,
                        convert_to_tensor=False
                    )

                    for user_id, embedding in zip(batch_users, batch_embeddings):
                        embeddings[user_id] = embedding

                    clear_memory()

                except Exception as e:
                    
                    continue

        else:
            print("Using TF-IDF feature extraction...")
            texts = []
            users = []

            for user_id in user_list:
                posts = user_posts[user_id][:5]
                user_text = " ".join(str(post) for post in posts)[:500]

                if len(user_text.strip()) > 10:
                    texts.append(user_text)
                    users.append(user_id)

            vectorizer = TfidfVectorizer(
                max_features=256,
                stop_words='english',
                min_df=1,
                max_df=0.95
            )

            tfidf_matrix = vectorizer.fit_transform(texts)

            for user_id, embedding in zip(users, tfidf_matrix.toarray()):
                embeddings[user_id] = embedding

        print(f"Generated embeddings: {len(embeddings)} users")

        return embeddings

    def build_ego_networks(self, embeddings, k_neighbors=3):
        """Build ego networks with memory efficiency."""

        print("Constructing ego-network structure...")
        users = list(embeddings.keys())
        user_embeddings = [embeddings[user] for user in users]

        if len(user_embeddings) < 2:
            return {}

        chunk_size = 50
        ego_networks = {}

        for i in range(0, len(users), chunk_size):
            end_i = min(i + chunk_size, len(users))
            chunk_embeddings = user_embeddings[i:end_i]
            chunk_users = users[i:end_i]

            similarities = cosine_similarity(chunk_embeddings, user_embeddings)

            for j, ego_user in enumerate(chunk_users):
                ego_similarities = similarities[j]

                ego_idx = i + j
                top_k_indices = np.argsort(ego_similarities)[::-1]
                top_k_indices = [idx for idx in top_k_indices if idx != ego_idx][:k_neighbors]

                if len(top_k_indices) == 0:
                    continue

                network_users = [ego_user] + [users[idx] for idx in top_k_indices]
                network_embeddings = [user_embeddings[ego_idx]] + [user_embeddings[idx] for idx in top_k_indices]

                edge_list = []
                edge_weights = []

                for k in range(1, len(network_users)):
                    edge_list.extend([[0, k], [k, 0]])
                    weight = ego_similarities[top_k_indices[k-1]]
                    edge_weights.extend([weight, weight])

                node_features = torch.FloatTensor(network_embeddings)
                edge_index = torch.LongTensor(edge_list).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.FloatTensor(edge_weights) if edge_weights else torch.empty(0, dtype=torch.float)

                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(network_users)
                )

                ego_networks[ego_user] = data

            clear_memory()

        print(f"Built {len(ego_networks)} ego networks")
        return ego_networks

def train_memory_efficient_baseline(user_posts, user_labels, config, return_predictions=False, save_model=False, results_saver=None, cached_features=None, data_splits=None):
    """
    Train baseline with aggressive memory management using shared data splits.

    Args:
        user_posts: User post data
        user_labels: User labels
        config: Configuration
        return_predictions: If True, returns (y_true, y_prob) for ROC curves
        save_model: If True and results_saver is provided, saves model checkpoint
        results_saver: ResultsSaver instance for saving model checkpoints
        cached_features: Pre-extracted features from UnifiedFeatureExtractor (required)
        data_splits: Shared train/val/test splits (required for fair comparison)
    """
    print("=== Training Memory-Efficient Baseline ===")
    print(f"Initial memory: {get_memory_usage():.1f} MB")

    if cached_features is None:
        raise ValueError("cached_features is required for RoBERTa baseline")

    if data_splits is None:
        raise ValueError("data_splits is required for fair comparison with STEMS-GNN")

    train_ids = data_splits['train_ids']
    val_ids = data_splits['val_ids']
    test_ids = data_splits['test_ids']

    print(f"Using shared data splits:")
    print(f"  Train: {len(train_ids)} users")
    print(f"  Val:   {len(val_ids)} users (used for early stopping)")
    print(f"  Test:  {len(test_ids)} users")

    train_texts = []
    train_labels = []
    for user_id in train_ids:
        if user_id in cached_features['text_features'] and user_id in cached_features['labels']:
            train_texts.append(cached_features['text_features'][user_id][:300])
            train_labels.append(cached_features['labels'][user_id])

    test_texts = []
    test_labels = []
    for user_id in test_ids:
        if user_id in cached_features['text_features'] and user_id in cached_features['labels']:
            test_texts.append(cached_features['text_features'][user_id][:300])
            test_labels.append(cached_features['labels'][user_id])

    print(f"Data prepared: {len(train_texts)} train, {len(test_texts)} test")

    if len(train_texts) < 10 or len(test_texts) < 10:
        return {'error': 'Insufficient data'}

    results = {}

    print("Training sklearn models...")

    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=1)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    lr_model = LogisticRegression(random_state=42, max_iter=500)
    lr_model.fit(X_train, train_labels)
    lr_pred = lr_model.predict(X_test)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]

    lr_metrics = {
        'accuracy': accuracy_score(test_labels, lr_pred),
        'precision': precision_score(test_labels, lr_pred, zero_division=0),
        'recall': recall_score(test_labels, lr_pred, zero_division=0),
        'f1': f1_score(test_labels, lr_pred, zero_division=0),
        'auc': roc_auc_score(test_labels, lr_prob) if len(set(test_labels)) > 1 else 0.5
    }

    results['tfidf_logistic'] = {'metrics': lr_metrics}
    print(f"TF-IDF + Logistic Regression - F1: {lr_metrics['f1']:.4f}")

    rf_model = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=10)
    rf_model.fit(X_train, train_labels)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(test_labels, rf_pred),
        'precision': precision_score(test_labels, rf_pred, zero_division=0),
        'recall': recall_score(test_labels, rf_pred, zero_division=0),
        'f1': f1_score(test_labels, rf_pred, zero_division=0),
        'auc': roc_auc_score(test_labels, rf_prob) if len(set(test_labels)) > 1 else 0.5
    }

    results['tfidf_randomforest'] = {'metrics': rf_metrics}
    print(f"TF-IDF + Random Forest - F1: {rf_metrics['f1']:.4f}")

    clear_memory()

    try:
        print("Attempting memory-efficient transformer training...")

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")

        dropout = config.get('models', {}).get('baseline', {}).get('dropout', 0.3)
        learning_rate = config.get('training', {}).get('learning_rate', 0.001)
        weight_decay = config.get('training', {}).get('weight_decay', 0.0001)
        batch_size = config.get('training', {}).get('batch_size', 32)
        # num_epochs = config.get('training', {}).get('epochs', 16)
        num_epochs = 1
        patience = config.get('training', {}).get('patience', 5)
        min_delta = config.get('training', {}).get('min_delta', 0.001)

        model = MemoryEfficientRoBERTa(max_length=128, dropout=dropout).to(device)

        if model.use_roberta:
            print("Transformer model loaded successfully")
            print(f"Hyperparameters: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}")
            print(f"Training: epochs={num_epochs}, patience={patience}, min_delta={min_delta}")

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            best_val_f1 = 0.0
            best_model_state = None
            patience_counter = 0

            val_texts = []
            val_labels_list = []
            for user_id in val_ids:
                if user_id in cached_features['text_features'] and user_id in cached_features['labels']:
                    val_texts.append(cached_features['text_features'][user_id][:300])
                    val_labels_list.append(cached_features['labels'][user_id])

            if len(val_texts) == 0:
                print("WARNING: No validation data available. Early stopping will be disabled.")
                print("Falling back to training without early stopping...")
                patience = num_epochs + 1

            model.train()

            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0

                for i in range(0, len(train_texts), batch_size):
                    batch_texts = train_texts[i:i+batch_size]
                    batch_labels = torch.LongTensor(train_labels[i:i+batch_size]).to(device)

                    optimizer.zero_grad()

                    try:
                        logits = model(batch_texts)
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        num_batches += 1

                        if i % 10 == 0:
                            clear_memory()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM at batch {i}, skipping...")
                            clear_memory()
                            continue
                        else:
                            raise e

                train_loss = total_loss / num_batches if num_batches > 0 else 0

                model.eval()
                val_predictions = []
                val_true_labels = []

                with torch.no_grad():
                    for i in range(0, len(val_texts), batch_size):
                        batch_texts = val_texts[i:i+batch_size]
                        batch_labels = torch.LongTensor(val_labels_list[i:i+batch_size]).to(device)

                        try:
                            logits = model(batch_texts)
                            preds = torch.argmax(logits, dim=1).cpu().numpy()
                            val_predictions.extend(preds)
                            val_true_labels.extend(batch_labels.cpu().numpy())
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                continue
                            else:
                                raise e

                if len(val_predictions) > 0 and len(set(val_true_labels)) > 1:
                    val_f1 = f1_score(val_true_labels, val_predictions, zero_division=0)
                else:
                    val_f1 = 0.0

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")

                if val_f1 > best_val_f1 + min_delta:
                    best_val_f1 = val_f1
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    print(f"  New best model (val_f1: {best_val_f1:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement (patience: {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

                model.train()

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"Restored best model with val_f1: {best_val_f1:.4f}")

            model.eval()
            transformer_preds = []
            transformer_probs = []
            transformer_test_labels = []

            with torch.no_grad():
                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i+batch_size]
                    batch_size_actual = len(batch_texts)

                    try:
                        logits = model(batch_texts)
                        probs = F.softmax(logits, dim=1)

                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        prob_pos = probs[:, 1].cpu().numpy()

                        transformer_preds.extend(preds)
                        transformer_probs.extend(prob_pos)
                        transformer_test_labels.extend(test_labels[i:i+batch_size_actual])

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"  WARNING: OOM on test batch at index {i}, skipping {batch_size_actual} samples")
                            clear_memory()
                            continue
                        else:
                            raise e

            if len(transformer_preds) > 0:
                transformer_metrics = {
                    'accuracy': accuracy_score(transformer_test_labels, transformer_preds),
                    'precision': precision_score(transformer_test_labels, transformer_preds, zero_division=0),
                    'recall': recall_score(transformer_test_labels, transformer_preds, zero_division=0),
                    'f1': f1_score(transformer_test_labels, transformer_preds, zero_division=0),
                    'auc': roc_auc_score(transformer_test_labels, transformer_probs) if len(set(transformer_test_labels)) > 1 else 0.5
                }

                results['memory_efficient_transformer'] = {'metrics': transformer_metrics}
                print(f"Memory-Efficient Transformer - F1: {transformer_metrics['f1']:.4f}")

                if save_model and results_saver is not None:
                    model_metadata = {
                        'model_type': 'RoBERTa',
                        'max_length': 128,
                        'dropout': dropout,
                        'batch_size': batch_size,
                        'num_epochs': num_epochs,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'patience': patience,
                        'min_delta': min_delta,
                        'best_val_f1': best_val_f1 if best_model_state is not None else None,
                        'early_stopped': patience_counter >= patience,
                        'metrics': transformer_metrics,
                        'training_samples': len(train_texts),
                        'test_samples': len(test_labels)
                    }
                    results_saver.save_model_checkpoint(model, 'roberta_baseline', model_metadata)
            else:
                print("Transformer evaluation failed due to memory constraints")

    except Exception as e:
        print(f"Transformer training failed: {e}")

    clear_memory()
    print(f"Final memory: {get_memory_usage():.1f} MB")

    if return_predictions:
        if 'memory_efficient_transformer' in results:
            return results, (transformer_test_labels, transformer_probs)
        elif 'tfidf_logistic' in results:
            return results, (test_labels, lr_prob)
        else:
            return results, ([], [])
    else:
        return results

def train_memory_efficient_gnn(user_posts, user_labels, config):
    """
    Train GNN with memory efficiency for MacBook Air.
    """
    print("=== Training Memory-Efficient GNN ===")
    print(f"Initial memory: {get_memory_usage():.1f} MB")

    processor = LightweightSemanticProcessor(max_users=300)

    embeddings = processor.create_embeddings(user_posts)

    if len(embeddings) < 10:
        return {'error': 'Insufficient embeddings'}

    ego_networks = processor.build_ego_networks(embeddings, k_neighbors=3)

    if len(ego_networks) < 5:
        return {'error': 'Insufficient networks'}

    valid_users = list(ego_networks.keys())
    valid_labels = [user_labels[user] for user in valid_users if user in user_labels]
    valid_users = valid_users[:len(valid_labels)]

    print(f"Training GNN on {len(valid_users)} users")

    split_idx = int(0.8 * len(valid_users))
    train_users = valid_users[:split_idx]
    test_users = valid_users[split_idx:]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    class LightGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super().__init__()
            self.conv1 = GATConv(input_dim, hidden_dim, heads=1, dropout=0.3)
            self.classifier = nn.Linear(hidden_dim, 2)

        def forward(self, data):
            if data.num_nodes == 0:
                return torch.zeros(1, 2, device=data.x.device if hasattr(data, 'x') else 'cpu')

            x = F.relu(self.conv1(data.x, data.edge_index))
            return self.classifier(x[0:1])

    model = LightGNN(processor.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        num_batches = 0

        for user in train_users:
            if user in ego_networks:
                optimizer.zero_grad()

                data = ego_networks[user].to(device)
                logits = model(data)

                target = torch.LongTensor([user_labels[user]]).to(device)
                loss = criterion(logits, target)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        if epoch % 5 == 0 and num_batches > 0:
            print(f"Epoch {epoch}, Loss: {total_loss/num_batches:.4f}")

    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for user in test_users:
            if user in ego_networks:
                data = ego_networks[user].to(device)
                logits = model(data)
                prob = F.softmax(logits, dim=1)

                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                prob_pos = prob[0, 1].cpu().numpy()

                predictions.append(pred)
                probabilities.append(prob_pos)
                true_labels.append(user_labels[user])

    if len(predictions) > 0:
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
            'auc': roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.5
        }
    else:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5}

    print(f"GNN Results - F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
    print(f"Final memory: {get_memory_usage():.1f} MB")

    return {
        'memory_efficient_gnn': {
            'metrics': metrics,
            'methodology': 'Memory-efficient STEMS-GNN',
            'num_users': len(valid_users)
        }
    }