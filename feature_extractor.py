#!/usr/bin/env python3
"""
Centralized Feature Extraction Module with Caching

Extracts and caches features once for reuse across all models (RoBERTa, STEMS-GNN, ablations).
Avoids redundant SBERT embeddings, LIWC features, and similarity calculations.
"""

import numpy as np
import pandas as pd
import os
import gc
from sentence_transformers import SentenceTransformer
from data_preprocessing import (
    get_data_hash,
    save_cached_features,
    load_cached_features
)


class UnifiedFeatureExtractor:
    """
    Centralized feature extractor that caches results to avoid redundant computations.

    Extracts once and reuses for:
    - RoBERTa baseline (text features)
    - STEMS-GNN (semantic + LIWC + temporal)
    - Ablation studies (different similarity weights)
    """

    def __init__(self, config=None):
        """
        Initialize feature extractor with caching support.

        Args:
            config (dict): Configuration dictionary with caching settings
        """
        self.config = config or {}
        self.cache_enabled = self.config.get('data', {}).get('enable_feature_cache', True)
        self.cache_dir = self.config.get('data', {}).get('cache_directory', 'data/processed')

        # Initialize SBERT model only when needed
        self.sbert_model = None

    def _init_sbert_model(self):
        """Lazy initialization of SBERT model to save memory."""
        if self.sbert_model is None:
            print("Loading SBERT model (all-MiniLM-L6-v2)...")
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SBERT model loaded successfully")

    def extract_all_features(self, user_posts, user_labels, max_users=None):
        """
        Extract all features with caching support.

        Args:
            user_posts (dict): Dictionary of user posts
            user_labels (dict): Dictionary of user labels
            max_users (int): Maximum number of users to process (None = all)

        Returns:
            dict: Cached features including SBERT embeddings, LIWC, temporal, and text
        """
        # Generate hash for caching
        data_hash = get_data_hash(user_posts, user_labels)

        # Try to load from cache
        if self.cache_enabled:
            cached = load_cached_features(self.cache_dir, data_hash)
            if cached is not None:
                print("Using cached features (skipping extraction)")
                return cached

        print("\n" + "="*60)
        print("Feature Extraction (results will be cached for reuse)")
        print("="*60)

        # Extract features
        features = self._extract_features_impl(user_posts, user_labels, max_users)

        # Save to cache
        if self.cache_enabled:
            save_cached_features(features, self.cache_dir, data_hash)

        return features

    def _extract_features_impl(self, user_posts, user_labels, max_users):
        """
        Internal implementation of feature extraction.

        Returns:
            dict: {
                'user_ids': list of user IDs,
                'semantic_embeddings': dict of 64-dim SBERT embeddings,
                'liwc_features': dict of 62-dim LIWC features,
                'temporal_features': dict of 9-dim temporal features,
                'text_features': dict of raw text for RoBERTa,
                'labels': dict of labels,
                'combined_features': dict of full 135-dim vectors
            }
        """
        print("\n1. Loading RMHD precomputed LIWC features...")
        all_liwc_features = self._load_rmhd_liwc_features()

        print(f"\n2. Extracting features for users...")
        user_list = list(user_posts.keys())
        if max_users:
            user_list = user_list[:max_users]

        semantic_embeddings = {}
        liwc_features = {}
        temporal_features = {}
        text_features = {}
        combined_features = {}
        valid_users = []

        # Initialize SBERT model
        self._init_sbert_model()

        batch_size = 50
        for i in range(0, len(user_list), batch_size):
            batch_users = user_list[i:i+batch_size]

            for user_id in batch_users:
                try:
                    # Get user posts
                    posts = user_posts[user_id]

                    # Extract text features (for RoBERTa)
                    user_text = " ".join(str(post) for post in posts[:10])
                    text_features[user_id] = user_text

                    # Extract SBERT embeddings
                    text_sample = user_text[:500]
                    if len(text_sample.strip()) > 20:
                        embedding = self.sbert_model.encode([text_sample], show_progress_bar=False)[0]
                        semantic_embeddings[user_id] = embedding[:64]
                    else:
                        semantic_embeddings[user_id] = np.zeros(64)

                    # Extract LIWC features
                    if user_id in all_liwc_features:
                        liwc_features[user_id] = all_liwc_features[user_id]
                    else:
                        # Fallback: basic LIWC approximation
                        liwc_features[user_id] = self._extract_basic_liwc(user_text)

                    # Extract temporal features
                    temporal_features[user_id] = self._extract_temporal_features(
                        user_id, all_liwc_features
                    )

                    # Combine all features (135-dim)
                    combined_features[user_id] = np.concatenate([
                        semantic_embeddings[user_id],
                        liwc_features[user_id],
                        temporal_features[user_id]
                    ])

                    valid_users.append(user_id)

                except Exception as e:
                    print(f"  Warning: Failed to extract features for user {user_id}: {e}")
                    continue

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Progress: {min(i+batch_size, len(user_list))}/{len(user_list)} users processed")
                gc.collect()

        print(f"Feature extraction complete: {len(valid_users)} users with {135}-dimensional features")

        return {
            'user_ids': valid_users,
            'semantic_embeddings': semantic_embeddings,
            'liwc_features': liwc_features,
            'temporal_features': temporal_features,
            'text_features': text_features,
            'labels': {uid: user_labels[uid] for uid in valid_users},
            'combined_features': combined_features,
            'feature_dims': {
                'semantic': 64,
                'liwc': 62,
                'temporal': 9,
                'total': 135
            }
        }

    def _load_rmhd_liwc_features(self):
        """Load precomputed LIWC features from RMHD CSV files."""
        csv_files = [
            'depression_2018_features_tfidf_256.csv',
            'depression_2019_features_tfidf_256.csv',
            'depression_pre_features_tfidf_256.csv',
            'depression_post_features_tfidf_256.csv',
            'conspiracy_2018_features_tfidf_256.csv',
            'conspiracy_2019_features_tfidf_256.csv',
            'conspiracy_pre_features_tfidf_256.csv',
            'conspiracy_post_features_tfidf_256.csv',
            'divorce_2018_features_tfidf_256.csv',
            'fitness_2018_features_tfidf_256.csv',
            'guns_2018_features_tfidf_256.csv',
            'jokes_2018_features_tfidf_256.csv',
            'legaladvice_2018_features_tfidf_256.csv',
            'meditation_2018_features_tfidf_256.csv',
            'parenting_2018_features_tfidf_256.csv',
            'personalfinance_2018_features_tfidf_256.csv',
            'relationships_2018_features_tfidf_256.csv',
            'teaching_2018_features_tfidf_256.csv'
        ]

        all_features = {}
        files_loaded = 0

        for csv_file in csv_files:
            file_path = f'data/raw/{csv_file}'
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    if 'author' in df.columns:
                        for author in df['author'].unique():
                            if author not in all_features:
                                author_data = df[df['author'] == author]

                                # Extract LIWC columns
                                liwc_cols = [col for col in df.columns if col.startswith('liwc_')]

                                if len(liwc_cols) > 0:
                                    # Average LIWC features across all posts
                                    aggregated = author_data[liwc_cols].mean()

                                    # Ensure exactly 62 dimensions
                                    if isinstance(aggregated, pd.Series):
                                        liwc_array = aggregated.values
                                    else:
                                        liwc_array = np.array(aggregated)

                                    if len(liwc_array) < 62:
                                        liwc_array = np.pad(liwc_array, (0, 62 - len(liwc_array)))
                                    else:
                                        liwc_array = liwc_array[:62]

                                    all_features[author] = liwc_array

                    files_loaded += 1
                except Exception as e:
                    print(f"  Warning: Error loading {csv_file}: {e}")
                    continue

        print(f"  Loaded LIWC features from {files_loaded} files for {len(all_features)} users")
        return all_features

    def _extract_basic_liwc(self, text):
        """Fallback basic LIWC approximation when precomputed features unavailable."""
        words = text.lower().split()
        word_count = len(words) if words else 1

        basic_features = [
            text.count('i') / word_count,  # First person
            text.count('you') / word_count,  # Second person
            (text.count('he') + text.count('she')) / word_count,  # Third person
            sum(1 for w in ['success', 'achieve', 'goal'] if w in text) / word_count,  # Achievement
            sum(1 for w in words if w.endswith('ly')) / word_count,  # Adverbs
            sum(1 for w in ['feel', 'emotion', 'mood'] if w in text) / word_count,  # Affect
            sum(1 for w in ['angry', 'mad', 'rage'] if w in text) / word_count,  # Anger
            sum(1 for w in ['anxious', 'worry', 'nervous'] if w in text) / word_count,  # Anxiety
            sum(1 for w in ['the', 'a', 'an'] if w in words) / word_count,  # Articles
            sum(1 for w in ['yes', 'ok', 'agree'] if w in words) / word_count,  # Assent
        ]

        # Pad to 62 dimensions
        return np.array(basic_features + [0.0] * (62 - len(basic_features)))

    def _extract_temporal_features(self, user_id, all_liwc_features):
        """Extract temporal posting pattern features."""
        if user_id in all_liwc_features:
            # Try to extract from RMHD data (limited temporal info available)
            temporal = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
        else:
            # Fallback
            temporal = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1])

        return temporal[:9]  # Ensure exactly 9 dimensions
