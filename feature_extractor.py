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

        self.sbert_model = None

    def _init_sbert_model(self):
        """Lazy initialization of SBERT model to save memory."""
        if self.sbert_model is None:
            print("Loading SBERT model (all-MiniLM-L6-v2)...")
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SBERT model loaded successfully")

    def extract_all_features(self, user_posts, user_labels, combined_df=None, max_users=None):
        """
        Extract all features with caching support.

        Args:
            user_posts (dict): Dictionary of user posts
            user_labels (dict): Dictionary of user labels
            combined_df (DataFrame): Full dataset with time_period and metadata columns
            max_users (int): Maximum number of users to process (None = all)

        Returns:
            dict: Cached features including SBERT embeddings, LIWC, temporal, and text
        """
        data_hash = get_data_hash(user_posts, user_labels)

        if self.cache_enabled:
            cached = load_cached_features(self.cache_dir, data_hash)
            if cached is not None:
                print("Using cached features (skipping extraction)")
                return cached

        print("\n" + "="*60)
        print("Feature Extraction (results will be cached for reuse)")
        print("="*60)

        features = self._extract_features_impl(user_posts, user_labels, combined_df, max_users)

        if self.cache_enabled:
            save_cached_features(features, self.cache_dir, data_hash)

        return features

    def _extract_features_impl(self, user_posts, user_labels, combined_df, max_users):
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
        semantic_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('semantic_dims', 64)
        liwc_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('liwc_dims', 60)
        temporal_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('temporal_dims', 9)
        total_dims = semantic_dims + liwc_dims + temporal_dims

        print("\n1. Loading RMHD precomputed LIWC features...")
        all_liwc_features = self._load_rmhd_liwc_features(liwc_dims)

        print(f"\n2. Performing stratified user sampling...")
        depression_users = [uid for uid in user_posts.keys() if user_labels[uid] == 1]
        control_users = [uid for uid in user_posts.keys() if user_labels[uid] == 0]

        print(f"Available users before sampling:")
        print(f"  Depression users: {len(depression_users)}")
        print(f"  Control users: {len(control_users)}")

        if max_users:
            users_per_class = max_users // 2

            if len(depression_users) > users_per_class:
                import random
                random.seed(42)
                depression_users = random.sample(depression_users, users_per_class)

            if len(control_users) > users_per_class:
                import random
                random.seed(42)
                control_users = random.sample(control_users, users_per_class)

            user_list = depression_users + control_users
            print(f"\nStratified sampling complete:")
            print(f"  Sampled depression users: {len(depression_users)}")
            print(f"  Sampled control users: {len(control_users)}")
            print(f"  Total users for feature extraction: {len(user_list)}")
        else:
            user_list = list(user_posts.keys())

        print(f"\n3. Extracting features for {len(user_list)} users...")

        semantic_embeddings = {}
        liwc_features = {}
        temporal_features = {}
        text_features = {}
        combined_features = {}
        valid_users = []

        semantic_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('semantic_dims', 64)
        liwc_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('liwc_dims', 62)
        temporal_dims = self.config.get('models', {}).get('semantic_gnn', {}).get('feature_breakdown', {}).get('temporal_dims', 9)
        total_dims = semantic_dims + liwc_dims + temporal_dims
        
        self._init_sbert_model()

        batch_size = 50
        for i in range(0, len(user_list), batch_size):
            batch_users = user_list[i:i+batch_size]

            for user_id in batch_users:
                try:
                    posts = user_posts[user_id]

                    user_text = " ".join(str(post) for post in posts[:10])
                    text_features[user_id] = user_text

                    text_sample = user_text[:500]
                    if len(text_sample.strip()) > 20:
                        embedding = self.sbert_model.encode([text_sample], show_progress_bar=False)[0]
                        semantic_embeddings[user_id] = embedding[:semantic_dims]
                    else:
                        semantic_embeddings[user_id] = np.zeros(semantic_dims)

                    if user_id in all_liwc_features:
                        liwc_features[user_id] = all_liwc_features[user_id]
                    else:
                        liwc_features[user_id] = self._extract_basic_liwc(user_text, liwc_dims)

                    if combined_df is not None and user_id in combined_df['author'].values:
                        user_group = combined_df[combined_df['author'] == user_id]
                        temporal_features[user_id] = self._extract_temporal_features(user_id, user_group, temporal_dims)
                    else:
                        temporal_features[user_id] = np.zeros(temporal_dims)

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

        print(f"Feature extraction complete: {len(valid_users)} users with {semantic_dims + liwc_dims + temporal_dims}-dimensional features")
        
        return {
            'user_ids': valid_users,
            'semantic_embeddings': semantic_embeddings,
            'liwc_features': liwc_features,
            'temporal_features': temporal_features,
            'text_features': text_features,
            'labels': {uid: user_labels[uid] for uid in valid_users},
            'combined_features': combined_features,
            'feature_dims': {
                'semantic': semantic_dims,
                'liwc': liwc_dims,
                'temporal': temporal_dims,
                'total': semantic_dims + liwc_dims + temporal_dims
            }
        }

    def _load_rmhd_liwc_features(self, liwc_dims):
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

                                ling_cats = [
                                    'liwc_1st_pers', 'liwc_2nd_pers', 'liwc_3rd_pers', 'liwc_articles_article', 
                                    'liwc_auxiliary_verbs', 'liwc_adverbs', 'liwc_conjunctions', 'liwc_fillers', 
                                    'liwc_future_tense', 'liwc_impersonal_pronouns', 'liwc_inclusive', 'liwc_negations', 
                                    'liwc_nonfluencies', 'liwc_numbers', 'liwc_past_tense', 'liwc_personal_pronouns', 
                                    'liwc_prepositions', 'liwc_present_tense', 'liwc_quantifiers', 'liwc_relativity', 
                                    'liwc_space', 'liwc_time', 'liwc_total_functional', 'liwc_total_pronouns', 
                                    'liwc_common_verbs'
                                ]
                                
                                psych_cats = [
                                    'liwc_achievement', 'liwc_affective_processes', 'liwc_anger', 'liwc_anxiety', 
                                    'liwc_assent', 'liwc_biological', 'liwc_body', 'liwc_causation', 'liwc_certainty', 
                                    'liwc_cognitive', 'liwc_death', 'liwc_discrepancy', 'liwc_exclusive', 'liwc_family', 
                                    'liwc_feel', 'liwc_friends', 'liwc_health', 'liwc_hear', 'liwc_home', 'liwc_humans', 
                                    'liwc_ingestion', 'liwc_inhibition', 'liwc_insight', 'liwc_leisure', 'liwc_money', 
                                    'liwc_motion', 'liwc_negative_emotion', 'liwc_perceptual_processes', 'liwc_positive_emotion', 
                                    'liwc_religion', 'liwc_sadness', 'liwc_see', 'liwc_sexual', 'liwc_social_processes', 
                                    'liwc_swear_words', 'liwc_tentative', 'liwc_work'
                                ]
                                
                                avail_ling = [c for c in ling_cats if c in author_data.columns]
                                avail_psych = [c for c in psych_cats if c in author_data.columns]
                                
                                ling_vals = author_data[avail_ling].mean().values if avail_ling else np.array([])
                                psych_vals = author_data[avail_psych].mean().values if avail_psych else np.array([])
                                
                                liwc_array = np.concatenate([ling_vals, psych_vals])
                                
                                if len(liwc_array) < liwc_dims:
                                    liwc_array = np.pad(liwc_array, (0, liwc_dims - len(liwc_array)))
                                elif len(liwc_array) > liwc_dims:
                                    liwc_array = liwc_array[:liwc_dims]

                                all_features[author] = liwc_array

                    files_loaded += 1
                except Exception as e:
                    print(f"  Warning: Error loading {csv_file}: {e}")
                    continue

        print(f"  Loaded LIWC features from {files_loaded} files for {len(all_features)} users")
        return all_features

    def _extract_basic_liwc(self, text, liwc_dims):
        """Fallback basic LIWC approximation when precomputed features unavailable."""
        words = text.lower().split()
        word_count = len(words) if words else 1

        basic_features = [
            text.count('i') / word_count,
            text.count('you') / word_count,
            (text.count('he') + text.count('she')) / word_count,
            sum(1 for w in ['success', 'achieve', 'goal'] if w in text) / word_count,
            sum(1 for w in words if w.endswith('ly')) / word_count,
            sum(1 for w in ['feel', 'emotion', 'mood'] if w in text) / word_count,
            sum(1 for w in ['angry', 'mad', 'rage'] if w in text) / word_count,
            sum(1 for w in ['anxious', 'worry', 'nervous'] if w in text) / word_count,
            sum(1 for w in ['the', 'a', 'an'] if w in words) / word_count,
            sum(1 for w in ['yes', 'ok', 'agree'] if w in words) / word_count,
        ]

        if len(basic_features) < liwc_dims:
            basic_features.extend([0.0] * (liwc_dims - len(basic_features)))

        return np.array(basic_features[:liwc_dims])

    def _extract_temporal_features(self, user_id, user_group_df, temporal_dims):
        """
        Extract temporal posting pattern features from user data.

        Args:
            user_id: User identifier
            user_group_df: DataFrame with user's posts including time_period column
            temporal_dims: The number of temporal dimensions to return

        Returns:
            np.array: N-dimensional temporal feature vector
        """
        temporal_features = {}

        temporal_features['posts_pre_pandemic'] = len(user_group_df[user_group_df['time_period'] == 'pre']) if 'time_period' in user_group_df.columns else 0
        temporal_features['posts_2018'] = len(user_group_df[user_group_df['time_period'] == '2018']) if 'time_period' in user_group_df.columns else 0
        temporal_features['posts_2019'] = len(user_group_df[user_group_df['time_period'] == '2019']) if 'time_period' in user_group_df.columns else 0
        temporal_features['posts_post_pandemic'] = len(user_group_df[user_group_df['time_period'] == 'post']) if 'time_period' in user_group_df.columns else 0

        total_posts = len(user_group_df)
        if total_posts > 0:
            temporal_features['posts_pre_pandemic'] /= total_posts
            temporal_features['posts_2018'] /= total_posts
            temporal_features['posts_2019'] /= total_posts
            temporal_features['posts_post_pandemic'] /= total_posts

        temporal_features['subreddit_count'] = len(user_group_df['subreddit'].unique()) if 'subreddit' in user_group_df.columns else 1
        temporal_features['subreddit_count'] = min(temporal_features['subreddit_count'] / 10.0, 1.0)

        period_counts = [
            temporal_features['posts_pre_pandemic'] * total_posts,
            temporal_features['posts_2018'] * total_posts,
            temporal_features['posts_2019'] * total_posts,
            temporal_features['posts_post_pandemic'] * total_posts
        ]
        temporal_features['posting_consistency'] = 1.0 / (1.0 + np.std(period_counts)) if total_posts > 0 else 0.0

        temporal_features['posting_volume'] = min(total_posts / 20.0, 1.0)

        if 'subreddit' in user_group_df.columns and len(user_group_df) > 0:
            subreddit_counts = user_group_df['subreddit'].value_counts()
            temporal_features['primary_subreddit_ratio'] = subreddit_counts.iloc[0] / len(user_group_df) if len(subreddit_counts) > 0 else 0.5
        else:
            temporal_features['primary_subreddit_ratio'] = 0.5

        non_zero_periods = sum(1 for c in period_counts if c > 0)
        temporal_features['posting_spread'] = non_zero_periods / 4.0

        feature_array = np.array([
            temporal_features['posts_pre_pandemic'],
            temporal_features['posts_2018'],
            temporal_features['posts_2019'],
            temporal_features['subreddit_count'],
            temporal_features['posting_consistency'],
            temporal_features['posting_volume'],
            temporal_features['primary_subreddit_ratio'],
            temporal_features['posting_spread']
        ])

        if len(feature_array) < temporal_dims:
            feature_array = np.pad(feature_array, (0, temporal_dims - len(feature_array)))
        
        return feature_array[:temporal_dims]
