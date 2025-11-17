"""
Data preprocessing for RMHD dataset.
Handles loading, text processing, and feature extraction.
"""

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime, timedelta
import warnings
import psutil
import gc
import pickle
import hashlib
warnings.filterwarnings('ignore')


def get_data_hash(user_posts, user_labels):
    """
    Generate a unique hash for the dataset to enable caching.

    Args:
        user_posts (dict): Dictionary of user posts
        user_labels (dict): Dictionary of user labels

    Returns:
        str: MD5 hash of the dataset
    """
    # Create a string representation of the dataset
    data_str = f"{len(user_posts)}_{len(user_labels)}_{sorted(user_posts.keys())[:10]}"
    return hashlib.md5(data_str.encode()).hexdigest()


def save_cached_features(features, cache_path, data_hash):
    """
    Save extracted features to disk for reuse.

    Args:
        features (dict): Dictionary of extracted features
        cache_path (str): Directory to save cached features
        data_hash (str): Unique hash for the dataset
    """
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f'features_cache_{data_hash}.pkl')

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Features cached to: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to cache features: {e}")


def load_cached_features(cache_path, data_hash):
    """
    Load previously extracted features from disk.

    Args:
        cache_path (str): Directory containing cached features
        data_hash (str): Unique hash for the dataset

    Returns:
        dict or None: Cached features if available, None otherwise
    """
    cache_file = os.path.join(cache_path, f'features_cache_{data_hash}.pkl')

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded cached features from: {cache_file}")
        return features
    except Exception as e:
        print(f"Warning: Failed to load cached features: {e}")
        return None


def get_available_memory_mb():
    """
    Get available system memory in megabytes.

    Returns:
        float: Available memory in MB
    """
    memory = psutil.virtual_memory()
    return memory.available / (1024 * 1024)


def load_rmhd_data_chunked(data_path, target_posts_per_class, chunk_size):
    """
    Load RMHD data in chunks to manage memory efficiently for large datasets.

    Args:
        data_path (str): Path to RMHD dataset files
        target_posts_per_class (int): Total number of posts to load per class
        chunk_size (int): Number of posts to load per chunk

    Returns:
        dict: Processed user data with posts, labels, metadata
    """
    print(f"Loading data in chunks of {chunk_size} posts per chunk...")

    depression_subreddits = ['depression']
    control_subreddits = [
        'conspiracy', 'divorce', 'fitness', 'guns', 'jokes',
        'legaladvice', 'meditation', 'parenting', 'personalfinance',
        'relationships', 'teaching'
    ]
    time_periods = ['2018', '2019', 'pre', 'post']

    # First pass: count available posts
    depression_files = []
    control_files = []

    for subreddit in depression_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                depression_files.append(filepath)

    for subreddit in control_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)
            if os.path.exists(filepath):
                control_files.append(filepath)

    # Load depression data in chunks
    depression_chunks = []
    depression_count = 0

    for filepath in depression_files:
        if depression_count >= target_posts_per_class:
            break

        try:
            # Load only what we need
            needed = min(chunk_size, target_posts_per_class - depression_count)
            df = pd.read_csv(filepath, nrows=needed, low_memory=False)

            # Extract metadata from filename
            filename = os.path.basename(filepath)
            parts = filename.replace('_features_tfidf_256.csv', '').split('_')
            subreddit = parts[0]
            period = parts[1] if len(parts) > 1 else 'unknown'

            df['subreddit'] = subreddit
            df['time_period'] = period
            df['is_depression'] = 1

            depression_chunks.append(df)
            depression_count += len(df)

            print(f"  Depression data: loaded {len(df)} posts from {filename} ({depression_count}/{target_posts_per_class} total)")

            # Free memory periodically
            if len(depression_chunks) >= 5:
                temp_df = pd.concat(depression_chunks, ignore_index=True)
                depression_chunks = [temp_df]
                gc.collect()

        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    # Load control data in chunks
    control_chunks = []
    control_count = 0

    for filepath in control_files:
        if control_count >= target_posts_per_class:
            break

        try:
            needed = min(chunk_size, target_posts_per_class - control_count)
            df = pd.read_csv(filepath, nrows=needed, low_memory=False)

            filename = os.path.basename(filepath)
            parts = filename.replace('_features_tfidf_256.csv', '').split('_')
            subreddit = parts[0]
            period = parts[1] if len(parts) > 1 else 'unknown'

            df['subreddit'] = subreddit
            df['time_period'] = period
            df['is_depression'] = 0

            control_chunks.append(df)
            control_count += len(df)

            print(f"  Control data: loaded {len(df)} posts from {filename} ({control_count}/{target_posts_per_class} total)")

            # Free memory periodically
            if len(control_chunks) >= 5:
                temp_df = pd.concat(control_chunks, ignore_index=True)
                control_chunks = [temp_df]
                gc.collect()

        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    # Combine all chunks
    if depression_chunks:
        depression_df = pd.concat(depression_chunks, ignore_index=True)
        if len(depression_df) > target_posts_per_class:
            depression_df = depression_df.sample(n=target_posts_per_class, random_state=42)
    else:
        depression_df = pd.DataFrame()

    if control_chunks:
        control_df = pd.concat(control_chunks, ignore_index=True)
        if len(control_df) > target_posts_per_class:
            control_df = control_df.sample(n=target_posts_per_class, random_state=42)
    else:
        control_df = pd.DataFrame()

    # Final combination
    if not depression_df.empty and not control_df.empty:
        combined_df = pd.concat([depression_df, control_df], ignore_index=True)
    elif not depression_df.empty:
        combined_df = depression_df
    elif not control_df.empty:
        combined_df = control_df
    else:
        raise ValueError("No valid data found after chunked loading")

    print(f"Chunked loading complete: {len(combined_df)} posts loaded")

    # Process users from dataset
    print(f"\nProcessing users from dataset...")

    user_posts = {}
    user_labels = {}
    user_metadata = {}
    temporal_windows = {}

    # Group by author
    for author, group in combined_df.groupby('author'):
        # Collect all posts for this user
        posts = group['post'].tolist()
        user_posts[author] = posts

        # Determine label (majority vote based on subreddit)
        depression_posts = sum(group['is_depression'])
        user_labels[author] = 1 if depression_posts > len(group) / 2 else 0

        # Store metadata
        user_metadata[author] = {
            'subreddits': group['subreddit'].unique().tolist(),
            'post_count': len(posts),
            'first_post_date': group['date'].min() if 'date' in group.columns else None,
            'last_post_date': group['date'].max() if 'date' in group.columns else None
        }

        # Temporal windows
        temporal_windows[author] = {
            'pre_pandemic': group[group['time_period'] == 'pre']['post'].tolist(),
            'year_2018': group[group['time_period'] == '2018']['post'].tolist(),
            'year_2019': group[group['time_period'] == '2019']['post'].tolist(),
            'post_pandemic': group[group['time_period'] == 'post']['post'].tolist()
        }

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total users: {len(user_posts)}")
    print(f"  Total posts: {len(combined_df)}")

    post_counts = [len(posts) for posts in user_posts.values()]
    if post_counts:
        avg_posts = np.mean(post_counts)
        median_posts = np.median(post_counts)
        print(f"  Average posts per user: {avg_posts:.1f}")
        print(f"  Median posts per user: {median_posts:.0f}")
        print(f"  Post count range: {min(post_counts)}-{max(post_counts)}")

        # Class distribution
        depression_users = sum(user_labels.values())
        control_users = len(user_labels) - depression_users
        print(f"\n  Class distribution:")
        print(f"    Depression users: {depression_users}")
        print(f"    Control users: {control_users}")
        print(f"    Class balance: {depression_users / len(user_labels) * 100:.1f}% depression")

    return {
        'user_posts': user_posts,
        'user_labels': user_labels,
        'user_metadata': user_metadata,
        'temporal_windows': temporal_windows,
        'combined_df': combined_df,
        'depression_subreddits': depression_subreddits,
        'control_subreddits': control_subreddits
    }


def estimate_memory_usage(num_posts, avg_post_length=500):
    """
    Estimate memory usage for loading a given number of posts.

    Args:
        num_posts (int): Number of posts to load
        avg_post_length (int): Average post length in characters

    Returns:
        float: Estimated memory usage in MB
    """
    # Rough estimate: text + metadata + features
    text_mb = (num_posts * avg_post_length) / (1024 * 1024)
    metadata_mb = (num_posts * 200) / (1024 * 1024)  # Assuming 200 bytes per post metadata
    features_mb = (num_posts * 300 * 8) / (1024 * 1024)  # 300 features * 8 bytes per float

    total_mb = text_mb + metadata_mb + features_mb
    return total_mb * 1.5  # Add 50% safety margin

def load_rmhd_data(data_path, target_posts_per_class=5000, config=None):
    """
    Load Reddit Mental Health Dataset with stratified sampling and memory-efficient chunked loading.

    Args:
        data_path (str): Path to RMHD dataset files
        target_posts_per_class (int): Number of posts to sample per class (depression/control)
        config (dict): Configuration dictionary with memory settings

    Returns:
        dict: Processed user data with posts, labels, metadata
    """
    print(f"Loading dataset from {data_path}...")

    # Check memory constraints
    available_memory = get_available_memory_mb()
    estimated_usage = estimate_memory_usage(target_posts_per_class * 2)

    print(f"Available memory: {available_memory:.0f} MB")
    print(f"Estimated memory needed: {estimated_usage:.0f} MB")

    # Determine if chunked loading is needed
    enable_chunked = False
    chunk_size = target_posts_per_class

    if config:
        enable_chunked = config.get('data', {}).get('enable_chunked_loading', False)
        chunk_size = config.get('data', {}).get('chunk_size', 10000)
        max_memory_mb = config.get('data', {}).get('max_memory_mb', 8192)

        if estimated_usage > max_memory_mb * 0.7:  # Use 70% threshold
            enable_chunked = True
            print(f"Warning: Estimated memory usage ({estimated_usage:.0f} MB) exceeds 70% of limit ({max_memory_mb} MB)")
            print(f"Enabling chunked loading to manage memory efficiently")

    if enable_chunked and target_posts_per_class > chunk_size:
        print(f"Using chunked loading strategy: {chunk_size} posts per chunk")
        return load_rmhd_data_chunked(data_path, target_posts_per_class, chunk_size)

    depression_subreddits = ['depression']

    control_subreddits = [
        'conspiracy', 'divorce', 'fitness', 'guns', 'jokes',
        'legaladvice', 'meditation', 'parenting', 'personalfinance',
        'relationships', 'teaching'
    ]
    time_periods = ['2018', '2019', 'pre', 'post']

    depression_data = []
    control_data = []

    # Load depression data
    print("Loading depression subreddit data...")
    for subreddit in depression_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    df['subreddit'] = subreddit
                    df['time_period'] = period
                    df['is_depression'] = 1
                    depression_data.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    # Load control data
    print("Loading control subreddit data...")
    for subreddit in control_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    df['subreddit'] = subreddit
                    df['time_period'] = period
                    df['is_depression'] = 0
                    control_data.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if not depression_data and not control_data:
        raise ValueError("No data files found. Check the data path.")

    # Combine and sample depression data
    if depression_data:
        print(f"Combining {len(depression_data)} depression data files...")
        depression_df = pd.concat(depression_data, ignore_index=True)
        print(f"Total depression posts available: {len(depression_df)}")

        if len(depression_df) > target_posts_per_class:
            print(f"Sampling {target_posts_per_class} depression posts...")
            depression_df = depression_df.sample(n=target_posts_per_class, random_state=42)
        else:
            print(f"Warning: Only {len(depression_df)} depression posts available (requested {target_posts_per_class})")

        # Free memory
        del depression_data
        gc.collect()
    else:
        depression_df = pd.DataFrame()

    # Combine and sample control data
    if control_data:
        print(f"Combining {len(control_data)} control data files...")
        control_df = pd.concat(control_data, ignore_index=True)
        print(f"Total control posts available: {len(control_df)}")

        if len(control_df) > target_posts_per_class:
            print(f"Sampling {target_posts_per_class} control posts...")
            control_df = control_df.sample(n=target_posts_per_class, random_state=42)
        else:
            print(f"Warning: Only {len(control_df)} control posts available (requested {target_posts_per_class})")

        # Free memory
        del control_data
        gc.collect()
    else:
        control_df = pd.DataFrame()

    if not depression_df.empty and not control_df.empty:
        combined_df = pd.concat([depression_df, control_df], ignore_index=True)
    elif not depression_df.empty:
        combined_df = depression_df
    elif not control_df.empty:
        combined_df = control_df
    else:
        raise ValueError("No valid data found after sampling")

    print(f"Dataset loaded successfully: {len(combined_df)} posts")

    # Process users - each post represents one user's contribution
    print(f"\nProcessing users from dataset...")

    user_posts = {}
    user_labels = {}
    user_metadata = {}
    temporal_windows = {}

    # Group by author
    for author, group in combined_df.groupby('author'):
        # Collect all posts for this user
        posts = group['post'].tolist()
        user_posts[author] = posts

        # Determine label (majority vote based on subreddit)
        depression_posts = sum(group['is_depression'])
        user_labels[author] = 1 if depression_posts > len(group) / 2 else 0

        # Store metadata
        user_metadata[author] = {
            'subreddits': group['subreddit'].unique().tolist(),
            'post_count': len(posts),
            'first_post_date': group['date'].min() if 'date' in group.columns else None,
            'last_post_date': group['date'].max() if 'date' in group.columns else None
        }

        # Temporal windows
        temporal_windows[author] = {
            'pre_pandemic': group[group['time_period'] == 'pre']['post'].tolist(),
            'year_2018': group[group['time_period'] == '2018']['post'].tolist(),
            'year_2019': group[group['time_period'] == '2019']['post'].tolist(),
            'post_pandemic': group[group['time_period'] == 'post']['post'].tolist()
        }

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total users: {len(user_posts)}")
    print(f"  Total posts: {len(combined_df)}")

    post_counts = [len(posts) for posts in user_posts.values()]
    if post_counts:
        avg_posts = np.mean(post_counts)
        median_posts = np.median(post_counts)
        print(f"  Average posts per user: {avg_posts:.1f}")
        print(f"  Median posts per user: {median_posts:.0f}")
        print(f"  Post count range: {min(post_counts)}-{max(post_counts)}")

        # Class distribution
        depression_users = sum(user_labels.values())
        control_users = len(user_labels) - depression_users
        print(f"\n  Class distribution:")
        print(f"    Depression users: {depression_users}")
        print(f"    Control users: {control_users}")
        print(f"    Class balance: {depression_users / len(user_labels) * 100:.1f}% depression")

    return {
        'user_posts': user_posts,
        'user_labels': user_labels,
        'user_metadata': user_metadata,
        'temporal_windows': temporal_windows,
        'combined_df': combined_df,
        'depression_subreddits': depression_subreddits,
        'control_subreddits': control_subreddits
    }

def preprocess_text(text):
    """
    Clean and normalize text data.

    Args:
        text (str): Raw text

    Returns:
        str: Cleaned text
    """
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = ' '.join(text.split())

    return text

def extract_liwc_features(texts, batch_size=1000):
    """
    Extract comprehensive LIWC-style linguistic features from texts with batch processing.
    Implements the full LIWC psychological category framework for depression detection.

    Args:
        texts (list): List of text documents
        batch_size (int): Number of texts to process per batch for memory efficiency

    Returns:
        pd.DataFrame: LIWC features per document
    """
    if len(texts) <= batch_size:
        # Process all at once if small enough
        return _extract_liwc_features_batch(texts)

    # Process in batches
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    all_features = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_features = _extract_liwc_features_batch(batch)
        all_features.append(batch_features)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
            gc.collect()

    return pd.concat(all_features, ignore_index=True)


def _extract_liwc_features_batch(texts):
    """
    Extract LIWC features for a single batch of texts.

    Args:
        texts (list): List of text documents

    Returns:
        pd.DataFrame: LIWC features per document
    """
    features = []

    liwc_categories = {
        'affect': ['feeling', 'emotion', 'mood', 'heart', 'soul', 'passion'],
        'posemo': ['love', 'nice', 'sweet', 'good', 'happy', 'joy', 'great', 'excellent', 'wonderful', 'amazing',
                   'fantastic', 'awesome', 'perfect', 'beautiful', 'brilliant', 'marvelous', 'outstanding'],
        'negemo': ['hate', 'kill', 'bad', 'nasty', 'awful', 'terrible', 'horrible', 'disgusting', 'annoying',
                   'frustrating', 'disappointing', 'ugly', 'stupid', 'worst', 'sucks', 'damn'],

        'anx': ['worry', 'anxious', 'stress', 'nervous', 'panic', 'fear', 'scared', 'concerned', 'troubled',
                'overwhelmed', 'tense', 'restless', 'uneasy', 'apprehensive', 'disturbed'],
        'anger': ['hate', 'kill', 'annoyed', 'furious', 'angry', 'mad', 'rage', 'pissed', 'irritated',
                  'frustrated', 'outraged', 'livid', 'enraged', 'bitter'],
        'sad': ['sad', 'depressed', 'down', 'lonely', 'empty', 'hopeless', 'despair', 'grief', 'sorrow',
                'miserable', 'gloomy', 'melancholy', 'blue', 'heartbroken', 'devastated'],

        'social': ['talk', 'they', 'child', 'aunt', 'brother', 'daughter', 'dad', 'family', 'grandpa', 'husband',
                   'mom', 'mother', 'nephew', 'son', 'wife', 'friend', 'buddy', 'neighbor'],
        'family': ['aunt', 'brother', 'child', 'dad', 'daughter', 'family', 'grandpa', 'grandma', 'husband',
                   'mom', 'mother', 'nephew', 'sister', 'son', 'wife'],
        'friends': ['friend', 'buddy', 'pal', 'mate', 'companion', 'acquaintance', 'peer'],

        'cogmech': ['cause', 'know', 'ought', 'think', 'because', 'how', 'when', 'why', 'reason', 'logic'],
        'insight': ['think', 'know', 'consider', 'understand', 'realize', 'recognize', 'learn', 'analyze',
                    'comprehend', 'grasp', 'perceive', 'acknowledge'],
        'cause': ['because', 'effect', 'hence', 'since', 'thus', 'therefore', 'consequently', 'result',
                  'due', 'reason', 'why', 'how'],
        'discrep': ['should', 'would', 'could', 'ought', 'need', 'may', 'might', 'suppose', 'hope',
                    'wish', 'want'],
        'tentat': ['maybe', 'perhaps', 'guess', 'probably', 'possibly', 'might', 'seem', 'appear',
                   'suppose', 'suggest'],
        'certain': ['always', 'never', 'completely', 'definitely', 'absolutely', 'certainly', 'sure',
                    'obvious', 'undoubtedly', 'clearly'],
        'inhib': ['block', 'constrain', 'stop', 'restrict', 'control', 'prevent', 'avoid', 'forbid',
                  'prohibit', 'restrain'],

        'percept': ['look', 'hear', 'feel', 'touch', 'taste', 'see', 'listen', 'sound', 'view', 'observe'],
        'see': ['look', 'see', 'view', 'saw', 'sight', 'blind', 'eyes', 'observe', 'watch', 'notice'],
        'hear': ['hear', 'listen', 'sound', 'music', 'noise', 'loud', 'quiet', 'voice', 'whisper', 'scream'],
        'feel': ['feel', 'touch', 'hand', 'finger', 'hold', 'grab', 'soft', 'hard', 'smooth', 'rough'],

        'bio': ['eat', 'blood', 'pain', 'doctor', 'hospital', 'medicine', 'health', 'sick', 'disease', 'body'],
        'body': ['hand', 'head', 'eye', 'foot', 'leg', 'arm', 'face', 'back', 'shoulder', 'finger',
                 'hair', 'skin', 'bone', 'muscle'],
        'health': ['clinic', 'doctor', 'nurse', 'hospital', 'pill', 'medicine', 'sick', 'ache', 'flu',
                   'therapy', 'treatment', 'symptom', 'pain', 'hurt'],
        'sexual': ['love', 'incest', 'sex', 'sexual', 'fuck', 'penis', 'intercourse', 'kiss', 'virgin'],
        'ingest': ['eat', 'consume', 'swallow', 'chew', 'taste', 'flavor', 'drink', 'sip', 'meal', 'food'],

        'work': ['work', 'job', 'employ', 'boss', 'career', 'office', 'business', 'company', 'salary', 'staff'],
        'achieve': ['win', 'success', 'better', 'top', 'best', 'effort', 'try', 'goal', 'aim', 'accomplish'],
        'leisure': ['cook', 'chat', 'movie', 'art', 'sing', 'dance', 'play', 'game', 'sport', 'vacation'],
        'home': ['home', 'house', 'kitchen', 'bathroom', 'bedroom', 'apartment', 'room', 'furniture'],
        'money': ['audit', 'cash', 'owe', 'earn', 'sold', 'buy', 'cost', 'cheap', 'dollar', 'pay'],
        'relig': ['altar', 'church', 'mosque', 'temple', 'church', 'synagogue', 'god', 'heaven', 'hell',
                  'sacred', 'spirit', 'soul', 'pray', 'worship'],
        'death': ['bury', 'coffin', 'kill', 'mourn', 'murder', 'dead', 'death', 'die', 'dying', 'funeral'],

        'time': ['ago', 'already', 'annual', 'daily', 'early', 'era', 'hour', 'late', 'quick', 'rate',
                 'season', 'time', 'today', 'week', 'year'],
        'past': ['ago', 'did', 'earlier', 'formerly', 'had', 'has', 'previous', 'prior', 'was', 'were',
                 'yesterday', 'used', 'been'],
        'present': ['today', 'is', 'now', 'current', 'moment', 'present', 'immediate', 'contemporary'],
        'future': ['may', 'will', 'soon', 'someday', 'tomorrow', 'upcoming', 'coming', 'next', 'shall',
                   'gonna', 'would', 'could'],

        'i': ['i', 'me', 'my', 'mine', 'myself'],
        'we': ['we', 'us', 'our', 'ours', 'ourselves'],
        'you': ['you', 'your', 'yours', 'yourself'],
        'shehe': ['she', 'her', 'hers', 'herself', 'he', 'him', 'his', 'himself'],
        'they': ['they', 'them', 'their', 'theirs', 'themselves'],

        'article': ['a', 'an', 'the'],
        'preps': ['to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into'],
        'auxverb': ['am', 'are', 'was', 'were', 'being', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did'],

        'period': ['.'],
        'comma': [','],
        'colon': [':'],
        'semic': [';'],
        'qmark': ['?'],
        'exclam': ['!'],
        'dash': ['-', '--'],
        'quote': ['"', "'"],
        'apostro': ["'"],
        'parenth': ['(', ')'],
        'otherp': ['[', ']', '{', '}', '<', '>']
    }

    for text in texts:
        doc_features = {}
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        if word_count == 0:
            for category in liwc_categories:
                doc_features[f'liwc_{category}'] = 0.0
            features.append(doc_features)
            continue

        for category, word_list in liwc_categories.items():
            category_count = 0

            for category_word in word_list:
                if category in ['period', 'comma', 'colon', 'semic', 'qmark', 'exclam', 'dash', 'quote', 'apostro', 'parenth', 'otherp']:
                    category_count += text.count(category_word)
                else:
                    if f' {category_word} ' in f' {text_lower} ':
                        category_count += 1
                    elif text_lower.startswith(f'{category_word} '):
                        category_count += 1
                    elif text_lower.endswith(f' {category_word}'):
                        category_count += 1
                    elif text_lower == category_word:
                        category_count += 1

            if category in ['period', 'comma', 'colon', 'semic', 'qmark', 'exclam', 'dash', 'quote', 'apostro', 'parenth', 'otherp']:
                doc_features[f'liwc_{category}'] = category_count / max(len(text), 1)
            else:
                doc_features[f'liwc_{category}'] = category_count / max(word_count, 1)

        doc_features['liwc_emotional_tone'] = (doc_features['liwc_posemo'] - doc_features['liwc_negemo'])
        doc_features['liwc_depression_indicators'] = (doc_features['liwc_sad'] + doc_features['liwc_anx'] +
                                                      doc_features['liwc_negemo'] + doc_features['liwc_past'] +
                                                      doc_features['liwc_i']) / 5
        doc_features['liwc_social_engagement'] = (doc_features['liwc_social'] + doc_features['liwc_family'] +
                                                  doc_features['liwc_friends'] + doc_features['liwc_we']) / 4
        doc_features['word_count'] = word_count
        doc_features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        doc_features['punctuation_density'] = sum(1 for char in text if char in '.,!?;:') / len(text) if text else 0
        doc_features['exclamation_count'] = text.count('!') / len(text) if text else 0
        doc_features['question_count'] = text.count('?') / len(text) if text else 0

        depression_indicators = ['depressed', 'suicide', 'kill myself', 'worthless', 'hopeless']
        anxiety_indicators = ['panic', 'anxiety', 'worry', 'stress', 'overwhelmed']

        doc_features['depression_indicators'] = sum(1 for indicator in depression_indicators if indicator in text_lower) / word_count
        doc_features['anxiety_indicators'] = sum(1 for indicator in anxiety_indicators if indicator in text_lower) / word_count

        time_words = ['today', 'yesterday', 'tomorrow', 'now', 'then', 'before', 'after', 'always', 'never']
        doc_features['temporal_references'] = sum(1 for word in time_words if word in text_lower) / word_count

        certain_words = ['always', 'never', 'definitely', 'absolutely', 'certainly', 'obviously']
        uncertain_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain']

        doc_features['certainty'] = sum(1 for word in certain_words if word in text_lower) / word_count
        doc_features['uncertainty'] = sum(1 for word in uncertain_words if word in text_lower) / word_count

        features.append(doc_features)

    return pd.DataFrame(features)

def extract_temporal_features(user_data):
    """
    Extract temporal behavioral patterns.

    Args:
        user_data (dict): User posting data with timestamps

    Returns:
        pd.DataFrame: Temporal features per user
    """
    temporal_features = []

    for user_id, posts in user_data['user_posts'].items():
        user_temporal = {}

        if not posts:
            temporal_features.append({f'temporal_{i}': 0 for i in range(20)})
            continue

        timestamps = extract_actual_timestamps_from_csv(user_id, posts)

        if not timestamps:
            temporal_features.append({f'temporal_{i}': 0 for i in range(20)})
            continue

        time_span = (max(timestamps) - min(timestamps)).days + 1
        user_temporal['posting_frequency'] = len(posts) / max(time_span, 1)

        hours = [t.hour for t in timestamps]
        user_temporal['night_posting'] = sum(1 for h in hours if h < 6 or h > 22) / len(hours)
        user_temporal['morning_posting'] = sum(1 for h in hours if 6 <= h < 12) / len(hours)
        user_temporal['afternoon_posting'] = sum(1 for h in hours if 12 <= h < 18) / len(hours)
        user_temporal['evening_posting'] = sum(1 for h in hours if 18 <= h < 22) / len(hours)

        weekdays = [t.weekday() for t in timestamps]
        user_temporal['weekday_posting'] = sum(1 for w in weekdays if w < 5) / len(weekdays)
        user_temporal['weekend_posting'] = sum(1 for w in weekdays if w >= 5) / len(weekdays)

        if len(timestamps) > 1:
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 for i in range(1, len(timestamps))]
            user_temporal['posting_regularity'] = 1 / (np.std(intervals) + 1)
            user_temporal['avg_posting_interval'] = np.mean(intervals) / 24
            user_temporal['max_posting_gap'] = max(intervals) / 24
        else:
            user_temporal['posting_regularity'] = 0
            user_temporal['avg_posting_interval'] = 0
            user_temporal['max_posting_gap'] = 0

        months = [t.month for t in timestamps]
        user_temporal['winter_posting'] = sum(1 for m in months if m in [12, 1, 2]) / len(months)
        user_temporal['spring_posting'] = sum(1 for m in months if m in [3, 4, 5]) / len(months)
        user_temporal['summer_posting'] = sum(1 for m in months if m in [6, 7, 8]) / len(months)
        user_temporal['fall_posting'] = sum(1 for m in months if m in [9, 10, 11]) / len(months)

        user_temporal['temporal_concentration'] = calculate_temporal_concentration(timestamps)

        hour_distribution = np.bincount(hours, minlength=24) / len(hours)
        user_temporal['peak_hour_concentration'] = np.max(hour_distribution)
        user_temporal['entropy_hours'] = -np.sum(hour_distribution * np.log(hour_distribution + 1e-10))

        user_temporal['burst_activity'] = calculate_burst_activity(timestamps)

        current_features = len(user_temporal)
        for i in range(current_features, 20):
            user_temporal[f'temporal_extra_{i}'] = 0.0

        temporal_features.append(user_temporal)

    return pd.DataFrame(temporal_features)


def extract_actual_timestamps_from_csv(user_id, posts):
    """
    Extract actual timestamps from CSV files for temporal analysis.
    Looks for timestamp columns in the RMHD dataset files.
    """
    timestamps = []

    timestamp_columns = ['timestamp', 'created_utc', 'date', 'time', 'post_time']

    try:
        for post in posts:
            if isinstance(post, dict):
                for col in timestamp_columns:
                    if col in post:
                        timestamp = parse_timestamp(post[col])
                        if timestamp:
                            timestamps.append(timestamp)
                            break
            elif isinstance(post, str):
                continue

        timestamps.sort()

    except Exception as e:
        print(f"Warning: Could not extract timestamps for user {user_id}: {e}")
        timestamps = []

    return timestamps


def parse_timestamp(timestamp_value):
    """
    Parse various timestamp formats commonly found in RMHD dataset.
    """
    if pd.isna(timestamp_value):
        return None

    try:
        if isinstance(timestamp_value, datetime):
            return timestamp_value

        if isinstance(timestamp_value, (int, float)):
            return datetime.fromtimestamp(timestamp_value)

        if isinstance(timestamp_value, str):
            try:
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except:
                pass

            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(timestamp_value, fmt)
                except:
                    continue

    except Exception:
        pass

    return None


def calculate_temporal_concentration(timestamps):
    """
    Calculate how concentrated posting activity is in time.
    Returns a value between 0 (evenly distributed) and 1 (highly concentrated).
    """
    if len(timestamps) < 2:
        return 0.0

    intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                for i in range(1, len(timestamps))]

    if not intervals:
        return 0.0

    intervals = np.array(sorted(intervals))
    n = len(intervals)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * intervals)) / (n * np.sum(intervals)) - (n + 1) / n

    return max(0.0, min(1.0, gini))


def calculate_burst_activity(timestamps):
    """
    Calculate burst activity patterns - periods of high posting activity.
    Returns the ratio of posts in the most active 24-hour window.
    """
    if len(timestamps) < 2:
        return 0.0

    timestamps = np.array(timestamps)

    max_posts_in_window = 0
    window_size = timedelta(hours=24)

    for i, start_time in enumerate(timestamps):
        end_time = start_time + window_size
        posts_in_window = np.sum((timestamps >= start_time) & (timestamps <= end_time))
        max_posts_in_window = max(max_posts_in_window, posts_in_window)

    return max_posts_in_window / len(timestamps)


def extract_features(users_data, config=None):
    """
    Main feature extraction function using pre-computed features from RMHD.

    Args:
        users_data (dict): Raw user data from RMHD
        config (dict): Configuration parameters

    Returns:
        pd.DataFrame: Combined feature matrix (users x features)
    """
    print("Extracting features from RMHD dataset...")

    combined_df = users_data['combined_df']

    metadata_cols = ['subreddit', 'author', 'date', 'post', 'time_period', 'is_depression']
    feature_cols = [col for col in combined_df.columns if col not in metadata_cols]

    print(f"Found {len(feature_cols)} feature columns:")
    print(f"  - Readability features: {len([col for col in feature_cols if 'tfidf' not in col])}")
    print(f"  - TF-IDF features: {len([col for col in feature_cols if 'tfidf' in col])}")

    print("Aggregating features per user...")
    user_features = []
    user_ids = []
    labels = []

    for author, group in combined_df.groupby('author'):
        user_feature_vector = group[feature_cols].mean()

        temporal_features = extract_user_temporal_features(group)

        combined_vector = pd.concat([user_feature_vector, temporal_features])

        user_features.append(combined_vector)
        user_ids.append(author)

        depression_posts = sum(group['is_depression'])
        labels.append(1 if depression_posts > len(group) / 2 else 0)

    features_df = pd.DataFrame(user_features, index=user_ids)
    features_df['label'] = labels

    print(f"Feature extraction complete. Shape: {features_df.shape}")
    print(f"Depression users: {sum(labels)}")
    print(f"Control users: {len(labels) - sum(labels)}")

    print("Normalizing features...")
    scaler = StandardScaler()
    feature_columns = [col for col in features_df.columns if col != 'label']
    features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])

    return features_df

def extract_user_temporal_features(user_group):
    """
    Extract temporal behavioral features for a single user.

    Args:
        user_group (pd.DataFrame): All posts for one user

    Returns:
        pd.Series: Temporal features
    """
    temporal_features = {}

    temporal_features['posts_pre_pandemic'] = len(user_group[user_group['time_period'] == 'pre'])
    temporal_features['posts_2018'] = len(user_group[user_group['time_period'] == '2018'])
    temporal_features['posts_2019'] = len(user_group[user_group['time_period'] == '2019'])
    temporal_features['posts_post_pandemic'] = len(user_group[user_group['time_period'] == 'post'])

    temporal_features['total_posts'] = len(user_group)

    temporal_features['subreddit_count'] = len(user_group['subreddit'].unique())

    temporal_features['depression_posting_ratio'] = user_group['is_depression'].mean()

    period_counts = [
        temporal_features['posts_pre_pandemic'],
        temporal_features['posts_2018'],
        temporal_features['posts_2019'],
        temporal_features['posts_post_pandemic']
    ]
    temporal_features['posting_consistency'] = np.std(period_counts) if np.sum(period_counts) > 0 else 0

    return pd.Series(temporal_features)