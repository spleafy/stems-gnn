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
warnings.filterwarnings('ignore')

def load_rmhd_data(data_path, target_posts_per_class=5000):
    """
    Load Reddit Mental Health Dataset with stratified sampling for depression detection.

    Args:
        data_path (str): Path to RMHD dataset files
        target_posts_per_class (int): Number of posts to sample per class (depression/control)

    Returns:
        dict: Processed user data with posts, labels, metadata
    """
    print(f"Loading dataset from {data_path}...")

    depression_subreddits = ['depression']

    control_subreddits = [
        'conspiracy', 'divorce', 'fitness', 'guns', 'jokes',
        'legaladvice', 'meditation', 'parenting', 'personalfinance',
        'relationships', 'teaching'
    ]
    time_periods = ['2018', '2019', 'pre', 'post']

    depression_data = []
    control_data = []

    for subreddit in depression_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['subreddit'] = subreddit
                    df['time_period'] = period
                    df['is_depression'] = 1
                    depression_data.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    for subreddit in control_subreddits:
        for period in time_periods:
            filename = f"{subreddit}_{period}_features_tfidf_256.csv"
            filepath = os.path.join(data_path, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['subreddit'] = subreddit
                    df['time_period'] = period
                    df['is_depression'] = 0
                    control_data.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if not depression_data and not control_data:
        raise ValueError("No data files found. Check the data path.")

    if depression_data:
        depression_df = pd.concat(depression_data, ignore_index=True)
        if len(depression_df) > target_posts_per_class:
            depression_df = depression_df.sample(n=target_posts_per_class, random_state=42)
    else:
        depression_df = pd.DataFrame()

    if control_data:
        control_df = pd.concat(control_data, ignore_index=True)
        if len(control_df) > target_posts_per_class:
            control_df = control_df.sample(n=target_posts_per_class, random_state=42)
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

    print(f"Dataset: {len(combined_df)} posts")
    user_posts = {}
    user_labels = {}
    user_metadata = {}
    temporal_windows = {}

    for author, group in combined_df.groupby('author'):
        posts = group['post'].tolist()
        user_posts[author] = posts

        depression_posts = sum(group['is_depression'])
        user_labels[author] = 1 if depression_posts > len(group) / 2 else 0

        user_metadata[author] = {
            'subreddits': group['subreddit'].unique().tolist(),
            'post_count': len(posts),
            'first_post_date': group['date'].min() if 'date' in group.columns else None,
            'last_post_date': group['date'].max() if 'date' in group.columns else None
        }

        temporal_windows[author] = {
            'pre_pandemic': group[group['time_period'] == 'pre']['post'].tolist(),
            'year_2018': group[group['time_period'] == '2018']['post'].tolist(),
            'year_2019': group[group['time_period'] == '2019']['post'].tolist(),
            'post_pandemic': group[group['time_period'] == 'post']['post'].tolist()
        }

    print(f"Processed {len(user_posts)} users")

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

def extract_liwc_features(texts):
    """
    Extract comprehensive LIWC-style linguistic features from texts.
    Implements the full LIWC psychological category framework for depression detection.

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