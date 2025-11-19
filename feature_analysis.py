import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def analyze_features(cache_path="data/processed/features_cache_b4a06071e1d4697c398a77ee79dbaf1e.pkl"):
    """
    Analyzes the cached features to understand their statistical properties and correlations.

    Args:
        cache_path (str): The path to the cached features file.
    """
    print(f"Loading cached features from {cache_path}...")
    with open(cache_path, "rb") as f:
        cached_features = pickle.load(f)

    combined_features = cached_features["combined_features"]
    user_ids = cached_features["user_ids"]
    feature_dims = cached_features["feature_dims"]

    # --- Scaling ---
    print("\n--- Scaling ---")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(np.array([combined_features[uid] for uid in user_ids]))
    print("Features scaled using RobustScaler.")

    # Separate the features into groups
    semantic_features = scaled_features[:, :feature_dims["semantic"]]
    liwc_features = scaled_features[:, feature_dims["semantic"]:feature_dims["semantic"] + feature_dims["liwc"]]
    temporal_features = scaled_features[:, -feature_dims["temporal"]:]

    print("\n--- Feature Analysis ---")
    print("\nShape of each feature group:")
    print(f"  Semantic:  {semantic_features.shape}")
    print(f"  LIWC:      {liwc_features.shape}")
    print(f"  Temporal:  {temporal_features.shape}")

    print("\nMean of each feature group:")
    print(f"  Semantic:  {np.mean(semantic_features):.4f}")
    print(f"  LIWC:      {np.mean(liwc_features):.4f}")
    print(f"  Temporal:  {np.mean(temporal_features):.4f}")

    print("\nVariance of each feature group:")
    print(f"  Semantic:  {np.var(semantic_features):.4f}")
    print(f"  LIWC:      {np.var(liwc_features):.4f}")
    print(f"  Temporal:  {np.var(temporal_features):.4f}")

    # --- LIWC Variance Analysis ---
    print("\n--- LIWC Variance Analysis ---")
    liwc_variances = np.var(liwc_features, axis=0)
    top_10_liwc_variances = np.argsort(liwc_variances)[-10:]
    
    print("\nTop 10 LIWC features with the highest variance:")
    # We don't have the names of the LIWC features, so we'll just use their indices
    for i in reversed(top_10_liwc_variances):
        print(f"  LIWC feature {i}: {liwc_variances[i]:.4f}")

    # --- Correlation Analysis ---
    print("\n--- Correlation Analysis ---")
    df_semantic = pd.DataFrame(semantic_features)
    df_liwc = pd.DataFrame(liwc_features)
    df_temporal = pd.DataFrame(temporal_features)

    # To simplify, let's take the mean of each feature set for each user
    mean_semantic = df_semantic.mean(axis=1)
    mean_liwc = df_liwc.mean(axis=1)
    mean_temporal = df_temporal.mean(axis=1)

    correlation_df = pd.DataFrame({
        "semantic": mean_semantic,
        "liwc": mean_liwc,
        "temporal": mean_temporal
    })

    print("\nCorrelation matrix of the mean of each feature group:")
    print(correlation_df.corr())

    # --- Similarity Analysis ---
    print("\n--- Similarity Analysis ---")
    linguistic_sim = cosine_similarity(np.concatenate([semantic_features, liwc_features[:, :30]], axis=1))
    temporal_sim = cosine_similarity(temporal_features)
    psychological_sim = cosine_similarity(liwc_features[:, 30:])
    
    # Get off-diagonal elements
    linguistic_sim_off_diag = linguistic_sim[~np.eye(linguistic_sim.shape[0], dtype=bool)]
    temporal_sim_off_diag = temporal_sim[~np.eye(temporal_sim.shape[0], dtype=bool)]
    psychological_sim_off_diag = psychological_sim[~np.eye(psychological_sim.shape[0], dtype=bool)]

    print("\nMean of each similarity score (off-diagonal):")
    print(f"  Linguistic:    {np.mean(linguistic_sim_off_diag):.4f}")
    print(f"  Temporal:      {np.mean(temporal_sim_off_diag):.4f}")
    print(f"  Psychological: {np.mean(psychological_sim_off_diag):.4f}")
    
    print("\nVariance of each similarity score (off-diagonal):")
    print(f"  Linguistic:    {np.var(linguistic_sim_off_diag):.4f}")
    print(f"  Temporal:      {np.var(temporal_sim_off_diag):.4f}")
    print(f"  Psychological: {np.var(psychological_sim_off_diag):.4f}")


if __name__ == "__main__":
    analyze_features()
