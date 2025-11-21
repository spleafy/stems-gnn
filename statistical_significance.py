import numpy as np
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to compare two classifiers.
    
    Args:
        y_true: Ground truth labels
        y_pred1: Predictions from classifier 1
        y_pred2: Predictions from classifier 2
        
    Returns:
        chi2, p_value
    """
    # Contingency table
    # n00: both correct
    # n01: 1 correct, 2 wrong
    # n10: 1 wrong, 2 correct
    # n11: both wrong
    
    # Note: Standard definition often uses:
    # b: model 1 correct, model 2 wrong
    # c: model 1 wrong, model 2 correct
    
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n00 = np.sum(correct1 & correct2)
    n01 = np.sum(correct1 & ~correct2) # Model 1 correct, Model 2 wrong
    n10 = np.sum(~correct1 & correct2) # Model 1 wrong, Model 2 correct
    n11 = np.sum(~correct1 & ~correct2)
    
    # McNemar's statistic with continuity correction
    # chi2 = (|b - c| - 1)^2 / (b + c)
    b = n01
    c = n10
    
    if b + c == 0:
        return 0.0, 1.0
        
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    p_value = stats.chi2.sf(chi2, 1)
    
    return chi2, p_value

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, ci=0.95):
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        metric_func: Function that takes (y_true, y_pred) and returns a score
        n_bootstraps: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95)
        
    Returns:
        (lower_bound, upper_bound)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
        
    scores = np.array(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    
    return lower, upper

def plot_confusion_matrix(y_true, y_pred, classes=['Control', 'Depression'], save_path=None):
    """
    Plot and optionally save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return cm

def analyze_misclassifications(y_true, y_pred, user_ids, text_features, save_path=None):
    """
    Identify and save misclassified examples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    user_ids = np.array(user_ids)
    
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    results = []
    for idx in misclassified_indices:
        uid = user_ids[idx]
        true_label = "Depression" if y_true[idx] == 1 else "Control"
        pred_label = "Depression" if y_pred[idx] == 1 else "Control"
        
        # Get text snippet if available
        text = text_features.get(uid, "")[:200] + "..." if uid in text_features else "N/A"
        
        results.append({
            'user_id': uid,
            'true_label': true_label,
            'predicted_label': pred_label,
            'text_snippet': text
        })
        
    if save_path:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        
    return results
