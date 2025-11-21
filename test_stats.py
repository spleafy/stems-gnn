from statistical_significance import mcnemar_test, bootstrap_confidence_interval
from sklearn.metrics import f1_score
import numpy as np

def test_stats():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred1 = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]) # 90% accuracy
    y_pred2 = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0]) # Worse

    chi2, p = mcnemar_test(y_true, y_pred1, y_pred2)
    print(f"McNemar: chi2={chi2}, p={p}")
    
    low, high = bootstrap_confidence_interval(y_true, y_pred1, f1_score, n_bootstraps=50)
    print(f"Bootstrap CI: ({low}, {high})")
    
    print("Statistical module test passed.")

if __name__ == "__main__":
    test_stats()
