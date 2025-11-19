from main import run_multi_seed_evaluation
import sys

import main

def mock_run(seed=42):
    print(f"Mock run with seed {seed}")
    results = {
        'roberta_baseline': {'metrics': {'accuracy': 0.8, 'precision': 0.8, 'recall': 0.8, 'f1': 0.8, 'auc': 0.8}},
        'correct_semantic_gnn': {'metrics': {'accuracy': 0.85, 'precision': 0.85, 'recall': 0.85, 'f1': 0.85, 'auc': 0.85}}
    }
    return results, {}, {}

main.run_comprehensive_comparison = mock_run

if __name__ == "__main__":
    print("Testing multi-seed evaluation logic...")
    run_multi_seed_evaluation(num_seeds=2)
    print("Test complete.")
