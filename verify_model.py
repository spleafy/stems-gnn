import yaml
import torch
from semantic_ego_gnn import CorrectSemanticGNN

def verify():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    input_dim = config['models']['semantic_gnn']['input_dim']
    temporal_dim = config['models']['semantic_gnn']['feature_breakdown']['temporal_dims']
    
    print(f"Config Input Dim: {input_dim}")
    print(f"Config Temporal Dim: {temporal_dim}")
    
    model = CorrectSemanticGNN(
        input_dim=input_dim,
        hidden_dim=128,
        dropout=0.3,
        has_temporal=True,
        temporal_dim=temporal_dim
    )
    
    print("Model initialized successfully.")
    print(f"Model Input Dim: {model.input_dim}")
    print(f"Model Temporal Dim: {model.temporal_dim}")
    
    from torch_geometric.data import Data
    x = torch.randn(10, input_dim)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn(2)
    feature_indices = list(range(input_dim))
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, feature_indices=feature_indices)
    
    try:
        out = model(data)
        print("Forward pass successful.")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    verify()
