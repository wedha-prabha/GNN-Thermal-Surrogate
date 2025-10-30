# =================================================================================
# File: src/data/preprocess.py
# Purpose: Defines data transformation functions (scaling) for stability.
# =================================================================================
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Path to save/load the fitted scaler
SCALER_PATH = './src/data/node_scaler.pkl'

def fit_and_transform_nodes(dataset, save_path=SCALER_PATH):
    """
    Fits a StandardScaler on all node positions (x, y, z) across the entire dataset 
    and returns a PyG transform function.
    """
    # 1. Collect all node features (x, y, z coordinates)
    all_nodes = []
    for data in dataset:
        all_nodes.append(data.x.cpu().numpy())
    
    # Concatenate all nodes into a single array
    all_nodes_flat = np.concatenate(all_nodes, axis=0)
    
    # 2. Fit the scaler
    scaler = StandardScaler()
    scaler.fit(all_nodes_flat)
    
    # 3. Save the fitted scaler object
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Node StandardScaler fitted and saved to {save_path}")

    # 4. Define the PyG-compatible transform function
    def transform_func(data):
        # Apply the transformation when loading data
        x_numpy = data.x.cpu().numpy()
        x_scaled = scaler.transform(x_numpy)
        data.x = torch.tensor(x_scaled, dtype=torch.float)
        return data

    return transform_func

# NOTE: The implementation of this scaling logic should ideally be integrated 
# into `data_loader.py`'s `process` method to ensure the same scaler is 
# used for both training data creation and single-point prediction in `app/ui.py`.