# =================================================================================
# File: src/models/gnn_model.py
# Purpose: Defines the High-Performance Graph Attention Network (GATv2Conv) model.
# FIX: Uses defensive attribute access (getattr) to bypass internal PyG storage errors.
# =================================================================================
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATv2Conv

# --- HYPERPARAMETERS ---
NODE_FEATURE_DIM = 3
GLOBAL_FEATURE_DIM = 6
HIDDEN_CHANNELS = 128 
NUM_LAYERS = 4
NUM_HEADS = 4      
FINAL_HEADS = 1    
DROPOUT_RATE = 0.3
# -----------------------

class GlobalFeatureGNN(torch.nn.Module):
    def __init__(self, node_in_channels=NODE_FEATURE_DIM, global_in_channels=GLOBAL_FEATURE_DIM,
                 hidden_channels=HIDDEN_CHANNELS, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, final_heads=FINAL_HEADS, dropout_rate=DROPOUT_RATE):
        
        super(GlobalFeatureGNN, self).__init__()
        
        self.lin_start = Linear(node_in_channels, hidden_channels)
        self.lin_global = Linear(global_in_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            in_c = hidden_channels if i == 0 else hidden_channels * num_heads
            out_heads = num_heads if i < num_layers - 1 else final_heads

            conv = GATv2Conv(
                in_c,
                hidden_channels, 
                heads=out_heads, 
                concat=(i < num_layers - 1), 
                dropout=dropout_rate,
                add_self_loops=True
            )
            self.convs.append(conv)
            
            bn_size = hidden_channels * out_heads
            self.bns.append(BatchNorm1d(bn_size))

        self.lin_final = Linear(hidden_channels * final_heads, 1)

    def forward(self, data):
        # Defensive Access: Use getattr() to safely access features and avoid deep proxy crash
        x = getattr(data, 'x', None)
        edge_index = getattr(data, 'edge_index', None)
        x_global = getattr(data, 'x_global', None)

        if x is None:
            raise ValueError('Data.x (node features) is required')

        # --- Step 1: Initial Feature Projection ---
        x = F.relu(self.lin_start(x))

        # --- Step 2: Global Feature Injection (Broadcasting) ---
        
        # Safely get batch vector; if none exists (single inference), create a dummy index (0...0)
        batch_tensor = getattr(data, 'batch', None)
        if batch_tensor is None:
            batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Project global features
        if x_global is None:
            projected_global = torch.zeros((x.size(0), self.lin_global.out_features), device=x.device)
        else:
            projected_global = self.lin_global(x_global)
            
        # Broadcast global features using the batch tensor
        try:
            global_features_broadcasted = projected_global[batch_tensor]
        except Exception:
            # Fallback if the Batch tensor is malformed or if Global is not indexed by batch (shouldn't happen with Batch object)
            global_features_broadcasted = projected_global.repeat(x.size(0), 1)

        x = x + global_features_broadcasted
        
        # --- Step 3: GNN Message Passing ---
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=DROPOUT_RATE, training=self.training)

        # --- Step 4: Final Prediction ---
        out = self.lin_final(x)
        return out
