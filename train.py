# =================================================================================
# File: src/training/train.py
# Purpose: Main script for high-performance training using AMP and GATConv.
# Fix: Suppresses PyTorch FutureWarnings and PyG UserWarnings for clean output.
# =================================================================================
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from src.models.gnn_model import GlobalFeatureGNN, NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, FINAL_HEADS
from src.data.data_loader import FilmCoolingDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import warnings

# Suppress all future warnings for a clean console
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- HYPERPARAMETERS & TARGETS ---
LEARNING_RATE = 0.0005  
EPOCHS = 300            
BATCH_SIZE = 32         
DATASET_ROOT = './data_gNN'
TRAIN_SPLIT_RATIO = 0.8
# ---------------------------------

# 1. Setup Data and Split
print("Setting up data loader...")
dataset = FilmCoolingDataset(root=DATASET_ROOT) 

# Ensure data is processed if it failed earlier
if dataset._data is None or len(dataset) == 0: # Use _data to suppress UserWarning
    print("Processed dataset missing or failed to load. Running processing step...")
    dataset.process()
    # Reload dataset after processing
    dataset = FilmCoolingDataset(root=DATASET_ROOT)
    
if dataset._data is None or len(dataset) == 0:
    raise FileNotFoundError("Dataset is empty. Ensure you have CFD .h5 files in 'data_gNN/raw/' and processing is successful.")

data_indices = list(range(len(dataset)))
train_indices, temp_indices = train_test_split(data_indices, train_size=TRAIN_SPLIT_RATIO, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42) 

train_set = dataset[torch.tensor(train_indices)]
val_set = dataset[torch.tensor(val_indices)]
test_set = dataset[torch.tensor(test_indices)]

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset Size: Total={len(dataset)} | Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")


# 2. Model, Optimizer, Loss, and AMP Scaler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GlobalFeatureGNN(
    NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, FINAL_HEADS
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) 
criterion = torch.nn.MSELoss() 
# FIX: Use recommended torch.amp.GradScaler API
scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda') 


# 3. Training Function (uses AMP)
def train():
    model.train()
    total_loss = 0
    
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # FIX: Use recommended torch.amp.autocast API
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            out = model(data)
            loss = criterion(out, data.y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_set)

# 4. Evaluation Function (Calculates MAE)
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_nodes = 0
    
    for data in tqdm(loader, desc="Evaluating"):
        data = data.to(device)
        # FIX: Use recommended torch.amp.autocast API
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            out = model(data)
            loss = criterion(out, data.y)
        
        total_loss += loss.float().item() * data.num_graphs
        
        abs_err = torch.abs(out.float() - data.y.float())
        total_mae += torch.sum(abs_err).item()
        total_nodes += data.y.size(0)

    avg_mse = total_loss / len(loader.dataset)
    avg_mae = total_mae / total_nodes
    return avg_mse, avg_mae


# 5. Main Training Loop
print("Starting Training...")
best_val_mae = float('inf')

os.makedirs('./models', exist_ok=True) # Ensure models folder exists

for epoch in range(1, EPOCHS + 1):
    train_loss = train()
    val_mse, val_mae = evaluate(val_loader)

    print(f'Epoch: {epoch:03d}, Train MSE: {train_loss:.6f}, Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}')
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        # Save the model
        torch.save(model.state_dict(), './models/best_surrogate_model.pt')
        print(f"--> New best model saved with Val MAE: {best_val_mae:.6f}")
        
    if epoch >= 300: # Stop if we hit the limit
        break

# 6. Final Test Evaluation
print("\n--- Final Test Evaluation ---")
# Use map_location to ensure it loads even if GPU isn't available later
model.load_state_dict(torch.load('./models/best_surrogate_model.pt', map_location=device)) 
test_mse, test_mae = evaluate(test_loader)
print(f'Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}')
