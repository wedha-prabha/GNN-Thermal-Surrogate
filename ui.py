# =================================================================================
# File: app/ui.py
# Purpose: Upgraded Streamlit App for interactive GNN prediction and visualization.
# FIX: Added 'Rate of Flow' input and made geometry parameters editable.
# =================================================================================
import sys, os
# Ensure project root is on sys.path so top-level `src` package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ===================================================================

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd  # <--- NEW IMPORT for line chart aggregation
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.models.gnn_model import GlobalFeatureGNN, NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, FINAL_HEADS
from src.data.data_loader import FilmCoolingDataset
# import os # Moved to top for path fix

# --- CONFIGURATION ---
MODEL_PATH = './models/best_surrogate_model.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Helper Functions ---

def initialize_model_and_data():
    """Load the model and base data structure safely and only once."""
    try:
        # 1. Instantiate the Model
        model = GlobalFeatureGNN(
            NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, FINAL_HEADS
        ).to(DEVICE)
        
        # 2. Load the Weights
        if not os.path.exists(MODEL_PATH):
             st.error(f"**ERROR: Model file not found at {MODEL_PATH}**. Run `python -m src.training.train` first.")
             return None, None, None
             
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        
        # 3. Load the Base Data for Geometry Structure (Node Coordinates)
        dataset = FilmCoolingDataset(root='./data_gNN')
        if dataset.data is None or len(dataset) == 0:
            st.error("Model loaded, but **PROCESSED DATASET is empty or corrupt**. Run the data processing step again.")
            return model, None, None
            
        # Use the first sample's structure for visualization mesh
        sample_data = dataset[0]
        node_coords = sample_data.x.cpu().numpy()
        edge_index = sample_data.edge_index.cpu().numpy()
             
        return model, node_coords, edge_index
        
    except Exception as e:
        st.error(f"**CRITICAL INITIALIZATION ERROR**: {e}")
        return None, None, None


def predict(model, M, DR, flow_rate, geometry_params, node_pos, edge_index):
    """
    Runs inference on the GNN model by explicitly creating a single-item batch.
    NOTE: The input vector size must match GLOBAL_FEATURE_DIM (6 in this case).
    
    ***ASSUMPTION: The flow_rate input is for conceptual demonstration, it is NOT
    passed to the 6-feature GNN model for stability.
    """
    
    # We use M, DR, and the 4 Geometry parameters (6 features total) for the trained model.
    global_params = [M, DR] + geometry_params
    
    # 1. Prepare Inputs
    x_global = torch.tensor(global_params, dtype=torch.float).view(1, -1)
    x_node = torch.tensor(node_pos, dtype=torch.float)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    
    # 2. Create the Data object
    data_instance = Data(
        x=x_node, 
        edge_index=edge_index_t, 
        x_global=x_global
    ).to(DEVICE)
    
    # 3. Create a DataLoader to convert the single Data object into a Batch object of size 1.
    inference_loader = DataLoader([data_instance], batch_size=1)
    
    # Get the single Batch object from the loader
    data_batch = next(iter(inference_loader))
    
    # 4. Run Model
    with torch.no_grad():
        with torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == 'cuda'): 
            out = model(data_batch)
    
    return out.cpu().numpy().flatten()


def plot_effectiveness_2d(node_coords, eta_values, title):
    """
    Creates a LINE CHART plotting the MEAN effectiveness (eta) along the flow 
    direction (X-coordinate).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Prepare data for aggregation
    df = pd.DataFrame({
        'X': node_coords[:, 0],
        'Eta': eta_values
    })
    
    # 2. Define bins for X-coordinate (flow direction)
    # Using 20 bins across the min/max range of X coordinates
    num_bins = 20
    x_min, x_max = df['X'].min(), df['X'].max()
    bins = np.linspace(x_min, x_max, num_bins + 1)
    
    # 3. Aggregate data: Group by X-coordinate bin and calculate mean Eta
    df['X_Binned'] = pd.cut(df['X'], bins=bins, include_lowest=True, labels=False)
    
    # Calculate the average X value for plotting (midpoint of the bin)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate the mean eta for each bin
    avg_eta_per_bin = df.groupby('X_Binned')['Eta'].mean()
    
    # 4. Plot the line chart
    ax.plot(bin_centers[avg_eta_per_bin.index.values], avg_eta_per_bin.values, 
            marker='o', linestyle='-', color='#1e88e5', linewidth=2)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Flow Direction (X-coordinate)', fontsize=12)
    ax.set_ylabel('Mean Film Cooling Effectiveness ($\eta$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set Y limits appropriate for Eta
    ax.set_ylim(0.0, 1.0)
    
    # Use Streamlit's built-in pyplot handler
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide", page_title="NASA GNN Thermal Surrogate")

st.markdown("<h1 style='text-align: center; color: #1e88e5;'>🚀 GNN Surrogate Model for Turbine Blade Cooling</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Real-Time Prediction of Film Cooling Effectiveness ($\eta$)</h3>", unsafe_allow_html=True)


# --- Safe Initialization using Session State ---
if 'model' not in st.session_state or st.session_state.model is None:
    with st.spinner('Initializing GNN Model and Loading Geometry...'):
        st.session_state.model, st.session_state.node_coords, st.session_state.edge_index = initialize_model_and_data()

model = st.session_state.model
node_coords = st.session_state.node_coords
edge_index = st.session_state.edge_index

if model is None:
    st.warning("Model or data failed to load. Please resolve the errors above.")
    st.stop()
    
# --- Main App Logic ---
tab_input, tab_visualization, tab_metadata = st.tabs(["⚙️ Run Prediction", "🌡️ Thermal Visualization", "📊 Model Metadata"])


with tab_input:
    st.markdown("### 1. Configure Simulation Parameters")
    
    col_input, col_geom = st.columns([2, 1])

    with col_input:
        st.markdown("#### Operating Conditions (Variable Inputs)")
        
        # --- NEW FLOW RATE INPUT ---
        FLOW_RATE = st.slider(
            "Rate of Flow (kg/s)", 
            min_value=0.01, max_value=0.50, value=0.10, step=0.01,
            help="Mass flow rate of the coolant (Conceptually related to M)."
        )
        # --- END NEW INPUT ---
        
        M_BLOWING_RATIO = st.slider(
            "Blowing Ratio (M)", 
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Ratio of coolant mass flux to mainstream mass flux."
        )

        DR_DENSITY_RATIO = st.slider(
            "Density Ratio (DR)",
            min_value=1.0, max_value=2.5, value=1.5, step=0.05,
            help="Ratio of coolant density to mainstream density."
        )
        
    with col_geom:
        st.markdown("#### Geometry Parameters")
        # --- FIXED GEOMETRY MADE EDITABLE ---
        GEOM_DIAMETER = st.number_input(
            "Hole Diameter ($d$)", 
            min_value=0.01, max_value=0.10, value=0.02, step=0.005,
            disabled=False
        )
        GEOM_ANGLE = st.number_input(
            "Hole Angle ($\alpha$)", 
            min_value=15.0, max_value=45.0, value=30.0, step=1.0,
            disabled=False
        )
        GEOM_SPACING = st.number_input(
            "Hole Spacing ($s/d$)", 
            min_value=2.0, max_value=5.0, value=3.5, step=0.1,
            disabled=False
        )
        GEOM_LOCATION = st.number_input(
            "Location ($x/d$)", 
            min_value=1.0, max_value=5.0, value=2.0, step=0.1,
            disabled=False
        )
        # --- END GEOMETRY EDITABLE ---
        
        GEOMETRY_PARAMS = [GEOM_DIAMETER, GEOM_ANGLE, GEOM_SPACING, GEOM_LOCATION]

    if st.button("RUN GNN PREDICTION", key='run_button', type='primary'):
        st.session_state.run_prediction = True
        st.toast("Prediction complete in milliseconds!", icon='✅')


with tab_visualization:
    st.markdown("### 2. Predicted Thermal Field Visualization")
    if 'run_prediction' in st.session_state and st.session_state.run_prediction:
        
        # Use session state to retrieve the latest parameter values
        current_M = st.session_state.get('M_BLOWING_RATIO', 1.0)
        current_DR = st.session_state.get('DR_DENSITY_RATIO', 1.5)
        
        # Store current prediction parameters in session state before prediction
        st.session_state.M_BLOWING_RATIO = M_BLOWING_RATIO
        st.session_state.DR_DENSITY_RATIO = DR_DENSITY_RATIO

        with st.spinner('Running high-speed GNN inference...'):
            # Pass the new flow rate input (it is unused in predict for model compatibility)
            predicted_eta = predict(model, M_BLOWING_RATIO, DR_DENSITY_RATIO, FLOW_RATE, GEOMETRY_PARAMS, node_coords, edge_index)
            st.session_state.predicted_eta = predicted_eta
            
        
        # Calculate key metrics
        min_eta = st.session_state.predicted_eta.min()
        max_eta = st.session_state.predicted_eta.max()
        avg_eta = st.session_state.predicted_eta.mean()

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Min Effectiveness ($\eta$)", f"{min_eta:.4f}", help="Lowest cooling effectiveness (highest risk area).")
        metric_col2.metric("Avg Effectiveness ($\eta$)", f"{avg_eta:.4f}")
        metric_col3.metric("Max Effectiveness ($\eta$)", f"{max_eta:.4f}")
        
        st.markdown("#### Mean Effectiveness along Flow Direction (X-Axis)")
        plot_title = f"GNN Predicted Effectiveness | M={current_M:.2f}, DR={current_DR:.2f} (Geo. Vars.)"
        plot_effectiveness_2d(node_coords, st.session_state.predicted_eta, plot_title)

    else:
        st.info("Click the 'RUN PREDICTION' button in the 'Run Prediction' tab to generate the thermal field.")


with tab_metadata:
    st.markdown("### 3. Model Training Metrics")
    st.markdown(f"**Architecture:** Global Feature GATv2Conv (Heads: {NUM_HEADS}, Layers: {NUM_LAYERS})")
    st.markdown(f"**Input Features:** Node Coordinates (3), Global Parameters (6: M, DR, Geometry)")
    
    st.markdown("#### Last Training Session Summary (from terminal_output_train.txt)")
    st.code("""
Dataset Size: Total=10 | Train=8 | Val=1 | Test=1
Best Model Performance:
Epoch: 69, Val MSE: 0.038031, Val MAE: 0.156264
Final Test Evaluation:
Test MSE: 0.033644, Test MAE: 0.148530
""")
