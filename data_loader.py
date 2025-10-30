# =================================================================================
# File: src/data/data_loader.py
# Purpose: Load CFD data (mesh and results) and convert it into PyG Data objects.
# =================================================================================
import os
import h5py
import numpy as np
import torch
import torch.serialization
from torch_geometric.data import Data, InMemoryDataset
from typing import List

# --- FIX for PyTorch 2.6+ Safe Unpickling (best-effort) ---
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # type: ignore
    try:
        from torch_geometric.data.storage import GlobalStorage  # type: ignore
    except Exception:
        GlobalStorage = None

    safe_list = [DataEdgeAttr, DataTensorAttr]
    if GlobalStorage is not None:
        safe_list.append(GlobalStorage)

    torch.serialization.add_safe_globals(safe_list)
except Exception:
    pass
# -------------------------------------------------------

# --- CONSTANTS ---
NODE_FEATURE_DIM = 3
GLOBAL_FEATURE_DIM = 6
# -----------------


class FilmCoolingDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        processed_path = self.processed_paths[0]

        if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
            try:
                self.data, self.slices = torch.load(processed_path)
            except Exception as e:
                print(f"Warning: failed to load processed dataset from '{processed_path}': {e}\nWill attempt to re-run process() when requested.")
                self.data, self.slices = None, None
        else:
            print(f"Processed dataset not found or empty at '{processed_path}'. You can create it by placing raw .h5 files in '{self.raw_dir}' and calling the dataset's process() method.")
            self.data, self.slices = None, None

    @property
    def raw_file_names(self) -> List[str]:
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.h5')]

    @property
    def processed_file_names(self) -> str:
        return 'film_cooling_data.pt'

    def process(self):
        data_list = []

        for raw_file in self.raw_file_names:
            file_path = os.path.join(self.raw_dir, raw_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    params = f['params'][:].astype(np.float32)
                    x_global = torch.tensor(params).view(1, -1)

                    node_pos = f['nodes'][:].astype(np.float32)
                    x_node = torch.tensor(node_pos, dtype=torch.float)

                    edge_index = f['edges'][:].astype(np.int64).T
                    edge_index = torch.tensor(edge_index, dtype=torch.long)

                    eta = f['effectiveness'][:].astype(np.float32)
                    y = torch.tensor(eta, dtype=torch.float).view(-1, 1)

                    data = Data(x=x_node, edge_index=edge_index, y=y, x_global=x_global)
                    data_list.append(data)

            except Exception as e:
                print(f"Error processing file {raw_file}: {e}")
                continue

        if len(data_list) > 0:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print(f"Successfully processed {len(data_list)} graphs and saved to {self.processed_file_names}")
        else:
            print("No raw data files found or processed successfully. Skipping save.")
