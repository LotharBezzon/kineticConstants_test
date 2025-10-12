### BUILD A GRAPH FROM DATA IN .CSV FILE

import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
import torch
import os

# --- LOAD DATA ---
# TODO: change format of the file to one better suited for graph construction
data = pd.read_csv(os.path.join('data', 'toy_steady_states_multistage_corrected.csv'))

# --- BUILD GRAPH ---

# Build node features matrix: shape (num_samples, 5, 2) where each node has [v0, vss]
v0 = torch.stack([torch.tensor(data[f'v0_{i}'].values, dtype=torch.float) for i in range(1, 6)], dim=1)  # (num_samples, 5)
vss = torch.stack([torch.tensor(data[f'vss_{i}'].values, dtype=torch.float) for i in range(1, 6)], dim=1)  # (num_samples, 5)
node_features_all = torch.stack([v0, vss], dim=2)  # (num_samples, 5, 2)

# Add a sixth node (particle_bath) with features [1, 1] for each sample
particle_bath = torch.ones((node_features_all.shape[0], 1, 2), dtype=torch.float)  # (num_samples, 1, 2)
node_features_all = torch.cat([particle_bath, node_features_all], dim=1)  # (num_samples, 6, 2)

# Create a list of Data objects, each with 6 nodes (1 sample per Data object), node features are [v0, vss, particle_bath]
# Identify all kij columns (e.g., k12, k23, etc.)
kij_cols = [col for col in data.columns if col.startswith('k') and len(col) == 3 and col[1].isdigit() and col[2].isdigit()]

data_list = []
for idx in range(node_features_all.shape[0]):
	node_features = node_features_all[idx]  # shape (6, 2)
	edge_index = []
	edge_attr = []
	for kij in kij_cols:
		i = int(kij[1])
		j = int(kij[2])
		edge_index.append([i, j])
		edge_attr.append([data[kij].iloc[idx]])
	if edge_index:
		edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape (2, num_edges)
		edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # shape (num_edges, 1)
	else:
		edge_index = torch.empty((2, 0), dtype=torch.long)
		edge_attr = torch.empty((0, 1), dtype=torch.float)
	data_obj = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
	data_list.append(data_obj)
print(f'Created {len(data_list)} Data objects, each with {data_list[0].num_nodes} nodes, node feature shape {data_list[0].x.shape}, and {data_list[0].num_edges} edges.')

# --- SPLIT INTO TRAIN/VAL AND BATCH FOR GNN ---
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# Split data_list into train and validation sets (e.g., 80% train, 20% val)
train_list, val_list = train_test_split(data_list, test_size=0.2, random_state=42)

# Set batch size
BATCH_SIZE = 32

# Create DataLoaders for batching
train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_list, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_list)}, Validation samples: {len(val_list)}")

