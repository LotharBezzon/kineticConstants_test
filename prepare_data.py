### BUILD A GRAPH FROM DATA IN .CSV FILE

import numpy as np
import pandas as pd
import torch_geometric
import torch
import os

# --- LOAD DATA ---
# TODO: change format of the file to one better suited for graph construction
data = pd.read_csv(os.path.join('data', 'toy_steady_states_multistage_corrected.csv'))

# --- BUILD GRAPH ---
x = torch.tensor([data[f'v0_{i}'].values for i in range(1, 6)], dtype=torch.float)  # Node features: v0 for each species
print(x.shape)
