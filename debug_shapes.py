
import torch
import numpy as np
import pandas as pd
import os
import sys
from torch_geometric.loader import DataLoader
from simulation.produce_simulations import SimulatedGraphDataset
from simulation.simulator import add_baths, get_biggest_submatrix

def debug():
    project_root = os.path.abspath(os.getcwd())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    db_path = 'simulation/simulated_graph_dataset_only_steady_state_free_energies'
    if not os.path.exists(db_path):
        print(f"Dataset not found at {db_path}")
        return

    dataset = SimulatedGraphDataset(root=db_path)
    
    loader = DataLoader(dataset[:2], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    
    print(f"Type of batch.parameters: {type(batch.parameters)}")
    if isinstance(batch.parameters, list):
        print(f"First element type: {type(batch.parameters[0])}")
        try:
            print(batch.parameters['sparse_deltaG'])
        except Exception as e:
            print(f"Accessing ['sparse_deltaG'] failed: {e}")
    else:
        print("batch.parameters is not a list")
        # Try to access it as if it was a dict of lists (which PyG might have collated it into if it was smart enough, but usually it's not for dicts)
        try:
            print(f"Keys: {batch.parameters.keys()}")
        except:
            pass
    
    mask = torch.kron(torch.eye(batch.num_graphs), torch.tensor(big_adj_matrix)).bool()
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum: {mask.sum()}")
    
    deltaG_masked = deltaG[mask]
    print(f"Masked deltaG shape: {deltaG_masked.shape}")
    
    deltaG_true = torch.from_numpy(sparse_deltaG).float().reshape(-1)
    print(f"deltaG_true shape: {deltaG_true.shape}")
    
    if deltaG_masked.shape[0] != deltaG_true.shape[0]:
        print("MISMATCH DETECTED!")
    else:
        print("Shapes match (unexpectedly)")

if __name__ == "__main__":
    debug()
