from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import lmdb
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

import sys
import os


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from classification.classifier import *
    from model import *
    from simulation.produce_simulations import SimulatedGraphDataset
    from simulation.simulator import add_baths, get_biggest_submatrix

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    db_path = 'simulation/simulated_graph_small_dataset_only_steady_state_free_energies'
    dataset = SimulatedGraphDataset(root=db_path)
    dataset = dataset[:10000]
    torch.manual_seed(42)
    dataset = dataset.shuffle()  # Shuffle the dataset
    print(f'Dataset size: {len(dataset)} graphs')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    batch_size = 32  # Adjust batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model, loss function, and optimizer
    #model = SimpleGNN(in_channels=2000, hidden_channels=256, node_out_channels=4, edge_out_channels=2).to(device)
    #model_steady_state = SimpleGNN(in_channels=1, hidden_channels=64, node_out_channels=4, edge_out_channels=2).to(device)
    model = SimpleGNN(in_channels=2, hidden_channels=64, node_out_channels=1, edge_out_channels=1).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    #optimizer_steady_state = torch.optim.Adam(model_steady_state.parameters(), lr=0.01)
    #scheduler_steady_state = torch.optim.lr_scheduler.StepLR(optimizer_steady_state, step_size=1, gamma=0.8)
    mse_loss = nn.MSELoss()

    ckpt_dir = 'model_checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    adj_matrix_df = pd.read_csv('simulation/adjacency_matrix.csv', index_col=0)
    all_lipids = adj_matrix_df.index.tolist()
    adj_matrix = np.array(adj_matrix_df)
    symmetric_adj_matrix = (adj_matrix + adj_matrix.T) > 0
    adj_matrix = symmetric_adj_matrix.astype(int)
    adj_matrix, _ = get_biggest_submatrix(adj_matrix)
    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    L = np.linalg.cholesky(correlation_matrix.values)
    big_adj_matrix = add_baths(adj_matrix)

    # Pre-compute mask as tensor on device (moved outside training loop for speed)
    symmetric_adj_mask = torch.tensor(symmetric_adj_matrix > 0, dtype=torch.bool, device=device)
    adj_shape_0, adj_shape_1 = symmetric_adj_matrix.shape[0], symmetric_adj_matrix.shape[1]

    epochs = 10  # Number of training epochs
    model_type = 'free_energies'  # 'kinetic_constants' or 'free_energies'

    '''for graph in train_dataset[1:]:
        fe_true = np.array(graph.parameters['free_energies'])
        row, col = graph.edge_index
        print(graph.x)
        #print(np.array(graph.parameters['sparse_all_deltaG']))# - (fe_true[col.cpu().numpy()] - fe_true[row.cpu().numpy()]))
        break'''

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            #batch.x[:, 0] = torch.log(batch.x[:, 0] + 1e-10)  # Log-transform concentrations
            optimizer.zero_grad()
            
            # Forward pass
            if model_type == 'free_energies':
                free_energy_true = torch.from_numpy(np.array(batch.parameters['free_energies'])).float().to(device)
                free_energy_true = free_energy_true.reshape(-1)
                deltaG_true = torch.from_numpy(np.array(batch.parameters['sparse_all_deltaG'])).float().to(device).reshape(-1)
                row, col = batch.edge_index

                node_out, edge_out = model(batch.x, batch.edge_index, batch.batch, free_energies=False, add_baths=False)
                free_energies = node_out.squeeze(-1)
                energy_barriers = edge_out.squeeze(-1)

                deltaG = free_energies[col] - free_energies[row]
                
                loss_fe = mse_loss(free_energies, free_energy_true)
                loss_deltaG = mse_loss(deltaG, deltaG_true)
                loss = loss_deltaG + loss_fe

            else:  # 'kinetic_constants'
                node_out, edge_out = model(batch.x, batch.edge_index, batch.batch)
                k_prod, k_deg, sigma_conc, dropout = node_out.split([1, 1, 1, 1], dim=-1)
                k, sigma_k = edge_out.split([1, 1], dim=-1)

                k_true = torch.tensor(np.array(batch.parameters['sparse_log_kinetic_constants']), dtype=torch.float32).reshape(-1).to(device)
                k_prod_true = torch.tensor(np.array(batch.parameters['production_constants']), dtype=torch.float32).reshape(-1).to(device)
                k_deg_true = torch.tensor(np.array(batch.parameters['degradation_constants']), dtype=torch.float32).reshape(-1).to(device)

                # Compute loss
                loss_k = mse_loss(k.squeeze(-1), k_true)
                loss_prod = mse_loss(k_prod.squeeze(-1), k_prod_true)
                loss_deg = mse_loss(k_deg.squeeze(-1), k_deg_true)
                loss = loss_k + loss_prod + loss_deg
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_checkpoint_epoch_{epoch+1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f'optimizer_checkpoint_epoch_{epoch+1}.pt'))
            
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    