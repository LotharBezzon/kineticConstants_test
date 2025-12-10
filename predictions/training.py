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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    db_path = 'simulation/simulated_graph_dataset'
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
    model = SimpleGNN(in_channels=2000, hidden_channels=256, node_out_channels=4, edge_out_channels=2).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()

    ckpt_dir = 'model_checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    adj_matrix_df = pd.read_csv('simulation/adjacency_matrix.csv', index_col=0)
    all_lipids = adj_matrix_df.index.tolist()
    adj_matrix = np.array(adj_matrix_df)
    symmetric_adj_matrix = (adj_matrix + adj_matrix.T) > 0
    adj_matrix = symmetric_adj_matrix.astype(int)
    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    L = np.linalg.cholesky(correlation_matrix.values)

    epochs = 10  # Number of training epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            node_out, edge_out = model(batch.x, batch.edge_index, batch.batch)
            k_prod, k_deg, sigma_conc, dropout = node_out.split([1, 1, 1, 1], dim=-1)
            k, sigma_k = edge_out.split([1, 1], dim=-1)
            #k = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes, edge_attr=k).squeeze(0)
            '''sigma_k = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes, edge_attr=sigma_k).squeeze(0)
            print(maxk:=k.max().item(), mink:=k.min().item())
            print(maxsk:=sigma_k.max().item(), minsk:=sigma_k.min().item())
            print(maxpc:=k_prod.max().item(), minpc:=k_prod.min().item())
            print(maxdc:=k_deg.max().item(), mindc:=k_deg.min().item())

            predicted_concentrations = run_simulation(
                simulator=Simulator(),
                adj_matrix=adj_matrix,
                noisy_steps=1000,
                log_kinetic_constants=k.squeeze(-1).cpu().detach().numpy(),
                log_degradation_constants=k_deg.squeeze(-1).cpu().detach().numpy(),
                log_production_constants=k_prod.squeeze(-1).cpu().detach().numpy(),
                concentration_noise=sigma_conc.squeeze(-1).cpu().detach().numpy(),
                log_kinetic_constants_noise=sigma_k.cpu().detach().numpy(),
                dropout=dropout.squeeze(-1).cpu().detach().numpy(),
                L=L
            )'''

            #k_true = torch.block_diag(*torch.tensor(np.array(batch.parameters['kinetic_constants']), dtype=torch.float32).to(device))
            k_true = torch.tensor(np.array(batch.parameters['sparse_log_kinetic_constants']), dtype=torch.float32).reshape(-1).to(device)
            k_prod_true = torch.tensor(np.array(batch.parameters['production_constants']), dtype=torch.float32).reshape(-1).to(device)
            k_deg_true = torch.tensor(np.array(batch.parameters['degradation_constants']), dtype=torch.float32).reshape(-1).to(device)

            # Compute loss
            loss_k = mse_loss(k.squeeze(-1), k_true)
            loss_prod = mse_loss(k_prod.squeeze(-1), k_prod_true)
            loss_deg = mse_loss(k_deg.squeeze(-1), k_deg_true)
            loss = loss_k + loss_prod + loss_deg
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_checkpoint_epoch_{epoch+1}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f'optimizer_checkpoint_epoch_{epoch+1}.pt'))
            
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    