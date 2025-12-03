from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
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
    torch.manual_seed(42)
    dataset = dataset.shuffle()  # Shuffle the dataset
    print(f'Dataset size: {len(dataset)} graphs')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    batch_size = 32  # Adjust batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for batch in train_loader:
        num_nodes = batch.x.size(0) / batch_size
        num_edges = batch.edge_index.size(1) / batch_size
        break

    # Initialize model, loss function, and optimizer
    model = mlp(in_channels=int(num_nodes), out_channel=int(num_edges), hidden_dim=[int(0.5*num_nodes), int(0.3*num_nodes), int(0.2*num_nodes), int(0.1*num_nodes)], hidden_num=4).to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    epochs = 10  # Number of training epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in tqdm(range(10)):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Flatten node features for the entire graph in the batch
                x_dense, mask = to_dense_batch(batch.x[:, i].clone(), batch.batch)   # [B, Nmax, F]
                x_flat = x_dense.view(batch.batch_size, -1)                  # [B, Nmax*F]
                outputs = model(x_flat)
                targets = torch.as_tensor(np.array(batch.parameters['kinetic_constants']), dtype=torch.float32, device=device)
                targets = torch.log10(targets[targets != 0]).view_as(outputs)
                loss = mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f'model_checkpoints/mlp_epoch_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), f'model_checkpoints/optimizer_epoch_{epoch+1}.pth')