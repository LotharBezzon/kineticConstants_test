from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import lmdb
import pickle
from tqdm import tqdm
from classification.classifier import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
data_path = 'simulation/simulation_files/simulated_data_0.parquet'
dataset = LMDBDataset(db_path=data_path)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Initialize model, loss function, and optimizer
model = SimpleGNN(in_channels=1000, hidden_channels=64, node_out_channels=4, edge_out_channels=2).to(device)
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

