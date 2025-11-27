import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout, ModuleList, PReLU, KLDivLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv, BatchNorm, LayerNorm, GraphNorm
from simulation.simulator import Simulator
import numpy as np

class mlp(torch.nn.Module):
    """
    A multi-layer perceptron (MLP) neural network module.

    Args:
        in_channels (int): Number of input features.
        out_channel (int): Number of output features.
        hidden_dim (int, optional): Number of hidden units in each hidden layer. Default is 64.
        hidden_num (int, optional): Number of hidden layers. Default is 2.
        normalize (bool, optional): Whether to apply batch normalization. Default is False.
        bias (bool, optional): Whether to include bias in the linear layers. Default is True.
    """
    def __init__(self, in_channels, 
                 out_channel, 
                 hidden_dim=64, 
                 hidden_num=2, 
                 normalize=False, 
                 bias=True):
        super().__init__()
        self.layers = [Linear(in_channels, hidden_dim), PReLU()]
        for _ in range(hidden_num):
            self.layers.append(Dropout(0.1))
            self.layers.append(Linear(hidden_dim, hidden_dim, bias=bias))
            if normalize:
                self.layers.append(BatchNorm(in_channels))
            self.layers.append(PReLU())
        self.layers.append(Linear(hidden_dim, out_channel))
        self.mlp = Sequential(*self.layers)
        self._init_parameters()

    def _init_parameters(self):
         for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        """Apply the MLP to the input tensor"""
        return self.mlp(x)

# --- 1. The GNN Part of the Model ---

class SimpleGNN(nn.Module):
    """
    A GNN model for node and edge predictions, with node embedding.
    """
    def __init__(self, in_channels, hidden_channels, node_out_channels, edge_out_channels, num_layers=2):
        super(SimpleGNN, self).__init__()
        self.embed = mlp(in_channels, hidden_channels, hidden_dim=2*hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.node_predictor = mlp(hidden_channels, node_out_channels)
        self.edge_predictor = mlp(2*hidden_channels, edge_out_channels)

    def forward(self, x, edge_index):
        x = self.embed(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        node_preds = self.node_predictor(x)
        row, col = edge_index
        edge_preds = self.edge_predictor(torch.cat([x[row], x[col]], dim=1))

        return node_preds, edge_preds
    
# --- 2. The Simulation ---

def run_simulation(simulator, adj_matrix, noisy_steps=1000, **kwargs):
    """
    Run the simulation part of the prediction pipeline. Uses the results from the GNN, run equilibration and returns concentrations from a noisy simulation.
    
    Args:
        simulator (Simulator): An instance of the Simulator class to run the simulation.
        adj_matrix (torch.Tensor): The adjacency matrix representing the graph structure.
        **kwargs: Additional keyword arguments for the simulator.
    """
    simulator.build_graph(adj_matrix)
    simulator.set_simulation_parameters(**kwargs)

    simulator.run_equilibration()
    simulated_concentrations = simulator.run_noisy_simulation(steps=noisy_steps)

    return simulated_concentrations

# --- 3. The Loss Function ---

def simulation_loss(predicted_concentrations, true_concentrations, loss_fn=KLDivLoss()):
    """
    Compute predicted and true concentrations normalized distributions, then evaluate the loss between the distributions.

    Args:
        predicted_concentrations (torch.Tensor): Predicted concentrations from the simulation.
        true_concentrations (torch.Tensor): True concentrations for comparison.
        loss_fn (callable): A loss function to compute the loss between distributions. Default is KLDivLoss.
    """
    # Binning into distributions
    all_values = torch.cat([predicted_concentrations, true_concentrations], dim=0)
    pos_values = all_values[all_values > 0]
    min_val = pos_values.min()
    max_val = pos_values.max()
    zero_bin_edge = torch.tensor([-1.0, 1e10]).to(predicted_concentrations.device)
    pos_bins_edges = torch.linspace(min_val, max_val, steps=50).to(predicted_concentrations.device)
    bins = torch.cat([zero_bin_edge, pos_bins_edges], dim=0)
    pred_hist = torch.histogram(predicted_concentrations, bins=bins, min=min_val, max=max_val)
    true_hist = torch.histogram(true_concentrations, bins=bins, min=min_val, max=max_val)
    pred_dist = pred_hist / pred_hist.sum()
    true_dist = true_hist / true_hist.sum()

    loss = loss_fn(pred_dist.log(), true_dist)
    return loss
