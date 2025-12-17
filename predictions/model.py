import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout, ModuleList, PReLU, KLDivLoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv, BatchNorm, LayerNorm, GraphNorm, global_mean_pool
#from simulation.simulator import Simulator
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
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]*(hidden_num+1)
        self.layers = [Linear(in_channels, hidden_dim[0], bias=bias), PReLU()]
        for i in range(hidden_num-1):
            self.layers.append(Dropout(0.1))
            self.layers.append(Linear(hidden_dim[i], hidden_dim[i+1], bias=bias))
            if normalize:
                self.layers.append(BatchNorm(in_channels))
            self.layers.append(PReLU())
        self.layers.append(Linear(hidden_dim[-1], out_channel, bias=bias))
        self.mlp = Sequential(*self.layers)
        self._init_parameters()

    def _init_parameters(self):
         for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        """Apply the MLP to the input tensor"""
        return self.mlp(x)
    
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Use nn.Embedding for learnable position vectors
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Lookup embeddings and add to input
        return x + self.position_embedding(positions)

# --- 1. The GNN Part of the Model ---

class MessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MessagePassingLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.mlp = mlp(2*in_channels, out_channels, hidden_dim=2*out_channels, hidden_num=1)
        self.mlp_out = mlp(out_channels, out_channels, hidden_dim=2*out_channels, hidden_num=1)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Start propagating messages
        return self.mlp_out(self.propagate(edge_index, x=x))

    def message(self, x_i, x_j):
        # x_j has shape [E, out_channels]
        return self.mlp(torch.cat([x_i, x_j], dim=1))


class SimpleGNN(nn.Module):
    """
    A GNN model for node and edge predictions, with node embedding.
    """
    def __init__(self, in_channels, hidden_channels, node_out_channels, edge_out_channels, num_layers=2):
        super(SimpleGNN, self).__init__()
        #self.pos_encoder = LearnablePositionalEncoding(d_model=in_channels, max_len=1000)
        self.in_channels = in_channels
        self.embed = mlp(in_channels, hidden_channels, hidden_dim=2*hidden_channels)
        self.graph_encoder = mlp(hidden_channels, hidden_channels, hidden_dim=2*hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(2*hidden_channels, 2*hidden_channels))
        self.convs.append(GCNConv(2*hidden_channels, hidden_channels))
        self.node_predictor = mlp(hidden_channels, node_out_channels, bias=False)
        self.edge_predictor = mlp(2*hidden_channels, edge_out_channels)

    def forward(self, x, edge_index, batch=None, return_embeddings=False, free_energies=False, add_baths=False):
        #x = self.pos_encoder(x.unsqueeze(1)).squeeze(1)
        if add_baths:   # for each node add a bath node
            baths_feat = torch.ones((x.size(0), x.size(1)), device=x.device)
            baths_idxes = torch.arange(x.size(0), 2 * x.size(0), device=x.device)
            x = torch.cat([x, baths_feat], dim=0)
            
            baths_edges = torch.tensor([[i, baths_idxes[i]] for i in range(baths_idxes.size(0))] + 
                                        [[baths_idxes[i], i] for i in range(baths_idxes.size(0))], dtype=torch.long, device=x.device).t()
            edge_index = torch.cat([edge_index, baths_edges], dim=1)
            if batch is not None:
                batch = torch.cat([batch, torch.tensor([batch.max()+1]*baths_idxes.size(0), device=x.device)], dim=0)
        x = self.embed(x)
        g = global_mean_pool(x, batch)
        h = torch.cat([x, self.graph_encoder(g)[batch]], dim=1)

        for conv in self.convs[:-1]:
            h = h + F.relu(conv(h, edge_index))
        h = self.convs[-1](h, edge_index)
        final_embedding = x + h
        node_preds = self.node_predictor(final_embedding)
        row, col = edge_index
        if free_energies:
            mask = row < col
            row, col = row[mask], col[mask]
        edge_preds = self.edge_predictor(torch.cat([final_embedding[row], final_embedding[col]], dim=1))

        if return_embeddings:
            return node_preds, edge_preds, final_embedding
        else:
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
    simulator.build_graph(adjacency_matrix=adj_matrix)
    simulator.set_simulation_parameters(**kwargs)

    simulator.run_equilibration()
    simulated_concentrations = simulator.run_noisy_simulation(steps=noisy_steps)

    return simulated_concentrations.T   # shape: (num_nodes, noisy_steps)

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
