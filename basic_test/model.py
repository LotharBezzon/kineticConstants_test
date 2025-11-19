# GNN model for edge feature prediction
import torch
from torch.nn import Sequential, Linear, Dropout, ModuleList, PReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing, GATConv, BatchNorm, LayerNorm, GraphNorm


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

class MyConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
        self.mlp = mlp(2*in_channels, out_channels)
        self.mlp_out = mlp(out_channels, out_channels)

    def forward(self, edge_index, x):
        aggregated_message = self.propagate(edge_index, x=x)
        return self.mlp_out(aggregated_message)
    
    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

class EdgePredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_out_channels):
        super().__init__()
        self.conv1 = MyConv(in_channels, hidden_channels)
        self.conv2 = MyConv(hidden_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        self.edge_mlp = mlp(2 * hidden_channels, edge_out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(edge_index, x)
        x = self.norm1(x)
        x = self.conv2(edge_index, x)
        x = self.norm2(x)

        # For each edge, concatenate the embeddings of the two nodes
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        pred_edge_attr = self.edge_mlp(edge_features)

        return pred_edge_attr
