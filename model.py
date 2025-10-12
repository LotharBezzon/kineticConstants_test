# GNN model for edge feature prediction
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EdgePredictorGNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, edge_out_channels):
		super().__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.edge_mlp = torch.nn.Sequential(
			torch.nn.Linear(2 * hidden_channels, hidden_channels),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_channels, edge_out_channels)
		)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		x = F.relu(x)

		# For each edge, concatenate the embeddings of the two nodes
		row, col = edge_index
		edge_features = torch.cat([x[row], x[col]], dim=1)
		pred_edge_attr = self.edge_mlp(edge_features)
		return pred_edge_attr
