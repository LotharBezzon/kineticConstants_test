import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_edge
from loader_classifier import create_multi_graph_dataloader

# --- 1. The Loss Function (Vectorized for any n_views) ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        features: [batch_size, n_views, output_dim]
        labels: [batch_size]
        """
        device = features.device
        
        # Check input shape
        if len(features.shape) < 3:
            # If user forgot to stack views, add the dimension manually
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # Flatten views to [batch_size * n_views, output_dim]
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = F.normalize(features, dim=1)

        # Similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Labels mask
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            # Repeat mask to match the flattened features shape
            mask = mask.repeat(n_views, n_views)
        else:
            # Fallback for self-supervised (identity mask)
            mask = torch.eye(batch_size * n_views, dtype=torch.float32).to(device)

        # Mask out self-contrast (diagonal)
        # We don't want the model to learn that sample i is similar to itself
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute Logits
        logits = similarity_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute Loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Prevent NaN if a class only has 1 sample in the batch (and n_views=1)
        # If mask.sum(1) is 0, it means this sample has no other positives in the batch.
        # We divide by 1 to avoid error, and the numerator (mask * log_prob) will be 0 anyway.
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()
        return loss

# --- 2. The Graph Encoder + Projector ---

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim=64, output_dim=32):
        super(GraphEncoder, self).__init__()
        
        # GNN Backbone
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Projection Head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        h = global_mean_pool(x, batch)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# --- 3. Graph Augmentation ---

def augment_graph(data, p=0.2):
    edge_index, _ = dropout_edge(data.edge_index, p=p, force_undirected=True)
    return edge_index

# --- 4. Training Loop ---

if __name__ == "__main__":
    # CONFIGURATION
    USE_AUGMENTATIONS = False  # <--- Set to False to use natural positives (other graphs)
    
    NUM_NODE_FEATURES = 1
    HIDDEN_DIM = 64
    BATCH_SIZE = 32 # Larger batch size is better for SupCon
    
    # Create Dummy Data
    # Ensure we have multiple graphs with the same label for USE_AUGMENTATIONS=False to work
    dataset = []
    for i in range(1):
        dataset += create_multi_graph_dataloader(
            adj_path='classification/adjacency_matrix.csv',
            parquet_path=f'simulation/simulation_files/simulated_data_{i}.parquet',
            batch_size=16,
            label_column='simulation_index'
        )
        

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = GraphEncoder(num_node_features=NUM_NODE_FEATURES, hidden_dim=HIDDEN_DIM)
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Starting Training (Augmentations: {USE_AUGMENTATIONS})...")
    
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            
            if USE_AUGMENTATIONS:
                # Approach A: Two Views (Augmentations)
                # Create two distorted versions of the SAME graph
                aug_edge_1 = augment_graph(batch, p=0.2)
                z1 = model(batch.x, aug_edge_1, batch.batch)
                
                aug_edge_2 = augment_graph(batch, p=0.2)
                z2 = model(batch.x, aug_edge_2, batch.batch)
                
                # Stack: [Batch, 2, Dim]
                features = torch.stack([z1, z2], dim=1)
                
            else:
                # Approach B: Natural Positives
                # Use the original graph exactly as is.
                # The Loss will find OTHER graphs in the batch with the same label.
                z = model(batch.x, batch.edge_index, batch.batch)
                
                # Add dummy view dimension: [Batch, 1, Dim]
                features = z.unsqueeze(1)

            # Calculate Loss
            loss = criterion(features, labels=batch.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    print("\nTraining Complete.")