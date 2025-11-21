import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def create_multi_graph_dataloader(adj_path, parquet_path, label_column=None):
    """
    Creates a DataLoader where each row in the parquet file is treated as a separate graph 
    with the same fixed topology but different node features.

    Args:
        adj_path (str): Path to the adjacency matrix CSV (defines topology).
        parquet_path (str): Path to the parquet file (rows=samples, cols=nodes).
        label_column (str, optional): Name of the column in parquet to be used as graph label (y).
        batch_size (int): Number of graphs per batch.

    Returns:
        DataLoader: Iterable over batches of graphs.
    """
    
    # --- 1. Load Data ---
    print(f"Loading topology from {adj_path}...")
    adj_df = pd.read_csv(adj_path, index_col=0)
    
    print(f"Loading samples from {parquet_path}...")
    try:
        samples_df = pd.read_parquet(parquet_path)
    except ImportError:
        raise ImportError("Install 'pyarrow' or 'fastparquet' to read the parquet file.")

    # --- 2. Align Columns (Nodes) ---
    # The columns of the parquet file should match the nodes in the adjacency matrix
    adj_nodes = set(adj_df.index)
    sample_columns = set(samples_df.columns)
    
    # Find intersection (lipid species present in both)
    common_nodes = sorted(list(adj_nodes.intersection(sample_columns)))
    
    if not common_nodes:
        raise ValueError("No overlap found between Adjacency nodes and Parquet columns.")
    
    print(f"Aligned {len(common_nodes)} nodes across {len(samples_df)} graph samples.")
    
    # Subset and sort to ensure index 0 in adj matches column 0 in features
    adj_aligned = adj_df.loc[common_nodes, common_nodes]
    node_features_df = samples_df[common_nodes]  # Only keep node columns for features
    
    # --- 3. Prepare Fixed Topology (Edge Index) ---
    # Since topology is shared, we compute edge_index only once
    adj_values = adj_aligned.values
    src_rows, dst_cols = np.where(adj_values != 0)
    # Convert the pair of numpy arrays into a single numpy array first
    edge_index_np = np.vstack((src_rows, dst_cols))
    edge_index = torch.from_numpy(edge_index_np).long()
    
    # --- 4. Generate Graph Objects ---
    data_list = []
    
    # Iterate over each row (sample) in the parquet file
    for idx, row in samples_df.iterrows():
        if idx % 100000 == 0:
            print(f"Processing graph sample {idx+1}/{len(samples_df)}...")
        # A. Extract Node Features for this sample
        # Shape: [Num_Nodes, Num_Features] -> Here [113, 1] since we have 1 value per node
        features = row[common_nodes].values.astype(float)
        x = torch.tensor(features, dtype=torch.float).unsqueeze(1) 
        
        # B. Extract Label (if applicable)
        y = None
        if label_column and label_column in row:
            # Assuming a single scalar label for graph classification/regression
            y = torch.tensor([row[label_column]], dtype=torch.float) # or torch.long for class
        
        # C. Create Data Object
        # Note: edge_index is reused for every sample (efficient referencing)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    # --- 5. Create DataLoader ---
    #loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    
    return data_list

# --- Example Usage ---
if __name__ == "__main__":
    ADJ_FILE = 'classification/adjacency_matrix.csv'
    PARQUET_FILE = 'simulation/simulation_files/simulated_data_3.parquet'

    loader = create_multi_graph_dataloader(ADJ_FILE, PARQUET_FILE, batch_size=32, label_column='simulation_index')
    
    print(f"\nCreated DataLoader with {len(loader)} batches.")
    
    # Check the first batch
    for batch in loader:
        print("\nBatch Structure:")
        print(batch)
        print(f" - Batch x shape: {batch.x.shape} (Nodes x Features)")
        print(f" - Batch edge_index shape: {batch.edge_index.shape}")
        print(f" - Graphs in batch: {batch.num_graphs}")
        break