import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import lmdb
import pickle
import time

def create_graphs_list(adj_path, parquet_path, label_column=None, progress=False):
    print(f"Loading topology from {adj_path}...")
    adj_df = pd.read_csv(adj_path, index_col=0)

    print(f"Loading samples from {parquet_path}...")
    try:
        samples_df = None
        for file in parquet_path:
            samples_df = pd.read_parquet(file) if samples_df is None else pd.concat([samples_df, pd.read_parquet(file)], ignore_index=True)
    except ImportError:
        raise ImportError("Install 'pyarrow' or 'fastparquet' to read the parquet file.")

    # --- 2. Align Columns (Nodes) ---
    adj_nodes = set(adj_df.index)
    sample_columns = set(samples_df.columns)
    common_nodes = sorted(list(adj_nodes.intersection(sample_columns)))

    if not common_nodes:
        raise ValueError("No overlap found between Adjacency nodes and Parquet columns.")

    print(f"Aligned {len(common_nodes)} nodes across {len(samples_df)} graph samples.")

    adj_aligned = adj_df.loc[common_nodes, common_nodes]
    node_features_df = samples_df[common_nodes]  # Only keep node columns for features

    # Convert features to a single contiguous numpy array (N_samples, N_nodes)
    features_np = node_features_df.to_numpy(dtype=np.float32)
    num_samples, num_nodes = features_np.shape

    # --- 3. Prepare Fixed Topology (Edge Index) ---
    adj_values = adj_aligned.values
    src_rows, dst_cols = np.nonzero(adj_values)  # faster than where != 0
    edge_index_np = np.vstack((src_rows, dst_cols)).astype(np.int64)
    edge_index = torch.from_numpy(edge_index_np).long()

    # --- 4. Prepare tensor views for node features (reuse memory) ---
    # Create a single torch tensor with shape (N_samples, N_nodes, 1)
    xs_all = torch.from_numpy(features_np).to(dtype=torch.float32).unsqueeze(2)

    # Prepare labels if requested
    labels_np = None
    if label_column and label_column in samples_df.columns:
        labels_np = samples_df[label_column].to_numpy()

    # --- 5. Generate Graph Objects (fast loop using prebuilt tensors) ---
    data_list = []
    iterator = range(num_samples)
    if progress:
        iterator = tqdm(iterator, desc="Building graphs", unit="graphs")

    for i in iterator:
        # x is a view into xs_all: shape (num_nodes, 1)
        x = xs_all[i].clone()
        y = None
        if labels_np is not None:
            # Use float for regression; cast to long if classification is desired
            y = torch.tensor([labels_np[i]], dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


def write_graphs_to_lmdb(adj_path, graph_files, lmdb_path, map_size=1e9, progress=False):
    """
    Writes a list of PyTorch Geometric Data objects to an LMDB database.

    Args:
        data_list (list of Data): List of graph Data objects to store.
        lmdb_path (str): Path to the LMDB database file.
    """
    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=False, sync=False, metasync=False, writemap=True)

    with env.begin(write=True) as txn:
        total_graphs = 0
        for file in graph_files:
            data_list = create_graphs_list(
                adj_path=adj_path,
                parquet_path=[file],
                progress=progress,
                label_column='simulation_index'
            )
            print(data_list[0].y)

            for idx, data in enumerate(tqdm(data_list, desc=f"Writing graphs from {os.path.basename(file)}", unit="graphs", disable=not progress)):
                key = f'graph_{total_graphs + idx:08d}'.encode('ascii')
                value = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                txn.put(key, value)

            txn.commit()
            total_graphs += len(data_list)
            txn = env.begin(write=True)  # Start a new transaction for the next file

        txn.put('length'.encode('ascii'), str(total_graphs).encode('ascii'))

    env.close()


if __name__ == "__main__":
    ADJ_FILE = 'classification/adjacency_matrix.csv'
    PARQUET_FILES = [f'simulation/simulation_files/simulated_data_{i}.parquet' for i in range(10)]
    LMDB_PATH = 'classification/graphs_database.lmdb'
    map_size = 1 * 1024**3  # 1 GB

    write_graphs_to_lmdb(ADJ_FILE, PARQUET_FILES, LMDB_PATH, map_size=map_size, progress=True)