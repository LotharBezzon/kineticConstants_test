from loader_classifier import create_multi_graph_dataloader
import torch
import os

def graphs_to_disk(adj_path, parquet_paths, output_path, label_column=None):
    """
    Loads graphs using create_multi_graph_dataloader and saves them to disk.

    Args:
        adj_path (str): Path to the adjacency matrix CSV (defines topology).
        parquet_paths (list of str): List of paths to parquet files (rows=samples, cols=nodes).
        output_path (str): Path to save the list of graph Data objects.
        label_column (str, optional): Name of the column in parquet to be used as graph label (y).
    """
    # Load graphs
    print("Creating graph dataset...")
    graph_list = []
    for file in parquet_paths:
        graph_list += create_multi_graph_dataloader(adj_path, file, label_column=label_column)
    
    # Save to disk
    print(f"Saving {len(graph_list)} graphs to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_list, output_path)
    print("Graphs saved successfully.")

if __name__ == "__main__":
    adjacency_matrix_path = 'classification/adjacency_matrix.csv'
    parquet_file_paths = [f'simulation/simulation_files/simulated_data_{i}.parquet' for i in range(10)]
    output_file_path = 'classification/graph_dataset.pt'
    
    graphs_to_disk(
        adj_path=adjacency_matrix_path,
        parquet_paths=parquet_file_paths,
        output_path=output_file_path,
        label_column='simulation_index'
    )