import pandas as pd
import numpy as np
import torch

class DataToEmbedder:
    """A class to produce data for the embedder model from simulation results."""
    def __init__(self, data, max_len, n_bins=30, shuffle=True, seed=42):
        """Initialize with simulation data.
        Args:
            data (pd.DataFrame): DataFrame containing simulation results. Its columns should be the names
            of the system components and its rows should be the time steps.
            max_len (int): Maximum number of time steps.
            n_bins (int, optional): Number of bins to use for binning the data. Default is 30.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.
            seed (int, optional): Random seed for shuffling. Default is 42.
        """
        if shuffle:
            self.data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)
        self.max_len = max_len
        self.n_bins = n_bins
        self.binning(n_bins=n_bins)
        self.single_node_tensors()

    def binning(self, n_bins=30):
        """Bin the data into n_bins bins."""
        binned_data = pd.DataFrame()
        for col in self.data.columns:
            counts, bin_edges = np.histogram(self.data[col], bins=n_bins, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_indexes = np.digitize(self.data[col], bin_edges) - 1    # Get bin indexes for each data point
            indexed_bins = np.zeros((n_bins, self.max_len), dtype=int)
            for i in range(len(self.data[col])):
                indexed_bins[bin_indexes[i], i] = 1
            binned_data[col + '_bin_centers'] = bin_centers
            binned_data[col + '_counts'] = counts
            binned_data[col + '_indexed_bins'] = list(indexed_bins)
        self.binned_data = binned_data

    def single_node_tensors(self):
        """Convert each component's binned data into individual tensors."""
        self.tensors = {}
        for col in self.data.columns:
            tensor = torch.zeros((self.n_bins, self.max_len + 1), dtype=torch.float32)
            for bin in range(self.n_bins):
                tensor[bin, 0] = self.binned_data[col + '_bin_centers'].values[bin]
                tensor[bin, 1:] = torch.tensor(self.binned_data[col + '_indexed_bins'].values[bin], dtype=torch.float32)
            self.tensors[col] = tensor

    