import pandas as pd
import numpy as np

class DataToEmbedder:
    """A class to produce data for the embedder model from simulation results."""
    def __init__(self, data, shuffle=True, seed=42):
        """Initialize with simulation data.
        Args:
            data (pd.DataFrame): DataFrame containing simulation results. Its columns should be the names
            of the system components and its rows should be the time steps.
        """
        if shuffle:
            self.data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)

        def binning(self, n_bins=30):
            """Bin the data into n_bins bins."""
            binned_data = pd.DataFrame()
        