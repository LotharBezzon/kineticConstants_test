import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from simulation.simulator import Simulator, get_biggest_submatrix, add_baths
import numpy as np
import pandas as pd
import os
from torch_geometric.data import Data, OnDiskDataset
from tqdm import tqdm

def simulations_for_predictor(n_samples=100, ks_per_sample=100, n_timesteps=100, adj_matrix=None, n_perturbations=20, random_seed=42, L=None, only_steady_state=False, add_baths=False, **kwargs):
    """Produce simulations given the number of set of kinetic constants to sample and timesteps.

    Args:
        n_ks (int): Number of kinetic constants sets to simulate.
        n_timesteps (int or sequence of int): Number of timesteps to simulate for each set.
        random_seed (int, optional): Seed for random number generation.
        adj_matrix (np.ndarray, optional): Adjacency matrix defining the graph structure.
        L (np.ndarray, optional): Cholesky decomposition matrix for correlated noise."""
    
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10000, size=n_samples)
    simulator = Simulator()
    if components_names := kwargs.get('components_names'):
        columns = components_names
    else:
        columns = (np.arange(adj_matrix.shape[0])).tolist()
        for col in range(len(columns)):
            columns[col] = f'component_{columns[col]}'

    data_list = []
    for i in tqdm(range(n_samples)):
        simulator.build_graph(adjacency_matrix=adj_matrix)
        simulator.sample_free_energies(random_seed=random_seeds[i], n_samples=ks_per_sample)
        simulator.run_equilibration()
        simulator.L = L if L is not None else np.eye(simulator.kinetic_constants.shape[1])
        if not only_steady_state:
            simulator.run_noisy_simulation(steps=n_timesteps, num_perturbations=n_perturbations)

        for n in range(ks_per_sample):
            data = Data()
            if only_steady_state:
                data.x = torch.tensor(simulator.concentrations[n, :].T, dtype=torch.float32).unsqueeze(-1)
                if add_baths:
                    baths_feat = torch.ones((data.x.size(0), data.x.size(1)), device=data.x.device)
                    data.x = torch.cat([torch.stack([data.x, torch.zeros_like(data.x)], dim=1), torch.stack([baths_feat, torch.ones_like(baths_feat)], dim=1)], dim=0).squeeze(-1)
            else:
                data.x = torch.tensor(simulator.simulated_data[:, n, :].T, dtype=torch.float32)

            data.edge_index = torch.tensor(np.vstack(np.nonzero(adj_matrix)), dtype=torch.long)
            if add_baths:
                baths_idxes = torch.arange(data.x.size(0) // 2, data.x.size(0), device=data.x.device)
                baths_edges = torch.tensor([[i, baths_idxes[i]] for i in range(baths_idxes.size(0))] + 
                                            [[baths_idxes[i], i] for i in range(baths_idxes.size(0))], dtype=torch.long, device=data.x.device).t()
                data.edge_index = torch.cat([data.edge_index, baths_edges], dim=1)
                # sort edges
                perm = data.edge_index[1].argsort(descending=False)
                data.edge_index = data.edge_index[:, perm]

                perm = data.edge_index[0].argsort(descending=False, stable=True)
                data.edge_index = data.edge_index[:, perm]
            data.parameters = simulator.get_simulation_parameters(only_steady_state=only_steady_state)[n]
            data_list.append(data)

    return data_list
        

class SimulatedGraphDataset(OnDiskDataset):
    def __init__(self, root, transform=None, pre_filter=None, random_seed=42, only_steady_state=False, add_baths=False, adj_matrix=None, L=None, n_samples=10000, ks_per_sample=100, n_perturbations=20, chunk_size=1000, n_timesteps=1000):
        self.random_seed = random_seed
        self.only_steady_state = only_steady_state
        self.add_baths = add_baths
        self.adj_matrix = adj_matrix
        self.L = L
        self.n_samples = n_samples
        self.ks_per_sample = ks_per_sample
        self.n_timesteps = n_timesteps
        self.n_perturbations = n_perturbations
        self.chunk_size = chunk_size
        super().__init__(root, transform, pre_filter)

    @property
    def raw_file_names(self):
        return []  # No raw files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass  # No download needed

    def process(self):
        np.random.seed(self.random_seed)
        n_chunks = (self.n_samples + self.chunk_size - 1) // self.chunk_size
        random_seeds = np.random.randint(0, 10000, size=n_chunks)
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks\n", unit="chunk"):
            current_chunk_size = min(self.chunk_size, self.n_samples - chunk_idx * self.chunk_size)
            data_list = simulations_for_predictor(
                n_samples=current_chunk_size,
                ks_per_sample=self.ks_per_sample,
                n_timesteps=self.n_timesteps,
                n_perturbations=self.n_perturbations,
                random_seed=random_seeds[chunk_idx],
                adj_matrix=self.adj_matrix,
                L=self.L,
                only_steady_state=self.only_steady_state,
                add_baths=self.add_baths
            )
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            self.extend(data_list)
        




if __name__ == "__main__":
    adj_matrix_pd = pd.read_csv('simulation/adjacency_matrix.csv', index_col=0)
    all_lipids = adj_matrix_pd.index.tolist()
    adj_matrix = np.array(adj_matrix_pd)
    symmetric_adj_matrix = (adj_matrix + adj_matrix.T) > 0
    adj_matrix = symmetric_adj_matrix.astype(int)
    adj_matrix, _ = get_biggest_submatrix(adj_matrix)
    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    for lipid in correlation_matrix_partial.index:
        for lipid2 in correlation_matrix_partial.columns:
            correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
    L = np.linalg.cholesky(correlation_matrix)

    #produce_simulations(10000, 1000, adj_matrix=adj_matrix, L=L, components_names=all_lipids, random_seed=12345)

    db_root = 'simulation/simulated_graph_big_dataset_only_steady_state_free_energies'
    os.makedirs(db_root, exist_ok=True)

    dataset = SimulatedGraphDataset(
        root=db_root,
        random_seed=123,
        adj_matrix=adj_matrix,
        L=L,
        n_samples=1000,
        ks_per_sample=100,
        n_perturbations=20,
        chunk_size=10,
        n_timesteps=100,
        only_steady_state=True,
        add_baths=True
    )