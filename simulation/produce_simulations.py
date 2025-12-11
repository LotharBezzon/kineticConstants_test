import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from simulation.simulator import Simulator
import numpy as np
import pandas as pd
import os
from torch_geometric.data import Data, OnDiskDataset
from tqdm import tqdm

def produce_simulations(n_samples, ks_per_sample, n_timesteps, n_perturbations=20, random_seed=42, adj_matrix=None, L=None, **kwargs):
    """Produce simulations given the number of kinetic constants and timesteps.

    Args:
        n_ks (int): Number of kinetic constants sets to simulate.
        n_timesteps (int or sequence of int): Number of timesteps to simulate for each set.
        random_seed (int, optional): Seed for random number generation.
        adj_matrix (np.ndarray, optional): Adjacency matrix defining the graph structure.
        L (np.ndarray, optional): Cholesky decomposition matrix for correlated noise."""
    
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10000, size=n_samples)
    if components_names := kwargs.get('components_names'):
        columns = components_names
    else:
        columns = (np.arange(adj_matrix.shape[0])).tolist()
        for col in range(len(columns)):
            columns[col] = f'component_{columns[col]}'
    #output = pd.DataFrame(columns=['simulation_index', 'timestep'] + columns)
    output = []

    for i in range(n_samples):
        if (i+1) % 100 == 0:
            print(f'Simulating kinetic constants set {i+1}/{n_samples}')
        if (i+1) % 1000 == 0 or (i+1) == n_samples:

            output_df = pd.DataFrame(output)
            sim_counter = 0
            while os.path.exists(f'simulation/simulation_files/simulated_data_{sim_counter}.parquet'):
                sim_counter += 1
            else:
                output_df.to_parquet(f'simulation/simulation_files/simulated_data_{sim_counter}.parquet', index=False)
                print(f'Saved simulation data to simulation/simulation_files/simulated_data_{sim_counter}.parquet')
            output = []
        simulator = Simulator(random_seed=random_seeds[i])
        simulator.build_graph(adjacency_matrix=adj_matrix)
        simulator.sample_free_energies(random_seed=random_seeds[i], n_samples=ks_per_sample)
        simulator.run_equilibration()
        simulator.concentration_noise = 0.6 * np.random.random()
        simulator.log_kinetic_constants_noise = 0.4 * np.random.random()
        simulator.L = L
        simulator.run_noisy_simulation(steps=n_timesteps, num_perturbations=n_perturbations)
        
        #simulated_data = pd.DataFrame(simulator.simulated_data, columns=columns)
        for n in range(ks_per_sample):
            for j in range(n_timesteps):
                row = {'simulation_index': i, 'timestep': j}
                for idx, col in enumerate(columns):
                    row[col] = simulator.simulated_data[j, n, idx]
                output.append(row)
        #output.append({'simulation_index': np.full(n_timesteps, i, dtype=int), 'timestep': np.arange(n_timesteps), **{col: simulator.simulated_data[:, idx] for idx, col in enumerate(columns)}})

def simulations_for_predictor(n_samples=100, ks_per_sample=100, n_timesteps=100, adj_matrix=None, n_perturbations=20, random_seed=42, L=None, only_steady_state=False, **kwargs):
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
            else:
                data.x = torch.tensor(simulator.simulated_data[:, n, :].T, dtype=torch.float32)
            data.edge_index = torch.tensor(np.vstack(np.nonzero(adj_matrix)), dtype=torch.long)
            data.parameters = simulator.get_simulation_parameters()[n]
            data_list.append(data)

    return data_list
        

class SimulatedGraphDataset(OnDiskDataset):
    def __init__(self, root, transform=None, pre_filter=None, random_seed=42, only_steady_state=False, adj_matrix=None, L=None, n_samples=10000, ks_per_sample=100, n_perturbations=20, chunk_size=1000, n_timesteps=1000):
        self.random_seed = random_seed
        self.only_steady_state = only_steady_state
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
                only_steady_state=self.only_steady_state
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
    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    for lipid in correlation_matrix_partial.index:
        for lipid2 in correlation_matrix_partial.columns:
            correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
    L = np.linalg.cholesky(correlation_matrix)

    #produce_simulations(10000, 1000, adj_matrix=adj_matrix, L=L, components_names=all_lipids, random_seed=12345)

    os.makedirs('simulation/simulated_graph_dataset_only_steady_state', exist_ok=True)
    db_root = 'simulation/simulated_graph_dataset_only_steady_state'
    dataset = SimulatedGraphDataset(
        root=db_root,
        random_seed=123,
        adj_matrix=adj_matrix,
        L=L,
        n_samples=500,
        ks_per_sample=100,
        n_perturbations=20,
        chunk_size=10,
        n_timesteps=100,
        only_steady_state=True
    )