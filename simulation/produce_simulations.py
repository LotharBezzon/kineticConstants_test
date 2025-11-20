from simulator import Simulator
import numpy as np
import pandas as pd
import os

def produce_simulations(n_ks, n_timesteps, random_seed=42, adj_matrix=None, L=None, **kwargs):
    """Produce simulations given the number of kinetic constants and timesteps.

    Args:
        n_ks (int): Number of kinetic constants sets to simulate.
        n_timesteps (int or sequence of int): Number of timesteps to simulate for each set.
        random_seed (int, optional): Seed for random number generation.
        adj_matrix (np.ndarray, optional): Adjacency matrix defining the graph structure.
        L (np.ndarray, optional): Cholesky decomposition matrix for correlated noise."""
    
    np.random.seed(random_seed)
    random_seeds = np.random.randint(0, 10000, size=n_ks)
    if components_names := kwargs.get('components_names'):
        columns = components_names
    else:
        columns = (np.arange(simulator.kinetic_constants.shape[0])).tolist()
        for col in range(len(columns)):
            columns[col] = f'component_{columns[col]}'
    #output = pd.DataFrame(columns=['simulation_index', 'timestep'] + columns)
    output = []

    for i in range(n_ks):
        simulator = Simulator(random_seed=random_seeds[i])
        simulator.build_graph(adjacency_matrix=adj_matrix)
        simulator.sample_kinetic_constants(random_seed=random_seeds[i])
        simulator.run_equilibration()
        concentration_noise = 0.1 * np.random.random()
        kinetic_noise = 0.04 * np.random.random()
        simulator.run_noisy_simulation(steps=n_timesteps, L=L, concentration_noise=concentration_noise, log_kinetic_constants_noise=kinetic_noise)
        
        #simulated_data = pd.DataFrame(simulator.simulated_data, columns=columns)
        for j in range(n_timesteps):
            row = {'simulation_index': i, 'timestep': j}
            for idx, col in enumerate(columns):
                row[col] = simulator.simulated_data[j, idx]
            output.append(row)
        #output.append({'simulation_index': np.full(n_timesteps, i, dtype=int), 'timestep': np.arange(n_timesteps), **{col: simulator.simulated_data[:, idx] for idx, col in enumerate(columns)}})

    output_df = pd.DataFrame(output)

    sim_counter = 0
    while os.path.exists(f'simulation/simulation_files/simulated_data_{sim_counter}.parquet'):
        sim_counter += 1
    else:
        output_df.to_parquet(f'simulation/simulation_files/simulated_data_{sim_counter}.parquet', index=False)
        print(f'Saved simulation data to simulation/simulation_files/simulated_data_{sim_counter}.parquet')

if __name__ == "__main__":
    adj_matrix_pd = pd.read_csv('simulation/adjacency_matrix.csv', index_col=0)
    all_lipids = adj_matrix_pd.index.tolist()
    adj_matrix = np.array(adj_matrix_pd)
    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    for lipid in correlation_matrix_partial.index:
        for lipid2 in correlation_matrix_partial.columns:
            correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
    L = np.linalg.cholesky(correlation_matrix)

    produce_simulations(100, 1000, adj_matrix=adj_matrix, L=L, components_names=all_lipids)