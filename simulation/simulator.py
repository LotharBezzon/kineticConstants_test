import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA

def add_baths(adjacency_matrix):
    """Adds bath connections to each node in the adjacency matrix.
    Args:
        adjacency_matrix (np.ndarray): The original adjacency matrix.
    """
    num_nodes = adjacency_matrix.shape[0]
    new_size = num_nodes * 2
    new_adj_matrix = np.zeros((new_size, new_size))
    new_adj_matrix[:num_nodes, :num_nodes] = adjacency_matrix
    for i in range(num_nodes):
        new_adj_matrix[i, num_nodes + i] = 1  # connection to bath
        new_adj_matrix[num_nodes + i, i] = 1  # connection from bath
    return new_adj_matrix

class Simulator:
    """A simulator to produce data to train a kinetic constants estimator.
    Randomly creates a system as a graph, then extract the kinetic constants and compute the per-node concentrations.
    methods:
        - build_graph: creates a random graph with given number of nodes and connection probability.
        - nth_nn: finds the n-th nearest neighbors in the graph.
        - graph_info: extracts information from the graph and visualizes nearest neighbors.
        - sample_kinetic_constants: samples kinetic constants for each edge in the graph.
        - graph_components: identifies connected components in the graph.
        - solve_system: computes per-node concentrations based on kinetic constants.
        - run_equilibration: runs a temporal simulation until steady state is reached.
        - run_noisy_simulation: runs a noisy simulation starting from equilibrium concentrations.
    """

    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def build_graph(self, num_nodes=10, connection_prob=0.1, adjacency_matrix=None):  # connection_prob is 0.024 in the data
        """Creates a random graph represented as an adjacency matrix. If `adjacency_matrix` is provided, it uses that instead of generating a new one.
        Args:
            num_nodes (int): Number of nodes in the graph.
            connection_prob (float): Probability of connection between nodes.
            adjacency_matrix (np.ndarray or pd.DataFrame): Optional adjacency matrix to use instead of generating a new one."""
        
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
            np.random.seed(self.random_seed)
            self.adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < connection_prob).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)  # No self-loops
        return self.adjacency_matrix
    
    def nth_nn(self, n, adj_matrix, indexes_to_remove=[]):
        """Find the n-th nearest neighbors given the adjacency matrix. Then remove rows and columns in `indexes_to_remove`.
        Args:
            n (int): The n-th nearest neighbor to find.
            adj_matrix (np.ndarray): The adjacency matrix of the graph.
            indexes_to_remove (list): List of indexes to remove from the resulting matrix."""
        
        self.connection_matrix = np.linalg.matrix_power(adj_matrix, n)
        self.connection_matrix = np.delete(self.connection_matrix, indexes_to_remove, axis=0)
        self.connection_matrix = np.delete(self.connection_matrix, indexes_to_remove, axis=1)
        return self.connection_matrix
    
    def graph_info(self):
        """Extracts information from the graph such as number of nodes and connection probability."""

        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        num_nodes = self.adjacency_matrix.shape[0]
        connection_prob = np.sum(self.adjacency_matrix) / (num_nodes ** 2 - num_nodes)
        fig = plt.figure(figsize=(12, 8))
        colors = ['black', 'red', 'orange', 'yellow']
        for n in reversed(range(1, 5)):
            nth_matrix = self.nth_nn(n, np.array(self.adjacency_matrix))
            nth_matrix[nth_matrix > 0] = 1  # Binarize
            cmap = ListedColormap([(1,1,1,0)] + [colors[n-1]])
            plt.imshow(nth_matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        plt.title(f'Graph Nearest Neighbors (1st to 4th)\nNum Nodes: {num_nodes}, Connection Prob: {connection_prob:.4f}', fontsize=16)
        plt.xlabel('Node Index', fontsize=14)
        plt.ylabel('Node Index', fontsize=14)
        plt.savefig(os.path.join('simulation', 'figures', 'graph_nearest_neighbors.png'))
        plt.show()

    def sample_kinetic_constants(self, mean=0, sigma=1.0, degr_sigma=1.0, prod_sigma=1.0, random_seed=42, rank=None, L=None, n_samples=1):
        """Samples the logarithm of kinetic constants for each edge in the graph.
        Args:
            mean (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution."""
        
        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        
        np.random.seed(random_seed)

        num_ks = np.sum(self.adjacency_matrix) + self.adjacency_matrix.shape[0] * 2  # edges + degradation + production

        if rank is None:
            rank = num_ks

        if L is None:
            L = np.random.randn(num_ks, rank)
        cov_matrix = (L @ L.T + np.eye(num_ks)) * sigma**2 + np.random.randn(num_ks, num_ks) * 1e-3
        mu = np.full(num_ks, mean)

        log_k = np.random.multivariate_normal(mu, cov_matrix, size=n_samples)

        kinetic_constants = np.zeros((n_samples, self.adjacency_matrix.shape[0], self.adjacency_matrix.shape[1]))
        kinetic_constants[:, self.adjacency_matrix > 0] = np.exp(log_k[:, :np.sum(self.adjacency_matrix)])
        self.sparse_kinetic_constants = np.exp(log_k[:, :np.sum(self.adjacency_matrix)])
        self.kinetic_constants = kinetic_constants
        self.degradation_constants = np.exp(log_k[:, np.sum(self.adjacency_matrix):np.sum(self.adjacency_matrix)+self.adjacency_matrix.shape[0]])
        self.production_constants = np.exp(log_k[:, np.sum(self.adjacency_matrix)+self.adjacency_matrix.shape[0]:])
        self.dropout = np.random.random() * 0.1
        self.concentration_noise = 0.1 * np.random.random()
        self.log_kinetic_constants_noise = 0.04 * np.random.random()
        
        return self.kinetic_constants, self.degradation_constants, self.production_constants
    
    def sample_free_energies(self, mean=0, sigma=1.0, mean_barrier=1.0, random_seed=42, c_rank=None, reax_rank=None, c_L=None, reax_L=None, n_samples=1):
        """Samples the free energies for each node in the graph.
        Args:
            mean (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution."""
        
        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        
        np.random.seed(random_seed)

        num_nodes = self.adjacency_matrix.shape[0] * 2 # add connections to baths

        symm_adjacency = self.adjacency_matrix + self.adjacency_matrix.T
        symm_adjacency[symm_adjacency > 0] = 1
        num_reactions = (np.sum(symm_adjacency) / 2).astype(int) + (num_nodes - self.adjacency_matrix.shape[0])  # undirected edges + connections to baths

        if c_rank is None:
            c_rank = num_nodes
        if reax_rank is None:
            reax_rank = num_reactions

        if c_L is None:
            c_L = np.random.randn(num_nodes, c_rank) / np.sqrt(c_rank) * sigma
        c_cov_matrix = c_L @ c_L.T + np.eye(num_nodes) * 1e-4
        c_mu = np.full(num_nodes, mean)

        if reax_L is None:
            reax_L = np.random.randn(num_reactions, reax_rank) / np.sqrt(reax_rank) * sigma
        reax_cov_matrix = reax_L @ reax_L.T + np.eye(num_reactions) * 1e-4
        reax_mu = np.full(num_reactions, mean_barrier)

        self.free_energies = np.random.multivariate_normal(c_mu, c_cov_matrix, size=n_samples)

        reaction_barriers = np.abs(np.random.multivariate_normal(reax_mu, reax_cov_matrix, size=n_samples))
        transition_free_energies = np.zeros((n_samples, num_nodes, num_nodes))
        big_adjacency = add_baths(symm_adjacency)
        idx = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if big_adjacency[i, j] > 0:
                    transition_free_energies[:, i, j] = np.maximum(self.free_energies[:, i], self.free_energies[:, j]) + reaction_barriers[:, idx] - self.free_energies[:, i]
                    transition_free_energies[:, j, i] = np.maximum(self.free_energies[:, j], self.free_energies[:, i]) + reaction_barriers[:, idx] - self.free_energies[:, j]
                    idx += 1
        self.kinetic_constants = np.exp(-transition_free_energies[:, :symm_adjacency.shape[0], :symm_adjacency.shape[0]]) * symm_adjacency[np.newaxis, :, :]
        self.sparse_kinetic_constants = self.kinetic_constants[:, symm_adjacency > 0]
        self.degradation_constants = np.exp(-transition_free_energies[:, :symm_adjacency.shape[0], symm_adjacency.shape[0]:][transition_free_energies[:, :symm_adjacency.shape[0], symm_adjacency.shape[0]:] > 0])  # from nodes to baths
        self.degradation_constants = np.reshape(self.degradation_constants, (n_samples, symm_adjacency.shape[0]))
        self.production_constants = np.exp(-transition_free_energies[:, symm_adjacency.shape[0]:, :symm_adjacency.shape[0]][transition_free_energies[:, symm_adjacency.shape[0]:, :symm_adjacency.shape[0]] > 0])  # from baths to nodes
        self.production_constants = np.reshape(self.production_constants, (n_samples, symm_adjacency.shape[0]))
        self.dropout = np.random.random() * 0.1
        self.concentration_noise = 0.6 * np.random.random()
        self.log_kinetic_constants_noise = 0.4 * np.random.random()
    
    def graph_components(self):
        """Identifies connected components in the graph. Needed to avoid singular matrices during simulation."""

        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        components = list(nx.strongly_connected_components(G))
        self.components = components
        return self.components
    
    def solve_system(self):
        """Runs the simulation to compute per-node concentrations based on kinetic constants."""

        if not hasattr(self, 'kinetic_constants'):
            raise ValueError("Kinetic constants not found. Please sample kinetic constants first.")
        
        num_nodes = self.kinetic_constants.shape[0]
        A = np.zeros((num_nodes, num_nodes))
        b = np.zeros(num_nodes)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    A[i, j] = self.kinetic_constants[j, i]
            A[i, i] = - np.sum(self.kinetic_constants[i, :])  # Sum of outgoing rates

        concentrations = np.linalg.solve(A, b)
        self.concentrations = concentrations
        return self.concentrations
    
    def set_simulation_parameters(self, **kwargs):
        """Set simulation parameters such as kinetic constants, production constants, degradation constants,
        noise correlation matrix, concentration noise, kinetic constants noise and dropout.
        Args:
            **kwargs: Keyword arguments for simulation parameters."""
        
        if 'log_kinetic_constants' in kwargs:
            self.kinetic_constants = np.exp(kwargs.get('log_kinetic_constants'))
        if 'log_production_constants' in kwargs:
            self.production_constants = np.exp(kwargs.get('log_production_constants'))
        if 'log_degradation_constants' in kwargs:
            self.degradation_constants = np.exp(kwargs.get('log_degradation_constants'))
        self.L = np.linalg.cholesky(kwargs.get('correlation_matrix', np.eye(self.kinetic_constants.shape[0])))
        if 'concentration_noise' in kwargs:
            self.concentration_noise = kwargs.get('concentration_noise')
        if 'log_kinetic_constants_noise' in kwargs:
            self.log_kinetic_constants_noise = kwargs.get('log_kinetic_constants_noise')
        if 'dropout' in kwargs:
            self.dropout = kwargs.get('dropout')

    def get_simulation_parameters(self):
        """Get current simulation parameters.
        Returns:
            dict: Dictionary of current simulation parameters."""
        
        if not hasattr(self, 'kinetic_constants'):
            raise ValueError("Kinetic constants not found. Please sample kinetic constants first.")
        
        params_dicts = []
        for sample_idx in range(self.kinetic_constants.shape[0]):
            params_dicts.append({
                'sparse_log_kinetic_constants': np.log(self.sparse_kinetic_constants[sample_idx]),
                'kinetic_constants': self.kinetic_constants[sample_idx],
                'production_constants': self.production_constants[sample_idx],
                'degradation_constants': self.degradation_constants[sample_idx],
                'L': self.L,
                'concentration_noise': self.concentration_noise,
                'log_kinetic_constants_noise': self.log_kinetic_constants_noise,
                'dropout': self.dropout
            })
        return params_dicts
    
    def run_equilibration(self, initial_concentrations=None, convergence_threshold=1e-4, max_iterations=10000, time_step=None, track_concentrations=[]):
        """Run a temporal simulation until steady state is reached.
        Args:
            initial_concentrations (np.ndarray): Initial concentrations of the nodes.
            convergence_threshold (float): Threshold for convergence.
            max_iterations (int): Maximum number of iterations.
            time_step (float): Time step for the simulation. If None, it will be set to $k_{max}^{-1}/10$.
        """

        if not hasattr(self, 'kinetic_constants'):
            raise ValueError("Kinetic constants not found. Please sample kinetic constants first.")
        
        if time_step is None:
            k_max = np.max(self.kinetic_constants)
            time_step = 1 / (10 * k_max)

        num_samples = self.kinetic_constants.shape[0]
        num_nodes = self.kinetic_constants.shape[1]
        if initial_concentrations is None:
            concentrations = np.ones((num_samples, num_nodes))

        tracked_concentrations = [[] for node in track_concentrations]

        k_out = np.einsum('bij->bi', self.kinetic_constants)
        for iteration in range(max_iterations):
            if iteration == max_iterations - 1:
                raise Exception("Simulation did not converge within the maximum number of iterations.")
            
            production = np.einsum('bji,bj->bi', self.kinetic_constants, concentrations) + self.production_constants
            degradation = (k_out + self.degradation_constants) * concentrations
            dCdt = production - degradation

            new_concentrations = concentrations + dCdt * time_step
            new_concentrations[new_concentrations < 0] = 0

            #if any value in new_concentrations is nan or inf, print concentrations, production, degradation, dCdt
            if np.any(np.isnan(new_concentrations)) or np.any(np.isinf(new_concentrations)):
                print("NaN or Inf detected in concentrations.")
                print("Concentrations:", concentrations)
                print("Production:", production)
                print("Degradation:", degradation)
                print("dCdt:", dCdt)
                raise ValueError("NaN or Inf detected in concentrations during equilibration.")

            for i, node in enumerate(track_concentrations):
                tracked_concentrations[i].append(new_concentrations[0, node])

            #print(f"Iteration {iteration}: Max Relative Change = {np.max(np.abs(new_concentrations - concentrations) / concentrations):.6f}, Index of Max Change = {np.argmax(np.abs(new_concentrations - concentrations) / concentrations)}")
            if np.max(np.abs(new_concentrations - concentrations) / (concentrations + 1e-10)) < convergence_threshold:
                break
            concentrations = new_concentrations

        self.concentrations = concentrations

        if track_concentrations != []:
            for i in range(len(tracked_concentrations)):
                plt.plot(tracked_concentrations[i])
            plt.xlabel('Time Steps')
            plt.ylabel('Concentration')
            plt.title('Concentration Trajectories of Tracked Nodes')
            if len(track_concentrations) < 15:
                plt.legend([f'Node {node}' for node in track_concentrations])
            plt.savefig(os.path.join('simulation', 'figures', 'equilibration_trajectories.png'))
            plt.show()
        return self.concentrations
    
    def run_noisy_simulation(self, steps=1000, num_perturbations=10, time_step=None, track_concentrations=[]):
        """Starting from euilibrium concentrations, run a noisy simulation to collect data. 
        Noise on concentrations in gaussian (to replicate pixel heterogeneity), while noise on kinetic constants is lognormal.
        Args:
            L (np.ndarray): Cholesky decomposition of the correlation matrix to introduce correlations in the noise.
            concentration_noise (float): Standard deviation of the noise added to concentrations.
            log_kinetic_constants_noise (float): Standard deviation of the noise added to kinetic constants.
            steps (int): Number of steps to run the simulation.
            time_step (float): Time step for the simulation. If None, it will be set to $k_{max}^{-1}/10$.
        """
        # Try to use CuPy if available for GPU acceleration, else NumPy.
        use_cupy = False
        try:
            import cupy as cp
            use_cupy = True
        except Exception:
            cp = None

        xp = cp if use_cupy else np

        if not hasattr(self, 'concentrations'):
            raise ValueError("Concentrations not found. Please run equilibration first.")

        if time_step is None:
            k_max = np.max(self.kinetic_constants)
            time_step = 1 / (10 * k_max)

        # Move arrays to chosen backend
        kin = xp.array(self.kinetic_constants)
        degr = xp.array(self.degradation_constants)
        prod = xp.array(self.production_constants)
        L = xp.array(self.L)
        # Precompute logs and masks to speed up the loop
        # Determine sparsity pattern from the first sample (assuming fixed topology)
        kin_mask = xp.any(kin > 0, axis=0)
        kin_nonzero = kin[:, kin_mask]
        # Precompute log terms
        log_kin_nonzero = xp.log(kin_nonzero + 1e-8)
        log_degr = xp.log(degr + 1e-8)
        log_prod = xp.log(prod + 1e-8)
        
        # Pre-allocate noisy_kin to avoid allocation in loop
        noisy_kin = xp.zeros_like(kin)

        concentration_data = []

        for perturbation in range(num_perturbations):
            concentrations = xp.array(self.concentrations, copy=True)
            # correlated gaussian noise: draw vector z ~ N(0,1), then L @ z
            z = xp.random.normal(0, self.concentration_noise, size=concentrations.shape)
            correlated = xp.einsum('ij,bj->bi', L, z)
            concentrations = concentrations + concentrations * correlated
            concentrations[concentrations < 0] = 0

            for step in range(steps):
                # Generate noise only for non-zero elements
                noise_k = xp.random.normal(0, self.log_kinetic_constants_noise, size=kin_nonzero.shape)
                noisy_kin[:, kin_mask] = kin_nonzero * xp.exp(log_kin_nonzero * noise_k)
                
                noise_d = xp.random.normal(0, self.log_kinetic_constants_noise, size=degr.shape)
                noisy_degr = degr * xp.exp(log_degr * noise_d)
                
                noise_p = xp.random.normal(0, self.log_kinetic_constants_noise, size=prod.shape)
                noisy_prod = prod * xp.exp(log_prod * noise_p)
                
                production = xp.einsum('bji,bj->bi', noisy_kin, concentrations) + noisy_prod
                degradation = (xp.einsum('bij->bi', noisy_kin) + noisy_degr) * concentrations
                dCdt = production - degradation

                concentrations = concentrations + dCdt * time_step
                concentrations[concentrations < 0] = 0

                if self.dropout > 0.0:
                    drop_mask = xp.random.rand(*concentrations.shape) < self.dropout
                    measured_concentrations = concentrations * (1 - drop_mask)
                else:
                    measured_concentrations = concentrations

                concentration_data.append(measured_concentrations.copy())

        if use_cupy:
            self.simulated_data = xp.asnumpy(xp.stack(concentration_data))
        else:
            self.simulated_data = np.array(concentration_data)  #shape (steps, num_nodes)
        
        np.random.shuffle(self.simulated_data)

        if track_concentrations != []:
            plt.plot(self.simulated_data[:, 0, track_concentrations])
            plt.xlabel('Time Steps')
            plt.ylabel('Concentration')
            plt.title('Noisy Simulation Trajectories of Tracked Nodes')
            if len(track_concentrations) < 15:
                plt.legend([f'Node {node}' for node in track_concentrations])
            plt.savefig(os.path.join('simulation', 'figures', 'noisy_simulation_trajectories.png'))
            plt.show()

        return self.simulated_data

        
    def analyze_results(self):
        """Analyze the simulated data."""

        if not hasattr(self, 'simulated_data'):
            raise ValueError("Simulated data not found. Please run a noisy simulation with collect_data=True first.")
        
        mean_concentrations = np.mean(self.simulated_data[:, 0, :], axis=0)
        std_concentrations = np.std(self.simulated_data[:, 0, :], axis=0)

        num_components = self.simulated_data.shape[2]
        num_rows = 8
        num_cols = (num_components + num_rows - 1) // num_rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(max(8, num_cols * 2), max(6, num_rows * 1.2)), constrained_layout=True)
        for i in range(num_rows * num_cols):
            if i >= num_components:
                axes[i // num_cols, i % num_cols].axis('off')
            else:
                ax = axes[i // num_cols, i % num_cols]
                ax.hist(self.simulated_data[:, 0, i], bins=30, color='blue', alpha=0.7)
            #ax.set_title(f'Component {i}')
        #fig.suptitle('Lipids concentrations distribution in all ReferenceAtlas sample', fontsize=16)
        plt.savefig(os.path.join('simulation', 'figures', 'concentration_distributions.png'))
        plt.show()

        return mean_concentrations, std_concentrations

        
    
    


if __name__ == "__main__":
    reactions = pd.read_csv(os.path.join('simulation', 'Supplementary Table 4 LBA.csv'), index_col=0)
    reactions.sort_index(inplace=True)
    reagents = reactions['reagent'].unique().tolist()
    products = reactions['product'].unique().tolist()
    all_lipids = sorted(list(set(reagents + products)))
    adj_matrix_pd = pd.DataFrame(np.zeros((len(all_lipids), len(all_lipids))), index=all_lipids, columns=all_lipids)

    for i, row in reactions.iterrows():
        reagent = row['reagent']
        product = row['product']
        adj_matrix_pd.loc[reagent, product] += 1

    adj_matrix = np.array(adj_matrix_pd)
    symmetric_adj_matrix = (adj_matrix + adj_matrix.T) > 0
    adj_matrix = symmetric_adj_matrix.astype(int)
    adj_matrix_pd = pd.DataFrame(adj_matrix, index=all_lipids, columns=all_lipids)
    adj_matrix_pd.to_csv('simulation/adjacency_matrix.csv')

    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    for lipid in correlation_matrix_partial.index:
        for lipid2 in correlation_matrix_partial.columns:
            correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
    L = np.linalg.cholesky(correlation_matrix)

    simulator = Simulator(random_seed=42)
    graph = simulator.build_graph(adjacency_matrix=adj_matrix)
    #simulator.graph_info()
    simulator.sample_free_energies(mean=0, sigma=1.0, mean_barrier=1.0, random_seed=22, c_rank=5, reax_rank=5, n_samples=10)

    nodes_to_track = [i for i in range(len(all_lipids))]
    concentrations = simulator.run_equilibration(track_concentrations=nodes_to_track)
    simulator.set_simulation_parameters(correlation_matrix=correlation_matrix)
    simulated_data = simulator.run_noisy_simulation(steps=100, num_perturbations=10, track_concentrations=nodes_to_track)

    #mean_concentrations, std_concentrations = simulator.analyze_results()

    correlation_matrix_simulated = np.corrcoef(simulated_data[:,0,:].T)
    correlation_df = pd.DataFrame(correlation_matrix_simulated, index=all_lipids, columns=all_lipids)
    sns.clustermap(correlation_df, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, annot=False)
    plt.tight_layout()
    plt.show()
    
    pca = PCA()
    pca.fit(simulator.free_energies)

    # 2. Extract the Eigenvalues (Variances)
    eigenvalues = pca.explained_variance_
    ratios = pca.explained_variance_ratio_

    # 3. Visualization (The "Scree Plot")
    plt.figure(figsize=(10, 5))

    # Plot the log of variances to see the drop-off clearly
    plt.plot(ratios, marker='o', linestyle='--')
    plt.yscale('log')  # <--- CRITICAL for low-rank data
    plt.title("Scree Plot: Revealing the Low Rank")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue (Log Scale)")
    plt.grid(True)
    plt.show()

    # 4. Check the "Intrinsic Dimension"
    # Count how many components are effectively non-zero (above machine noise)
    # A common threshold for numerical noise is 1e-10
    effective_rank = np.sum(eigenvalues > 1e-2)
    print(f"PCA estimates the rank is: {effective_rank}")