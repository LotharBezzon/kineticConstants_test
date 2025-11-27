import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import seaborn as sns

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

    def sample_kinetic_constants(self, mean=0, sigma=1.0, degr_sigma=1.0, prod_sigma=1.0, random_seed=42):
        """Samples the logarithm of kinetic constants for each edge in the graph.
        Args:
            mean (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution."""
        
        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        
        np.random.seed(random_seed)
        
        log_kinetic_constants = np.random.normal(mean, sigma, size=self.adjacency_matrix.shape)
        log_degradation = np.random.normal(mean, degr_sigma, size=self.adjacency_matrix.shape[0])
        log_production = np.random.normal(mean, prod_sigma, size=self.adjacency_matrix.shape[0])
        self.kinetic_constants = np.exp(log_kinetic_constants)
        self.kinetic_constants *= np.array(self.adjacency_matrix)  # Zero out non-edges
        self.degradation_constants = np.exp(log_degradation)
        self.production_constants = np.exp(log_production)
        
        return self.kinetic_constants, self.degradation_constants, self.production_constants
    
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
        
        self.kinetic_constants = kwargs.get('kinetic_constants', None)
        self.production_constants = kwargs.get('production_constants', None)
        self.degradation_constants = kwargs.get('degradation_constants', None)
        self.L = np.linalg.cholesky(kwargs.get('correlation_matrix', np.eye(self.kinetic_constants.shape[0])))
        self.concentration_noise = kwargs.get('concentration_noise', 0.05)
        self.log_kinetic_constants_noise = kwargs.get('log_kinetic_constants_noise', 0.01)
        self.dropout = kwargs.get('dropout', 0.0)
    
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

        num_nodes = self.kinetic_constants.shape[0]
        if initial_concentrations is None:
            concentrations = np.ones(num_nodes) 

        tracked_concentrations = [[] for node in track_concentrations]

        k_out = np.sum(self.kinetic_constants, axis=1)
        for iteration in range(max_iterations):
            if iteration == max_iterations - 1:
                raise Exception("Simulation did not converge within the maximum number of iterations.")
            
            production = self.kinetic_constants.T @ concentrations + self.production_constants
            degradation = (k_out + self.degradation_constants) * concentrations
            dCdt = production - degradation

            new_concentrations = concentrations + dCdt * time_step
            new_concentrations[new_concentrations < 0] = 0

            for i, node in enumerate(track_concentrations):
                tracked_concentrations[i].append(new_concentrations[node])

            #print(f"Iteration {iteration}: Max Relative Change = {np.max(np.abs(new_concentrations - concentrations) / concentrations):.6f}, Index of Max Change = {np.argmax(np.abs(new_concentrations - concentrations) / concentrations)}")
            if np.max(np.abs(new_concentrations - concentrations) / concentrations) < convergence_threshold:
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
    
    def run_noisy_simulation(self, steps=1000, time_step=None, track_concentrations=[]):
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
        concentrations = xp.array(self.concentrations, copy=True)
        kin = xp.array(self.kinetic_constants)
        degr = xp.array(self.degradation_constants)
        prod = xp.array(self.production_constants)
        L = xp.array(self.L)

        concentration_data = []

        for step in range(steps):
            # correlated gaussian noise: draw vector z ~ N(0,1), then L @ z
            z = xp.random.normal(0, self.concentration_noise, size=concentrations.shape)
            correlated = z @ L.T
            concentrations = concentrations + concentrations * correlated
            concentrations[concentrations < 0] = 0

            noisy_kin = kin * xp.exp(xp.random.normal(0, self.log_kinetic_constants_noise, size=kin.shape))
            noisy_degr = degr * xp.exp(xp.random.normal(0, self.log_kinetic_constants_noise, size=degr.shape))
            noisy_prod = prod * xp.exp(xp.random.normal(0, self.log_kinetic_constants_noise, size=prod.shape))

            production = noisy_kin.T @ concentrations + noisy_prod
            degradation = (xp.sum(noisy_kin, axis=1) + noisy_degr) * concentrations
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
            self.simulated_data = np.array(concentration_data)

        if track_concentrations != []:
            plt.plot(self.simulated_data[:, track_concentrations])
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
        
        mean_concentrations = np.mean(self.simulated_data, axis=0)
        std_concentrations = np.std(self.simulated_data, axis=0)

        num_components = self.simulated_data.shape[1]
        num_rows = 8
        num_cols = (num_components + num_rows - 1) // num_rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(max(8, num_cols * 2), max(6, num_rows * 1.2)), constrained_layout=True)
        for i in range(num_rows * num_cols):
            if i >= num_components:
                axes[i // num_cols, i % num_cols].axis('off')
            else:
                ax = axes[i // num_cols, i % num_cols]
                ax.hist(self.simulated_data[:, i], bins=30, color='blue', alpha=0.7)
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
    adj_matrix_pd.to_csv('simulation/adjacency_matrix.csv')

    correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
    correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
    for lipid in correlation_matrix_partial.index:
        for lipid2 in correlation_matrix_partial.columns:
            correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
    L = np.linalg.cholesky(correlation_matrix)

    simulator = Simulator(random_seed=12)
    graph = simulator.build_graph(adjacency_matrix=adj_matrix)
    #simulator.graph_info()
    simulator.sample_kinetic_constants()

    nodes_to_track = [i for i in range(len(all_lipids))]
    concentrations = simulator.run_equilibration(track_concentrations=nodes_to_track)
    simulated_data = simulator.run_noisy_simulation(steps=2000, track_concentrations=nodes_to_track, L=L)

    #mean_concentrations, std_concentrations = simulator.analyze_results()

    correlation_matrix_simulated = np.corrcoef(simulated_data.T)
    correlation_df = pd.DataFrame(correlation_matrix_simulated, index=all_lipids, columns=all_lipids)
    sns.clustermap(correlation_df, cmap='coolwarm', cbar=True, vmin=-1, vmax=1, annot=False)
    plt.tight_layout()
    plt.show()