import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
import pandas as pd
import os

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
        """Creates a random graph represented as an adjacency matrix. If adjacency_matrix is provided, it uses that instead of generating a new one.
        Args:
            num_nodes (int): Number of nodes in the graph.
            connection_prob (float): Probability of connection between nodes."""
        
        np.random.seed(self.random_seed)
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
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
            nth_matrix = self.nth_nn(n, self.adjacency_matrix)
            nth_matrix[nth_matrix > 0] = 1  # Binarize
            cmap = ListedColormap([(1,1,1,0)] + [colors[n-1]])
            plt.imshow(nth_matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        plt.title(f'Graph Nearest Neighbors (1st to 4th)\nNum Nodes: {num_nodes}, Connection Prob: {connection_prob:.4f}', fontsize=16)
        plt.xlabel('Node Index', fontsize=14)
        plt.ylabel('Node Index', fontsize=14)
        plt.show()

    def sample_kinetic_constants(self, mean=0, sigma=0.5, self_sigma=0.5):  # TODO: add self-loops for production and degradation
        """Samples the logarithm of kinetic constants for each edge in the graph.
        Args:
            mean (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution."""
        
        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        
        log_kinetic_constants = np.random.normal(mean, sigma, size=self.adjacency_matrix.shape)
        self.kinetic_constants = np.exp(log_kinetic_constants)
        self.kinetic_constants *= self.adjacency_matrix  # Zero out non-edges
        np.fill_diagonal(self.kinetic_constants, np.random.normal(0, self_sigma, size=self.adjacency_matrix.shape[0]))   # Self-loops
        return self.kinetic_constants
    
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

        for iteration in range(max_iterations):
            if iteration == max_iterations - 1:
                raise ValueError("Simulation did not converge within the maximum number of iterations.")
            
            dCdt = np.zeros(num_nodes)
            for i in range(num_nodes):
                production = np.sum(self.kinetic_constants[:, i] * concentrations)
                degradation = np.sum(self.kinetic_constants[i, :] * concentrations[i])
                dCdt[i] = production - degradation
            new_concentrations = concentrations + dCdt * time_step
            new_concentrations[new_concentrations < 0] = 0  # No negative concentrations

            for i, node in enumerate(track_concentrations):
                tracked_concentrations[i].append(new_concentrations[node])

            if np.max(np.abs(new_concentrations - concentrations)) < convergence_threshold:
                break
            concentrations = new_concentrations

        self.concentrations = concentrations

        if track_concentrations != []:
            for i in range(len(tracked_concentrations)):
                plt.plot(tracked_concentrations[i])
            plt.xlabel('Time Steps')
            plt.ylabel('Concentration')
            plt.show()
        return self.concentrations
    
    def run_noisy_simulation(self, concentration_noise=0.05, log_kinetic_constants_noise=0.01, steps=1000, time_step=None):
        """Starting from euilibrium concentrations, run a noisy simulation to collect data. 
        Noise on concentrations in gaussian (to replicate pixel heterogeneity), while noise on kinetic constants is lognormal.
        Args:
            concentration_noise (float): Standard deviation of the noise added to concentrations.
            log_kinetic_constants_noise (float): Standard deviation of the noise added to kinetic constants.
            steps (int): Number of steps to run the simulation.
            time_step (float): Time step for the simulation. If None, it will be set to $k_{max}^{-1}/10$.
        """

        if not hasattr(self, 'concentrations'):
            raise ValueError("Concentrations not found. Please run equilibration first.")
        
        if time_step is None:
            k_max = np.max(self.kinetic_constants)
            time_step = 1 / (10 * k_max)
        
        num_nodes = self.kinetic_constants.shape[0]
        concentrations = self.concentrations.copy()

        concentration_data = []

        for step in range(steps):
            concentrations += concentrations * np.random.normal(0, concentration_noise, size=concentrations.shape)  # Add noise
            concentrations[concentrations < 0] = 0  # No negative concentrations
            noisy_kinetic_constants = self.kinetic_constants * np.exp(np.random.normal(0, log_kinetic_constants_noise, size=self.kinetic_constants.shape))
            dCdt = np.zeros(num_nodes)
            for i in range(num_nodes):
                production = np.sum(noisy_kinetic_constants[:, i] * concentrations)
                degradation = np.sum(noisy_kinetic_constants[i, :] * concentrations[i])
                dCdt[i] = production - degradation
            concentrations += dCdt * time_step
            concentrations[concentrations < 0] = 0  # No negative concentrations

            concentration_data.append(concentrations.copy())

        self.simulated_data = np.array(concentration_data)
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

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 1 * num_rows), constrained_layout=True)
        for i in range(num_components):
            ax = axes[i // num_cols, i % num_cols]
            ax.hist(self.simulated_data[:, i], bins=30, color='blue', alpha=0.7)
            #ax.set_title(f'Component {i}')
        #fig.suptitle('Lipids concentrations distribution in all ReferenceAtlas sample', fontsize=16)
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

    simulator = Simulator(random_seed=123)
    graph = simulator.build_graph(adjacency_matrix=adj_matrix)
    simulator.graph_info()
    simulator.sample_kinetic_constants()
    concentrations = simulator.run_equilibration(track_concentrations=[i for i in range(len(all_lipids))])
    simulated_data = simulator.run_noisy_simulation(steps=2000)
    plt.plot(simulated_data)
    plt.xlabel('Time Steps')
    plt.ylabel('Concentration')
    plt.show()

    mean_concentrations, std_concentrations = simulator.analyze_results()