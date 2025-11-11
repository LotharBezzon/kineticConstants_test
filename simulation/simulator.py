import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import ListedColormap
import pandas as pd
import os

class Simulator:
    '''A simulator to produce data to train a kinetic constants estimator.
    Randomly creates a system as a graph, then extract the kinetic constants and compute the per-node concentrations.
    methods:
        - build_graph: creates a random graph with given number of nodes and connection probability.
    '''
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def build_graph(self, num_nodes=10, connection_prob=0.1, adjacency_matrix=None):  # connection_prob is 0.024 in the data
        '''Creates a random graph represented as an adjacency matrix. If adjacency_matrix is provided, it uses that instead of generating a new one.
        Args:
            num_nodes (int): Number of nodes in the graph.
            connection_prob (float): Probability of connection between nodes.'''
        np.random.seed(self.random_seed)
        if adjacency_matrix is not None:
            self.adjacency_matrix = adjacency_matrix
        else:
            self.adjacency_matrix = (np.random.rand(num_nodes, num_nodes) < connection_prob).astype(int)
        np.fill_diagonal(self.adjacency_matrix, 0)  # No self-loops
        return self.adjacency_matrix
    
    def nth_nn(self, n, adj_matrix, indexes_to_remove=[]):
        '''Find the n-th nearest neighbors given the adjacency matrix. Then remove rows and columns in `indexes_to_remove`.
        Args:
            n (int): The n-th nearest neighbor to find.
            adj_matrix (np.ndarray): The adjacency matrix of the graph.
            indexes_to_remove (list): List of indexes to remove from the resulting matrix.'''
        self.connection_matrix = np.linalg.matrix_power(adj_matrix, n)
        self.connection_matrix = np.delete(self.connection_matrix, indexes_to_remove, axis=0)
        self.connection_matrix = np.delete(self.connection_matrix, indexes_to_remove, axis=1)
        return self.connection_matrix
    
    def graph_info(self):
        '''Extracts information from the graph such as number of nodes and connection probability.'''
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

    def sample_kinetic_constants(self, mean=0, sigma=0.5):  # TODO: add self-loops for production and degradation
        '''Samples the logarithm of kinetic constants for each edge in the graph.
        Args:
            mean (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.'''
        if not hasattr(self, 'adjacency_matrix'):
            raise ValueError("Adjacency matrix not found. Please build the graph first.")
        log_kinetic_constants = np.random.normal(mean, sigma, size=self.adjacency_matrix.shape)
        kinetic_constants = np.exp(log_kinetic_constants)
        kinetic_constants *= self.adjacency_matrix  # Zero out non-edges
        return kinetic_constants

        



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