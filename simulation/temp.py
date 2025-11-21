from produce_simulations import produce_simulations
import numpy as np
import pandas as pd

adj_matrix_pd = pd.read_csv('simulation/adjacency_matrix.csv', index_col=0)
all_lipids = adj_matrix_pd.index.tolist()
adj_matrix = np.array(adj_matrix_pd)
correlation_matrix_partial = pd.read_csv('simulation/correlation_matrix.csv', index_col=0)
correlation_matrix = pd.DataFrame(np.eye(len(all_lipids)), index=all_lipids, columns=all_lipids)
for lipid in correlation_matrix_partial.index:
    for lipid2 in correlation_matrix_partial.columns:
        correlation_matrix.loc[lipid, lipid2] = correlation_matrix_partial.loc[lipid, lipid2]
L = np.linalg.cholesky(correlation_matrix)

produce_simulations(10, 1000, adj_matrix=adj_matrix, L=L, components_names=all_lipids)