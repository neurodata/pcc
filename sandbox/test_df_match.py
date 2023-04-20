#%%
import pandas as pd
from graspologic.match import graph_match
from graspologic.simulations import er_corr, sbm_corr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = 20
r = 0.65

B_mat = [[0.5, 0.3], [0.2, 0.4]]
A1, A2 = sbm_corr([n, n], B_mat, r, directed=True)
pad = np.zeros((2 * n + 5, 2 * n + 5))
pad[: 2 * n, : 2 * n] = A2
A2 = pad
A1 = pd.DataFrame(A1)
A2 = pd.DataFrame(A2)

res1 = graph_match(A1, A2, transport=False, n_init=10)
res2 = graph_match(A1, A2, transport=True, n_init=10)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

sns.heatmap(res1[3][0]["total_convex_solution"], ax=axs[0], square=True, cbar=False)
sns.heatmap(res2[3][0]["total_convex_solution"], ax=axs[1], square=True, cbar=False)

# %%
