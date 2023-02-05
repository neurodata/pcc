#%%

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import DATA_PATH
from pkg.plot import SIDE_PALETTE, set_theme

from neuropull.graph import AdjacencyFrame

dataset = "flywire"
version = "526"
data_dir = DATA_PATH / "flywire" / "526"

# %%
nodes = pd.read_csv(data_dir / "nodes.csv.gz", index_col=0)
edges = pd.read_csv(data_dir / "edgelist.csv.gz", header=None)
edges.rename(columns={0: "source", 1: "target", 2: "weight"}, inplace=True)

g = nx.from_pandas_edgelist(edges, edge_attr="weight", create_using=nx.DiGraph())

#%%

adj = nx.to_scipy_sparse_array(g)

# %%

af = AdjacencyFrame(adj, nodes)
af.nodes

#%%
af = af.query("side.isin(['left', 'right'])")

#%%

for name, side_af in af.groupby("side"):
    print(name)
    print(side_af.shape)
