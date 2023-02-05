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

class NetworkFrame:
    def __init__():

#%%

adj = nx.to_scipy_sparse_array(g)

#%%
for name, data in af.nodes.groupby("side"):
    print(len(data))

# %%

af = AdjacencyFrame(adj, nodes)

af.nodes

#%%

for name, side_af in af.groupby("side"):
    print(name)
    print(side_af.shape)


#%%
af = af.query("side.isin(['left', 'right'])")
af

# %%
for name, side_af in af.groupby("side"):
    print(name)
    print(side_af.shape)

#%%

#%% [markdown]
# ```{note}
# In the plot below, it looks like there is a huge discrepancy in L/R class counts for
# the optic regions.
# ```

#%%


set_theme()

#%%


def plot_counts_by_side(nodes, col, figsize=(8, 4), log=False):
    counts = nodes.groupby(["side", col]).size().reset_index(name="counts")
    counts.sort_values("counts", ascending=False, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.barplot(data=counts, x=col, y="counts", hue="side", ax=ax, palette=SIDE_PALETTE)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    if log:
        ax.set_yscale("log")

    return fig, ax


plot_counts_by_side(af.nodes, "class")
plot_counts_by_side(af.nodes, "class", log=True)
plot_counts_by_side(af.nodes, "io")
plot_counts_by_side(af.nodes, "nt_type")

#%%
counts = nodes.groupby("group").size().reset_index(name="counts")
counts.sort_values("counts", ascending=False, inplace=True)
counts

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(data=counts, x="counts", ax=ax, log_scale=True)
ax.set(xlabel="# neurons in 'group'")

# %%
for name, side_af in af.groupby("side"):
    print(name)
    print(side_af.shape)

#%%
