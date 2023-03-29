#%%

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import DATA_PATH
from pkg.plot import SIDE_PALETTE, set_theme


dataset = "flywire"
version = "526"
data_dir = DATA_PATH / "flywire" / "526"

# %%
og_nodes = pd.read_csv(data_dir / "nodes.csv.gz", index_col=0)
og_edges = pd.read_csv(data_dir / "edgelist.csv.gz", header=None)
og_edges.rename(columns={0: "source", 1: "target", 2: "weight"}, inplace=True)
nodes = og_nodes.copy()
#%%

nodes = nodes.query("side.isin(['left', 'right'])")

for name, side_nodes in nodes.groupby("side"):
    print(name)
    print(side_nodes.shape[0])


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


plot_counts_by_side(nodes, "class")
plot_counts_by_side(nodes, "class", log=True)
plot_counts_by_side(nodes, "io")
plot_counts_by_side(nodes, "nt_type")

#%%
counts = nodes.groupby("group").size().reset_index(name="counts")
counts.sort_values("counts", ascending=False, inplace=True)
counts

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.histplot(data=counts, x="counts", ax=ax, log_scale=True, discrete=True)
ax.set(xlabel="# neurons in 'group'")

#%%
nodes = nodes[nodes["class"] != "optic"]

#%%

plot_counts_by_side(nodes, "class", log=True)
plot_counts_by_side(nodes, "io")
plot_counts_by_side(nodes, "nt_type")

#%%

left_nodes = nodes.query("side == 'left'")
right_nodes = nodes.query("side == 'right'")

#%%
g = nx.from_pandas_edgelist(
    og_edges, "source", "target", "weight", create_using=nx.DiGraph
)

#%%
left_adj = nx.to_scipy_sparse_array(g, nodelist=left_nodes.index, weight="weight")
right_adj = nx.to_scipy_sparse_array(g, nodelist=right_nodes.index, weight="weight")

#%%
from graspologic.utils import largest_connected_component

left_adj, left_ilocs = largest_connected_component(left_adj, return_inds=True)
left_nodes = left_nodes.iloc[left_ilocs]

right_adj, right_ilocs = largest_connected_component(right_adj, return_inds=True)
right_nodes = right_nodes.iloc[right_ilocs]

#%%
from graspologic.embed import LaplacianSpectralEmbed
from graspologic.utils import pass_to_ranks


left_adj_ptr = pass_to_ranks(left_adj)
right_adj_ptr = pass_to_ranks(right_adj)

#%%
lse = LaplacianSpectralEmbed(n_components=16, form="R-DAD", concat=True)
left_embed = lse.fit_transform(left_adj_ptr)
right_embed = lse.fit_transform(right_adj_ptr)

#%%

from graspologic.plot import pairplot

# subsample = np.random.choice(left_embed.shape[0], 10000, replace=False)

pairplot(left_embed[:, :4], title="Left", alpha=0.03)
pairplot(right_embed[:, :4], title="Right", alpha=0.03)

#%%

edges = og_edges.copy()
left_edges = edges.query(
    "source in @left_nodes.index and target in @left_nodes.index"
).copy()
right_edges = edges.query(
    "source in @right_nodes.index and target in @right_nodes.index"
).copy()
left_edges["side"] = "left"
right_edges["side"] = "right"

#%%
examples = 10
count = 0
for class1, class1_nodes in nodes.groupby("class"):
    left_class1_nodes = class1_nodes.query("side == 'left'")
    right_class1_nodes = class1_nodes.query("side == 'right'")
    for class2, class2_nodes in nodes.groupby("class"):
        left_class2_nodes = class2_nodes.query("side == 'left'")
        right_class2_nodes = class2_nodes.query("side == 'right'")

        sub_left_edges = left_edges.query(
            "source in @left_class1_nodes.index and target in @left_class2_nodes.index"
        )
        sub_right_edges = right_edges.query(
            "source in @right_class1_nodes.index and target in @right_class2_nodes.index"
        )
        cat_edges = pd.concat([sub_left_edges, sub_right_edges])

        count += 1

        if count < examples and len(cat_edges) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            sns.histplot(
                data=cat_edges,
                hue='side',
                x="weight",
                ax=ax,
                log_scale=True,
            )
            ax.set(title=f'{class1} -> {class2}')
