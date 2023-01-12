#%% [markdown]
# # Toy simulation of connectome edge perturbations

#%% [markdown]
# ## Load and process the data
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from neuropull.graph import AdjacencyFrame
from tqdm.autonotebook import tqdm


data_path = Path("neuropull/processing/raw_data/maggot/2022-11-03")

g = nx.read_edgelist(
    data_path / "G_edgelist.txt",
    delimiter=" ",
    data=[("weight", float)],
    create_using=nx.DiGraph,
    nodetype=int,
)
nodes = pd.read_csv(data_path / "meta_data.csv", index_col=0)
nodes = nodes[nodes.index.isin(g.nodes)]
nodes.index.name = "node_id"
adj = nx.to_pandas_adjacency(g, nodelist=nodes.index)

af = AdjacencyFrame(adj.copy(), nodes.copy(), nodes.copy())
af

#%% [markdown]
# ### Get the paired nodes/edges
#%%
pair_counts = af.source_nodes["pair_id"].value_counts()
af.source_nodes["pair_count"] = af.source_nodes["pair_id"].map(pair_counts) == 2
af.target_nodes["pair_count"] = af.target_nodes["pair_id"].map(pair_counts) == 2
pair_af = af.query("pair_count", axis="both")
pair_af = pair_af.sort_values(["hemisphere", "pair_id"], axis="both")
# pair_af = pair_af.set_index(["hemisphere", "pair_id"])
pair_af.source_nodes

# %%
for hemisphere, side_af in pair_af.groupby("hemisphere", axis="both"):
    print(hemisphere)
    print(side_af)

#%% [markdown]
# ### Select the MBON $\rightarrow$ CN connections

#%%


source_group = "MBONs"
target_group = "CNs"
query_af = pair_af.query(source_group, axis=0).query(target_group, axis=1)
gb = query_af.groupby("hemisphere", axis="both")


def density(data):
    return np.count_nonzero(data) / data.size


query_densities_by_side = gb.apply(density, data=True)
query_densities_by_side

#%%
subgraph_map = dict(zip(*list(zip(*list(gb)))))

fig, axs = plt.subplots(2, 1, figsize=(8, 4))

heatmap_kws = dict(
    square=True,
    cmap="RdBu_r",
    center=0,
    cbar=False,
    xticklabels=False,
    yticklabels=False,
)
ax = axs[0]
sns.heatmap(subgraph_map[("L", "L")].data, ax=ax, **heatmap_kws)
ax.set_ylabel(r"L $\rightarrow$ L")
ax = axs[1]
sns.heatmap(subgraph_map[("R", "R")].data, ax=ax, **heatmap_kws)
ax.set_ylabel(r"R $\rightarrow$ R")

#%%
ll_af = subgraph_map[("L", "L")]
rr_af = subgraph_map[("R", "R")]

#%% [markdown]
# ### Get the edge weights for edges that are on both sides
#%%
both_mask = (ll_af.data > 0) & (rr_af.data > 0)
ll_edge_weights = ll_af.data[both_mask]
rr_edge_weights = rr_af.data[both_mask]


#%%
fig, ax = plt.subplots(1, figsize=(8, 6))
sns.histplot(ll_edge_weights, ax=ax, label="Left", binwidth=2)
sns.histplot(rr_edge_weights, ax=ax, label="Right", binwidth=2)
ax.legend()

#%%
ll_edge_df = pd.Series(ll_edge_weights)
ll_edge_df.index.name = "edge_id"
ll_edge_df.name = "weight"
ll_edge_df = ll_edge_df.to_frame().reset_index()
ll_edge_df["side"] = "L"

rr_edge_df = pd.Series(rr_edge_weights)
rr_edge_df.index.name = "edge_id"
rr_edge_df.name = "weight"
rr_edge_df = rr_edge_df.to_frame().reset_index()
rr_edge_df["side"] = "R"

edge_df = pd.concat([ll_edge_df, rr_edge_df], axis=0)
edge_df

#%%


def matched_stripplot(
    data,
    x=None,
    y=None,
    jitter=0.2,
    hue=None,
    match=None,
    ax=None,
    matchline_kws=None,
    order=None,
    **kwargs,
):
    data = data.copy()
    if ax is None:
        ax = plt.gca()

    if order is None:
        unique_x_var = data[x].unique()
    else:
        unique_x_var = order
    ind_map = dict(zip(unique_x_var, range(len(unique_x_var))))
    data["x"] = data[x].map(ind_map)
    if match is not None:
        groups = data.groupby(match)
        for _, group in groups:
            perturb = np.random.uniform(-jitter, jitter)
            data.loc[group.index, "x"] += perturb
    else:
        data["x"] += np.random.uniform(-jitter, jitter, len(data))

    sns.scatterplot(data=data, x="x", y=y, hue=hue, ax=ax, zorder=1, **kwargs)

    if match is not None:
        unique_match_var = data[match].unique()
        fake_palette = dict(zip(unique_match_var, len(unique_match_var) * ["black"]))
        if matchline_kws is None:
            matchline_kws = dict(alpha=0.2, linewidth=1)
        sns.lineplot(
            data=data,
            x="x",
            y=y,
            hue=match,
            ax=ax,
            legend=False,
            palette=fake_palette,
            zorder=-1,
            **matchline_kws,
        )
    ax.set(xlabel=x, xticks=np.arange(len(unique_x_var)), xticklabels=unique_x_var)
    return ax


ax = matched_stripplot(
    data=edge_df, x="side", y="weight", match="edge_id", hue="side", legend=False, s=20
)
ax.set(yscale="log")

#%% [markdown]
# ## Comparing edge weights

#%% [markdown]
# To start, we can just run Wilcoxon's rank sum test on the observed left and right
# edge weights.
#%%

wilcoxon(ll_edge_weights, rr_edge_weights)

#%% [markdown]
# Now, I wanted to create a simulation of sampled pair
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(ll_edge_weights - rr_edge_weights, ax=ax, binwidth=2, stat="density")

from scipy.stats import norm

mean, sd = norm.fit(ll_edge_weights - rr_edge_weights)

x = np.linspace(-20, 20, 1000)

ax.plot(x, norm.pdf(x, mean, sd), color="red")

#%%

edge_pairs = {
    "Extremely strong": [[7840791, 16699958], [14082322, 14000746]],
    "Very strong": [[7802210, 16699958], [16797672, 14000746]],
    "Strong": [[8297018, 16699958], [8922644, 14000746]],
    "Medium": [[7840791, 16699958], [17013073, 14000746]],
    "Weak": [[9109799, 6290724], [9074101, 15741865]],
}


#%%
ll_edge_interest = ll_af.loc[14082322, 14000746]
rr_edge_interest = rr_af.loc[7840791, 16699958]

mean_edge = (ll_edge_interest + rr_edge_interest) / 2

#%%

from scipy.stats import mannwhitneyu

effect_size = 1.5
n_sims = 1000
replace = True
rng = np.random.default_rng()
sample_sizes = np.arange(1, 40)
null_means = [5, 10, 25, 50]

total_runs = len(sample_sizes) * n_sims * len(null_means)

pbar = tqdm(total=total_runs)
rows = []
for mean_edge in null_means:
    # left_inds = edge_pair[1]
    # right_inds = edge_pair[0]
    # try:
    #     ll_edge_interest = ll_af.loc[left_inds[0], left_inds[1]]
    #     rr_edge_interest = rr_af.loc[right_inds[0], right_inds[1]]
    # except KeyError:
    #     print(f"Edge {edge_name} not found in one of the hemispheres")
    #     continue

    # mean_edge = (ll_edge_interest + rr_edge_interest) / 2
    # print(edge_name, mean_edge)

    for sample_size in sample_sizes:
        for i in range(n_sims):
            edges1 = rng.normal(mean_edge, scale=sd, size=sample_size)
            edges2 = effect_size * rng.normal(mean_edge, scale=sd, size=sample_size)
            edges1[edges1 < 0] = 0
            edges2[edges2 < 0] = 0

            stat, pvalue = mannwhitneyu(edges1, edges2, alternative="two-sided")

            rows.append(
                {
                    "stat": stat,
                    "pvalue": pvalue,
                    "sample_size": sample_size,
                    "null_mean": mean_edge,
                }
            )
            pbar.update(1)

results = pd.DataFrame(rows)
results

pbar.close()


#%%

sns.set_context("talk")

alpha = 0.05


def compute_power(x):
    return (x < alpha).sum() / len(x)


power = results.groupby(["null_mean", "sample_size"])["pvalue"].agg(compute_power)
power.name = "power"
power = power.to_frame().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=power, x="sample_size", y="power", hue="null_mean", ax=ax)
ax.set(
    ylabel=r"Power @ $\alpha=0.05$",
    xlabel="Sample size",
    title=f"Effect weight = {effect_size}",
)
sns.move_legend(ax, "lower right", title="Null weight")

ax.axhline(0.8, ls="--", color="grey")
for null_mean, group_data in power.groupby("null_mean"):
    above_thresh_data = group_data[group_data["power"] > 0.8]
    if len(above_thresh_data) > 0:
        min_sample_size = above_thresh_data["sample_size"].min()
        ax.axvline(min_sample_size, -0.08, 0.775, ls="--", color="grey", clip_on=False)
        ax.text(
            min_sample_size,
            -0.15,
            f"{min_sample_size}",
            rotation=0,
            va="top",
            ha="center",
        )


# ax.set(xlim=(0, 6))
#%% [markdown]
# ## junk below this
#%%
effect_size = 1.5
n_sims = 1000
replace = True
rng = np.random.default_rng()
sample_sizes = np.arange(10, 200, 10)

total_runs = len(sample_sizes) * n_sims

pbar = tqdm(total=total_runs)
rows = []
for sample_size in sample_sizes:
    for i in range(n_sims):
        subsampled_weights = rng.choice(
            scaled_edge_weights, size=sample_size, replace=True
        )
        other_weights = subsampled_weights.copy()
        other_weights *= effect_size * rng.normal(0, 1, size=sample_size)
        other_weights[other_weights < 1] = 1

        stat, pvalue = wilcoxon(subsampled_weights, other_weights)
        rows.append({"stat": stat, "pvalue": pvalue, "sample_size": sample_size})
        pbar.update(1)

results = pd.DataFrame(rows)
results

pbar.close()

#%%

#%%


sns.set_context("talk")

nonzero_indices = np.nonzero(sub_af.data)
edge_weights = sub_af.data[nonzero_indices]


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(edge_weights, ax=ax)
ax.axvline(edge_of_interest, color="darkred", linestyle="--")
# sub_mean = (edge_weights - 1).mean()
# inds = np.arange(0, 80)
# density = poisson(sub_mean).pmf(inds)
# ax.plot(inds + 1, density, color="red")

# density = expon(sub_mean).pdf(inds)
# ax.plot(inds + 1, density, color="orange")

#%%
norm_edge_weights = edge_weights / edge_weights.max()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(norm_edge_weights, ax=ax)

#%%
scaled_edge_weights = norm_edge_weights * edge_of_interest
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(scaled_edge_weights, ax=ax)

# %%
