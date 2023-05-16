# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from graspologic.match import graph_match

from neuropull import NetworkFrame

DATA_PATH = Path("./neuropull/data")


def load_networkframe(dataset):
    nodes = pd.read_csv(DATA_PATH / dataset / "nodes.csv", index_col=0)
    edges = pd.read_csv(DATA_PATH / dataset / "edgelist.csv")
    return NetworkFrame(nodes, edges)


summary_rows = []
for dataset in [
    "c_elegans_hermaphrodite",
    "c_elegans_male",
    "maggot_brain",
    "flywire_central",
    "flywire",
]:
    frame = load_networkframe(dataset)
    frame.query_nodes("side.isin(['left', 'right'])", inplace=True)

    sizes = frame.nodes.groupby("side").size()
    sizes.name = dataset

    if "pair" in frame.nodes.columns:
        paired_nodes = frame.nodes.groupby("pair").filter(
            lambda x: (len(x) == 2)
            and ("left" in x["side"].values)
            and ("right" in x["side"].values)
        )
        paired_frame = frame.loc[paired_nodes.index, paired_nodes.index]

    else:
        paired_frame = frame

    left_frame = paired_frame.query_nodes("side == 'left'")
    right_frame = paired_frame.query_nodes("side == 'right'")

    left_density = len(left_frame.edges) / (len(left_frame.nodes) ** 2)
    right_density = len(right_frame.edges) / (len(right_frame.nodes) ** 2)

    summary_rows.append(
        {
            "dataset": dataset,
            "side": "left",
            "n_nodes": len(left_frame.nodes),
            "density": left_density,
        }
    )
    summary_rows.append(
        {
            "dataset": dataset,
            "side": "right",
            "n_nodes": len(right_frame.nodes),
            "density": right_density,
        }
    )
# %%
summary = pd.DataFrame(summary_rows)


sns.set_context("talk")

fig, axs = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
ax = axs[0]
sns.barplot(data=summary, y="n_nodes", x="dataset", hue="side", ax=ax)
ax.set_yscale("log")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax = axs[1]
sns.barplot(data=summary, y="density", x="dataset", hue="side", ax=ax)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# %%

n_init = 5

for dataset in [
    "c_elegans_hermaphrodite",
    "c_elegans_male",
    "maggot_brain",
]:
    frame = load_networkframe(dataset)
    frame.query_nodes("side.isin(['left', 'right'])", inplace=True)
    for i in range(n_init):
        frame.nodes = frame.nodes.sample(frac=1)

        paired_nodes = frame.nodes.groupby("pair").filter(
            lambda x: (len(x) == 2)
            and ("left" in x["side"].values)
            and ("right" in x["side"].values)
        )
        paired_frame = frame.loc[paired_nodes.index, paired_nodes.index]

        left_frame = paired_frame.query_nodes("side == 'left'")
        right_frame = paired_frame.query_nodes("side == 'right'")

        left_adjacency = left_frame.to_sparse_adjacency()
        right_adjacency = right_frame.to_sparse_adjacency()

        timer = time.time()
        left_indices, right_indices, score, misc = graph_match(
            left_adjacency, right_adjacency, shuffle_input=False
        )
        elapsed = time.time() - timer

        left_matched_nodes = left_frame.nodes.iloc[left_indices]
        right_matched_nodes = right_frame.nodes.iloc[right_indices]
        match_acc = (
            left_matched_nodes["pair"].values == right_matched_nodes["pair"].values
        ).mean()

        print(f"{dataset}: accuracy={match_acc:.2f}, elapsed={elapsed:.2f}s")

# %%
