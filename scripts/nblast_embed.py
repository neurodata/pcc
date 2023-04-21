#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import polars as pl
import seaborn as sns
from pkg.data import DATA_PATH
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from scipy.optimize import linear_sum_assignment
from tqdm.autonotebook import tqdm

from giskard.plot import confusionplot, matrixplot
from graspologic.match import graph_match
from neuropull.graph import NetworkFrame
from scipy.cluster.hierarchy import linkage
from graspologic.embed import AdjacencySpectralEmbed
from giskard.plot import pairplot
from umap import UMAP
import seaborn as sns

import ot

DISPLAY_FIGS = True
FILENAME = "nblast_embed"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


# %%


def load_flywire_networkframe():
    data_dir = DATA_PATH / "hackathon"

    dataset = "fafb_flywire"
    dataset_dir = data_dir / dataset

    nodes = pd.read_csv(dataset_dir / f"{dataset}_meta.csv", low_memory=False)
    nodes.drop("row_id", axis=1, inplace=True)
    nodes.rename(columns={"root_id": "node_id"}, inplace=True)

    # NOTE:
    # some nodes have multiple rows in the table
    # strategy here is to keep the first row that has a hemibrain type, though that
    # could be changed
    node_counts = nodes.value_counts("node_id")  # noqa: F841
    dup_nodes = nodes.query(
        "node_id.isin(@node_counts[@node_counts > 1].index)"
    ).sort_values("node_id")
    keep_rows = (
        dup_nodes.sort_values("hemibrain_type")
        .drop_duplicates("node_id", keep="first")
        .index
    )
    drop_rows = dup_nodes.index.difference(keep_rows)
    nodes.drop(drop_rows, inplace=True)

    nodes["cell_type_filled"] = nodes["cell_type"].fillna(nodes["hemibrain_type"])

    nodes.set_index("node_id", inplace=True)

    edges = pd.read_feather(dataset_dir / f"{dataset}_edges.feather")
    edges.rename(
        columns={
            "pre_pt_root_id": "source",
            "post_pt_root_id": "target",
            "syn_count": "weight",
            "neuropil": "region",
        },
        inplace=True,
    )

    # NOTE: there are some edges that reference nodes that are not in the node table
    referenced_node_ids = np.union1d(edges["source"].unique(), edges["target"].unique())
    isin_node_table = np.isin(referenced_node_ids, nodes.index)
    missing_node_ids = referenced_node_ids[~isin_node_table]  # noqa: F841

    edges.query(
        "~((source in @missing_node_ids) or (target in @missing_node_ids))",
        inplace=True,
    )

    flywire = NetworkFrame(nodes.copy(), edges.copy())
    return flywire


def load_flywire_nblast_subset(queries):
    data_dir = DATA_PATH / "hackathon"

    nblast = pl.scan_ipc(
        data_dir / "nblast" / "nblast_flywire_all_right_aba_comp.feather",
        memory_map=False,
    )
    index = pd.Index(nblast.select("index").collect().to_pandas()["index"])
    columns = pd.Index(nblast.columns[1:])
    index_ids = index.str.split(",", expand=True).get_level_values(0).astype(int)
    column_ids = columns.str.split(",", expand=True).get_level_values(0).astype(int)
    index_ids_map = dict(zip(index_ids, index))
    column_ids_map = dict(zip(column_ids, columns))
    index_ids_reverse_map = dict(zip(index, index_ids))
    column_ids_reverse_map = dict(zip(columns, column_ids))

    query_node_ids = np.concatenate(queries)
    query_index = pd.Series([index_ids_map[i] for i in query_node_ids])
    query_columns = pd.Series(["index"] + [column_ids_map[i] for i in query_node_ids])

    nblast = nblast.with_columns(
        pl.col("index").is_in(query_index).alias("select_index")
    )

    mini_nblast = (
        nblast.filter(pl.col("select_index"))
        .select(query_columns)
        .collect()
        .to_pandas()
    ).set_index("index")

    mini_nblast.index = mini_nblast.index.map(index_ids_reverse_map)
    mini_nblast.columns = mini_nblast.columns.map(column_ids_reverse_map)

    mini_nblast = mini_nblast.loc[query_node_ids, query_node_ids]

    return mini_nblast


def select_al(flywire):
    al_types = [
        "ALPN",
        "olfactory",
        "thermosensory",
        "hygrosensory",
        "ALLN",
        "ALON",
        "ALIN",
    ]
    al = flywire.query_nodes(f"cell_class.isin({al_types})")
    return al


score_col = "cell_type_filled"

flywire = load_flywire_networkframe()
al = select_al(flywire)
al_left = al.query_nodes("side == 'left'").copy()
al_right = al.query_nodes("side == 'right'").copy()
al_left.nodes.sort_values(score_col, inplace=True)
al_right.nodes.sort_values(score_col, inplace=True)

#%%
nblast = load_flywire_nblast_subset((al_left.nodes.index, al_right.nodes.index))

nblast_within_left = nblast.loc[al_left.nodes.index, al_left.nodes.index].values
nblast_within_right = nblast.loc[al_right.nodes.index, al_right.nodes.index].values
nblast_between = nblast.loc[al_left.nodes.index, al_right.nodes.index].values
adjacency_left = al_left.to_adjacency().values.astype(float)
adjacency_right = al_right.to_adjacency().values.astype(float)


#%%

nodes = pd.concat((al_left.nodes, al_right.nodes))
side_labels = nodes["side"]
broad_type_labels = nodes["cell_class"]
fine_type_labels = nodes["cell_type_filled"].fillna("unknown")


def clustermap(X, D, method="ward", metric="euclidean", **kwargs):

    linkage_matrix = linkage(X, method=method, metric=metric)

    side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
    side_colors = pd.Series(side_labels, name="Side").map(side_palette)
    broad_class_palette = dict(
        zip(broad_type_labels.unique(), sns.color_palette("tab10"))
    )
    broad_class_colors = pd.Series(broad_type_labels, name="Cell class").map(
        broad_class_palette
    )
    colors = pd.concat((side_colors, broad_class_colors), axis=1)

    cgrid = sns.clustermap(
        D,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        row_colors=colors,
        col_colors=colors,
        xticklabels=False,
        yticklabels=False,
    )
    return cgrid


def side_pairplot(X, n_show=8, title=None):
    fig, ax = pairplot(
        X[:, :n_show],
        labels=side_labels,
        palette="Set2",
        hue_order=["left", "right"],
        s=3,
        alpha=0.5,
        title=title
    )
    return fig, ax


def class_pairplot(X, n_show=8, title=None):
    fig, ax = pairplot(
        X[:, :n_show],
        labels=broad_type_labels,
        palette="tab10",
        s=3,
        alpha=0.5,
        title=title
    )
    return fig, ax


#%%

n_components = 128
ase = AdjacencySpectralEmbed(n_components=n_components, check_lcc=False)
X_nblast = ase.fit_transform(nblast.values)


#%%

side_pairplot(X_nblast, title='Raw')
class_pairplot(X_nblast, title='Raw')
clustermap(X_nblast, nblast)


# %%

X_nblast_left = X_nblast[: len(al_left.nodes)]
X_nblast_right = X_nblast[len(al_left.nodes) :]

emd = ot.da.EMDTransport()
X_nblast_right_emd = emd.fit_transform(Xs=X_nblast_right, Xt=X_nblast_left)
X_nblast_emd = np.concatenate((X_nblast_left, X_nblast_right_emd))

#%%

side_pairplot(X_nblast_emd, title='EMD')
class_pairplot(X_nblast_emd, title='EMD')
clustermap(X_nblast_emd, nblast)

#%%



#%%
# # # # # # # # # # # # # # # # J U N K # # # # # # # # # # # # # # # # # # # # # # # #
#%%
sinkhorn = ot.da.SinkhornTransport(reg_e=1e-2, tol=1e-8, max_iter=10000)
X_nblast_right_sinkhorn = sinkhorn.fit_transform(Xs=X_nblast_right, Xt=X_nblast_left)
X_nblast_sinkhorn = np.concatenate((X_nblast_left, X_nblast_right_sinkhorn))

pairplot(
    X_nblast_sinkhorn[:, :n_show],
    labels=side_labels,
    palette="Set2",
    hue_order=["left", "right"],
    s=3,
    alpha=0.5,
    title="Sinkhorn",
)

# %%

umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.5)
Y_nblast = umap.fit_transform(X_nblast)

plot_df = pd.DataFrame(Y_nblast, index=nodes.index, columns=["x", "y"])
plot_df["side"] = side_labels
plot_df["cell_class"] = broad_type_labels


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(
    data=plot_df,
    x="x",
    y="y",
    hue="side",
    palette="Set2",
    s=10,
    alpha=0.5,
)
ax.set(xticks=[], yticks=[])
