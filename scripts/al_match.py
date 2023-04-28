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
from scipy.spatial.distance import squareform
from graspologic.embed import AdjacencySpectralEmbed
from giskard.plot import pairplot
from umap import UMAP
import seaborn as sns

import ot

DISPLAY_FIGS = True
FILENAME = "al_match"

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

nodes = pd.concat((al_left.nodes, al_right.nodes))
side_labels = nodes["side"]
broad_type_labels = nodes["cell_class"]
fine_type_labels = nodes["cell_type_filled"].fillna("unknown")

side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
broad_class_palette = dict(zip(broad_type_labels.unique(), sns.color_palette("tab10")))

#%%
nblast = load_flywire_nblast_subset((al_left.nodes.index, al_right.nodes.index))

nblast_within_left = nblast.loc[al_left.nodes.index, al_left.nodes.index].values
nblast_within_right = nblast.loc[al_right.nodes.index, al_right.nodes.index].values
nblast_between = nblast.loc[al_left.nodes.index, al_right.nodes.index].values
adjacency_left = al_left.to_adjacency().values.astype(float)
adjacency_right = al_right.to_adjacency().values.astype(float)


#%%
#%%
# rescaling everything...

desired_norm = 1

adjacency_left_scaled = adjacency_left * desired_norm / np.linalg.norm(adjacency_left)
adjacency_right_scaled = (
    adjacency_right * desired_norm / np.linalg.norm(adjacency_right)
)

# scale so that the maximum trace is the same as the norm of one of the above...
# don't know how to do this in a principled way
rows, cols = linear_sum_assignment(nblast_between, maximize=True)
nblast_between_scaled = (
    nblast_between * desired_norm / np.sum(nblast_between[rows, cols])
)


# %%


def create_matched_nodes(row_inds, col_inds):
    left_nodes_matched = al_left.nodes.iloc[row_inds].copy().reset_index()
    left_nodes_matched["matching"] = range(len(left_nodes_matched))
    right_nodes_matched = al_right.nodes.iloc[col_inds].copy().reset_index()
    right_nodes_matched["matching"] = range(len(right_nodes_matched))
    left_nodes_matched.set_index("matching", inplace=True)
    right_nodes_matched.set_index("matching", inplace=True)
    matched_nodes = left_nodes_matched.join(
        right_nodes_matched, lsuffix="_left", rsuffix="_right"
    )
    return matched_nodes


def compute_scores(indices1, indices2):
    conn_score = np.linalg.norm(
        adjacency_left[indices1][:, indices1] - adjacency_right[indices2][:, indices2]
    )

    nblast_between_score = nblast_between[indices1, indices2].sum()

    return conn_score, nblast_between_score


#%%

labels_left = al_left.nodes.query(f"~{score_col}.isna()")[score_col].values
labels_right = al_right.nodes.query(f"~{score_col}.isna()")[score_col].values

co_label_mat = labels_left[:, None] == labels_right[None, :]

row_inds, col_inds = linear_sum_assignment(co_label_mat, maximize=True)

print("Oracle accuracy:")
print(co_label_mat[row_inds, col_inds].mean())


#%%
currtime = time.time()
indices1, indices2, score, misc = graph_match(
    adjacency_left_scaled,
    adjacency_right_scaled,
    # S=nblast_between_scaled,
    # init="similarity",
    solver="sinkhorn",
    transport_regularizer=1e-2,
)
print("Converged: ", misc[0]["converged"])
print(f"{time.time() - currtime:.3f} seconds elapsed.")

type_col = "cell_type_filled"
matched_nodes = create_matched_nodes(indices1, indices2)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna()
acc = np.mean(colabeling[f"{score_col}_left"] == colabeling[f"{score_col}_right"])
conn_score, nblast_between_score = compute_scores(indices1, indices2)

print(f"Accuracy: {acc:.3f}")
print(f"Connectivity score: {conn_score:.3f}")
print(f"NBLAST score: {nblast_between_score:.3f}")

# # %%
# from graspologic.match.solver import _GraphMatchSolver

# solver = _GraphMatchSolver(
#     adjacency_left_scaled,
#     adjacency_right_scaled,
#     S=nblast_between_scaled,
#     transport_regularizer=1e0,
#     solver="sinkhorn",
#     init='similarity',
#     transport_tol=1e-8,
# )
# rng = np.random.default_rng(888)

# P = solver.initialize(rng)
# sns.heatmap(P[:10, :10])

# P.sum(axis=0)

#%%

currtime = time.time()
indices1, indices2, score, misc = graph_match(
    adjacency_left_scaled,
    adjacency_right_scaled,
    # S=nblast_between_scaled,
    # init="similarity",
    solver="sinkhorn",
    transport_regularizer=7e-4,
    shuffle_input=False,
    max_iter=50,
    tol=1e-7,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print("Converged: ", misc[0]["converged"])

type_col = "cell_type_filled"
matched_nodes = create_matched_nodes(indices1, indices2)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna(
    how="any"
)
acc = np.mean(colabeling[f"{score_col}_left"] == colabeling[f"{score_col}_right"])
conn_score, nblast_between_score = compute_scores(indices1, indices2)

print(f"Accuracy: {acc:.3f}")
print(f"Connectivity score: {conn_score:.3f}")
print(f"NBLAST score: {nblast_between_score:.3f}")

# %%

currtime = time.time()
left_dummy = np.zeros_like(adjacency_left_scaled)
right_dummy = np.zeros_like(adjacency_right_scaled)
indices1, indices2, score, misc = graph_match(
    left_dummy,
    right_dummy,
    S=nblast_between_scaled,
    init="similarity",
    solver="sinkhorn",
    transport_regularizer=1e3,
    shuffle_input=False,
    max_iter=50,
    tol=1e-7,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print("Converged: ", misc[0]["converged"])

type_col = "cell_type_filled"
matched_nodes = create_matched_nodes(indices1, indices2)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna(
    how="any"
)
acc = np.mean(colabeling[f"{score_col}_left"] == colabeling[f"{score_col}_right"])
conn_score, nblast_between_score = compute_scores(indices1, indices2)

print(f"Accuracy: {acc:.3f}")
print(f"Connectivity score: {conn_score:.3f}")
print(f"NBLAST score: {nblast_between_score:.3f}")

#%%


def normalize(X, axis=1):
    if axis == 1:
        norm = np.linalg.norm(X, axis=axis)[:, None]
    else:
        norm = np.linalg.norm(X, axis=axis)[None, :]
    norm[norm == 0] = 1
    X = X / norm
    return X


adjacency_left_out_scaled = normalize(adjacency_left_scaled, axis=1)
adjacency_right_out_scaled = normalize(adjacency_right_scaled, axis=1)
adjacency_left_in_scaled = normalize(adjacency_left_scaled, axis=0)
adjacency_right_in_scaled = normalize(adjacency_right_scaled, axis=0)

currtime = time.time()
indices1, indices2, score, misc = graph_match(
    [adjacency_left_out_scaled, adjacency_left_in_scaled],
    [adjacency_right_out_scaled, adjacency_right_in_scaled],
    S=nblast_between_scaled,
    init="similarity",
    solver="lap",
    transport_regularizer=1e3,
    shuffle_input=False,
    max_iter=80,
    tol=1e-5,
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

print("Converged: ", misc[0]["converged"])

type_col = "cell_type_filled"
matched_nodes = create_matched_nodes(indices1, indices2)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna(
    how="any"
)
acc = np.mean(colabeling[f"{score_col}_left"] == colabeling[f"{score_col}_right"])
conn_score, nblast_between_score = compute_scores(indices1, indices2)

print(f"Accuracy: {acc:.3f}")
print(f"Connectivity score: {conn_score:.3f}")
print(f"NBLAST score: {nblast_between_score:.3f}")
# %%
