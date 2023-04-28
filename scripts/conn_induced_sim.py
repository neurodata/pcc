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

DISPLAY_FIGS = True
FILENAME = "conn_induced_sim"

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
    S=nblast_between_scaled,
    init="similarity",
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


#%%


def scores_to_predictions(score_matrix, sort_left_nodes, sort_right_nodes):
    score_to_col_matrix = score_matrix * 1 / score_matrix.sum(axis=0)[None, :]
    score_to_col_matrix = pd.DataFrame(
        data=score_to_col_matrix,
        index=sort_left_nodes.index,
        columns=sort_right_nodes.index,
    )
    score_to_col_matrix["source_group"] = sort_left_nodes[score_col]
    group_score_matrix = score_to_col_matrix.groupby("source_group", dropna=False).sum()
    predictions = group_score_matrix.idxmax()
    prediction_confs = {}
    for node_id, group in predictions.items():
        prediction_conf = group_score_matrix.at[group, node_id]
        prediction_confs[node_id] = prediction_conf
    prediction_confs = pd.Series(prediction_confs)
    sort_right_nodes["predicted_group"] = predictions
    sort_right_nodes["prediction_conf"] = prediction_confs

    score_to_row_matrix = score_matrix * 1 / score_matrix.sum(axis=1)[:, None]
    score_to_row_matrix = pd.DataFrame(
        data=score_to_row_matrix,
        index=sort_left_nodes.index,
        columns=sort_right_nodes.index,
    )
    score_to_row_matrix = score_to_row_matrix.T
    score_to_row_matrix["source_group"] = sort_right_nodes[score_col]
    group_score_matrix = score_to_row_matrix.groupby("source_group", dropna=False).sum()
    predictions = group_score_matrix.idxmax()
    prediction_confs = {}
    for node_id, group in predictions.items():
        prediction_conf = group_score_matrix.at[group, node_id]
        prediction_confs[node_id] = prediction_conf
    prediction_confs = pd.Series(prediction_confs)
    sort_left_nodes["predicted_group"] = predictions
    sort_left_nodes["prediction_conf"] = prediction_confs

    return sort_left_nodes, sort_right_nodes


#%%

#%%

nblast_cost = nblast.loc[al_left.nodes.index, al_right.nodes.index].values.copy()

n = max(len(al_left.nodes), len(al_right.nodes))
reg = 0.033
a = np.ones(n)
b = np.ones(len(al_right.nodes)) * n / len(al_right.nodes)

currtime = time.time()
sinkhorn_sol = ot.sinkhorn(a, b, -nblast_cost, reg)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


#%%


def normalize(X, axis=1):
    if axis == 1:
        norm = np.linalg.norm(X, axis=axis)[:, None]
    else:
        norm = np.linalg.norm(X, axis=axis)[None, :]
    norm[norm == 0] = 1
    X = X / norm
    return X


# V1

# X_left = adjacency_left.copy()
# X_right = adjacency_right.copy()
# Y_left = adjacency_left.copy().T
# Y_right = adjacency_right.copy().T
# X_left = normalize(X_left)
# X_right= normalize(X_right)
# Y_left = normalize(Y_left)
# Y_right = normalize(Y_right)
# conn_cost = (X_left @ all_T @ X_right.T) + (Y_left @ all_T @ Y_right.T)

# V2
A_in = adjacency_left.copy()
B_in = adjacency_right.copy()
A_out = adjacency_left.copy()
B_out = adjacency_right.copy()

A_in = normalize(A_in, axis=0)
B_in = normalize(B_in, axis=0)
A_out = normalize(A_out, axis=1)
B_out = normalize(B_out, axis=1)

in_cost = A_in.T @ sinkhorn_sol @ B_in
out_cost = A_out @ sinkhorn_sol @ B_out.T

conn_cost = in_cost + out_cost

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

matrixplot(
    nblast_cost,
    square=True,
    title="NBLAST",
    row_sort_class=al_left.nodes["cell_class"],
    col_sort_class=al_right.nodes["cell_class"],
    cbar=False,
    ax=axs[0],
)

matrixplot(
    conn_cost,
    square=True,
    title="Connectivity",
    row_sort_class=al_left.nodes["cell_class"],
    col_sort_class=al_right.nodes["cell_class"],
    cbar=False,
    ax=axs[1],
)
gluefig("1-step-cost-comparison-v2", fig, formats=["png"])

#%%
show_slice = slice(0, 50)


#%%

reg = 0.10
a = np.ones(n)
b = np.ones(len(al_right.nodes)) * n / len(al_right.nodes)

currtime = time.time()
sinkhorn_conn_sol = ot.sinkhorn(a, b, -conn_cost, reg)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%

show_slice = slice(1200, 1300)

matrixplot_kws = dict(
    row_sort_class=al_left.nodes.iloc[show_slice][score_col],
    col_sort_class=al_right.nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    col_ticks=False,
    cbar=False,
    tick_fontsize=7,
)


def quick_matrixplot(A, ax=None, title=None, **kwargs):
    ax, _, _, _ = matrixplot(
        A[show_slice, show_slice],
        row_ticks=True,
        ax=ax,
        title=title,
        **kwargs,
        **matrixplot_kws,
    )
    ax.tick_params(axis="y", rotation=0)


fig, axs = plt.subplots(2, 2, figsize=(20, 20))
quick_matrixplot(nblast_cost, title="NBLAST", ax=axs[0, 0])
quick_matrixplot(conn_cost, title="Connectivity", ax=axs[0, 1])
quick_matrixplot(sinkhorn_sol, title="NBLAST Sinkhorn", ax=axs[1, 0])
quick_matrixplot(sinkhorn_conn_sol, title="Connectivity Sinkhorn", ax=axs[1, 1])

# %%
