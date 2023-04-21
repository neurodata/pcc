#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from pkg.data import DATA_PATH
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from tqdm.autonotebook import tqdm

from giskard.plot import confusionplot, upset_catplot
from graspologic.match import graph_match
from neuropull.graph import NetworkFrame

DISPLAY_FIGS = True
FILENAME = "al_explore"

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

    nodes = pd.read_csv(dataset_dir / f"{dataset}_meta.csv")
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

    flywire = NetworkFrame(nodes, edges)
    return flywire


def load_flywire_nblast_subset(queries):
    data_dir = DATA_PATH / "hackathon"

    nblast = pl.scan_ipc(
        data_dir / "nblast" / "nblast_flywire_all_right_aba_comp.feather"
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
al_left = al.query_nodes("side == 'left'")
al_right = al.query_nodes("side == 'right'")
al_left.nodes.sort_values(score_col, inplace=True)
al_right.nodes.sort_values(score_col, inplace=True)

nblast = load_flywire_nblast_subset((al_left.nodes.index, al_right.nodes.index))

nblast_within_left = nblast.loc[al_left.nodes.index, al_left.nodes.index].values
nblast_within_right = nblast.loc[al_right.nodes.index, al_right.nodes.index].values
nblast_between = nblast.loc[al_left.nodes.index, al_right.nodes.index].values
adjacency_left = al_left.to_adjacency().values.astype(float)
adjacency_right = al_right.to_adjacency().values.astype(float)

#%%
# rescaling everything...

desired_norm = 10_000

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
    transport=True,
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

sigma = 0.03
nblast_cost = nblast.loc[al_left.nodes.index, al_right.nodes.index].values.copy()

a = np.ones(len(al_left.nodes)) / len(al_left.nodes)
b = np.ones(len(al_right.nodes)) / len(al_right.nodes)

currtime = time.time()

all_T = np.zeros(nblast_cost.shape)
for i in tqdm(range(100)):
    M = nblast_cost + np.random.normal(0, sigma, size=nblast_cost.shape)
    noisy_T = ot.emd(a, b, -M)
    all_T += noisy_T
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%

# X_left = adjacency_left.copy()
# X_right = adjacency_right.copy()
# Y_left = adjacency_left.copy().T
# Y_right = adjacency_right.copy().T
# X_left = normalize(X_left)
# X_right= normalize(X_right)
# Y_left = normalize(Y_left)
# Y_right = normalize(Y_right)
# conn_cost = (X_left @ all_T @ X_right.T) + (Y_left @ all_T @ Y_right.T)


A_in = adjacency_left.copy()
B_in = adjacency_right.copy()
A_out = adjacency_left.copy()
B_out = adjacency_right.copy()


def normalize(X, axis=1):
    if axis == 1:
        norm = np.linalg.norm(X, axis=axis)[:, None]
    else:
        norm = np.linalg.norm(X, axis=axis)[None, :]
    norm[norm == 0] = 1
    X = X / norm
    return X


A_in = normalize(A_in, axis=0)
B_in = normalize(B_in, axis=0)
A_out = normalize(A_out, axis=1)
B_out = normalize(B_out, axis=1)

in_cost = A_in.T @ all_T @ B_in

from giskard.plot import matrixplot

out_cost = A_out @ all_T @ B_out.T

matrixplot(in_cost, square=True, title="Out cost")
matrixplot(out_cost, square=True, title="Out cost")

conn_cost = in_cost + out_cost

#%%

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
gluefig("1-step-cost-comparison", fig, fmts=["png"])

#%%


fig, axs = plt.subplots(1, 2, figsize=(20, 10))

matrixplot(cost, ax=axs[0], square=True, cbar=False, title="NBLAST")
matrixplot(conn_cost, ax=axs[1], square=True, title="Connectivity")

#%%

conn_cost = adjacency_left.T @ all_T @ adjacency_right


#%%
import ot

sort_left_nodes = al_left.nodes.sort_values(score_col)
sort_right_nodes = al_right.nodes.sort_values(score_col)

a = np.ones(len(al_left.nodes)) / len(al_left.nodes)
b = np.ones(len(al_right.nodes)) / len(al_right.nodes)
M = nblast.loc[sort_left_nodes.index, sort_right_nodes.index].values
reg = 100

currtime = time.time()
T = ot.emd(a, b, -M)  # exact linear program
print(f"{time.time() - currtime:.3f} seconds elapsed.")

currtime = time.time()
T_reg = ot.sinkhorn(a, b, -M, 1e-1)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
import seaborn as sns


show_slice = slice(50, 100)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

matrixplot(
    M[show_slice, show_slice],
    row_sort_class=sort_left_nodes.iloc[show_slice][score_col],
    col_sort_class=sort_right_nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    cbar_kws=dict(shrink=0.5),
    row_ticks=False,
    col_ticks=False,
    ax=axs[0],
)


#%%

from tqdm.autonotebook import tqdm

sigma = 0.03
cost = nblast.loc[sort_left_nodes.index, sort_right_nodes.index].values.copy()

T = ot.emd(a, b, -cost)

currtime = time.time()

all_T = np.zeros(cost.shape)
for i in tqdm(range(100)):
    M = cost + np.random.normal(0, sigma, size=cost.shape)
    noisy_T = ot.emd(a, b, -M)  # exact linear program
    all_T += noisy_T
print(f"{time.time() - currtime:.3f} seconds elapsed.")


#%%

show_slice = slice(0, 100)

fig, axs = plt.subplots(1, 3, figsize=(30, 10), constrained_layout=True)

matrixplot(
    cost[show_slice, show_slice],
    row_sort_class=sort_left_nodes.iloc[show_slice][score_col],
    col_sort_class=sort_right_nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    cbar_kws=dict(shrink=0.5),
    row_ticks=True,
    col_ticks=False,
    ax=axs[0],
    title="NBLAST",
    cbar=False,
)
axs[0].tick_params(axis="y", rotation=0)

matrixplot(
    T[show_slice, show_slice],
    row_sort_class=sort_left_nodes.iloc[show_slice][score_col],
    # col_sort_class=sort_right_nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    cbar_kws=dict(shrink=0.5),
    row_ticks=False,
    col_ticks=False,
    ax=axs[1],
    title="Transport",
    cbar=False,
)

matrixplot(
    all_T[show_slice, show_slice],
    row_sort_class=sort_left_nodes.iloc[show_slice][score_col],
    # col_sort_class=sort_right_nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    cbar_kws=dict(shrink=0.5),
    row_ticks=False,
    col_ticks=False,
    ax=axs[2],
    title="Noisy Transport",
    cbar=False,
)

gluefig("nblast-noisy-transport-compare", fig)

#%%

score_matrix = all_T.copy()


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


sort_left_nodes, sort_right_nodes = scores_to_predictions(
    score_matrix, sort_left_nodes, sort_right_nodes
)

sort_right_nodes[[score_col, "predicted_group", "prediction_conf"]]


#%%

labeled_right_nodes = sort_right_nodes.dropna(axis=0, subset=score_col).copy()
labeled_right_nodes[[score_col, "predicted_group", "prediction_conf"]]

true_labels = labeled_right_nodes[score_col].values
pred_labels = labeled_right_nodes["predicted_group"].values

from sklearn.metrics import accuracy_score

accuracy_score(true_labels, pred_labels)

confusionplot(true_labels, pred_labels)

#%%
rows = []
for thresh in np.linspace(0.05, 0.99, 50):
    sub_labeled_right_nodes = labeled_right_nodes.query(f"prediction_conf > {thresh}")
    true_labels = sub_labeled_right_nodes[score_col].values
    pred_labels = sub_labeled_right_nodes["predicted_group"].values
    acc = accuracy_score(true_labels, pred_labels)
    rows.append(
        {"thresh": thresh, "acc": acc, "n_classified": len(sub_labeled_right_nodes)}
    )
acc_results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=acc_results, x="n_classified", y="acc", ax=ax)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=acc_results, x="thresh", y="acc", ax=ax)

#%%

labeled_right_nodes["is_correct"] = (
    labeled_right_nodes[score_col] == labeled_right_nodes["predicted_group"]
)
bins = pd.qcut(labeled_right_nodes["prediction_conf"], 5)
labeled_right_nodes.groupby(bins)["is_correct"].mean()  # [1:].mean()

#%%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(data=acc_results, x="n_classified", y="acc", ax=ax)

#%%
query = sort_right_nodes.query("cell_type_filled.isna() & predicted_group.isna()")[
    [score_col, "predicted_group", "prediction_conf", "cell_class"]
].sort_values("prediction_conf", ascending=False)

#%%
left_query = sort_left_nodes.query("cell_type_filled.isna() & predicted_group.isna()")
right_query = sort_right_nodes.query("cell_type_filled.isna() & predicted_group.isna()")

#%%
score_df = pd.DataFrame(
    data=score_matrix,
    index=sort_left_nodes.index,
    columns=sort_right_nodes.index,
)
score_df.loc[left_query.index, right_query.index]

sns.clustermap(score_df.loc[left_query.index, right_query.index])

#%%
sub_score_df = score_df.loc[left_query.index, right_query.index]

#%%
from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=32, random_state=0)
model.fit(sub_score_df)

row_labels = model.row_labels_
col_labels = model.column_labels_

matrixplot(sub_score_df.values, row_sort_class=row_labels, col_sort_class=col_labels)

matrixplot(
    nblast.loc[left_query.index, right_query.index].values,
    row_sort_class=row_labels,
    col_sort_class=col_labels,
)

#%%
left_res = pd.concat(
    (
        pd.Series(data=sub_score_df.index, name="node_id"),
        pd.Series(data=row_labels, name="cluster"),
        pd.Series(data=sub_score_df.shape[0] * ["left"], name="side"),
    ),
    axis=1,
)
right_res = pd.concat(
    (
        pd.Series(data=sub_score_df.columns, name="node_id"),
        pd.Series(data=col_labels, name="cluster"),
        pd.Series(data=sub_score_df.shape[1] * ["right"], name="side"),
    ),
    axis=1,
)
res = pd.concat((left_res, right_res), axis=0, ignore_index=True)
res = res.sort_values(["cluster", "side"])

res.to_clipboard()
