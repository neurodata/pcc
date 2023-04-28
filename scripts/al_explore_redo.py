#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from scipy import sparse
from pkg.data import DATA_PATH
from pkg.io import OUT_PATH

from graspologic.match import graph_match
from scipy.optimize import linear_sum_assignment
from giskard.plot import confusionplot

from pkg.io import glue as default_glue
from pkg.io import savefig
from neuropull.graph import NetworkFrame

import time
from joblib import Parallel, delayed
from giskard.plot import upset_catplot
from pkg.plot import set_theme
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import LinearSegmentedColormap


DISPLAY_FIGS = True
FILENAME = "al_explore_redo"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%% [markdown]

# ## FAFB Flywire

# %%

dataset = "fafb_flywire"
dataset_dir = data_dir / dataset

#%%
nodes = pd.read_csv(dataset_dir / f"{dataset}_meta.csv")
nodes.drop("row_id", axis=1, inplace=True)
nodes.rename(columns={"root_id": "node_id"}, inplace=True)

# NOTE:
# some nodes have multiple rows in the table
# strategy here is to keep the first row that has a hemibrain type, though that
# could be changed
node_counts = nodes.value_counts("node_id")
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
nodes.head()

#%%
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
edges.head()

#%%

# NOTE: there are some edges that reference nodes that are not in the node table

referenced_node_ids = np.union1d(edges["source"].unique(), edges["target"].unique())
isin_node_table = np.isin(referenced_node_ids, nodes.index)
missing_node_ids = referenced_node_ids[~isin_node_table]
print("Number of missing nodes: ", len(missing_node_ids))

edges.query(
    "~((source in @missing_node_ids) or (target in @missing_node_ids))", inplace=True
)

flywire = NetworkFrame(nodes, edges)

#%%

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

#%%
al_left = al.query_nodes("side == 'left'")
al_right = al.query_nodes("side == 'right'")

#%%

#%%
nblast = pl.scan_ipc(data_dir / "nblast" / "nblast_flywire_all_right_aba_comp.feather")
index = pd.Index(nblast.select("index").collect().to_pandas()["index"])
columns = pd.Index(nblast.columns[1:])
index_ids = index.str.split(",", expand=True).get_level_values(0).astype(int)
column_ids = columns.str.split(",", expand=True).get_level_values(0).astype(int)
index_ids_map = dict(zip(index_ids, index))
column_ids_map = dict(zip(column_ids, columns))
index_ids_reverse_map = dict(zip(index, index_ids))
column_ids_reverse_map = dict(zip(columns, column_ids))

query_node_ids = np.concatenate((al_left.nodes.index, al_right.nodes.index))
query_index = pd.Series([index_ids_map[i] for i in query_node_ids])
query_columns = pd.Series(["index"] + [column_ids_map[i] for i in query_node_ids])

nblast = nblast.with_columns(pl.col("index").is_in(query_index).alias("select_index"))

mini_nblast = (
    nblast.filter(pl.col("select_index")).select(query_columns).collect().to_pandas()
).set_index("index")

mini_nblast.index = mini_nblast.index.map(index_ids_reverse_map)
mini_nblast.columns = mini_nblast.columns.map(column_ids_reverse_map)

nblast = mini_nblast

#%%

nblast_within_left = nblast.loc[al_left.nodes.index, al_left.nodes.index].values.copy()
nblast_within_right = nblast.loc[
    al_right.nodes.index, al_right.nodes.index
].values.copy()
nblast_between_left_right = nblast.loc[
    al_left.nodes.index, al_right.nodes.index
].values.copy()

nblast_within_left = sparse.csr_array(nblast_within_left)
nblast_within_right = sparse.csr_array(nblast_within_right)

#%%

adjacency_left = al_left.to_sparse_adjacency().astype(float)
adjacency_right = al_right.to_sparse_adjacency().astype(float)


#%%
# rescaling everything...

desired_norm = 1

adjacency_left *= desired_norm / sparse.linalg.norm(adjacency_left)
adjacency_right *= desired_norm / sparse.linalg.norm(adjacency_right)
nblast_within_left *= desired_norm / sparse.linalg.norm(nblast_within_left)
nblast_within_right *= desired_norm / sparse.linalg.norm(nblast_within_right)

rows, cols = linear_sum_assignment(nblast_between_left_right, maximize=True)
nblast_between_left_right *= desired_norm / np.sum(
    nblast_between_left_right[rows, cols]
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


#%%


dummy = False
if dummy:
    adjacency1 = adjacency_left[:100, :100]
    adjacency2 = adjacency_right[:100, :100]
    similarity1 = nblast_within_left[:100, :100]
    similarity2 = nblast_within_right[:100, :100]
    similarity12 = nblast_between_left_right[:100, :100]
else:
    adjacency1 = adjacency_left
    adjacency2 = adjacency_right
    similarity1 = nblast_within_left
    similarity2 = nblast_within_right
    similarity12 = nblast_between_left_right

n_jobs = -2
graph_match_kws = dict(transport_regularizer=200, shuffle_input=False)


def match_experiment(
    nblast_between, nblast_within, connectivity, transport, warm_start, n_init=1
):
    adjs1 = []
    adjs2 = []
    if connectivity:
        adjs1.append(adjacency1)
        adjs2.append(adjacency2)
    if nblast_within:
        adjs1.append(similarity1)
        adjs2.append(similarity2)

    # Should reduce to the LAP basically, but just wanted to keep the code the same
    if len(adjs1) == 0:
        adjs1.append(np.zeros(adjacency1.shape))
        adjs2.append(np.zeros(adjacency2.shape))

    if nblast_between:
        S = similarity12
    else:
        S = None

    if warm_start:
        init = "similarity"
    else:
        init = None

    if transport:
        solver = "sinkhorn"
    else:
        solver = "lap"

    if n_init == 1:
        outs = graph_match(
            adjs1,
            adjs2,
            S=S,
            init=init,
            solver=solver,
            **graph_match_kws,
        )
        return [outs]
    else:

        def run():
            return graph_match(
                adjs1,
                adjs2,
                S=S,
                init=init,
                solver=solver,
                **graph_match_kws,
            )

        outs = Parallel(n_jobs=n_jobs)(delayed(run)() for _ in range(n_init))
        return outs


def compute_scores(indices1, indices2):
    conn_score = np.linalg.norm(
        adjacency1.todense()[indices1][:, indices1]
        - adjacency2.todense()[indices2][:, indices2]
    )
    nblast_score = np.linalg.norm(
        similarity1.todense()[indices1][:, indices1]
        - similarity2.todense()[indices2][:, indices2]
    )
    similarity12 = nblast.loc[al_left.nodes.index, al_right.nodes.index].values

    nblast_between_score = similarity12[indices1, indices2].sum()

    return conn_score, nblast_score, nblast_between_score


n_init = 10

type_col = "cell_type_filled"

RERUN_EXPERIMENT = False

if RERUN_EXPERIMENT:
    metrics = []
    for nblast_between in [True, False]:
        for nblast_within in [False]:
            for connectivity in [True, False]:
                for transport in [False]:
                    for warm_start in [True, False]:
                        print("----")
                        print(
                            "NBLAST between, NBLAST within, connectivity, transport, warm_start"
                        )
                        print(
                            nblast_between,
                            nblast_within,
                            connectivity,
                            transport,
                            warm_start,
                        )

                        if not (nblast_between or nblast_within or connectivity):
                            continue

                        if (not nblast_between) and warm_start:
                            continue

                        currtime = time.time()

                        options = (
                            nblast_between,
                            nblast_within,
                            connectivity,
                            transport,
                            warm_start,
                        )
                        outs = match_experiment(*options, n_init=n_init)
                        for out in outs:

                            indices1, indices2, score, misc = out

                            matched_nodes = create_matched_nodes(indices1, indices2)
                            colabeling = matched_nodes[
                                [f"{type_col}_left", f"{type_col}_right"]
                            ].dropna()

                            acc = np.mean(
                                colabeling[f"{type_col}_left"]
                                == colabeling[f"{type_col}_right"]
                            )

                            (
                                conn_score,
                                nblast_score,
                                nblast_between_score,
                            ) = compute_scores(indices1, indices2)

                            metrics.append(
                                {
                                    "nblast_between": nblast_between,
                                    "nblast_within": nblast_within,
                                    "connectivity": connectivity,
                                    "transport": transport,
                                    "warm_start": warm_start,
                                    "accuracy": acc,
                                    "conn_score": conn_score,
                                    "nblast_score": nblast_score,
                                    "nblast_between_score": nblast_between_score,
                                }
                            )
                        print(f"{time.time() - currtime:.3f} seconds elapsed.")
                        print()
        results = pd.DataFrame(metrics)
        results.to_csv(OUT_PATH / FILENAME / "match_results.csv")

else:
    results = pd.read_csv(OUT_PATH / FILENAME / "match_results.csv", index_col=0)


#%%

group_cols = ["nblast_between", "connectivity", "warm_start"]

mean_results = results.groupby(group_cols).mean()

mean_results = mean_results.sort_values("accuracy", ascending=False)

#%%

set_theme()

out = upset_catplot(
    mean_results.reset_index(),
    x=group_cols,
    y="accuracy",
    kind="bar",
    dodge=False,
    estimator_labels=True,
)
gluefig("al_accuracy", out.fig)

out = upset_catplot(
    mean_results.reset_index(),
    x=group_cols,
    y="conn_score",
    kind="bar",
    dodge=False,
)
gluefig("conn_score", out.fig)

out = upset_catplot(
    mean_results.reset_index(),
    x=group_cols,
    y="nblast_score",
    kind="bar",
    dodge=False,
)
gluefig("nblast_within_score", out.fig)

out = upset_catplot(
    mean_results.reset_index(),
    x=group_cols,
    y="nblast_between_score",
    kind="bar",
    dodge=False,
)
gluefig("nblast_between_score", out.fig)

#%%
from scipy.spatial.distance import cosine


def compute_metrics(indices1, indices2):

    perm_adjacency1 = adjacency1[indices1][:, indices1].todense()
    perm_adjacency2 = adjacency2[indices2][:, indices2].todense()
    similarity12 = nblast.loc[al_left.nodes.index, al_right.nodes.index].values
    perm_sim = similarity12[indices1][:, indices2]

    rows = []
    for i in range(perm_adjacency1.shape[0]):
        if (perm_adjacency1[i].sum() != 0) and (perm_adjacency2[i].sum() != 0):
            output_cos_sim = 1 - cosine(perm_adjacency1[i], perm_adjacency2[i])
        else:
            output_cos_sim = 1
        if (perm_adjacency1[:, i].sum() != 0) and (perm_adjacency2[:, i].sum() != 0):
            input_cos_sim = 1 - cosine(perm_adjacency1[:, i], perm_adjacency2[:, i])
        else:
            input_cos_sim = 1
        nblast_sim = perm_sim[i, i]
        rows.append(
            {
                "output_cos_sim": output_cos_sim,
                "input_cos_sim": input_cos_sim,
                "nblast_sim": nblast_sim,
                "sum_cos_sim": (output_cos_sim + input_cos_sim) / 2,
            }
        )

    node_metrics = pd.DataFrame(rows)
    return node_metrics


def summarize_node_matching(indices1, indices2, score_col):
    matched_nodes = create_matched_nodes(indices1, indices2).copy()
    node_metrics = compute_metrics(indices1, indices2)
    matched_nodes = matched_nodes.join(node_metrics)
    match_residuals = matched_nodes.query(f"{score_col}_left != {score_col}_right")
    matched_nodes["is_correct"] = True
    matched_nodes.loc[match_residuals.index, "is_correct"] = False
    return matched_nodes


def train_model(X, y):
    model = RandomForestClassifier(n_estimators=200, max_depth=2)
    model.fit(X.values, y)
    return model


def plot_decision_boundary(model, X, ax=None, plot_step=0.02, palette=None):
    if isinstance(X, pd.DataFrame):
        X = X.values
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colors = [palette[False], palette[True]]
    cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
    cs = ax.contourf(xx, yy, Z, cmap=cm, zorder=-2, alpha=0.2)


palette = {True: "green", False: "darkred"}


def plot_node_metrics(matched_nodes, title=None):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10, 10),
        sharex="col",
        sharey="row",
        gridspec_kw=dict(width_ratios=[2, 1], height_ratios=[1, 2]),
    )

    ax = axs[1, 0]

    model = train_model(
        matched_nodes[["sum_cos_sim", "nblast_sim"]], matched_nodes["is_correct"]
    )
    plot_decision_boundary(
        model, matched_nodes[["sum_cos_sim", "nblast_sim"]], ax=ax, palette=palette
    )

    sns.scatterplot(
        data=matched_nodes,
        x="sum_cos_sim",
        y="nblast_sim",
        s=10,
        alpha=0.5,
        hue="is_correct",
        ax=ax,
        palette=palette,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.set(xlabel="Match cosine similarity", ylabel="Match NBLAST similarity")
    # sns.move_legend(ax, "upper left", frameon=True)

    x_quantiles = matched_nodes["sum_cos_sim"].quantile([0.2, 0.4, 0.6, 0.8])
    y_quantiles = matched_nodes["nblast_sim"].quantile([0.2, 0.4, 0.6, 0.8])

    for q, quant in x_quantiles.items():
        ax.axvline(quant, ls="-", c="grey", zorder=-1, lw=2)

    for q, quant in y_quantiles.items():
        ax.axhline(quant, ls="-", c="grey", zorder=-1, lw=2)

    ax = axs[0, 0]
    sns.histplot(
        data=matched_nodes,
        x="sum_cos_sim",
        hue="is_correct",
        common_norm=False,
        stat="density",
        ax=ax,
        legend=False,
        palette=palette,
    )

    ax = axs[1, 1]
    sns.histplot(
        data=matched_nodes,
        y="nblast_sim",
        hue="is_correct",
        common_norm=False,
        stat="density",
        ax=ax,
        legend=False,
        palette=palette,
    )

    fig.suptitle(title, y=0.92)

    ax = axs[0, 1]
    ax.legend(handles, labels, loc="lower left", frameon=True, title="Correct match")
    ax.axis("off")
    # ax.axis("off")

    return fig, axs


#%%

score_col = "cell_type_filled"

#%%
nblast_between = True
nblast_within = False
connectivity = False
transport = False
warm_start = False

options = (nblast_between, nblast_within, connectivity, transport, warm_start)
out = match_experiment(*options, n_init=1)[0]
indices1, indices2, score, misc = out

matched_nodes = summarize_node_matching(indices1, indices2, score_col)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna()

confusionplot(
    colabeling[f"{score_col}_left"],
    colabeling[f"{score_col}_right"],
    annot=True,
    figsize=(20, 20),
    title="NBLAST between only",
)

fig, axs = plot_node_metrics(matched_nodes, title="NBLAST between only")
gluefig("post-hoc-metrics-nblast-between-only", fig)

#%%
nblast_between = True
nblast_within = False
connectivity = True
transport = False
warm_start = True

options = (nblast_between, nblast_within, connectivity, transport, warm_start)
currtime = time.time()
out = match_experiment(*options, n_init=1)[0]
print(f"{time.time() - currtime:.3f} seconds elapsed.")
indices1, indices2, score, misc = out

matched_nodes = summarize_node_matching(indices1, indices2, score_col)
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna()

confusionplot(
    colabeling[f"{score_col}_left"],
    colabeling[f"{score_col}_right"],
    annot=True,
    figsize=(20, 20),
    title="NBLAST between + connectivity",
)

fig, axs = plot_node_metrics(matched_nodes, title="NBLAST between + connectivity")
gluefig("post-hoc-metrics-nblast-between-connectivity", fig)

#%%
colabeling = matched_nodes[[f"{score_col}_left", f"{score_col}_right"]].dropna()
mislabeling = colabeling.query(f"{score_col}_left != {score_col}_right")

ax = confusionplot(
    mislabeling[f"{score_col}_left"],
    mislabeling[f"{score_col}_right"],
    annot=True,
    figsize=(20, 20),
    title="NBLAST between + connectivity",
)
gluefig("mislabeling_con_nblast", ax.figure)


#%%
conf_mat = pd.crosstab(
    mislabeling[f"{score_col}_left"], mislabeling[f"{score_col}_right"]
)
conf_mat.index.name = "left"
conf_mat.columns.name = "right"
mismatches = conf_mat.reset_index().melt(
    id_vars="left", value_vars=conf_mat.columns, value_name="count"
)
mismatches = mismatches.query("count > 0")
mismatches.sort_values("count", ascending=False).head(30)


#%%

labels_left = al_left.nodes[score_col].values
labels_right = al_right.nodes[score_col].values

co_label_mat = labels_left[:, None] == labels_right[None, :]

from scipy.optimize import linear_sum_assignment

row_inds, col_inds = linear_sum_assignment(co_label_mat, maximize=True)

co_label_mat[row_inds, col_inds].mean()

#%%
matched_nodes.query("~is_correct & nblast_sim > 6 & sum_cos_sim > 1.5")[
    [
        "cell_type_filled_left",
        "cell_type_filled_right",
        "nblast_sim",
        "sum_cos_sim",
        "node_id_left",
        "node_id_right",
    ]
].to_clipboard(excel=False)

#%%
matched_nodes.query("is_correct").sort_values("nblast_sim")[
    [
        "cell_type_filled_left",
        "cell_type_filled_right",
        "nblast_sim",
        "sum_cos_sim",
        "node_id_left",
        "node_id_right",
    ]
].head(20).to_clipboard(excel=False)


#%%
residual_summary = matched_nodes.query(f"{score_col}_left != {score_col}_right")[
    [
        "node_id_left",
        "node_id_right",
        "cell_class_left",
        "cell_class_right",
        "hemibrain_type_left",
        "hemibrain_type_right",
        "cell_type_left",
        "cell_type_right",
        "cell_type_filled_left",
        "cell_type_filled_right",
        "status_left",
        "status_right",
    ]
]

# TODO: check how much of the neuron's connectivity is in the subgraph

#%%
import ot

sort_left_nodes = al_left.nodes.sort_values(score_col)
sort_right_nodes = al_right.nodes.sort_values(score_col)

a = np.ones(len(al_left.nodes))
b = np.ones(len(al_right.nodes)) / len(al_right.nodes) * len(al_left.nodes)

M = nblast.loc[sort_left_nodes.index, sort_right_nodes.index].values

currtime = time.time()
T_reg = ot.sinkhorn(a, b, -M, reg=0.03)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
import seaborn as sns
from giskard.plot import matrixplot


#%%

from tqdm.autonotebook import tqdm


#%%

score_matrix = T_reg.copy()


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

labeled_right_nodes = sort_right_nodes.dropna(
    axis=0, subset=[score_col, "predicted_group"]
).copy()
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
ax.set(xlabel="Threshold on transport weight", ylabel="Accuracy")
gluefig("sinkhorn_acc_vs_thresh", fig)

#%%
bins = pd.cut(labeled_right_nodes["prediction_conf"], np.linspace(-0.001, 1.001, 50))
for bin, bin_data in labeled_right_nodes.groupby(bins):
    true_labels = sub_labeled_right_nodes[score_col].values
    pred_labels = sub_labeled_right_nodes["predicted_group"].values
    acc = accuracy_score(true_labels, pred_labels)

#%%

labeled_right_nodes["is_correct"] = (
    labeled_right_nodes[score_col] == labeled_right_nodes["predicted_group"]
)
# bins = pd.qcut(labeled_right_nodes["prediction_conf"], 5)
bins = pd.cut(labeled_right_nodes["prediction_conf"], np.linspace(0.00, 1.00, 11))
bin_accs = labeled_right_nodes.groupby(bins)["is_correct"].mean()  # [1:].mean()

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.barplot(x=bin_accs.index, y=bin_accs, width=1, color=sns.color_palette()[0], ax=ax)
ax.set(xlabel="Confidence", ylabel="Accuracy")
ax.set_xticklabels([f"{x.left:.1f}" for x in bin_accs.index], rotation=0)

gluefig("sinkhorn_acc_vs_conf", fig)

#%%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
counts = labeled_right_nodes.groupby(bins).size() / len(labeled_right_nodes)
sns.barplot(
    x=counts.index,
    y=counts,
    width=1,
    color=sns.color_palette()[0],
    ax=ax,
)
ax.set(xlabel="Confidence", ylabel="Proportion")
ax.set_xticklabels([f"{x.left:.1f}" for x in bin_accs.index], rotation=0)
gluefig('sinkhorn_prop_vs_conf', fig)

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

# #%%
# sub_score_df = score_df.loc[left_query.index, right_query.index]
# n = sub_score_df.shape[0]
# m = sub_score_df.shape[1]
# biadjacency = np.zeros((n + m, n + m))
# biadjacency[:n, n:] = sub_score_df.values


# #%%
# from graspologic.embed import LaplacianSpectralEmbed

# lse = LaplacianSpectralEmbed(n_components=24, concat=True, check_lcc=False)
# X_hat = lse.fit_transform(biadjacency)

# from umap import UMAP

# umap = UMAP(n_components=2)
# Y_hat = umap.fit_transform(X_hat)

# sns.scatterplot(x=Y_hat[:, 0], y=Y_hat[:, 1])

# #%%
# from sklearn.cluster import SpectralCoclustering

# model = SpectralCoclustering(n_clusters=32, random_state=0)
# model.fit(sub_score_df)

# row_labels = model.row_labels_
# col_labels = model.column_labels_

# matrixplot(sub_score_df.values, row_sort_class=row_labels, col_sort_class=col_labels)

# matrixplot(
#     nblast.loc[left_query.index, right_query.index].values,
#     row_sort_class=row_labels,
#     col_sort_class=col_labels,
# )

# #%%
# left_res = pd.concat(
#     (
#         pd.Series(data=sub_score_df.index, name="node_id"),
#         pd.Series(data=row_labels, name="cluster"),
#         pd.Series(data=sub_score_df.shape[0] * ["left"], name="side"),
#     ),
#     axis=1,
# )
# right_res = pd.concat(
#     (
#         pd.Series(data=sub_score_df.columns, name="node_id"),
#         pd.Series(data=col_labels, name="cluster"),
#         pd.Series(data=sub_score_df.shape[1] * ["right"], name="side"),
#     ),
#     axis=1,
# )
# res = pd.concat((left_res, right_res), axis=0, ignore_index=True)
# res = res.sort_values(["cluster", "side"])

# res.to_clipboard()


# #%%
# biadjacency[biadjacency < thresh] = 0
# biadjacency = sparse.csr_matrix(biadjacency)


# #%%


# # # %%
# # Tvals = T.ravel()
# # sns.histplot(Tvals[Tvals > 0], bins=100)

# # #%%
# # labels_left = sort_left_nodes[score_col].values
# # labels_right = sort_right_nodes[score_col].values

# # co_label_mat = labels_left[:, None] == labels_right[None, :]

# # # %%
# # thresh = 0.0001

# # for thresh in [
# #     0,
# #     1e-5,
# #     0.0004,
# #     0.0005,
# #     0.00055,
# #     0.000555,
# #     0.00056,
# #     0.000565,
# #     0.000566,
# #     0.0005665,
# # ]:
# #     n = len(al_left.nodes)
# #     m = len(al_right.nodes)
# #     biadjacency = np.zeros((n + m, n + m))
# #     biadjacency[:n, n:] = T
# #     biadjacency[biadjacency < thresh] = 0
# #     biadjacency = sparse.csr_matrix(biadjacency)

# #     n_components, labels = sparse.csgraph.connected_components(
# #         biadjacency, return_labels=True
# #     )
# #     print(n_components)

# #     cc_labels_left = labels[:n]
# #     cc_labels_right = labels[n:]
# #     matched = cc_labels_left[:, None] == cc_labels_right[None, :]
# #     metric = (matched & co_label_mat).sum() / matched.sum()
# #     print(metric)

# # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# # Tvals = T.ravel()
# # sns.histplot(Tvals[Tvals > 0], bins=100, log_scale=True, ax=ax)
# # ax.axvline(thresh, color="red")
