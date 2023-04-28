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
FILENAME = "transport"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, formats=["png"], **kwargs)

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

side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
broad_class_palette = dict(zip(broad_type_labels.unique(), sns.color_palette("tab10")))

#%%

nblast_cost = nblast.loc[al_left.nodes.index, al_right.nodes.index].values.copy()

n = max(nblast_cost.shape)
nblast_cost_padded = np.zeros((n, n))
nblast_cost_padded[: nblast_cost.shape[0], : nblast_cost.shape[1]] = nblast_cost


show_slice = slice(0, 50)


matrixplot_kws = dict(
    row_sort_class=al_left.nodes.iloc[show_slice][score_col],
    col_sort_class=al_right.nodes.iloc[show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    col_ticks=False,
    cbar=False,
    tick_fontsize=6,
)


def quick_matrixplot(A, ax=None, title=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matrixplot(
        A[show_slice, show_slice],
        row_ticks=True,
        ax=ax,
        **kwargs,
        **matrixplot_kws,
    )
    ax.set_xlabel("Right neuron")
    ax.set_ylabel("Left neuron")
    ax.set_title(title, pad=30)
    return ax


ax = quick_matrixplot(nblast_cost_padded, title="NBLAST")
gluefig("nblast", ax.figure)

#%%
from scipy.optimize import linear_sum_assignment

row_inds, col_inds = linear_sum_assignment(-nblast_cost_padded)
lap_sol = np.eye(n)[col_inds][:, row_inds]

ax = quick_matrixplot(lap_sol, title="LAP solution")
gluefig("lap_solution", ax.figure)

#%%

# a = np.ones(len(al_left.nodes)) / len(al_left.nodes)
# b = np.ones(len(al_right.nodes)) / len(al_right.nodes)

a = np.ones(n)
b = np.ones(n)

emd_sol = ot.emd(a, b, -nblast_cost_padded)

ax = quick_matrixplot(emd_sol, title="EMD solution")
gluefig("emd_solution", ax.figure)

#%%

sigma = 0.05

currtime = time.time()

noisy_emd_solution = np.zeros(nblast_cost_padded.shape)
for i in tqdm(range(100)):
    M = nblast_cost_padded + np.random.normal(0, sigma, size=nblast_cost_padded.shape)
    noisy_T = ot.emd(a, b, -M)
    noisy_emd_solution += noisy_T
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
ax = quick_matrixplot(
    noisy_emd_solution, title="Noisy EMD solution " + r"($\sigma = $" + f"{sigma:0.2f})"
)
gluefig("noisy_emd_solution", ax.figure)


#%%
if False:
    regularizers = np.geomspace(5e-3, 1e-1, 20)
    i = 12
    reg = regularizers[i]

    sinkhorn_sol = ot.sinkhorn(a, b, -nblast_cost_padded, reg)
    quick_matrixplot(
        sinkhorn_sol, title="Sinkhorn solution " + r"($\lambda = $" + f"{reg:0.2e})"
    )

#%%
reg = 0.03
a = np.ones(n)
b = np.ones(len(al_right.nodes)) * n / len(al_right.nodes)

currtime = time.time()
sinkhorn_sol = ot.sinkhorn(a, b, -nblast_cost, reg)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

ax = quick_matrixplot(
    sinkhorn_sol, title="Sinkhorn solution " + r"($\lambda = $" + f"{reg:0.2f})"
)
gluefig("sinkhorn_solution", ax.figure)

#%%
row_show_slice = slice(401, 461)
col_show_slice = slice(399, 459)

matrixplot_kws = dict(
    row_sort_class=al_left.nodes.iloc[row_show_slice][score_col],
    col_sort_class=al_right.nodes.iloc[col_show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    col_ticks=False,
    cbar=False,
    tick_fontsize=10,
)


def quick_matrixplot(A, ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    matrixplot(
        A[row_show_slice, col_show_slice],
        row_ticks=True,
        ax=ax,
        **kwargs,
        **matrixplot_kws,
    )
    ax.set_xlabel("Right neuron")
    ax.set_ylabel("Left neuron")
    ax.set_title(title, pad=30)
    ax.tick_params("y", rotation=0)
    return ax


fig, axs = plt.subplots(
    2, 2, figsize=(10, 10), gridspec_kw=dict(wspace=0.5, hspace=0.05)
)
ax = quick_matrixplot(nblast_cost_padded, ax=axs[0, 0], title="NBLAST")
ax = quick_matrixplot(lap_sol, ax=axs[0, 1], title="LAP solution")
ax = quick_matrixplot(
    noisy_emd_solution,
    title="Noisy EMD solution " + r"($\sigma = $" + f"{sigma:0.2f})",
    ax=axs[1, 0],
)
ax = quick_matrixplot(
    sinkhorn_sol,
    title="Sinkhorn solution " + r"($\lambda = $" + f"{reg:0.2f})",
    ax=axs[1, 1],
)
gluefig("composite_view_2", fig)


row_show_slice = slice(550, 650)
col_show_slice = slice(550, 650)
matrixplot_kws = dict(
    row_sort_class=al_left.nodes.iloc[row_show_slice][score_col],
    col_sort_class=al_right.nodes.iloc[col_show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    col_ticks=False,
    cbar=False,
    tick_fontsize=10,
)
fig, axs = plt.subplots(
    2, 2, figsize=(10, 10), gridspec_kw=dict(wspace=0.5, hspace=0.05)
)
ax = quick_matrixplot(nblast_cost_padded, ax=axs[0, 0], title="NBLAST")
ax = quick_matrixplot(lap_sol, ax=axs[0, 1], title="LAP solution")
ax = quick_matrixplot(
    noisy_emd_solution,
    title="Noisy EMD solution " + r"($\sigma = $" + f"{sigma:0.2f})",
    ax=axs[1, 0],
)
ax = quick_matrixplot(
    sinkhorn_sol,
    title="Sinkhorn solution " + r"($\lambda = $" + f"{reg:0.2f})",
    ax=axs[1, 1],
)
gluefig("composite_view_3", fig)


row_show_slice = slice(650, 780)
col_show_slice = slice(650, 780)
matrixplot_kws = dict(
    row_sort_class=al_left.nodes.iloc[row_show_slice][score_col],
    col_sort_class=al_right.nodes.iloc[col_show_slice][score_col],
    cmap="Blues",
    center=None,
    square=True,
    col_ticks=False,
    cbar=False,
    tick_fontsize=10,
)
fig, axs = plt.subplots(
    2, 2, figsize=(10, 10), gridspec_kw=dict(wspace=0.5, hspace=0.05)
)
ax = quick_matrixplot(nblast_cost_padded, ax=axs[0, 0], title="NBLAST")
ax = quick_matrixplot(lap_sol, ax=axs[0, 1], title="LAP solution")
ax = quick_matrixplot(
    noisy_emd_solution,
    title="Noisy EMD solution " + r"($\sigma = $" + f"{sigma:0.2f})",
    ax=axs[1, 0],
)
ax = quick_matrixplot(
    sinkhorn_sol,
    title="Sinkhorn solution " + r"($\lambda = $" + f"{reg:0.2f})",
    ax=axs[1, 1],
)
gluefig("composite_view_4", fig)


#%%

sinkhorn_sol_df = pd.DataFrame(
    sinkhorn_sol, index=al_left.nodes.index, columns=al_right.nodes.index
)
sinkhorn_sol_df.index.name = "node_id_left"
sinkhorn_sol_df.columns.name = "node_id_right"
sinkhorn_sol_df.to_csv(OUT_PATH / FILENAME / f"sinkhorn_solution_reg={reg}.csv")

#%%
left_nas = al_left.nodes.query(f"{score_col}.isna()").index
right_nas = al_right.nodes.query(f"{score_col}.isna()").index

matrixplot(
    sinkhorn_sol_df.loc[left_nas, right_nas].values,
    cmap="Blues",
    square=True,
    center=None,
    cbar=False,
)

sub_sol = sinkhorn_sol_df.loc[left_nas, right_nas].values

from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=15)
model.fit(sub_sol)

row_labels = model.row_labels_
col_labels = model.column_labels_

matrixplot(sub_sol, row_sort_class=row_labels, col_sort_class=col_labels)

sub_left_nodes = al_left.nodes.loc[left_nas].copy()
sub_right_nodes = al_right.nodes.loc[right_nas].copy()

sub_left_nodes["cluster"] = row_labels
sub_right_nodes["cluster"] = col_labels
sub_nodes = pd.concat([sub_left_nodes, sub_right_nodes])


#%%
nblast_ll = nblast.loc[al_left.nodes.index, al_left.nodes.index].values.copy()
nblast_lr = nblast.loc[al_left.nodes.index, al_right.nodes.index].values.copy()
nblast_rr = nblast.loc[al_right.nodes.index, al_right.nodes.index].values.copy()
nblast_rl = nblast.loc[al_right.nodes.index, al_left.nodes.index].values.copy()

D = sinkhorn_sol
nblast_lr = nblast_lr @ D.T
nblast_rl = nblast_lr.T
nblast_rr = D @ nblast_rr @ D.T
nblast_rr = (nblast_rr + nblast_rr.T) / 2

nblast_transformed = np.block([[nblast_ll, nblast_lr], [nblast_rl, nblast_rr]])


#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(nblast_ll[:10, :10], square=True, cmap="Blues", center=None, ax=axs[0])
sns.heatmap(nblast_rr[:10, :10], square=True, cmap="Blues", center=None, ax=axs[1])

#%%
nodes = pd.concat((al_left.nodes, al_right.nodes))
side_labels = nodes["side"]
broad_type_labels = nodes["cell_class"]
fine_type_labels = nodes["cell_type_filled"].fillna("unknown")

side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
broad_class_palette = dict(zip(broad_type_labels.unique(), sns.color_palette("tab10")))


def clustermap(X, D, method="ward", metric="euclidean", **kwargs):
    if X is None:
        X = squareform(1 - D)
    linkage_matrix = linkage(X, method=method, metric=metric)

    side_colors = pd.Series(side_labels, name="Side").map(side_palette)
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
    cgrid.ax_heatmap.set_ylabel(None)
    return cgrid


clustermap(None, nblast_transformed, method="ward")

#%%
if True:
    a = np.ones(n)
    b = np.ones(n)

    eta = 10
    labels_a = al_left.nodes[score_col].values
    labels_a = pd.Categorical(labels_a).codes
    sinkhorn_gl_sol = ot.da.sinkhorn_l1l2_gl(
        a, labels_a, b, -nblast_cost_padded, reg, eta, verbose=True
    )

    quick_matrixplot(
        sinkhorn_gl_sol,
        title="Sinkhorn GL solution "
        + r"($\lambda = $"
        + f"{reg:0.2e}, $\eta = $"
        + f"{eta:0.2e})",
    )
