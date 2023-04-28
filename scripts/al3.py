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
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from tqdm.autonotebook import tqdm

from giskard.plot import confusionplot, matrixplot
from graspologic.match import graph_match
from neuropull.graph import NetworkFrame


DISPLAY_FIGS = True
FILENAME = "al3"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%%
def load_al_nblast(queries=None):
    data_dir = DATA_PATH / "hackathon"

    nblast = pl.scan_ipc(
        data_dir / "al" / "AL_three_hemisphere_nblast.feather",
        memory_map=False,
    )
    index = pd.Index(nblast.select("query").collect().to_pandas()["query"])
    columns = pd.Index(nblast.columns[1:])
    # index_ids = index.str.split(",", expand=True).get_level_values(0).astype(int)
    # column_ids = columns.str.split(",", expand=True).get_level_values(0).astype(int)
    index_ids = index
    column_ids = columns
    index_ids_map = dict(zip(index_ids, index))
    column_ids_map = dict(zip(column_ids, columns))
    index_ids_reverse_map = dict(zip(index, index_ids))
    column_ids_reverse_map = dict(zip(columns, column_ids))

    if queries is None:
        query_node_ids = index
    elif not isinstance(queries, tuple):
        query_node_ids = queries
    else:
        query_node_ids = np.concatenate(queries)

    query_index = pd.Series([index_ids_map[i] for i in query_node_ids])
    query_columns = pd.Series(["query"] + [column_ids_map[i] for i in query_node_ids])

    nblast = nblast.with_columns(
        pl.col("query").is_in(query_index).alias("select_index")
    )

    mini_nblast = (
        nblast.filter(pl.col("select_index"))
        .select(query_columns)
        .collect()
        .to_pandas()
    ).set_index("query")

    mini_nblast.index = mini_nblast.index.map(index_ids_reverse_map)
    mini_nblast.columns = mini_nblast.columns.map(column_ids_reverse_map)

    return mini_nblast


# %%
nodes = pd.read_csv(
    DATA_PATH / "hackathon" / "al" / "AL_three_hemisphere_meta.csv", index_col=0
).set_index("id")
nodes["side"] = nodes["side"].map(lambda x: np.nan if x in ["na", "none"] else x)
nodes = nodes[~nodes["side"].isna()]
nodes["dataset"] = nodes["source"] + "-" + nodes["side"]
nodes["dataset"].unique()

nodes = nodes[~nodes["cell_class"].isin(["olfactory", "thermosensory", "hygrosensory"])]
nodes["cell_class"].value_counts()
#%%
nblast = pd.read_feather(
    DATA_PATH / "hackathon" / "al" / "AL_three_hemisphere_nblast.feather"
).set_index("query")
nblast.columns = nblast.columns.astype(int)

index = nodes.index.intersection(nblast.index)
nodes = nodes.loc[index]

nblast = nblast.loc[index, index]
# optional...
nblast[nblast < 0] = 0


# %%


dataset_labels = nodes["dataset"]
broad_type_labels = nodes["cell_class"]

dataset_palette = dict(
    zip(
        ["flywire-left", "flywire-right", "hemibrain-right"],
        sns.color_palette("Set2", 3),
    )
)
broad_class_palette = dict(zip(broad_type_labels.unique(), sns.color_palette("tab10")))


def clustermap(X, D, method="ward", metric="euclidean"):
    if X is None:
        X = squareform(1 - D)
    linkage_matrix = linkage(X, method=method, metric=metric)

    side_colors = pd.Series(dataset_labels, name="Side").map(dataset_palette)
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


clustermap(None, nblast)

#%%

