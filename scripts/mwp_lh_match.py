#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import polars as pl
import seaborn as sns
from pkg.data import DATA_PATH, load_flywire_networkframe
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from scipy.optimize import linear_sum_assignment
from tqdm.autonotebook import tqdm

from graspologic.match import graph_match


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
    al = flywire.query_nodes(f"cell_class.isin({al_types})").copy()
    return al


score_col = "cell_type_filled"

flywire = load_flywire_networkframe()
al = select_al(flywire)
al.nodes.sort_values(score_col, inplace=True)



#%%
al_left = al.query_nodes("side == 'left'").copy()
al_right = al.query_nodes("side == 'right'").copy()

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
