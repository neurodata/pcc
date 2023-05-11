# %%
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
from pkg.data import load_flywire_networkframe
from pkg.data import load_flywire_nblast_subset

DISPLAY_FIGS = True
FILENAME = "try_whole_brains"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


# %%


def load_nblast():
    data_dir = DATA_PATH / "hackathon"

    nblast = pl.scan_ipc(
        data_dir / "nblast" / "nblast_flywire_mcns_comp.feather",
        memory_map=False,
    )
    index = pd.Index(nblast.select("index").collect().to_pandas()["index"])
    columns = pd.Index(nblast.columns[1:])

    if index.dtype == "object":
        index_ids = index.str.split(",", expand=True).get_level_values(0).astype(int)
    else:
        index_ids = index
    column_ids = columns.str.split(",", expand=True).get_level_values(0).astype(int)

    print(column_ids)

    index_ids_map = dict(zip(index_ids, index))
    column_ids_map = dict(zip(column_ids, columns))
    index_ids_reverse_map = dict(zip(index, index_ids))
    column_ids_reverse_map = dict(zip(columns, column_ids))

    query_index = pd.Series([index_ids_map[i] for i in index])
    query_columns = pd.Series(["index"] + [column_ids_map[i] for i in columns])

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


currtime = time.time()
# nblast = load_nblast()
nblast = pd.read_feather(
    data_dir / "nblast" / "nblast" / "nblast_flywire_mcns_comp.feather",
)
print(
    f"{time.time() - currtime:.3f} seconds elapsed to load FlyWire vs. MaleCNS NBLASTs."
)
quit()

# %%
a = np.ones(n_left)
b = np.ones(n_right) / n_right * n_left
nblast_between = nblast.loc[left_nodes.index, right_nodes.index]
S = nblast_between.values.astype(float)


currtime = time.time()
sinkhorn_sol, log = ot.sinkhorn(
    a, b, -S, reg=0.03, numItermax=2000, verbose=True, log=True
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])) * 10, ax=ax)
ax.set_yscale("log")
