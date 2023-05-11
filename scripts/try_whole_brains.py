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

currtime = time.time()
# nblast = load_nblast()

nblast = pd.read_feather(
    data_dir / "nblast" / "nblast" / "nblast_flywire_mcns_comp.feather",
)
print(
    f"{time.time() - currtime:.3f} seconds elapsed to load FlyWire vs. MaleCNS NBLASTs."
)


# %%

n_left = nblast.shape[0]
n_right = nblast.shape[1]

a = np.ones(n_left)
b = np.ones(n_right) / n_right * n_left
S = nblast.values.astype(float)

currtime = time.time()
sinkhorn_sol, log = ot.sinkhorn(
    a, b, -S, reg=0.03, numItermax=200, verbose=True, log=True
)
print(f"{time.time() - currtime:.3f} seconds elapsed to sinkhorn.")

currtime = time.time()
threshold = 1e-3
sinkhorn_sol[sinkhorn_sol < threshold] = 0
print(f"{time.time() - currtime:.3f} seconds elapsed to threshold.")

currtime = time.time()
sinkhorn_sol = pd.DataFrame(sinkhorn_sol, index=nblast.index, columns=nblast.columns)
sinkhorn_sol.to_feather(data_dir / "nblast" / "nblast" / "sinkhorn_sol.feather")
print(f"{time.time() - currtime:.3f} seconds elapsed to save.")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])) * 10, ax=ax)
ax.set_yscale("log")
