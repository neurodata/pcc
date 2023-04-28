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
from pkg.data import load_flywire_networkframe
from pkg.data import load_flywire_nblast_subset

DISPLAY_FIGS = True
FILENAME = "bigger"

data_dir = DATA_PATH / "hackathon"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


#%%

currtime = time.time()
flywire = load_flywire_networkframe()
print(f"{time.time() - currtime:.3f} seconds elapsed to load FlyWire network.")

flywire.query_nodes("(super_class == 'central') & (side != 'center')", inplace=True)
flywire.nodes.sort_values(["side", "cell_type_filled"], inplace=True)

print(len(flywire.nodes), " remaining after filtering.")

currtime = time.time()
nblast = load_flywire_nblast_subset(flywire.nodes.index)
print(
    f"{time.time() - currtime:.3f} seconds elapsed to load FlyWire central brain NBLASTs."
)

#%%
left_nodes = flywire.nodes.query("side == 'left'")
right_nodes = flywire.nodes.query("side == 'right'")
n_left = len(left_nodes)
n_right = len(right_nodes)
#%%
a = np.ones(n_left)
b = np.ones(n_right) / n_right * n_left
nblast_between = nblast.loc[left_nodes.index, right_nodes.index]
S = nblast_between.values.astype(float)

# sinkhorn_sol, log = ot.sinkhorn(
#     sinkhorn.mu_t, sinkhorn.mu_s, -S, reg=0.003, numItermax=500, verbose=True, log=True
# )

sinkhorn_sol, log = ot.sinkhorn(
    a, b, -S, reg=0.03, numItermax=500, verbose=True, log=True
)

#%%
np.count_nonzero(sinkhorn_sol) / sinkhorn_sol.size

#%%
flywire_left = flywire.query_nodes("side == 'left'")
flywire_right = flywire.query_nodes("side == 'right'")

#%%


def normalize(X, axis=1):
    if axis == 1:
        norm = np.linalg.norm(X, axis=axis)[:, None]
    else:
        norm = np.linalg.norm(X, axis=axis)[None, :]
    norm[norm == 0] = 1
    X = X / norm
    return X


# V2
A_in = flywire_left.to_sparse_adjacency().copy()
B_in = flywire_right.to_sparse_adjacency().copy()
A_in = normalize(A_in, axis=0)
B_in = normalize(B_in, axis=0)

in_sim = A_in.T @ sinkhorn_sol @ B_in

del A_in, B_in

A_out = flywire_left.to_sparse_adjacency().copy()
B_out = flywire_right.to_sparse_adjacency().copy()

A_out = normalize(A_out, axis=1)
B_out = normalize(B_out, axis=1)

out_sim = A_out @ sinkhorn_sol @ B_out.T

del A_out, B_out

conn_sim = in_sim + out_sim

#%%
show_slice = slice(300, 400)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(in_sim[show_slice][:, show_slice], ax=axs[0], cmap="Blues")
sns.heatmap(out_sim[show_slice][:, show_slice], ax=axs[1], cmap="Blues")


#%%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])), ax=ax)
ax.set_yscale("log")

#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

show_slice = slice(300, 400)
sns.heatmap(nblast_between.values[show_slice][:, show_slice], ax=axs[0], cmap="Blues")
sns.heatmap(sinkhorn_sol[show_slice][:, show_slice], ax=axs[1], cmap="Blues")

#%%
from graspologic.embed import AdjacencySpectralEmbed
from scipy import sparse

n_components = 16
ase = AdjacencySpectralEmbed(n_components=n_components, diag_aug=True, check_lcc=False)

currtime = time.time()
X_nblast = ase.fit_transform(sparse.csr_array(nblast.values))
print(f"{time.time() - currtime:.3f} seconds elapsed for sparse/randomized SVD.")


#%%

from giskard.plot import pairplot

labels = flywire.nodes["side"].values
side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
pairplot(
    X_nblast[:, :8], title="NBLAST", alpha=0.1, s=5, palette=side_palette, labels=labels
)
#%%
# 1e-3 was decent but looked over-smoothed
# 5e-3 looks decent, better than above
#
sinkhorn = ot.da.SinkhornTransport(reg_e=1e-4, max_iter=1, verbose=True, tol=1e-5)

n_left = len(flywire.nodes.query('side == "left"'))
X_nblast_left = X_nblast[:n_left].astype(float)
X_nblast_right = X_nblast[n_left:].astype(float)

currtime = time.time()
X_nblast_right_sinkhorn = sinkhorn.fit(Xs=X_nblast_right, Xt=X_nblast_left)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
sinkhorn.coupling_ = sinkhorn_sol.T

X_nblast_right_sinkhorn = sinkhorn.transform(Xs=X_nblast_right)

#%%

X_nblast_sinkhorn = np.concatenate((X_nblast_left, X_nblast_right_sinkhorn))

pairplot(
    X_nblast_sinkhorn[:, :8],
    title="NBLAST",
    alpha=0.1,
    s=5,
    palette=side_palette,
    labels=labels,
)

#%%
# from hyppo.ksample import KSample


# rng = np.random.default_rng(8888)
# stat = 0
# n_subsamples = 100
# for i in tqdm(range(n_subsamples)):
#     left_inds = rng.choice(len(X_nblast_left), size=1000, replace=False)
#     right_inds = rng.choice(len(X_nblast_right), size=1000, replace=False)
#     new_stat = KSample("Dcorr").statistic(
#         X_nblast_left[left_inds], X_nblast_right_sinkhorn[right_inds]
#     )
#     stat += new_stat / n_subsamples
