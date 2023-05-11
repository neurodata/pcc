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
FILENAME = "bigger"

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

# %%
left_nodes = flywire.nodes.query("side == 'left'")
right_nodes = flywire.nodes.query("side == 'right'")
n_left = len(left_nodes)
n_right = len(right_nodes)
# %%
a = np.ones(n_left)
b = np.ones(n_right) / n_right * n_left
nblast_between = nblast.loc[left_nodes.index, right_nodes.index]
S = nblast_between.values.astype(float)

# sinkhorn_sol, log = ot.sinkhorn(
#     sinkhorn.mu_t, sinkhorn.mu_s, -S, reg=0.003, numItermax=500, verbose=True, log=True
# )

currtime = time.time()
sinkhorn_sol, log = ot.sinkhorn(
    a, b, -S, reg=0.03, numItermax=2000, verbose=True, log=True
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])) * 10, ax=ax)
ax.set_yscale("log")

# %%
transport_df = pd.DataFrame(
    sinkhorn_sol, index=left_nodes.index, columns=right_nodes.index
)


# %%
label_col = "cell_type_filled"
left_labels = left_nodes[label_col]
right_labels = right_nodes[label_col]
valid_left_labels = left_labels[left_labels.notna()]
valid_right_labels = right_labels[right_labels.notna()]
intersect_labels = np.intersect1d(
    valid_left_labels.unique(), valid_right_labels.unique()
)
valid_left_labels = valid_left_labels[valid_left_labels.isin(intersect_labels)]
valid_right_labels = valid_right_labels[valid_right_labels.isin(intersect_labels)]

# %%
left_label_counts = valid_left_labels.value_counts()
right_label_counts = valid_right_labels.value_counts()
left_label_singles = left_label_counts[left_label_counts == 1].index
right_label_singles = right_label_counts[right_label_counts == 1].index
label_singles = np.intersect1d(left_label_singles, right_label_singles)

left_singles = valid_left_labels[valid_left_labels.isin(label_singles)]
right_singles = valid_right_labels[valid_right_labels.isin(label_singles)]

assert (left_singles.values == right_singles.values).all()

scores = np.diag(transport_df.loc[left_singles.index, right_singles.index])

known_singletons_df = pd.concat(
    (
        left_singles.index.to_series(name="left_id").reset_index(drop=True),
        right_singles.index.to_series(name="right_id").reset_index(drop=True),
        pd.Series(scores, name="score"),
    ),
    axis=1,
).sort_values("score", ascending=False)
known_singletons_df.to_clipboard()

# %%
bad_known_singletons = known_singletons_df.query("score < 0.5")
k = 5
cols1 = [f"rank_{i}_id" for i in range(1, k + 1)]
cols2 = [f"rank_{i}_score" for i in range(1, k + 1)]
rows = transport_df.loc[bad_known_singletons["left_id"]].rank(axis=1, ascending=False)
all_scores = []
for idx, row in rows.iterrows():
    top_ranks = row[row <= 5].sort_values()
    top_scores = transport_df.loc[idx, top_ranks.index]
    top_scores.name = idx
    top_score_ids = top_scores.index.to_series()
    top_score_ids.index = cols1
    top_score_ids.name = idx
    top_score_ids = top_score_ids.to_frame().T
    top_score_values = pd.Series(top_scores.values, index=cols2, name=idx).to_frame().T
    top_scores = pd.concat((top_score_ids, top_score_values), axis=1)
    all_scores.append(top_scores)
left_other_matches = pd.concat(all_scores, axis=0)
left_other_matches.to_clipboard()
# %%
valid_transport_df = transport_df.loc[valid_left_labels.index, valid_right_labels.index]

sns.heatmap(valid_transport_df.iloc[300:400, 300:400], cmap="Blues")

# %%
collapsed_on_left = valid_transport_df.groupby(valid_left_labels).sum()
best_guess_per_right = collapsed_on_left.idxmax()
best_guess_per_right.name = "predicted_label"
best_guess_scores_per_right = collapsed_on_left.max()
best_guess_scores_per_right.name = "score"

acc = (best_guess_per_right == valid_right_labels).mean()
acc

# %%
mismatch_mask = (best_guess_per_right != valid_right_labels) & (
    best_guess_scores_per_right > 0.8
)
mismatches_high_conf = pd.concat(
    (
        best_guess_per_right[mismatch_mask],
        valid_right_labels[mismatch_mask],
        best_guess_scores_per_right[mismatch_mask],
    ),
    axis=1,
)
mismatches_high_conf.to_clipboard()


# %%
thresh = 0.8
thresh_acc = (
    best_guess_per_right[best_guess_scores_per_right > thresh]
    == valid_right_labels[best_guess_scores_per_right > thresh]
).mean()

thresh_amount = (best_guess_scores_per_right > thresh).mean()

print(f"Accuracy for threshold of {thresh}: {thresh_acc:.3f}")
print(f"Proportion of data above threshold of {thresh}: {thresh_amount:.3f}")

# %%
unlabeled_transport_df = transport_df.loc[left_labels.isna(), right_labels.isna()]

# %%

mask = unlabeled_transport_df > 0.8

row_inds, col_inds = np.nonzero(mask.values)
row_ids = (
    unlabeled_transport_df.index[row_inds]
    .to_series(name="left_id")
    .reset_index(drop=True)
)
col_ids = (
    unlabeled_transport_df.columns[col_inds]
    .to_series(name="right_id")
    .reset_index(drop=True)
)
score = pd.Series(unlabeled_transport_df.values[row_inds, col_inds], name="score")
summary_df = (
    pd.concat((row_ids, col_ids, score), axis=1)
    .sort_values("score", ascending=False)
    .reset_index(drop=True)
)
summary_df.to_clipboard()

# %%
from hyppo.discrim import DiscrimOneSample

DiscrimOneSample(is_dist=True).statistic()

# %%
np.count_nonzero(sinkhorn_sol) / sinkhorn_sol.size

# %%
flywire_left = flywire.query_nodes("side == 'left'")
flywire_right = flywire.query_nodes("side == 'right'")


# %%


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

# %%
show_slice = slice(300, 400)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

sns.heatmap(in_sim[show_slice][:, show_slice], ax=axs[0], cmap="Blues")
sns.heatmap(out_sim[show_slice][:, show_slice], ax=axs[1], cmap="Blues")


# %%

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(y=log["err"], x=np.arange(len(log["err"])), ax=ax)
ax.set_yscale("log")

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

show_slice = slice(300, 400)
sns.heatmap(nblast_between.values[show_slice][:, show_slice], ax=axs[0], cmap="Blues")
sns.heatmap(sinkhorn_sol[show_slice][:, show_slice], ax=axs[1], cmap="Blues")

# %%
from graspologic.embed import AdjacencySpectralEmbed
from scipy import sparse

n_components = 16
ase = AdjacencySpectralEmbed(n_components=n_components, diag_aug=True, check_lcc=False)

currtime = time.time()
X_nblast = ase.fit_transform(sparse.csr_array(nblast.values))
print(f"{time.time() - currtime:.3f} seconds elapsed for sparse/randomized SVD.")


# %%

from giskard.plot import pairplot

labels = flywire.nodes["side"].values
side_palette = dict(zip(["left", "right"], sns.color_palette("Set2", 2)))
pairplot(
    X_nblast[:, :8], title="NBLAST", alpha=0.1, s=5, palette=side_palette, labels=labels
)
# %%
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

# %%
sinkhorn.coupling_ = sinkhorn_sol.T

X_nblast_right_sinkhorn = sinkhorn.transform(Xs=X_nblast_right)

# %%

X_nblast_sinkhorn = np.concatenate((X_nblast_left, X_nblast_right_sinkhorn))

pairplot(
    X_nblast_sinkhorn[:, :8],
    title="NBLAST",
    alpha=0.1,
    s=5,
    palette=side_palette,
    labels=labels,
)

# %%
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
