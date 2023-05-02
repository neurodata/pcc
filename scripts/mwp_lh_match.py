#%%
import time

import numpy as np
import pandas as pd
from pkg.data import DATA_PATH
from pkg.io import OUT_PATH
from sklearn.metrics.pairwise import cosine_similarity

from graspologic.match import graph_match


FILENAME = "mwp_lh_match"
data_dir = DATA_PATH / "hackathon" / "mwp_lh"

# %%


def load_adjacency(side):
    if side == "left":
        token = "LHS"
    elif side == "right":
        token = "RHS"
    left_adj = pd.read_csv(data_dir / f"{token}_LH_adjx.csv", index_col=0)
    non_dupe_cols = (
        left_adj.columns.str.split(".", expand=True).get_level_values(0).unique()
    )
    left_adj = left_adj[non_dupe_cols]
    left_adj.columns = left_adj.columns.astype(int)
    left_adj = left_adj[~left_adj.index.duplicated(keep="first")]
    left_adj = left_adj.loc[left_adj.columns]
    return left_adj


left_adj = load_adjacency("left")
right_adj = load_adjacency("right")

#%%

n_init = 20

graph_match_kws = dict(
    max_iter=20,
    shuffle_input=True,
    n_jobs=-2,
    n_init=n_init,
    init_perturbation=0,
    verbose=3,
    rng=8888,
)

currtime = time.time()
indices_left, indices_right, score, misc = graph_match(
    left_adj.values,
    right_adj.values,
    **graph_match_kws,
)
print(
    f"{time.time() - currtime:.3f} seconds elapsed for {n_init} runs of graph matching."
)


#%%
index_left = left_adj.index[indices_left]
index_right = right_adj.index[indices_right]
left_adj_perm = left_adj.loc[index_left, index_left]
right_adj_perm = right_adj.loc[index_right, index_right]

#%%
pair_df = pd.DataFrame()
pair_df["left_id"] = index_left
pair_df["right_id"] = index_right

out_cos_sims = np.diag(cosine_similarity(left_adj_perm, right_adj_perm))
pair_df["out_cos_sim"] = out_cos_sims

in_cos_sims = np.diag(cosine_similarity(left_adj_perm.T, right_adj_perm.T))
pair_df["in_cos_sim"] = in_cos_sims

pair_df["cos_sim"] = (pair_df["out_cos_sim"] + pair_df["in_cos_sim"]) / 2

pair_df.sort_values("cos_sim", ascending=False, inplace=True)
pair_df

#%%
pair_df.to_csv(OUT_PATH / FILENAME / "mwp_lh_match.csv", index=False)
