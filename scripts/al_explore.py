#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

from tqdm.autonotebook import tqdm
import time
from joblib import Parallel, delayed
from giskard.plot import upset_catplot


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

nblast_within_left = nblast.loc[al_left.nodes.index, al_left.nodes.index].values
nblast_within_right = nblast.loc[al_right.nodes.index, al_right.nodes.index].values
nblast_between_left_right = nblast.loc[al_left.nodes.index, al_right.nodes.index].values

nblast_within_left = sparse.csr_array(nblast_within_left)
nblast_within_right = sparse.csr_array(nblast_within_right)

#%%

adjacency_left = al_left.to_sparse_adjacency().astype(float)
adjacency_right = al_right.to_sparse_adjacency().astype(float)


#%%
# rescaling everything...

desired_norm = 10_000

adjacency_left *= desired_norm / sparse.linalg.norm(adjacency_left)
adjacency_right *= desired_norm / sparse.linalg.norm(adjacency_right)
nblast_within_left *= desired_norm / sparse.linalg.norm(nblast_within_left)
nblast_within_right *= desired_norm / sparse.linalg.norm(nblast_within_right)

rows, cols = linear_sum_assignment(nblast_between_left_right, maximize=True)
nblast_between_left_right *= desired_norm / np.sum(
    nblast_between_left_right[rows, cols]
)

# #%%

# left_inputs = [adjacency_left, nblast_within_left]
# right_inputs = [adjacency_right, nblast_within_right]
# between_input = nblast_between_left_right

# row_inds, col_inds, score, misc = graph_match(
#     left_inputs, right_inputs, S=between_input, verbose=3
# )

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


dummy = True
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
graph_match_kws = dict(transport_regularizer=200)


def match_experiment(nblast_between, nblast_within, connectivity, transport, n_init=1):
    n_query1 = adjacency1.shape[0]
    n_query2 = adjacency2.shape[0]

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

    if n_init == 1:
        outs = graph_match(
            adjs1,
            adjs2,
            S=S,
            transport=transport,
            **graph_match_kws,
        )
        return [outs]
    else:

        def run():
            return graph_match(
                adjs1,
                adjs2,
                S=S,
                transport=transport,
                **graph_match_kws,
            )

        outs = Parallel(n_jobs=n_jobs)(delayed(run)() for _ in range(n_init))
        return outs

    # transport_mat = np.zeros((n_query1, n_query2))
    # total_scores = 0
    # match_mat = np.zeros((n_query1, n_query2))
    # for _ in tqdm(range(n_init)):
    # inds1, inds2, score, misc = graph_match(
    #     adjs1,
    #     adjs2,
    #     S=S,
    #     transport=transport,
    #     transport_regularizer=500,
    #     rng=8888,
    #     **graph_match_kws,
    # )

    # total_scores += score

    # P = misc[0]["convex_solution"]
    # P = P[:n_query1, :n_query2]
    # transport_mat += P * score

    # transport_mat = transport_mat / total_scores

    # return match_mat, transport_mat


def compute_scores(indices1, indices2):
    conn_score = np.linalg.norm(
        adjacency1.todense()[indices1][:, indices1]
        - adjacency2.todense()[indices2][:, indices2]
    )
    nblast_score = np.linalg.norm(
        similarity1.todense()[indices1][:, indices1]
        - similarity2.todense()[indices2][:, indices2]
    )
    nblast_between_score = similarity12[indices1, indices2].sum()

    return conn_score, nblast_score, nblast_between_score


n_init = 10

type_col = "hemibrain_type"

RERUN_EXPERIMENT = True

if RERUN_EXPERIMENT:
    metrics = []
    for nblast_between in [True, False]:
        for nblast_within in [True, False]:
            for connectivity in [True, False]:
                for transport in [True, False]:
                    print("----")
                    print("NBLAST between, NBLAST within, connectivity, transport")
                    print(nblast_between, nblast_within, connectivity, transport)

                    if not (nblast_between or nblast_within or connectivity):
                        continue

                    currtime = time.time()

                    options = (nblast_between, nblast_within, connectivity, transport)
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

                        conn_score, nblast_score, nblast_between_score = compute_scores(
                            indices1, indices2
                        )

                        metrics.append(
                            {
                                "nblast_between": nblast_between,
                                "nblast_within": nblast_within,
                                "connectivity": connectivity,
                                "transport": transport,
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


upset_catplot(
    results.sort_values("accuracy", ascending=False),
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="accuracy",
)

upset_catplot(
    results.sort_values("accuracy", ascending=False),
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="conn_score",
)

upset_catplot(
    results.sort_values("accuracy", ascending=False),
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="nblast_score",
)

upset_catplot(
    results.sort_values("accuracy", ascending=False),
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="nblast_between_score",
)

# %%
#%%
# matched_nodes = create_matched_nodes(row_inds, col_inds)
# colabeling = matched_nodes[["hemibrain_type_left", "hemibrain_type_right"]].dropna()

# confusionplot(
#     colabeling["hemibrain_type_left"],
#     colabeling["hemibrain_type_right"],
#     annot=True,
#     figsize=(20, 20),
# )
