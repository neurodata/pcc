#%%
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.autonotebook import tqdm

from graspologic.match import graph_match
from graspologic.plot import adjplot

#%%


def load_data(dataset):
    # Load the edges
    edges_df = pd.read_csv(f"pcc/data/flywire-philipp/2023-02-05/{dataset}_edges.csv")[
        ["pre", "post", "weight"]
    ]

    # Load the nodes
    nodes_df = pd.read_csv(
        f"pcc/data/flywire-philipp/2023-02-05/{dataset}_meta_data.csv"
    )[["id", "type"]]
    # TODO  why are there some dupes?
    # I assume these did not have a single label assigned to them?
    # perhaps could add half the weights to each of the two labels?
    nodes_df.drop_duplicates(["id"], inplace=True)
    nodes_df["id"] = nodes_df["id"].astype(int)
    nodes_df = nodes_df.set_index("id")

    # Annotate the nodes which are in the lineage we are interested in
    lineage_labels = pd.read_csv(
        f"pcc/data/flywire-philipp/2023-02-05/{dataset}_LHAV4_labels.csv"
    )[["id", "type"]]
    lineage_labels["id"] = lineage_labels["id"].astype(int)
    lineage_labels = lineage_labels.set_index("id")
    in_lineage_ids = lineage_labels.index
    nodes_df["query"] = False
    nodes_df.loc[in_lineage_ids, "query"] = True
    nodes_df.sort_values("query", inplace=True, ascending=False)
    nodes_df["type"].fillna("unknown", inplace=True)

    g = nx.from_pandas_edgelist(
        edges_df,
        edge_attr="weight",
        create_using=nx.DiGraph(),
        source="pre",
        target="post",
    )

    return edges_df, nodes_df, g


def load_nblast(dataset1, dataset2=None):
    if dataset2 is None:
        filename = f"{dataset1}_allbyall.csv"
    else:
        filename = f"{dataset1}_vs_{dataset2}.csv"
    nblasts = pd.read_csv(
        "pcc/data/flywire-philipp/2023-02-05/nblast_" + filename, index_col=0
    )
    nblasts.index = nblasts.index.astype(str)
    nblasts.columns = nblasts.columns.astype(str)
    return nblasts


#%%

dataset1 = "flywire"
dataset2 = "hemibrain"

edges1, nodes1, g1 = load_data(dataset1)
adj1 = nx.to_numpy_array(g1, nodelist=nodes1.index)

edges2, nodes2, g2 = load_data(dataset2)
adj2 = nx.to_numpy_array(g2, nodelist=nodes2.index)

#%%


adjplot(adj1, plot_type="scattermap", sizes=(0.1, 0.1))


#%%
intersect_labels = np.intersect1d(
    nodes1.query("~query")["type"], nodes2.query("~query")["type"]
)

#%%
# collapse labels graph

#%%


def collapse_adjacency_on_labels(adj, nodes_df, labels_to_squash):
    n_query = len(nodes_df[nodes_df["query"]])
    n_non_query_groups = len(labels_to_squash)
    n_total = n_query + n_non_query_groups
    new_adj = np.zeros((n_total, n_total))
    new_adj[:n_query, :n_query] = adj[:n_query, :n_query]

    new_index = np.array(
        list(nodes_df[nodes_df["query"]].index) + list(labels_to_squash)
    )

    for i, label in enumerate(labels_to_squash):
        inds = np.where(nodes_df["type"] == label)[0]
        int_index = i + n_query
        new_adj[:n_query, int_index] = adj[:n_query, inds].sum(axis=1)
        new_adj[int_index, :n_query] = adj[inds, :n_query].sum(axis=0)

    return new_adj, new_index, n_query


squashed_adj1, index1, n_query1 = collapse_adjacency_on_labels(
    adj1, nodes1, intersect_labels
)
print(squashed_adj1.shape)
squashed_adj2, index2, n_query2 = collapse_adjacency_on_labels(
    adj2, nodes2, intersect_labels
)
print(squashed_adj2.shape)

assert (index1[n_query1:] == index2[n_query2:]).all()

#%%

seeds1 = np.arange(n_query1, len(index1))
seeds2 = np.arange(n_query2, len(index2))
seeds = np.stack((seeds1, seeds2)).T

#%%

nblast_between = True
nblast_within = False

if nblast_between:
    nblast_1_to_2 = load_nblast(dataset1, dataset2)
    nblast_1_to_2 = nblast_1_to_2.loc[index1[:n_query1], index2[:n_query2]]
    nblast_1_to_2 = nblast_1_to_2.reindex(index=index1, columns=index2).fillna(0)

if nblast_within:
    nblast_1_to_1 = load_nblast(dataset1)
    nblast_1_to_1 = nblast_1_to_1.loc[index1[:n_query1], index1[:n_query1]]
    nblast_2_to_2 = load_nblast(dataset2)
    nblast_2_to_2 = nblast_2_to_2.loc[index2[:n_query2], index2[:n_query2]]

#%%
# TODO how to rescale these to match the scale of the adjacencies?

#%%

# TODO make this a function

graph_match_kws = dict(
    partial_match=seeds,
    max_iter=100,
    tol=1e-3,
    init_perturbation=0.5,
)
match_mat = np.zeros((n_query1, n_query2))
n_init = 10
for i in tqdm(range(n_init)):
    if (not nblast_between) and (not nblast_within):
        inds1, inds2, score, misc = graph_match(
            squashed_adj1,
            squashed_adj2,
            **graph_match_kws,
        )
    elif nblast_between and (not nblast_within):
        inds1, inds2, score, misc = graph_match(
            squashed_adj1,
            squashed_adj2,
            S=nblast_1_to_2.values,
            **graph_match_kws,
        )
    elif (not nblast_between) and nblast_within:
        inds1, inds2, score, misc = graph_match(
            squashed_adj1,
            squashed_adj2,
            **graph_match_kws,
        )
    else:  # nblast_between and nblast_within:
        inds1, inds2, score, misc = graph_match(
            squashed_adj1,
            squashed_adj2,
            **graph_match_kws,
        )

    # cleanup
    nonseed_mask = np.isin(inds1, np.setdiff1d(inds1, seeds1))
    nonseed_inds1 = inds1[nonseed_mask]
    nonseed_inds2 = inds2[nonseed_mask]
    match_mat[nonseed_inds1, nonseed_inds2] += 1 / n_init

#%%
match_mat = pd.DataFrame(
    index=index1[:n_query1], columns=index2[:n_query2], data=match_mat
)
match_mat.index.name = f"{dataset1}_id"
match_mat.columns.name = f"{dataset2}_id"
match_mat

#%%

sns.heatmap(match_mat, square=True, cmap="RdBu_r", vmin=0, center=0, vmax=n_init)
# %%

row_inds, col_inds = np.nonzero(match_mat.values)
match_mat.values[row_inds, col_inds]

#%%

match_summary = pd.DataFrame(index=match_mat.index)
for d1_id, row in match_mat.iterrows():
    sort_inds = np.argsort(-row)

    matches = row.index[sort_inds]
    scores = row.values[sort_inds]
    for i, (d2_id, score) in enumerate(zip(matches, scores)):
        if score == 0:
            break
        match_summary.loc[d1_id, f"{dataset2}_match_{i+1}"] = d2_id
        match_summary.loc[d1_id, f"{dataset2}_score_{i+1}"] = np.round(score, 2)

match_summary

#%%
match_summary.to_csv(
    f"pcc/results/outputs/flywire_match/{dataset1}_to_{dataset2}_match_summary.csv"
)
# %%
