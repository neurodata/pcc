#%%
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.autonotebook import tqdm
from scipy.stats import rankdata

from graspologic.match import graph_match
from graspologic.plot import adjplot

from pkg.io import glue as default_glue
from pkg.io import savefig
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


FILENAME = "flywire_match"
DISPLAY_FIGS = True


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


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


def load_nblast(dataset1, dataset2=None, normalize=False, renormalize=False):
    if dataset2 is None:
        filename = f"{dataset1}_allbyall"
    else:
        filename = f"{dataset1}_vs_{dataset2}"
    if normalize:
        filename = filename + "_nonnorm.csv"
    else:
        filename = filename + ".csv"
    nblasts = pd.read_csv(
        "pcc/data/flywire-philipp/2023-02-05/nblast_" + filename, index_col=0
    )
    nblasts.index = nblasts.index.astype(str)
    nblasts.columns = nblasts.columns.astype(str)

    values = nblasts.values

    if dataset2 is None:
        values -= np.diag(np.diag(values))

    if renormalize:
        inds = np.nonzero(values)
        values[inds] = rankdata(values[inds]) / len(values[inds])

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

    # NOTE this could be sped up a lot with broadcasting but my late night numpy-fu was
    # not up to the task. I doubt this is a bottleneck, though
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

normalize = False
renormalize = True

nblast_1_to_2 = load_nblast(
    dataset1, dataset2, normalize=normalize, renormalize=renormalize
)
nblast_1_to_1 = load_nblast(dataset1, normalize=normalize, renormalize=renormalize)
nblast_2_to_2 = load_nblast(dataset2, normalize=normalize, renormalize=renormalize)

# TODO not sure what the correct weighting is here
nblast_1_to_2 = (
    nblast_1_to_2 / np.linalg.norm(nblast_1_to_2) * np.linalg.norm(squashed_adj1)
)

nblast_1_to_1 = (
    nblast_1_to_1 / np.linalg.norm(nblast_1_to_1) * np.linalg.norm(squashed_adj1)
)
nblast_2_to_2 = (
    nblast_2_to_2 / np.linalg.norm(nblast_2_to_2) * np.linalg.norm(squashed_adj2)
)

# put things in the same orientation as the adjacencies
nblast_1_to_2 = nblast_1_to_2.reindex(index=index1, columns=index2).fillna(0)
nblast_1_to_1 = nblast_1_to_1.reindex(index=index1, columns=index1).fillna(0)
nblast_2_to_2 = nblast_2_to_2.reindex(index=index2, columns=index2).fillna(0)

#%%
# TODO how to rescale these to match the scale of the adjacencies?

#%%

n_init = 20
graph_match_kws = dict(
    partial_match=seeds,
    max_iter=100,
    tol=1e-3,
    init_perturbation=0.5,
)


def mapper(x):
    if isinstance(x, float):
        return np.nan
    elif isinstance(x, str):
        x = int(x)
        return nodes2.loc[x, "type"]


def compute_metrics(transport_mat):
    def collapse_adjacency_on_labels(adj, labels, axis=0):
        if axis == "both":
            semi_squashed_adj = collapse_adjacency_on_labels(adj, labels, axis=0)
            new_adjacency = collapse_adjacency_on_labels(
                semi_squashed_adj.values, labels, axis=1
            )
            new_adjacency.index = new_adjacency.columns
            return new_adjacency

        unique_labels = np.unique(labels)
        new_pieces = []
        for i, label in enumerate(unique_labels):
            inds = np.where(labels == label)[0]
            if axis == 0:
                squashed_adj_for_label = adj[inds].sum(axis=0)
            else:
                squashed_adj_for_label = adj[:, inds].sum(axis=1)
            new_pieces.append(squashed_adj_for_label)
        new_adjacency = np.stack(new_pieces, axis=axis)
        if axis == 0:
            new_adjacency = pd.DataFrame(new_adjacency, index=unique_labels)
        elif axis == 1:
            new_adjacency = pd.DataFrame(new_adjacency, columns=unique_labels)
        return new_adjacency

    # row_inds, col_inds = linear_sum_assignment(transport_mat.values, maximize=True)

    top_matches = transport_mat.idxmax(axis=1)
    top_matches.map(mapper)

    new_to_new1 = squashed_adj1[:n_query1, :n_query1]
    new_to_old1 = squashed_adj1[:n_query1, n_query1:]
    old_to_new1 = squashed_adj1[n_query1:, :n_query1]

    mapped_labels = top_matches.map(mapper)
    squashed_new_to_old1 = collapse_adjacency_on_labels(
        new_to_old1, mapped_labels, axis=0
    )
    squashed_old_to_new1 = collapse_adjacency_on_labels(
        old_to_new1, mapped_labels, axis=1
    )
    squashed_new_to_new1 = collapse_adjacency_on_labels(
        new_to_new1, mapped_labels, axis="both"
    )

    new_to_new2 = squashed_adj2[:n_query2, :n_query2]
    new_to_old2 = squashed_adj2[:n_query2, n_query2:]
    old_to_new2 = squashed_adj2[n_query2:, :n_query2]

    hemibrain_labels = nodes2.query("query")["type"].values
    squashed_new_to_old2 = collapse_adjacency_on_labels(
        new_to_old2, hemibrain_labels, axis=0
    )
    squashed_old_to_new2 = collapse_adjacency_on_labels(
        old_to_new2, hemibrain_labels, axis=1
    )
    squashed_new_to_new2 = collapse_adjacency_on_labels(
        new_to_new2, hemibrain_labels, axis="both"
    )

    squashed_new_to_new1 = squashed_new_to_new1.reindex(
        index=squashed_new_to_new2.index, columns=squashed_new_to_new2.columns
    ).fillna(0)
    squashed_new_to_old1 = squashed_new_to_old1.reindex(
        index=squashed_new_to_old2.index, columns=squashed_new_to_old2.columns
    ).fillna(0)
    squashed_old_to_new1 = squashed_old_to_new1.reindex(
        index=squashed_old_to_new2.index, columns=squashed_old_to_new2.columns
    ).fillna(0)

    cos_new_to_new = np.diag(
        cosine_similarity(squashed_new_to_new1, squashed_new_to_new2)
    )
    cos_new_to_old = np.diag(
        cosine_similarity(squashed_new_to_old1, squashed_new_to_old2)
    )
    cos_old_to_new = np.diag(
        cosine_similarity(squashed_old_to_new1, squashed_old_to_new2)
    )

    return (
        np.nanmean(cos_new_to_new),
        np.nanmean(cos_new_to_old),
        np.nanmean(cos_old_to_new),
    )


def match_experiment(nblast_between, nblast_within, connectivity, transport):
    adjs1 = []
    adjs2 = []
    if connectivity:
        adjs1.append(squashed_adj1)
        adjs2.append(squashed_adj2)
    if nblast_within:
        adjs1.append(nblast_1_to_1.values)
        adjs2.append(nblast_2_to_2.values)

    # Should reduce to the LAP basically, but just wanted to keep the code the same
    if len(adjs1) == 0:
        adjs1.append(np.zeros(squashed_adj1.shape))
        adjs2.append(np.zeros(squashed_adj2.shape))

    if nblast_between:
        S = nblast_1_to_2.values
    else:
        S = None

    transport_mat = np.zeros((n_query1, n_query2))
    total_scores = 0
    match_mat = np.zeros((n_query1, n_query2))
    for _ in tqdm(range(n_init)):
        inds1, inds2, score, misc = graph_match(
            adjs1,
            adjs2,
            S=S,
            transport=transport,
            transport_regularizer=500,
            rng=8888,
            **graph_match_kws,
        )

        total_scores += score

        P = misc[0]["convex_solution"]
        P = P[:n_query1, :n_query2]
        transport_mat += P * score

        # TODO add an option for soft matching here

        nonseed_mask = np.isin(inds1, np.setdiff1d(inds1, seeds1))
        nonseed_inds1 = inds1[nonseed_mask]
        nonseed_inds2 = inds2[nonseed_mask]
        match_mat[nonseed_inds1, nonseed_inds2] += 1 / n_init

    # cleanup
    match_mat = pd.DataFrame(
        index=index1[:n_query1], columns=index2[:n_query2], data=match_mat
    )
    match_mat.index.name = f"{dataset1}_id"
    match_mat.columns.name = f"{dataset2}_id"

    transport_mat /= total_scores
    transport_mat = pd.DataFrame(
        index=index1[:n_query1], columns=index2[:n_query2], data=transport_mat
    )
    transport_mat.index.name = f"{dataset1}_id"
    transport_mat.columns.name = f"{dataset2}_id"

    return match_mat, transport_mat


match_mats = {}
transport_mats = {}
metrics = []
for nblast_between in [True, False]:
    for nblast_within in [True, False]:
        for connectivity in [True, False]:
            for transport in [True, False]:
                if not (nblast_between or nblast_within or connectivity):
                    continue
                options = (nblast_between, nblast_within, connectivity, transport)
                match_mat, transport_mat = match_experiment(*options)
                match_mats[options] = match_mat
                transport_mats[options] = transport_mat
                cos_new_new, cos_new_old, cos_old_new = compute_metrics(transport_mat)
                row = {
                    "nblast_between": nblast_between,
                    "nblast_within": nblast_within,
                    "connectivity": connectivity,
                    "transport": transport,
                    "cos_new_new": cos_new_new,
                    "cos_new_old": cos_new_old,
                    "cos_old_new": cos_old_new,
                }
                metrics.append(row)

#%%


def summarize_match_ranks(match_mat):

    match_summary = pd.DataFrame(index=match_mat.index)
    for d1_id, row in match_mat.iterrows():
        sort_inds = np.argsort(-row)

        matches = row.index[sort_inds][:5]
        scores = row.values[sort_inds][:5]
        for i, (d2_id, score) in enumerate(zip(matches, scores)):
            if score == 0:
                break
            match_summary.loc[d1_id, f"{dataset2}_match_{i+1}"] = d2_id
            match_summary.loc[d1_id, f"{dataset2}_score_{i+1}"] = np.round(score, 2)
            match_summary.loc[d1_id, f"{dataset2}_type_{i+1}"] = mapper(d2_id)
    match_summary.columns.name = "match_info"
    return match_summary


summaries = []
opts_list = []
for options, match_mat in transport_mats.items():
    match_summary = summarize_match_ranks(match_mat)
    summaries.append(match_summary)
    opts_list.append(options)

full_match_summary = pd.concat(
    summaries,
    keys=opts_list,
    names=("nblast_between", "nblast_within", "connectivity"),
    axis=1,
)

full_match_summary = full_match_summary.reorder_levels([3, 0, 1, 2], axis=1).sort_index(
    axis=1
)

full_match_summary.to_csv("pcc/results/outputs/flywire_match/full_match_summary.csv")

#%%

full_match_summary.to_clipboard()

#%%

rows = []
for method_keys1, match_summary1 in full_match_summary["hemibrain_type_1"].T.iterrows():
    for method_keys2, match_summary2 in full_match_summary[
        "hemibrain_type_1"
    ].T.iterrows():

        ari = adjusted_rand_score(match_summary1, match_summary2)

        rows.append({"method1": method_keys1, "method2": method_keys2, "ari": ari})


#%%

ari_df = pd.DataFrame(rows)
ari_df = ari_df.pivot(index="method1", columns="method2", values="ari")

sns.set_theme("talk")


palette = dict(zip([True, False], ["green", "lightgrey"]))
all_colors = []
for i in range(3):
    index = pd.MultiIndex.from_tuples(ari_df.index)
    labels = index.get_level_values(i)
    colors = [palette[x] for x in labels]
    colors = pd.Series(
        colors, name=full_match_summary["hemibrain_type_1"].columns.names[i]
    )
    all_colors.append(colors)
all_colors = pd.concat(all_colors, axis=1)
all_colors.index = ari_df.index

cgrid = sns.clustermap(
    ari_df,
    row_colors=all_colors,
    col_colors=all_colors,
    cmap="RdBu_r",
    center=0,
    vmin=0,
    annot=True,
    xticklabels=False,
    yticklabels=False,
)
cgrid.ax_heatmap.set(xlabel="", ylabel="")
# cgrid.ax_row_colors.tick_params(axis="x", length=5, width=5, pad=0)

cgrid.ax_col_colors.tick_params(axis="y", length=0, width=0, pad=5)

cgrid.ax_cbar.set_ylabel("ARI", rotation=0, va="bottom", ha="left", fontsize="large")

gluefig("ari_heatmap", cgrid.fig)

#%%

metrics = pd.DataFrame(metrics)
metrics

#%%
from giskard.plot import upset_catplot

#%%
metrics = metrics.sort_values("cos_new_new", ascending=False)
sns.set_context("talk")
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
ax = axs[0]
uc = upset_catplot(
    metrics,
    x=["nblast_between", "nblast_within", "connectivity", 'transport'],
    y="cos_new_new",
    ax=ax,
    upset_size=30,
)
ax = axs[1]
uc = upset_catplot(
    metrics,
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="cos_new_old",
    ax=ax,
    upset_size=30,
)
uc.set_upset_ticklabels([])

ax = axs[2]
uc = upset_catplot(
    metrics,
    x=["nblast_between", "nblast_within", "connectivity", "transport"],
    y="cos_old_old",
    ax=ax,
    upset_size=30,
)
uc.set_upset_ticklabels([])

#%%
