#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import DATA_PATH

#%%

data_dir = DATA_PATH / "hackathon"

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


# %% [markdown]

# ## Male CNS

#%%

dataset = "male_cns"
dataset_dir = data_dir / dataset

#%%
nodes = pd.read_csv(dataset_dir / f"{dataset}_meta.csv", index_col=0)
nodes.rename(
    columns={"somaSide": "side", "class": "super_class", "bodyId": "node_id"},
    inplace=True,
)
nodes["side"] = nodes["side"].map({"L": "left", "R": "right", "C": "center"})


def super_class_renamer(name):
    if name == "Ascending Interneuron":
        return "ascending"
    if name == "Descending":
        return "descending"
    else:
        return name


nodes["super_class"] = nodes["super_class"].map(super_class_renamer)

nodes.set_index("node_id", inplace=True)

nodes.head()

# %%

edges = pd.read_csv(dataset_dir / f"{dataset}_edges.csv", index_col=0)
edges.rename(
    columns={"bodyId_pre": "source", "bodyId_post": "target", "roi": "region"},
    inplace=True,
)
edges.head()

#%%

referenced_node_ids = np.union1d(edges["source"].unique(), edges["target"].unique())
isin_node_table = np.isin(referenced_node_ids, nodes.index)
missing_node_ids = referenced_node_ids[~isin_node_table]
print("Number of missing nodes: ", len(missing_node_ids))

#%%
# ! ! !
# NOTE: past this point you'll need to have neuropull and giskard installed... both of
# which are not "real" packages yet... just code I have on Github
# ! ! !

#%%
from neuropull.graph import NetworkFrame

flywire = NetworkFrame(nodes, edges)

malecns = NetworkFrame(nodes, edges)

#%%

print("FAFB Flywire:")
print("Number of nodes: ", len(flywire.nodes))
print("Number of edges: ", len(flywire.edges))

print()

print("Male CNS:")
print("Number of nodes: ", len(malecns.nodes))
print("Number of edges: ", len(malecns.edges))

#%%

flywire.nodes["dataset"] = "fafb_flywire"
malecns.nodes["dataset"] = "male_cns"
joint_nodes = pd.concat([flywire.nodes, malecns.nodes], axis=0)

#%%

from giskard.plot import rotate_labels

counts = joint_nodes.groupby(["dataset", "super_class"]).size()
counts.name = "count"
counts = counts.reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=counts, x="super_class", y="count", hue="dataset")
rotate_labels(ax)


#%%
flywire.largest_connected_component(inplace=True, verbose=True)
malecns.largest_connected_component(inplace=True, verbose=True)


# %%

from graspologic.embed import LaplacianSpectralEmbed

n_components = 8
lse = LaplacianSpectralEmbed(n_components=n_components, form="R-DAD", concat=True)
flywire_adj = flywire.to_sparse_adjacency()
flywire_X_lse = lse.fit_transform(flywire_adj)

lse = LaplacianSpectralEmbed(n_components=n_components, form="R-DAD", concat=True)
malecns_adj = malecns.to_sparse_adjacency()
malecns_X_lse = lse.fit_transform(malecns_adj)

# %%

from giskard.plot import pairplot

palette = dict(
    zip(flywire.nodes["super_class"].value_counts().index, sns.color_palette("tab20"))
)

pairplot(
    flywire_X_lse[:, :8],
    labels=flywire.nodes["super_class"],
    subsample=0.1,
    palette=palette,
    alpha=0.1,
)

#%%

flywire_X_lse = pd.DataFrame(index=flywire.nodes.index, data=flywire_X_lse)
malecns_X_lse = pd.DataFrame(index=malecns.nodes.index, data=malecns_X_lse)

#%%

# %%
singletons = pd.read_csv(data_dir / "singletons" / "singletons-2023-04-17.csv")
singletons.rename(
    columns={
        "root_left": "flywire_left",
        "root_right": "flywire_right",
        "match_left": "malecns_left",
        "match_right": "malecns_right",
        "Added to Clio": "added",
    },
    inplace=True,
)

good_singletons = singletons.query("added.isin(['yes', 'already in Clio'])").copy()
good_singletons["malecns_left"] = good_singletons["malecns_left"].astype(int)
good_singletons["malecns_right"] = good_singletons["malecns_right"].astype(int)
left_matches = good_singletons[["flywire_left", "malecns_left"]]
right_matches = good_singletons[["flywire_right", "malecns_right"]]

left_counts = left_matches["flywire_left"].value_counts()
right_counts = right_matches["flywire_right"].value_counts()
good_left_flywire = left_counts[(left_counts == 1)].index
good_right_flywire = right_counts[(right_counts == 1)].index

left_counts = left_matches["malecns_left"].value_counts()
right_counts = right_matches["malecns_right"].value_counts()
good_left_malecns = left_counts[(left_counts == 1)].index
good_right_malecns = right_counts[(right_counts == 1)].index

is_valid_match = (
    left_matches["flywire_left"].isin(flywire.nodes.index)
    & left_matches["malecns_left"].isin(malecns.nodes.index)
    & right_matches["flywire_right"].isin(flywire.nodes.index)
    & right_matches["malecns_right"].isin(malecns.nodes.index)
    & left_matches["flywire_left"].isin(good_left_flywire)
    & right_matches["flywire_right"].isin(good_right_flywire)
    & left_matches["malecns_left"].isin(good_left_malecns)
    & right_matches["malecns_right"].isin(good_right_malecns)
)

left_matches = left_matches[is_valid_match]
right_matches = right_matches[is_valid_match]
len(left_matches)

#%%
matches = pd.concat([left_matches, right_matches], axis=1)

frac = 0.8
train_matches = matches.sample(frac=frac)
test_matches = matches.drop(train_matches.index)

#%%

train_flywire_ids = np.concatenate(
    (train_matches["flywire_left"], train_matches["flywire_right"])
)
train_malecns_ids = np.concatenate(
    (train_matches["malecns_left"], train_matches["malecns_right"])
)
test_flywire_ids = np.concatenate(
    (test_matches["flywire_left"], test_matches["flywire_right"])
)
test_malecns_ids = np.concatenate(
    (test_matches["malecns_left"], test_matches["malecns_right"])
)

flywire_template = flywire_X_lse.loc[train_flywire_ids]
malecns_template = malecns_X_lse.loc[train_malecns_ids]

#%%
from sklearn.metrics import pairwise_distances

metric = "cosine"

flywire_Y = pairwise_distances(flywire_X_lse, flywire_template, metric=metric)
flywire_Y = pd.DataFrame(index=flywire_X_lse.index, data=flywire_Y)

malecns_Y = pairwise_distances(malecns_X_lse, malecns_template, metric=metric)
malecns_Y = pd.DataFrame(index=malecns_X_lse.index, data=malecns_Y)

#%%

joint_Y = pd.concat((flywire_Y, malecns_Y), axis=0)


from sklearn.decomposition import PCA

pca = PCA(n_components=64)
low_d_joint_Y = pca.fit_transform(joint_Y)

#%%

from sklearn.neighbors import NearestNeighbors

metric = "cosine"
joint_Y_nn = NearestNeighbors(n_neighbors=1000, metric="cosine")
joint_Y_nn.fit(joint_Y)

test_joint_Y = joint_Y.loc[test_matches.iloc[0]]

indices = joint_Y_nn.kneighbors(test_joint_Y, return_distance=False)

joint_Y.index[indices]


#%%

from sklearn.neighbors import NearestNeighbors

metric = "cosine"
flywire_nn = NearestNeighbors(n_neighbors=1, metric="cosine")
