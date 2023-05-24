# %%
import pandas as pd
from pkg.data import DATA_PATH
from pathlib import Path
import numpy as np

data_path = DATA_PATH / "microns"

edges = pd.read_feather(data_path / "prf_soma_soma_connections_v661.feather")
edges.set_index("index", inplace=True)

identifier = "nucid"

sources = edges[f"pre_{identifier}"]
targets = edges[f"post_{identifier}"]
index = np.union1d(sources, targets)

index_to_iloc = dict(zip(index, range(len(index))))
source_ilocs = np.array(list(map(index_to_iloc.get, sources)))
target_ilocs = np.array(list(map(index_to_iloc.get, targets)))
weights = edges["sum_size"].values

from scipy.sparse import csr_array

sparse_adj = csr_array(
    (weights, (source_ilocs, target_ilocs)), shape=(len(index), len(index))
)


# %%
from graspologic.embed import AdjacencySpectralEmbed

ase = AdjacencySpectralEmbed(n_components=12)
X_out, X_in = ase.fit_transform(sparse_adj)


# %%
from statsmodels.multivariate.factor_rotation import rotate_factors

import time

currtime = time.time()
# L, T = rotate_factors(X_out, "varimax")
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# pip install cave client
import caveclient


# client.auth.setup_token()
# client.CAVEclinent()
# %%
# client.auth.save_token(token="86d87e6ff45b0df7fbce83b082421675", overwrite=True)

# %%
client = caveclient.CAVEclient("minnie65_public")

# %%

model_preds = client.materialize.query_table("aibs_soma_nuc_metamodel_preds_v117")

# %%

model_preds = client.materialize.query_table("aibs_soma_nuc_metamodel_preds_v117")
# %%
model_preds.set_index("target_id")

# %%
hand_labels = client.materialize.query_table("allen_v1_column_types_slanted_ref")
merged_nodes = (
    pd.concat(
        (hand_labels.set_index("target_id"), model_preds.set_index("target_id")), axis=0
    )
    .reset_index()
    .groupby("target_id")
    .first()
)

# index on target_id
# pre/post nuc_id


# aibs_soma_nuc_exc_mtype_preds_v117


# allen_v1_column_types_slanted_ref - labeled by hand
# trust these ahead of the ones in the above

# get_views

# synapse table
# neur

# %%
merged_nodes["cell_type"].value_counts()

# %%
merged_nodes = merged_nodes.reindex(index)
# %%
index = pd.Index(index)
valid_index = index.intersection(merged_nodes.index)

# %%
labels = merged_nodes["cell_type"].fillna("Unknown")
# %%
from giskard.plot import pairplot

pairplot(X_out, labels=labels, palette="tab20")

# %%
pairplot(X_in, labels=labels, palette="tab20")

# %%
from giskard.plot import simple_umap_scatterplot

simple_umap_scatterplot(X_in, labels=labels, palette="tab20", metric="cosine")
s
