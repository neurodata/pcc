# %%
import pandas as pd
from pkg.data import DATA_PATH

data_path = DATA_PATH / "flywire-model-missing"

filename = "flywire_adj_coo_extrapolated_630_71820_86184.feather"

edges = pd.read_feather(data_path / filename)

# %%
group_sizes = edges.groupby("syn_ix").size()
# %%
counts = edges.groupby("syn_ix").size().value_counts()

# %%
sns.barplot(data=counts.reset_index(), x="index", y="count")
# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.set_context("talk")
sns.histplot(x=group_sizes, ax=ax, discrete=True, binwidth=1, element="poly")
ax.set_xlabel("Number of potential postsynaptic partners")
