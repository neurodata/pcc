#%%

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import DATA_PATH
from pkg.plot import SIDE_PALETTE, set_theme
import pandas as pd
from scipy.sparse import csr_array

from neuropull.graph import AdjacencyFrame


#%%
n = 100
nodes = pd.DataFrame(index=np.arange(n))
nodes["feature"] = np.random.uniform(0, 1, len(nodes))
matrix = csr_array(np.random.uniform(size=(n, n)))
index = nodes.index.copy()

index_map = pd.Series(index=index, data=np.arange(matrix.shape[0]))
columns_map = pd.Series(index=index, data=np.arange(matrix.shape[1]))

pandas_matrix = pd.DataFrame(matrix.toarray(), index=index, columns=index)

#%%

nodes = nodes.sort_values("feature", ascending=False)

#%%


new_index = nodes.index.copy()

# this is a map from new index (in new index sorted order) to old index
reordered_index_map = index_map.reindex(new_index).astype("Int64")
reordered_columns_map = columns_map.reindex(new_index).astype("Int64")
# so this is a map from old positional index to new positional index
old_to_new_index_pos_map = pd.Series(
    index=reordered_index_map.values, data=np.arange(len(reordered_index_map))
)
old_to_new_columns_pos_map = pd.Series(
    index=reordered_columns_map.values, data=np.arange(len(reordered_columns_map))
)

old_edge_row_indices, old_edge_col_indices = np.nonzero(matrix)

new_edge_row_indices = old_to_new_index_pos_map.loc[old_edge_row_indices].values
new_edge_col_indices = old_to_new_columns_pos_map.loc[old_edge_col_indices].values
edge_data = matrix[old_edge_row_indices, old_edge_col_indices]

new_matrix = csr_array((edge_data, (new_edge_row_indices, new_edge_col_indices)))

# # all the elements which existed in the old array
# valid_new_index_map = reordered_index_map[reordered_index_map.notna()]
# valid_new_row_positions = row_positions[reordered_index_map.notna()]
# valid_new_columns_map = reordered_columns_map[reordered_columns_map.notna()]
# valid_new_col_positions = col_positions[reordered_columns_map.notna()]

#%%
numpy_new_matrix = pandas_matrix.reindex(index=new_index, columns=new_index).values

#%%
numpy_new_matrix

#%%
new_matrix.toarray()

#%%
np.array_equal(new_matrix.toarray(), numpy_new_matrix)

#%%
from neuropull.graph import AdjacencyFrame

af = AdjacencyFrame(matrix)
int_index = [9, 8, 7, 4, 5, 2, 1]
new_af = af.reindex(index=int_index, columns=int_index)
#%%
print(new_af.data.toarray().max())
#%%
matrix[int_index][:, int_index].toarray()

#%%
new_af.data.toarray()

#%%
np.array_equal(matrix[int_index][:, int_index].toarray(), new_af.data.toarray())
