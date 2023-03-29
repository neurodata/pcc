#%%
from typing import Literal, Optional, Union

import pandas as pd
from beartype import beartype

from neuropull.graph.base_frame import BaseFrame
from neuropull.graph.network_frame import BaseNetworkFrame


from scipy.sparse import csr_array

AxisType = Union[
    Literal[0], Literal[1], Literal["index"], Literal["columns"], Literal['both']
]

EdgeAxisType = Union[Literal['source'], Literal['target'], Literal['both']]

ColumnsType = Union[list, str]

NetworkFrameReturn = Union["NetworkFrame", None]


class NetworkFrame:
    @beartype
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        directed=True,
        sources=None,
        targets=None,
    ):
        # TODO checks ensuring that nodes and edges are valid.
        # should probably assume things like "source" and "target" columns
        # and that these elements are in the nodes dataframe
        # TODO are multigraphs allowed?
        # TODO assert that sources and targets and node index are all unique?
        self.nodes = nodes
        self.edges = edges
        if sources is None and targets is None:
            self.induced = True
            self._sources = None
            self._targets = None
        else:
            self.induced = False
            self._sources = sources
            self._targets = targets
        # TODO some checks on repeated edges if not directed
        self.directed = directed

    @property
    def sources(self):
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._sources, sort=False)
            # all_sources = self.edges["source"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_sources, sort=False)

    @property
    def targets(self):
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._targets, sort=False)
            # all_targets = self.edges["target"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_targets, sort=False)

    @property
    def source_nodes(self):
        return self.nodes.loc[self.sources]

    @property
    def target_nodes(self):
        return self.nodes.loc[self.targets]

    def __repr__(self) -> str:
        out = f"NetworkFrame(nodes={self.nodes.shape}, edges={self.edges.shape}, "
        out += f"induced={self.induced}, directed={self.directed})"
        return out

    def reindex_nodes(self, index: pd.Index) -> "NetworkFrame":
        nodes = self.nodes.reindex(index=index, axis=0)
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        return NetworkFrame(nodes, edges, directed=self.directed)

    # def reindex_edges(self, index: pd.Index, axis: AxisType = 0) -> "NetworkFrame":
    #     if axis != 'both':
    #         self.edges = self.edges.reindex(index=index, axis=axis)
    #     else:
    #         self.edges = self.edges.reindex(index=index, axis=0)
    #         self.edges = self.edges.reindex(index=index, axis=1)
    #     return self

    # def reindex_like(self, other: "NetworkFrame") -> "NetworkFrame":
    #     self.reindex_nodes(other.nodes.index, axis='both')
    #     self.reindex_edges(other.edges.index, axis='both')
    #     return self

    def query_nodes(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        nodes = self.nodes.query(query)
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def query_edges(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        edges = self.edges.query(query)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def remove_unused_nodes(self, inplace=False) -> Optional["NetworkFrame"]:
        index = self.nodes.index
        new_index = index.intersection(
            self.edges.source.append(self.edges.target).unique()
        )
        nodes = self.nodes.loc[new_index]
        if inplace:
            self.nodes = nodes
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def apply_node_features(
        self, columns: ColumnsType, axis: EdgeAxisType = 'both', inplace=False
    ) -> Optional["NetworkFrame"]:
        if not inplace:
            edges = self.edges.copy()
        else:
            edges = self.edges
        if isinstance(columns, str):
            columns = [columns]
        if axis.isin(['source', 'both']):
            for col in columns:
                edges[f'source_{col}'] = self.edges['source'].map(self.nodes[col])
        if axis.isin(['target', 'both']):
            for col in columns:
                edges[f'target_{col}'] = self.edges['target'].map(self.nodes[col])
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def to_adjacency(self, weight_col: str = 'weight', aggfunc='sum') -> pd.DataFrame:
        # TODO: wondering if the sparse method of doing this would actually be faster
        # here too...
        adj_df = self.edges.pivot_table(
            index='source',
            columns='target',
            values=weight_col,
            fill_value=0,
            aggfunc=aggfunc,
            sort=False,
        )
        adj_df = adj_df.reindex(
            index=self.sources,
            columns=self.targets,
            fill_value=0,
        )
        adj_df.index = adj_df.index.set_names('source')
        adj_df.columns = adj_df.columns.set_names('target')
        return adj_df

    def to_networkx(self):
        import networkx as nx

        if self.directed:
            create_using = nx.MultiDiGraph
        else:
            create_using = nx.MultiGraph

        g = nx.from_pandas_edgelist(
            self.edges,
            source='source',
            target='target',
            edge_attr=True,
            create_using=create_using,
        )
        nx.set_node_attributes(g, self.nodes.to_dict(orient='index'))
        return g

    def to_sparse_adjacency(
        self, weight_col: str = 'weight', aggfunc='sum'
    ) -> csr_array:
        edges = self.edges
        # TODO only necessary since there might be duplicate edges
        # might be more efficient to have a attributed checking this, e.g. set whether
        # this is a multigraph or not
        effective_edges = edges.groupby(['source', 'target'])[weight_col].agg(aggfunc)

        data = effective_edges.values
        source_indices = effective_edges.index.get_level_values('source')
        target_indices = effective_edges.index.get_level_values('target')

        source_indices = pd.Categorical(source_indices, categories=self.sources)
        target_indices = pd.Categorical(target_indices, categories=self.targets)

        adj = csr_array(
            (data, (source_indices.codes, target_indices.codes)),
            shape=(len(self.sources), len(self.targets)),
        )
        return adj

    def groupby_nodes(self, by=None, axis='both', **kwargs):
        """Group the frame by data in the row or column (or both) metadata.

        Parameters
        ----------
        by : _type_, optional
            _description_, by default None
        axis : str, optional
            _description_, by default 'both'

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if axis == 0:
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
        elif axis == 1:
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        elif axis == 'both':
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")

        return NodeGroupBy(self, source_nodes_groupby, target_nodes_groupby)

    @property
    def loc(self):
        return LocIndexer(self)


class NodeGroupBy:
    """A class for grouping a NetworkFrame by a set of labels."""

    def __init__(self, frame, source_groupby, target_groupby):
        self._frame = frame
        self._source_groupby = source_groupby
        self._target_groupby = target_groupby

        if source_groupby is None:
            self._axis = 1
        elif target_groupby is None:
            self._axis = 0
        else:
            self._axis = 'both'

        if self.has_source_groups:
            self.source_group_names = list(source_groupby.groups.keys())
        if self.has_target_groups:
            self.target_group_names = list(target_groupby.groups.keys())

    @property
    def has_source_groups(self):
        """Whether the frame has row groups."""
        return self._source_groupby is not None

    @property
    def has_target_groups(self):
        """Whether the frame has column groups."""
        return self._target_groupby is not None

    def __iter__(self):
        """Iterate over the groups."""
        if self._axis == 'both':
            for source_group, source_objects in self._source_groupby:
                for target_group, target_objects in self._target_groupby:
                    yield (source_group, target_group), self._frame.loc[
                        source_objects.index, target_objects.index
                    ]
        elif self._axis == 0:
            for source_group, source_objects in self._source_groupby:
                yield source_group, self._frame.loc[source_objects.index]
        elif self._axis == 1:
            for target_group, target_objects in self._target_groupby:
                yield target_group, self._frame.loc[:, target_objects.index]

    # def apply(self, func, *args, data=False, **kwargs):
    #     """Apply a function to each group."""
    #     if self._axis == 'both':
    #         answer = pd.DataFrame(
    #             index=self.source_group_names, columns=self.target_group_names
    #         )

    #     else:
    #         if self._axis == 0:
    #             answer = pd.Series(index=self.source_group_names)
    #         else:
    #             answer = pd.Series(index=self.target_group_names)
    #     for group, frame in self:
    #         if data:
    #             value = func(frame.data, *args, **kwargs)
    #         else:
    #             value = func(frame, *args, **kwargs)
    #         answer.at[group] = value
    #     return answer

    @property
    def source_groups(self):
        """Return the row groups."""
        if self._axis == 'both' or self._axis == 0:
            return self._source_groupby.groups
        else:
            raise ValueError('No source groups, groupby was on targets only')

    @property
    def target_groups(self):
        """Return the column groups."""
        if self._axis == 'both' or self._axis == 1:
            return self._target_groupby.groups
        else:
            raise ValueError('No target groups, groupby was on sources only')


class LocIndexer:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, args):
        if isinstance(args, tuple):
            if len(args) != 2:
                raise ValueError("Must provide at most two indexes.")
            else:
                row_index, col_index = args
        else:
            raise NotImplementedError()

        if isinstance(row_index, int):
            row_index = [row_index]
        if isinstance(col_index, int):
            col_index = [col_index]

        if isinstance(row_index, slice):
            row_index = self._frame.nodes.index[row_index]
        if isinstance(col_index, slice):
            col_index = self._frame.nodes.index[col_index]

        row_index = pd.Index(row_index)
        col_index = pd.Index(col_index)

        source_nodes = self._frame.nodes.loc[row_index]
        target_nodes = self._frame.nodes.loc[col_index]

        edges = self._frame.edges.query(
            "source in @source_nodes.index and target in @target_nodes.index"
        )

        if row_index.equals(col_index):
            nodes = source_nodes
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
            )
        else:
            nodes = pd.concat([source_nodes, target_nodes], copy=False, sort=False)
            nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
                sources=row_index,
                targets=col_index,
            )


edges = pd.DataFrame()
edges['source'] = [1, 2, 3, 4, 5, 2, 3, 4]
edges['target'] = [2, 3, 3, 5, 1, 3, 1, 3]
edges['weight'] = [1, 2, 1, 2, 1, 1, 2, 2]

nodes = pd.DataFrame()
nodes['node_class'] = ['a', 'b', 'b', 'b', 'a']
nodes['node_feature'] = [0.5, 0.4, 0.2, 0.1, 0.3]
nodes.index = [1, 2, 3, 4, 5]

nf = NetworkFrame(nodes, edges)
print(nf.to_adjacency())

nf.nodes = nf.nodes.sort_values('node_class')

print(nf.to_adjacency())

nf = nf.query_nodes('node_feature >= 0.2')
print(nf.to_adjacency())

for group, frame in nf.groupby_nodes('node_class'):
    print(frame.to_adjacency())

print(nf.to_sparse_adjacency().toarray())

#%%
from pathlib import Path
import numpy as np

data_dir = Path("neuropull/data/flywire/526")

nodes = pd.read_csv(data_dir / "nodes.csv.gz", index_col=0)
nodes.rename(columns={'class': 'cell_class'}, inplace=True)
edges = pd.read_csv(data_dir / "edgelist.csv.gz", header=None)
edges.columns = ['source', 'target', 'weight']


#%%
nf = NetworkFrame(nodes, edges)

#%%

nf.query_nodes("side.isin(['left', 'right'])", inplace=True)
# nf.query_nodes("cell_class != 'optic'", inplace=True)
nf.query_nodes("cell_class.notna()", inplace=True)
nf.to_sparse_adjacency()

# %%
import time 

# for group, frame in nf.groupby_nodes('cell_class'):

