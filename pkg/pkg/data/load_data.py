import json
from pathlib import Path
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.graph import MaggotGraph
from giskard.utils import to_pandas_edgelist
from sklearn.utils import Bunch
from neuropull.graph import NetworkFrame
import polars as pl

version_loc = Path(__file__).parent / "version.txt"
with open(version_loc) as f:
    version = f.readline()

processed_loc = Path(__file__).parent / "processed_version.txt"
with open(processed_loc) as f:
    processed_version = f.readline()

DATA_VERSION = version
DATA_PATH = Path(__file__).parent.parent.parent.parent  # don't judge me judge judy
DATA_PATH = DATA_PATH / "data"


def _get_folder(path, version):
    if path is None:
        path = DATA_PATH
    if version is None:
        version = DATA_VERSION
    folder = path / version
    return folder


def load_node_meta(path=None, version=None):
    folder = _get_folder(path, version)
    meta = pd.read_csv(folder / "meta_data.csv", index_col=0)
    meta.sort_index(inplace=True)
    return meta


def load_edgelist(graph_type="G", path=None, version=None):
    folder = _get_folder(path, version)
    edgelist = pd.read_csv(
        folder / f"{graph_type}_edgelist.txt",
        delimiter=" ",
        header=None,
        names=["source", "target", "weight"],
    )
    return edgelist


def load_networkx(graph_type="G", node_meta=None, path=None, version=None):
    edgelist = load_edgelist(graph_type, path=path, version=version)
    g = nx.from_pandas_edgelist(edgelist, edge_attr="weight", create_using=nx.DiGraph())
    if node_meta is not None:
        meta_data_dict = node_meta.to_dict(orient="index")
        nx.set_node_attributes(g, meta_data_dict)
    return g


def load_adjacency(
    graph_type="G", nodelist=None, output="numpy", path=None, version=None
):
    g = load_networkx(graph_type=graph_type, path=path, version=version)
    if output == "numpy":
        adj = nx.to_numpy_array(g, nodelist=nodelist)
    elif output == "pandas":
        adj = nx.to_pandas_adjacency(g, nodelist=nodelist)
    return adj


def load_palette(path=None, version=None):
    folder = _get_folder(path, version)
    with open(folder / "simple_color_map.json", "r") as f:
        palette = json.load(f)
    return palette


def load_unmatched(side="left", weights=False):
    side = side.lower()
    dir = DATA_PATH / processed_version
    if weights:
        data = (("weight", int),)
    else:
        data = False
    g = nx.read_edgelist(
        dir / f"unmatched_{side}_edgelist.csv",
        create_using=nx.DiGraph,
        delimiter=",",
        nodetype=int,
        data=data,
    )
    nodes = pd.read_csv(dir / f"unmatched_{side}_nodes.csv", index_col=0)
    adj = nx.to_numpy_array(g, nodelist=nodes.index)
    return adj, nodes


def load_matched(side="left"):
    side = side.lower()
    dir = DATA_PATH / processed_version
    g = nx.read_edgelist(
        dir / f"matched_{side}_edgelist.csv",
        create_using=nx.DiGraph,
        delimiter=",",
        nodetype=int,
    )
    nodes = pd.read_csv(dir / f"matched_{side}_nodes.csv", index_col=0)
    adj = nx.to_numpy_array(g, nodelist=nodes.index)
    return adj, nodes


def load_maggot_graph(path=None, version=None):
    nodes = load_node_meta()
    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes.index)
    nx.set_node_attributes(g, nodes.to_dict(orient="index"))
    graph_types = ["Gaa", "Gad", "Gda", "Gdd"]
    for graph_type in graph_types:
        g_type = load_networkx(graph_type=graph_type)
        for u, v, data in g_type.edges(data=True):
            g.add_edge(u, v, key=graph_type[1:], edge_type=graph_type[1:], **data)

    g_type = load_networkx(graph_type="G")
    for u, v, data in g_type.edges(data=True):
        g.add_edge(u, v, key="sum", edge_type="sum", **data)

    edges = to_pandas_edgelist(g)

    return MaggotGraph(g, nodes, edges)


def load_network_palette():
    colors = sns.color_palette("Set2")
    palette = dict(zip(["Left", "Right"], [colors[0], colors[1]]))
    return palette, "Side"


def load_node_palette(key="simple_group"):
    if key == "merge_class":
        from src.visualization import CLASS_COLOR_DICT

        return CLASS_COLOR_DICT, "merge_class"
    elif key == "simple_group":
        palette = load_palette()
        return palette, "simple_group"


def load_navis_neurons(ids=None, path=None, version=None):
    folder = _get_folder(path, version)
    with open(folder / "neurons.pickle", "rb") as f:
        neuron_list = pickle.load(f)
    return neuron_list


# def load_networkx(graph_type, base_path=None, version=DATA_VERSION):
#     if base_path is None:
#         base_path = DATA_PATH
#     data_path = Path(base_path)
#     data_path = data_path / version
#     file_path = data_path / (graph_type + ".graphml")
#     graph = nx.read_graphml(file_path, node_type=str, edge_key_type="str")
#     return graph


def load_data(graph_type, base_path=None, version=None):
    # TODO deprecate this and pull out of old scripts
    if base_path is None:
        base_path = DATA_PATH
    if version is None:
        version = DATA_VERSION

    data_path = Path(base_path)
    data_path = data_path / version

    edgelist_path = data_path / (graph_type + ".edgelist")
    meta_path = data_path / "meta_data.csv"

    graph = nx.read_edgelist(
        edgelist_path, create_using=nx.DiGraph, nodetype=int, data=[("weight", int)]
    )
    meta = pd.read_csv(meta_path, index_col=0)
    adj = nx.to_numpy_array(graph, nodelist=meta.index.values, dtype=float)
    missing_nodes = np.setdiff1d(meta.index, list(graph.nodes()))
    for node in missing_nodes:
        graph.add_node(node)

    return Bunch(graph=graph, adj=adj, meta=meta)


def load_flywire_networkframe():
    data_dir = DATA_PATH / "hackathon"

    dataset = "fafb_flywire"
    dataset_dir = data_dir / dataset

    nodes = pd.read_csv(dataset_dir / f"{dataset}_meta.csv", low_memory=False)
    nodes.drop("row_id", axis=1, inplace=True)
    nodes.rename(columns={"root_id": "node_id"}, inplace=True)

    # NOTE:
    # some nodes have multiple rows in the table
    # strategy here is to keep the first row that has a hemibrain type, though that
    # could be changed
    node_counts = nodes.value_counts("node_id")  # noqa: F841
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

    nodes["cell_type_filled"] = nodes["cell_type"].fillna(nodes["hemibrain_type"])

    nodes.set_index("node_id", inplace=True)

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

    # NOTE: there are some edges that reference nodes that are not in the node table
    referenced_node_ids = np.union1d(edges["source"].unique(), edges["target"].unique())
    isin_node_table = np.isin(referenced_node_ids, nodes.index)
    missing_node_ids = referenced_node_ids[~isin_node_table]  # noqa: F841

    edges.query(
        "~((source in @missing_node_ids) or (target in @missing_node_ids))",
        inplace=True,
    )

    flywire = NetworkFrame(nodes.copy(), edges.copy())
    return flywire


def load_flywire_nblast_subset(queries=None):
    data_dir = DATA_PATH / "hackathon"

    nblast = pl.scan_ipc(
        data_dir / "nblast" / "nblast_flywire_all_right_aba_comp.feather",
        memory_map=False,
    )
    index = pd.Index(nblast.select("index").collect().to_pandas()["index"])
    columns = pd.Index(nblast.columns[1:])
    index_ids = index.str.split(",", expand=True).get_level_values(0).astype(int)
    column_ids = columns.str.split(",", expand=True).get_level_values(0).astype(int)
    index_ids_map = dict(zip(index_ids, index))
    column_ids_map = dict(zip(column_ids, columns))
    index_ids_reverse_map = dict(zip(index, index_ids))
    column_ids_reverse_map = dict(zip(columns, column_ids))

    if queries is None:
        query_node_ids = index
    elif not isinstance(queries, tuple):
        query_node_ids = queries
    else:
        query_node_ids = np.concatenate(queries)

    query_index = pd.Series([index_ids_map[i] for i in query_node_ids])
    query_columns = pd.Series(["index"] + [column_ids_map[i] for i in query_node_ids])

    nblast = nblast.with_columns(
        pl.col("index").is_in(query_index).alias("select_index")
    )

    mini_nblast = (
        nblast.filter(pl.col("select_index"))
        .select(query_columns)
        .collect()
        .to_pandas()
    ).set_index("index")

    mini_nblast.index = mini_nblast.index.map(index_ids_reverse_map)
    mini_nblast.columns = mini_nblast.columns.map(column_ids_reverse_map)

    mini_nblast = mini_nblast.loc[query_node_ids, query_node_ids]

    return mini_nblast
