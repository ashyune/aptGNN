"""
smoke_test_hetero.py

Sanity check for hetero_aptgnn_model.py -- confirms the edge-type-aware
HeteroConv stack runs end to end, BEFORE wiring it into the real
graph_construction.py output or writing a training loop.

Builds a small synthetic HeteroData graph with node/edge types modeled on
DARPA TC CADETS provenance graphs (process, file, socket nodes; read/write/
execute/connect edges). Swap these type names for your actual
EDGE_TYPE_MAP from graph_construction.py once you're ready to test on real
windowed data.

Usage:
    python smoke_test_hetero.py
"""

import torch
from torch_geometric.data import HeteroData

from hetero_aptgnn_model import HeteroAPTGNN


def build_dummy_graph():
    data = HeteroData()

    # Node features: (num_nodes, feature_dim) -- feature_dim mimics
    # ThreaTrace-style histogram features, dims don't need to match across
    # node types.
    data["process"].x = torch.randn(10, 16)
    data["file"].x = torch.randn(15, 12)
    data["socket"].x = torch.randn(5, 8)

    # node_id: used later to map anomalous predictions back to real entity
    # ids. Here just an identity range per type.
    data["process"].node_id = torch.arange(10)
    data["file"].node_id = torch.arange(15)
    data["socket"].node_id = torch.arange(5)

    # Edges: (src_type, relation, dst_type) -> edge_index [2, num_edges]
    data["process", "reads", "file"].edge_index = torch.randint(0, 10, (2, 20))
    data["process", "reads", "file"].edge_index[1] = torch.randint(0, 15, (20,))

    data["process", "writes", "file"].edge_index = torch.stack(
        [torch.randint(0, 10, (18,)), torch.randint(0, 15, (18,))]
    )

    data["process", "connects", "socket"].edge_index = torch.stack(
        [torch.randint(0, 10, (7,)), torch.randint(0, 5, (7,))]
    )

    data["process", "executes", "process"].edge_index = torch.stack(
        [torch.randint(0, 10, (6,)), torch.randint(0, 10, (6,))]
    )

    return data


def main():
    data = build_dummy_graph()
    print(data)

    node_types = data.node_types
    edge_types = data.edge_types
    in_channels_dict = {nt: data[nt].x.size(-1) for nt in node_types}

    model = HeteroAPTGNN(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict=in_channels_dict,
        hidden_channels=32,
        num_layers=2,
    )

    x_dict = {nt: data[nt].x for nt in node_types}
    edge_index_dict = {et: data[et].edge_index for et in edge_types}

    out_dict = model(x_dict, edge_index_dict)

    print("\nForward pass succeeded. Output logits per node type:")
    for nt, out in out_dict.items():
        expected_shape = (data[nt].x.size(0), len(node_types))
        status = "OK" if tuple(out.shape) == expected_shape else "SHAPE MISMATCH"
        print(f"  {nt}: {tuple(out.shape)}  (expected {expected_shape})  [{status}]")


if __name__ == "__main__":
    main()