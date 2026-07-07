"""
hetero_aptgnn_model.py

APTGNN model definition -- Phase A: edge-type-aware convs only.
(Cross-window GRU memory deliberately deferred.)

Novelty over ThreaTrace baseline:
  1. Edge-type-aware message passing: a separate SAGEConv weight matrix per
     edge type via HeteroConv, instead of collapsing all relations into a
     single relation type the way ThreaTrace's GraphSAGE does.
  2. Operates natively on PyG HeteroData (typed nodes + typed edges).

Self-supervised classification target (adapted from ThreaTrace):
  ThreaTrace has each node predict its own coarse entity type (process/file/
  socket/...). That doesn't transfer to HeteroData: PyG segregates nodes into
  per-type stores by construction, so "predict your own type" becomes
  trivially 100% accurate and the anomaly signal disappears.

  Instead, each node predicts a finer-grained BEHAVIORAL TAG: entity-type
  nodes are clustered (via KMeans over their ThreaTrace-style edge-count
  histogram features, see graph_construction.py) into k sub-types per node
  type. The classifier head therefore needs a DIFFERENT output dimension per
  node type (num_classes_dict[node_type] = k for that type), not one shared
  dimension across all types.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroAPTGNN(nn.Module):
    def __init__(
        self,
        node_types,
        edge_types,
        in_channels_dict,
        num_classes_dict,
        hidden_channels=32,
        num_layers=2,
    ):
        """
        node_types: list[str]                e.g. ["SUBJECT_PROCESS", "FILE_OBJECT_FILE", ...]
        edge_types: list[tuple(str,str,str)]  PyG-style (src_type, relation, dst_type)
        in_channels_dict: dict[str, int]      feature dim per node type
                                               (2 * num_edge_types, ThreaTrace-style
                                               in/out edge-count histogram)
        num_classes_dict: dict[str, int]      NUMBER OF BEHAVIORAL CLUSTERS (k) per
                                               node type -- NOT shared across types.
                                               Comes from graph_construction.py's
                                               per-type KMeans fit.
        """
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels

        self.input_proj = nn.ModuleDict(
            {nt: Linear(in_channels_dict[nt], hidden_channels) for nt in node_types}
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {et: SAGEConv((-1, -1), hidden_channels) for et in edge_types},
                aggr="mean",
            )
            self.convs.append(conv)

        # Separate classifier head per node type, each with its OWN output
        # dimension (that type's cluster count k) -- this is the key fix.
        self.classifiers = nn.ModuleDict(
            {nt: Linear(hidden_channels, num_classes_dict[nt]) for nt in node_types}
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {nt: self.input_proj[nt](x).relu() for nt, x in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}

        out_dict = {nt: self.classifiers[nt](x) for nt, x in x_dict.items()}
        return out_dict