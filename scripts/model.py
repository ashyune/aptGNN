import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

MEM_DIM = 64        # size of the global cross-window memory vector
NUM_BASES = 8        # basis-decomposition rank for RGCNConv's per-relation
                      # weights. CADETS has 28 relation types with heavy
                      # imbalance (EVENT_READ alone is ~33% of all edges,
                      # several types are <1%) -- giving every relation its
                      # own full weight matrix starves the sparse ones of
                      # gradient signal and risks overfitting the common
                      # ones. num_bases shares a small set of basis matrices
                      # across all relations, each relation just learns a
                      # coefficient vector over them. Tune this once you see
                      # validation behavior; start here, not at num_bases=None
                      # (which means "no decomposition, full per-relation
                      # matrices").


class SAGEMemNet(torch.nn.Module):
    """Two-layer relation-aware GraphSAGE-style classifier + a persistent,
    GRU-updated GLOBAL memory vector carried across time windows.

    Message passing is now relation-aware (RGCNConv instead of SAGEConv):
    a Process->File edge and a Process->Socket edge go through different
    per-relation weights instead of being collapsed into one shared
    transformation. See edge_type in data_process_train.py / 
    data_process_test.py -- this is the consumer of that tensor.

    Why global rather than per-node memory: this vector is meant to
    transfer between the training scene graph and the separate test scene
    graph (validate()), which don't share a node-id space. A per-node
    memory bank couldn't cross that boundary; a single "what has the whole
    system looked like so far" vector can. This part is UNCHANGED by the
    heterogeneous message passing -- see the architecture doc, Task 4.4:
    the readout pools over post-conv1 embeddings regardless of which
    relation produced them, so it stays relation-independent and mem_dim
    doesn't need to grow.

    The memory update (readout + GRU) happens INSIDE forward(), before
    broadcasting back into node features for classification -- so the
    ordinary classification loss is what trains the GRU/readout weights.
    Only the carried-over `state` tensor gets detached by the caller
    between calls (truncated BPTT of depth 1), which is what keeps the
    backward graph bounded no matter how many windows/batches a full
    training run covers.

    forward() accepts a NeighborLoader batch's x/edge_index/edge_type plus
    the incoming memory state, and returns (log_probs, updated_state). The
    caller slices out[:batch.batch_size] before loss/accuracy/threshold
    logic, same convention as before.

    NOTE: this class is imported by BOTH train_darpatc.py and
    test_darpatc.py so checkpoints saved during training always load
    correctly at test time. Previously test_darpatc.py carried its own
    hand-copied duplicate of this class -- that's gone now; edit only
    here.
    """
    def __init__(self, in_channels, out_channels, num_relations,
                 mem_dim=MEM_DIM, num_bases=NUM_BASES):
        super().__init__()
        self.mem_dim = mem_dim
        num_bases = min(num_bases, num_relations)  # can't exceed num_relations
        self.conv1 = RGCNConv(in_channels, 32, num_relations, num_bases=num_bases)
        self.readout = torch.nn.Linear(32, mem_dim)
        self.readout_norm = torch.nn.LayerNorm(mem_dim)   # bounds the GRU's input
        self.gru = torch.nn.GRUCell(mem_dim, mem_dim)
        self.ctx_proj = torch.nn.Linear(mem_dim, 32)
        # Learnable scalar gate (init near 0) so the context term starts as
        # a no-op and the model only leans on memory once it's earned a
        # role in the loss, instead of immediately swamping h with an
        # untrained, potentially large ctx vector from step 1.
        self.ctx_gate = torch.nn.Parameter(torch.tensor(-2.0))
        self.conv2 = RGCNConv(32, out_channels, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type, prev_state):
        h = F.relu(self.conv1(x, edge_index, edge_type))                  # [N, 32]

        g = self.readout_norm(self.readout(h).mean(dim=0, keepdim=True))  # [1, mem_dim], bounded
        new_state = torch.tanh(self.gru(g, prev_state))                  # keep state in [-1, 1]

        ctx = self.ctx_proj(new_state).expand(h.size(0), -1)              # broadcast to every node
        gate = torch.sigmoid(self.ctx_gate)                               # starts ~0.12, learns upward
        h = F.dropout(h + gate * ctx, p=0.5, training=self.training)

        out = self.conv2(h, edge_index, edge_type)
        return F.log_softmax(out, dim=1), new_state

    def init_state(self, device):
        return torch.zeros(1, self.mem_dim, device=device)