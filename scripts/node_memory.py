"""Cross-window persistent node memory with exponential decay.

Extension 1 (Gap 1 -- persistent node state in the DTDG paradigm).

ThreaTrace's original pipeline (data_process_train.py / data_process_test.py)
treats an entire provenance file as one static graph. Stage 2
(windowing.py / windowed_data.py) breaks a file into a chronological
sequence of independent per-window Data objects, but nothing carries a
node's state from one window into the next -- each window is processed as
if it were the only graph that ever existed. NodeMemory adds that missing
piece: one embedding per node, keyed by Stage 1's global_id, that persists
across the whole window sequence and is decayed according to how long
it's been since the node was last active.

Why decay lives here and nowhere else
--------------------------------------
Decay is applied ONLY to the embeddings this module hands back for reuse
as model input -- never to sampling, masking, or which nodes get
processed at all. This was a deliberate choice made when evaluating
STARE's importance-based sampling as a possible third component: decay-
gated *sampling* would systematically deprioritize dormant-then-active
nodes, which is exactly the long-dwell-time pattern APTs rely on, before
the model even sees them. Keeping decay entirely inside the embedding
path means a node that reactivates after a long absence is still
processed completely normally in every window -- it simply arrives
carrying a heavily-decayed (never deleted, never resampled-around)
memory vector alongside its ordinary features.

Why decay is lazy
------------------
A node's stored embedding and last-seen timestep are left untouched until
the node is looked up again. At that point,
    elapsed = current_timestep - last_seen
and the decay factor exp(-decay_rate * elapsed) is applied on the fly.
Nothing is proactively decayed or evicted for nodes absent from a window
-- there would be ~696k of them to touch on every single CADETS window,
almost all no-ops, for no benefit.

Why timesteps are window indices, not real time
--------------------------------------------------
generate_windows() (windowing.py) chunks by a fixed *event count*, not a
fixed wall-clock duration, so windows are not evenly spaced in real time.
Decaying by window index rather than by real elapsed nanoseconds is a
deliberate simplification consistent with the DTDG framing this whole
extension rests on (KAIROS/TFLAG get cross-window memory by staying in
continuous time (CTDG), at real scalability cost; this project's
contribution is bringing persistent state into the cheaper, snapshot-
based DTDG paradigm instead). NodeMemory itself doesn't hard-code this
choice: get_decayed()/update() take a plain `current_timestep` integer
and don't care what it counts, so real-time decay could be substituted
later (e.g. passing a window's mean timestamp instead of its index)
without changing this class at all.

Why embeddings are detached before storage
---------------------------------------------
update() refuses a tensor with requires_grad=True. A node's memory is
treated as a fixed, non-differentiable input to whichever window reuses
it -- not a parameter trained end-to-end across the whole 87-window
sequence. Storing a tensor that still required grad would silently chain
every window's computation graph onto the next, growing one unbounded
graph across a whole training epoch for no intended purpose.
"""

import torch


class NodeMemory:
    """A fixed-size, dense, exponentially-decayed memory bank over a
    Stage 1 global node vocabulary.

    Parameters
    ----------
    num_nodes : int
        Size of the Stage 1 global node vocabulary (e.g. len(node_vocab)
        from node_vocab.load_node_vocab -- 696,358 for the real CADETS
        train+test vocabulary). Global ids are assumed dense in
        [0, num_nodes), matching how node_vocab.build_node_vocab assigns
        them.
    embedding_dim : int
        Width of the stored embedding. In Stage 4 this matches
        SAGENetWithMemory's hidden width, not the raw feature width --
        see train_windowed.py's module docstring for why memory is
        injected at the hidden layer rather than concatenated onto raw
        input features.
    decay_rate : float
        Non-negative exponential decay constant. decay_rate=0 disables
        decay entirely (memory never weakens); larger values decay
        faster. Untuned -- treat any default as a starting point for a
        sweep, not a derived value.
    device : torch.device, optional
        Defaults to CPU. Use .to(device) to move an existing instance.
    """

    def __init__(self, num_nodes, embedding_dim, decay_rate, device=None, max_norm=None):
        if num_nodes <= 0:
            raise ValueError(f'num_nodes must be positive, got {num_nodes}')
        if embedding_dim <= 0:
            raise ValueError(f'embedding_dim must be positive, got {embedding_dim}')
        if decay_rate < 0:
            raise ValueError(f'decay_rate must be >= 0, got {decay_rate}')

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.device = device or torch.device('cpu')
        self.max_norm = max_norm

        self.memory = torch.zeros(num_nodes, embedding_dim, device=self.device)
        # -1 sentinel means "never updated". Any real timestep is >= 0.
        self.last_seen = torch.full((num_nodes,), -1, dtype=torch.long, device=self.device)

    def to(self, device):
        """Move stored tensors to *device* in place; returns self."""
        self.device = device
        self.memory = self.memory.to(device)
        self.last_seen = self.last_seen.to(device)
        return self

    def reset(self):
        """Clear all stored memory and last-seen timestamps.

        Call this at the start of every independent pass through a window
        sequence from timestep 0 -- each training epoch, and separately
        before a test-set evaluation pass (unless deliberately continuing
        training's memory forward -- see train_windowed.py's
        --continue-memory-into-test). Skipping this between epochs would
        let the last window of one epoch leak into the first window of
        the next as though it were "one window later", which isn't true
        (an epoch restarts the same real timeline from the beginning) and
        would silently corrupt the elapsed-time computation.
        """
        self.memory.zero_()
        self.last_seen.fill_(-1)

    def get_decayed(self, global_ids, current_timestep):
        """Look up decayed memory for a batch of nodes.

        Parameters
        ----------
        global_ids : LongTensor [N] (or anything torch.as_tensor accepts)
            Stage 1 global node ids -- for a window's Data object this is
            simply data.global_id.
        current_timestep : int
            The window index (or other monotonically increasing counter,
            see module docstring) being processed right now.

        Returns
        -------
        FloatTensor [N, embedding_dim]
            Row i holds global_ids[i]'s stored embedding scaled by
            exp(-decay_rate * elapsed), elapsed clamped to >= 0, or an
            all-zero row if the node has no prior memory. Negative
            elapsed (current_timestep before a node's last_seen -- i.e.
            get_decayed called out of chronological order) is clamped to
            0 rather than left to amplify the stored embedding.
        """
        ids = self._as_id_tensor(global_ids)
        last_seen = self.last_seen[ids]
        seen_mask = (last_seen >= 0).to(torch.float32)

        elapsed = (current_timestep - last_seen).clamp(min=0).to(torch.float32)
        decay = torch.exp(-self.decay_rate * elapsed) * seen_mask

        return self.memory[ids] * decay.unsqueeze(1)

    def update(self, global_ids, embeddings, current_timestep):
        """Overwrite stored memory for a batch of nodes.

        Parameters
        ----------
        global_ids : LongTensor [N]
        embeddings : FloatTensor [N, embedding_dim]
            New embeddings to store. Must not require grad -- see the
            module docstring for why detaching is mandatory rather than
            done silently here (a caller passing a grad-tracked tensor by
            accident is a bug worth surfacing, not papering over).
        current_timestep : int

        Notes
        -----
        Assumes global_ids has no repeated entries within one call (true
        for every window Data object build_window_data() produces, since
        each uuid maps to exactly one local index and hence exactly one
        global_id per window). A repeated id would simply be written
        twice with last-write-wins semantics; this is not validated
        against here.
        """
        if embeddings.requires_grad:
            raise ValueError(
                'update() received a tensor with requires_grad=True -- '
                'detach embeddings before storing them in memory (memory '
                'is a fixed input to future windows, not a parameter '
                'trained end-to-end across the whole window sequence).'
            )
        ids = self._as_id_tensor(global_ids)
        if embeddings.size(0) != ids.size(0):
            raise ValueError(
                f'global_ids has {ids.size(0)} entries but embeddings has '
                f'{embeddings.size(0)} rows'
            )
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f'embeddings has width {embeddings.size(1)}, expected '
                f'{self.embedding_dim}'
            )

        if self.max_norm is not None:
            norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = (self.max_norm / norms).clamp(max=1.0)
            embeddings = embeddings * scale

        self.memory[ids] = embeddings.to(self.device)
        self.last_seen[ids] = current_timestep

    def __len__(self):
        """Number of nodes ever updated (not the vocabulary size)."""
        return int((self.last_seen >= 0).sum().item())

    def _as_id_tensor(self, global_ids):
        if torch.is_tensor(global_ids):
            return global_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(global_ids, dtype=torch.long, device=self.device)
