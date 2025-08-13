
from typing import List, Tuple, Dict
import numpy as np
import torch
from .graph import DCGMGraph
from .message_passing import MessagePassing
from .pooling import pool_max, pool_attention
from .utils import adaptive_threshold

class DCGM:
    import scipy.sparse as sp
    import numpy as np
    def __init__(self, d: int, B: int, M: int, L: int, pool: str = "max", device="cpu"):
        self.graph = DCGMGraph(d=d, B=B, M=M)
        self.msg = MessagePassing(d=d, L=L).to(device)
        self.pool_kind = pool
        self.device = device

    def _pool(self, feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        if self.pool_kind == "max":
            return pool_max(feats)
        else:
            # attention pooling with degree as score proxy
            scores = {i: len(self.graph.adj.get(i, [])) for i in feats}
            return pool_attention(feats, scores)

    def update_step(self, new_nodes: List[np.ndarray], gamma: np.ndarray):
        """Now also computes spectral deviation and delta_t mass."""
        """One streaming step:
         - add new nodes with embeddings
         - build edges using gamma (C x C) and adaptive threshold + top-M
         - prune to B and run message passing to produce pooled memory
        Returns pooled memory vector as torch.Tensor (d,) and stats dict.
        """
        # 1) Insert nodes
        ids = []
        for vec in new_nodes:
            i = self.graph.add_node(vec)
            ids.append(i)

        # 2) Build edges
        V = list(self.graph.nodes.keys())
        tau = adaptive_threshold(len(V), eta=0.5)
        C = gamma.shape[0]
        # map local chunk ids [0..C-1] to global node ids for the new step
        if len(ids) != C:
            # if mismatch, fall back: use min(C, len(ids))
            C = min(C, len(ids))
        for local_i in range(C):
            i = ids[local_i]
            nbrs = []
            for local_j in range(C):
                j = ids[local_j]
                w = float(gamma[local_i, local_j])
                if w > tau:
                    nbrs.append((j, w))
            self.graph.set_topM_out_edges(i, nbrs)

        # 3) Prune to B
        self.graph.prune_to_B()

        # 4) Message passing over current graph
        node_vecs = {i: torch.tensor(self.graph.nodes[i].vec, dtype=torch.float32, device=self.device) for i in self.graph.nodes}
        feats = self.msg(node_vecs, self.graph.adj)
        g = self._pool(feats)  # (d_hidden,) because of MessagePassing expansion

# Spectral deviation: compare current Laplacian with fully connected Laplacian over same nodes
L_sparse = self.graph.laplacian()
n = L_sparse.shape[0]
if n > 0:
    rows, cols = zip(*[(i,j) for i in range(n) for j in range(n) if i != j])
    full_w = np.ones(len(rows), dtype=float)
    from scipy.sparse import csr_matrix, diags
    A_full = csr_matrix((full_w, (rows, cols)), shape=(n, n))
    AT = A_full.transpose()
    A_sym_full = (A_full + AT) * 0.5
    d = np.array(A_sym_full.sum(axis=1)).reshape(-1)
    D = diags(d)
    L_full = D - A_sym_full
    diff = (L_sparse - L_full)
    num = np.linalg.norm(diff.toarray())
    den = np.linalg.norm(L_full.toarray()) + 1e-9
    spectral_dev = num / den
else:
    spectral_dev = 0.0
# delta_t mass: track removed edges' total weight fraction
total_mass = 0.0
removed_mass = 0.0
for i, nbrs in self.graph.adj.items():
    total_mass += sum(w for (_, w) in nbrs)
# simulate removal: keep only top-M, measure rest
for i, nbrs in self.graph.adj.items():
    sorted_nbrs = sorted(nbrs, key=lambda x: -x[1])
    removed_mass += sum(w for (_, w) in sorted_nbrs[self.graph.M:])
delta_mass_frac = (removed_mass / (total_mass + 1e-9)) if total_mass > 0 else 0.0
        return g.detach().cpu(), {
            "num_nodes": len(self.graph.nodes),
            "num_edges": self.graph.edge_count(),
            "tau": tau
        }
