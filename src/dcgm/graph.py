
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import scipy.sparse as sp

@dataclass
class Node:
    id: int
    vec: np.ndarray           # d-dim embedding
    meta: dict = field(default_factory=dict)

class DCGMGraph:
    """Sparse directed graph with at most B nodes and up to M out-edges per node."""
    def __init__(self, d: int, B: int, M: int):
        self.d = d
        self.B = B
        self.M = M
        self.nodes: Dict[int, Node] = {}
        # adjacency as dict: i -> list[(j, weight)]
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self.next_id = 0

    def add_node(self, vec: np.ndarray, meta=None) -> int:
        i = self.next_id
        self.next_id += 1
        self.nodes[i] = Node(id=i, vec=np.asarray(vec).astype(np.float32), meta=meta or {})
        if i not in self.adj:
            self.adj[i] = []
        return i

    def set_topM_out_edges(self, i: int, nbrs: List[Tuple[int, float]]):
        # keep only top-M by weight, excluding self loops
        nbrs = [(j, w) for (j, w) in nbrs if j != i]
        nbrs.sort(key=lambda x: -x[1])
        self.adj[i] = nbrs[: self.M]

    def degree_out(self, i: int) -> int:
        return len(self.adj.get(i, []))

    def degree_in(self, j: int) -> int:
        deg = 0
        for i in self.adj:
            for (jj, _) in self.adj[i]:
                if jj == j:
                    deg += 1
        return deg

    def prune_to_B(self):
        if len(self.nodes) <= self.B:
            return
        # simple centrality: in-degree + out-degree
        scores = []
        for i in self.nodes:
            s = self.degree_out(i) + self.degree_in(i)
            scores.append((i, s))
        # keep top-B by centrality, drop rest
        scores.sort(key=lambda x: -x[1])
        keep = set([i for i,_ in scores[: self.B]])
        drop = [i for i in self.nodes if i not in keep]
        for i in drop:
            self.nodes.pop(i, None)
            self.adj.pop(i, None)
        # remove edges pointing to dropped nodes
        for i in list(self.adj.keys()):
            self.adj[i] = [(j,w) for (j,w) in self.adj[i] if j in keep]

    def laplacian(self) -> sp.csr_matrix:
        """Symmetric Laplacian of the undirected version (A + A^T)/2."""
        n = max(self.nodes.keys()) + 1 if self.nodes else 0
        if n == 0:
            return sp.csr_matrix((0,0))
        rows, cols, data = [], [], []
        for i, nbrs in self.adj.items():
            for (j, w) in nbrs:
                rows.append(i); cols.append(j); data.append(w)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
        AT = A.transpose()
        A_sym = (A + AT) * 0.5
        d = np.array(A_sym.sum(axis=1)).reshape(-1)
        D = sp.diags(d)
        L = D - A_sym
        return L

    def edge_count(self) -> int:
        return sum(len(nbrs) for nbrs in self.adj.values())
