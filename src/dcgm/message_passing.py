
import numpy as np
import torch
import torch.nn as nn

class MessagePassing(nn.Module):
    """L-hop message passing over a sparse graph stored as adjacency lists."""
    def __init__(self, d: int, L: int = 2, hidden_scale: int = 4):
        super().__init__()
        self.L = L
        self.lin0 = nn.Linear(d, d * hidden_scale)
        self.lin1 = nn.Linear(d, d)
        self.act = nn.ReLU()

    def forward(self, node_vecs, adj):
        # node_vecs: dict[i] = torch.Tensor(d)
        # adj: dict[i] = list[(j, weight)]
        feats = {i: v for i, v in node_vecs.items()}
        for _ in range(self.L):
            new = {}
            for i, v in feats.items():
                msg = 0.0
                if i in adj:
                    denom = sum(w for (_, w) in adj[i]) + 1e-9
                    for (j, w) in adj[i]:
                        msg = msg + (w / denom) * feats[j]
                h = self.act(self.lin0(v) + self.lin1(msg))
                new[i] = h
            feats = new
        return feats
