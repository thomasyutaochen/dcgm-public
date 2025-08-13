
import numpy as np
from numpy.linalg import norm
from src.dcgm.graph import DCGMGraph

def test_laplacian_small():
    g = DCGMGraph(d=4, B=10, M=2)
    for _ in range(4):
        g.add_node(np.random.randn(4))
    # fully connect with weights, then top-M will restrict
    for i in list(g.nodes.keys()):
        nbrs = [(j, np.random.rand()) for j in g.nodes if j != i]
        g.set_topM_out_edges(i, nbrs)
    L = g.laplacian()
    assert L.shape[0] == L.shape[1] >= 4
    # Laplacian PSD check: x^T L x >= 0 for random x
    x = np.random.randn(L.shape[0])
    val = x @ (L @ x)
    assert val >= -1e-6
