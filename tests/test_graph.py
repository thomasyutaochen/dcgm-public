
import numpy as np
from src.dcgm.graph import DCGMGraph

def test_add_and_prune():
    g = DCGMGraph(d=8, B=5, M=3)
    ids = [g.add_node(np.ones(8)*i) for i in range(10)]
    # connect a chain
    for i in range(9):
        g.set_topM_out_edges(i, [(i+1, 1.0)])
    g.prune_to_B()
    assert len(g.nodes) <= 5
    # edges must only point to kept nodes
    for i, nbrs in g.adj.items():
        for (j, w) in nbrs:
            assert j in g.nodes
