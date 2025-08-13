
import numpy as np, torch
from src.dcgm import DCGM

def test_stream_small():
    dcgm = DCGM(d=16, B=8, M=2, L=1, pool="max", device="cpu")
    for t in range(5):
        C = 4
        X = np.random.randn(C,16).astype('float32')
        X /= (np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
        attn = torch.rand(2, 16, 16)
        tok2chunk = np.random.randint(0, C, size=(16,))
        from src.dcgm.attention import compute_chunk_causal_scores
        gamma = compute_chunk_causal_scores(attn, tok2chunk)
        g, stats = dcgm.update_step([X[i] for i in range(C)], gamma)
        assert stats["num_nodes"] <= 8
        assert stats["num_edges"] <= 8*2
        assert g.ndim == 1
