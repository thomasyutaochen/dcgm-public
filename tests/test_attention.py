
import numpy as np, torch
from src.dcgm.attention import compute_chunk_causal_scores

def test_gamma_shape():
    H, T = 3, 10
    attn = torch.rand(H, T, T)
    tok2chunk = np.random.randint(0, 4, size=(T,))
    gamma = compute_chunk_causal_scores(attn, tok2chunk)
    assert gamma.shape == (tok2chunk.max()+1, tok2chunk.max()+1)
    assert (gamma >= 0).all()
