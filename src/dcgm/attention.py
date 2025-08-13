
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class AttentionBundle:
    # head-wise attention over tokens, shape: (H, T, T)
    attn: torch.Tensor
    # mapping from tokens -> chunk id (e.g., sentence/paragraph index)
    token_to_chunk: np.ndarray

def compute_chunk_causal_scores(attn: torch.Tensor, token_to_chunk: np.ndarray) -> np.ndarray:
    """Aggregate token-level multi-head attention into per-chunk causal scores γ_ij.
    Args:
        attn: (H, T, T), attn[h, p, q] = weight from token p -> token q
        token_to_chunk: (T,), integer chunk ids for each token
    Returns:
        gamma: (C, C) array where C is # distinct chunks in token_to_chunk
    """
    H, T, _ = attn.shape
    chunks = np.unique(token_to_chunk)
    C = chunks.max() + 1
    gamma = np.zeros((C, C), dtype=np.float64)
    counts = np.zeros((C, C), dtype=np.int64)

    # average over heads and tokens
    attn_mean = attn.mean(dim=0).detach().cpu().numpy()  # (T, T)
    for p in range(T):
        i = token_to_chunk[p]
        for q in range(T):
            j = token_to_chunk[q]
            gamma[i, j] += attn_mean[p, q]
            counts[i, j] += 1

    # normalize by token pair counts
    nz = counts > 0
    gamma[nz] /= counts[nz]
    return gamma
