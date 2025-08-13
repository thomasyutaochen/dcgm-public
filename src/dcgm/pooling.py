
import torch

def pool_max(feats: dict) -> torch.Tensor:
    if not feats:
        return torch.zeros(1)
    stacked = torch.stack(list(feats.values()), dim=0)
    return torch.amax(stacked, dim=0)

def pool_attention(feats: dict, scores: dict) -> torch.Tensor:
    # scores[i] assumed >= 0; normalize, then weighted sum
    if not feats:
        return torch.zeros(1)
    keys = list(feats.keys())
    vals = torch.stack([feats[i] for i in keys], dim=0)
    sc = torch.tensor([scores.get(i, 0.0) for i in keys], dtype=vals.dtype, device=vals.device)
    if sc.sum() <= 1e-9:
        sc = torch.ones_like(sc) / sc.numel()
    else:
        sc = sc / sc.sum()
    return (sc.view(-1,1) * vals).sum(dim=0)
