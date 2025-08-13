
import numpy as np
import torch, time, math, random
from typing import Iterable

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Stopwatch:
    def __init__(self):
        self.start = time.time()
    def lap(self):
        t = time.time()
        dt = t - self.start
        self.start = t
        return dt

def adaptive_threshold(num_nodes: int, eta: float) -> float:
    # τ_t = η * (log |V_t| / |V_t|), with guardrails
    if num_nodes <= 1:
        return 1.0
    return float(eta * (math.log(num_nodes) / max(2, num_nodes)))

def topk_indices(values, k):
    # returns indices of largest k entries
    values = np.asarray(values)
    if k >= len(values):
        return np.argsort(-values)
    return np.argpartition(-values, k)[:k]
