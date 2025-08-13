
from dataclasses import dataclass

@dataclass
class DCGMConfig:
    B: int = 1024               # node budget
    M: int = 16                 # out-edges per node
    L: int = 3                  # message passing hops
    d: int = 128                # feature dimension
    tau_eta: float = 0.5        # threshold scale for adaptive τ_t
    pool: str = "max"           # 'max' or 'attention'
    seed: int = 1337            # RNG seed
    device: str = "cpu"         # 'cpu' or 'cuda'
    topk_new: int = 8           # K: new chunks per step (synthetic demo)
    gate: bool = True           # use memory gating adapter
    verbose: bool = False
