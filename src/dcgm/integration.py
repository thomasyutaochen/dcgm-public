
import torch
import torch.nn as nn

class MemoryAdapter(nn.Module):
    """Gating adapter that injects pooled memory into a decoder hidden state."""
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_t: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, d), g_t: (d,)
        g = self.gate(g_t).unsqueeze(0).expand_as(x_t)
        return self.ln(x_t + g)
