
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / (self.d_head ** 0.5)  # (B,H,T,T)
        mask = torch.triu(torch.ones(T,T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.out(y)
        return y, att

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab=128, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([MultiHeadSelfAttn(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx)
        attns = []
        for layer in self.layers:
            x, a = layer(x)
            attns.append(a)  # list of (B,H,T,T)
        x = self.ln(x)
        logits = self.lm_head(x)
        # Return the last layer's attention for γ_ij
        last_attn = attns[-1]
        return logits, last_attn
