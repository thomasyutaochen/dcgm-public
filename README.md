
# DCGM-Public

A **scientific-computing–focused** reference implementation of **Dynamic Causal-Graph Memory (DCGM)** —
a structured retrieval module for million-token reasoning. 

- **Chen, T. Y.** (2025, July). **Dynamic Causal-Graph Memory: Structured Retrieval for Million–Token Reasoning**. *Proceedings of the International Conference on Machine Learning (ICML)* | Workshop on Long-Context Foundation Models. [**Paper Link**](https://openreview.net/forum?id=0Us7om0vhZ&noteId=0Us7om0vhZ)



This library provides:

- A *streaming* causal-graph maintainer with strict memory budgets (`B` nodes, `M` out-edges per node)
- Attention-derived edge weights `γ_ij` aggregated across heads and tokens
- Logarithmic-size subgraph maintenance with heaps and centrality-based pruning
- Message passing over the sparse graph and a pooled "causal memory" vector
- Laplacian checks and spectral approximation probes for the sparsifier
- A tiny Transformer that exposes head-wise attention for `γ_ij` construction
- Clean APIs, unit tests, and reproducible scripts

⚠️ This code is a research-oriented prototype to accompany a paper on DCGM. It is not an official product.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (Synthetic)

1) Run a tiny LM to generate attention on a toy corpus:

```bash
python scripts/train_tiny_lm.py --epochs 1 --seq_len 64 --batch_size 16
```

This saves head-wise attention tensors for a few batches to `artifacts/attn/`.

2) Stream the DCGM maintainer over the attention-derived chunks:

```bash
python scripts/run_synthetic.py --B 64 --M 8 --L 2 --max_steps 50
```

This prints memory usage, edge counts, sparsification statistics, and produces:
- `artifacts/graphs/` with snapshots (edge lists, Laplacians)
- `artifacts/metrics.json` with approximation/error metrics

## Design highlights

- **Graph construction**: `γ_ij` is the mean of per-head, per-token attention from chunk `i` to `j`. Thresholding is adaptive in |V_t|.
- **Sparsification**: retain top-`M` outgoing edges per node; prune to `B` nodes via heap of centrality.
- **Message passing**: `L` hops of linear layers with gating; pooled memory vector fed to a small adapter.
- **Theory probes**: numerical checks of spectral deviation `‖L̃-L‖` and a simple bound relating truncated mass to pooled error.

## References
- Original paper introducing DCGM (used as the implementation reference).
