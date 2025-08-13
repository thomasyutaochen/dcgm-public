
import os, json, argparse, numpy as np, torch
from tqdm import tqdm
from src.dcgm import DCGM, compute_chunk_causal_scores

def synthetic_chunks(k, d):
    X = np.random.randn(k, d).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--save_dir", type=str, default="artifacts")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dcgm = DCGM(d=args.d, B=args.B, M=args.M, L=args.L, pool="max", device="cpu")
    metrics = []
    for t in tqdm(range(args.max_steps), desc="streaming"):
        C = np.random.randint(4, 8)
        X = synthetic_chunks(C, args.d)
        attn = torch.rand(4, 32, 32)  # (H,T,T) synthetic
        tok2chunk = np.random.randint(0, C, size=(32,))
        gamma = compute_chunk_causal_scores(attn, tok2chunk)
        g, stats = dcgm.update_step([X[i] for i in range(C)], gamma)
        metrics.append(dict(step=t, **stats))
        if (t % 5) == 0:
            print(f"step={t} nodes={stats['num_nodes']} edges={stats['num_edges']} "
                  f"tau={stats['tau']:.4f} delta={stats['delta_mass']:.4f} "
                  f"rel_spec_err={stats['rel_spec_err']:.4f}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Wrote metrics to", os.path.join(args.save_dir, "metrics.json"))

if __name__ == "__main__":
    main()
