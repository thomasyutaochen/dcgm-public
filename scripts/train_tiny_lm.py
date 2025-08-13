
import os, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.dcgm.tiny_transformer import TinyTransformerLM

class ToyDataset(Dataset):
    def __init__(self, text, seq_len=64):
        vocab = sorted(set(text))
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.ids = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids) - self.seq_len - 1
    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.seq_len]
        y = self.ids[idx+1:idx+self.seq_len+1]
        return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save_dir", type=str, default="artifacts/attn")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    text = "To be, or not to be, that is the question. " * 200
    ds = ToyDataset(text, seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = TinyTransformerLM(vocab=len(ds.stoi))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for step, (x, y) in enumerate(pbar):
            logits, attn = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss.item()))
            if step % 10 == 0:
                # save a small attention bundle
                torch.save(attn.detach().cpu(), os.path.join(args.save_dir, f"attn_{epoch}_{step}.pt"))
    print("Saved attention tensors in", args.save_dir)

if __name__ == "__main__":
    main()
