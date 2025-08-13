
import numpy as np
from typing import List, Tuple

class BM25Retriever:
    """Very small BM25-like retriever for demonstration (not optimized)."""
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.vocab = {}
        self.doc_freq = {}
        self.doc_tfs = []
        self._build()

    def _tokenize(self, s: str) -> List[str]:
        return [t.lower() for t in s.split()]

    def _build(self):
        for d in self.docs:
            tfs = {}
            for tok in self._tokenize(d):
                self.vocab.setdefault(tok, len(self.vocab))
                tfs[tok] = tfs.get(tok, 0) + 1
            self.doc_tfs.append(tfs)
            for tok in tfs:
                self.doc_freq[tok] = self.doc_freq.get(tok, 0) + 1

    def query(self, q: str, k: int = 8) -> List[int]:
        N = len(self.docs)
        scores = []
        q_tokens = self._tokenize(q)
        for i, tfs in enumerate(self.doc_tfs):
            s = 0.0
            for tok in q_tokens:
                df = self.doc_freq.get(tok, 0) + 1e-9
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
                tf = tfs.get(tok, 0)
                s += idf * (tf / (tf + 1.5))
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return [i for (i,_) in scores[:k]]

class ChunkEncoder:
    """Encodes chunks into fixed-d vectors using TF-IDF-style bag-of-words (toy)."""
    def __init__(self, docs: List[str], d: int = 128, seed: int = 1337):
        self.docs = docs
        self.d = d
        rng = np.random.default_rng(seed)
        # random hash projection for tokens
        self.hash_vecs = {}
        vocab = set()
        for d_ in docs:
            vocab.update(d_.lower().split())
        for tok in vocab:
            self.hash_vecs[tok] = rng.standard_normal(d) / np.sqrt(d)

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.d, dtype=np.float32)
        toks = text.lower().split()
        for t in toks:
            vec += self.hash_vecs.get(t, 0)
        if np.linalg.norm(vec) > 0:
            vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec
