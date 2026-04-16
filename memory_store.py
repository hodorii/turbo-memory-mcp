"""
Compressed vector memory store using TurboQuant.

Stores (text, compressed_embedding) pairs and supports
similarity search via inner-product estimation.
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from turbo_quant import TurboQuantState, build_turbo_quant, quant_prod, dequant_prod


@dataclass
class MemoryEntry:
    id: str
    text: str
    # Compressed representation
    idx: np.ndarray    # MSE quantization indices
    norm: float        # L2 norm of original vector
    qjl: np.ndarray   # QJL sign bits
    r_norm: float      # residual norm


class CompressedMemoryStore:
    """
    Memory store that compresses embeddings with TurboQuant.

    Compression ratio: bits/16 (e.g., 3-bit → ~5x vs FP16)
    Inner-product queries remain unbiased due to QJL correction.
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.state: TurboQuantState = build_turbo_quant(dim, bits, seed)
        self.entries: List[MemoryEntry] = []
        self._counter = 0

    def add(self, text: str, embedding: np.ndarray) -> str:
        """Compress and store an embedding. Returns entry id."""
        assert embedding.shape == (self.dim,), f"Expected dim={self.dim}"
        idx, norm, qjl, r_norm = quant_prod(embedding, self.state)
        entry_id = f"mem_{self._counter:06d}"
        self._counter += 1
        self.entries.append(MemoryEntry(
            id=entry_id, text=text,
            idx=idx, norm=norm, qjl=qjl, r_norm=r_norm,
        ))
        return entry_id

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find top-k most similar entries using inner-product estimation.
        Returns list of (id, text, score).
        """
        if not self.entries:
            return []

        scores = []
        for entry in self.entries:
            x_hat = dequant_prod(entry.idx, entry.norm, entry.qjl, entry.r_norm, self.state)
            score = float(np.dot(query, x_hat))
            scores.append((entry.id, entry.text, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def delete(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.id != entry_id]
        return len(self.entries) < before

    def save(self, path: str):
        data = {
            "dim": self.dim,
            "bits": self.bits,
            "counter": self._counter,
            "state": self.state,
            "entries": self.entries,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "CompressedMemoryStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls.__new__(cls)
        store.dim = data["dim"]
        store.bits = data["bits"]
        store._counter = data["counter"]
        store.state = data["state"]
        store.entries = data["entries"]
        return store

    def stats(self) -> dict:
        n = len(self.entries)
        # bits per entry: (bits-1)*dim (MSE idx) + dim (QJL) + 32+32 (norm, r_norm)
        compressed_bits = n * ((self.bits - 1) * self.dim + self.dim + 64)
        original_bits = n * self.dim * 32  # FP32
        return {
            "entries": n,
            "dim": self.dim,
            "bits": self.bits,
            "compression_ratio": round(original_bits / max(compressed_bits, 1), 2),
        }
