"""Embedder interface and implementations for Layer 5 (app/index/embedder.py).

Provides:
- Embedder (ABC): standard interface returning L2-normalized numpy embeddings.
- HashingEmbedder: fast, deterministic, 100% offline fallback using hashed word + bigram
  counts with TF weighting. Requires no models, no network, and fixed dimension (512).
- SentenceTransformerEmbedder: uses sentence-transformers with local model
  (default 'BAAI/bge-small-en-v1.5', configurable via EMBED_MODEL env var). Falls back
  automatically to HashingEmbedder with a clear warning if sentence-transformers is missing
  or model loading fails.
- get_embedder(): helper factory selecting the appropriate embedder.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Embedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifying name of the embedder (e.g. 'hashing-512', 'BAAI/bge-small-en-v1.5')."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into an L2-normalized 2D numpy array of shape (N, dim)."""


class HashingEmbedder(Embedder):
    """Deterministic offline fallback embedder.

    Computes hashed bag-of-words and bigrams with sublinear TF weighting,
    projected into a fixed-dimension vector and L2-normalized.
    Deterministic across all Python runs, platforms, and processes.
    """

    def __init__(self, dim: int = 512, name: Optional[str] = None) -> None:
        self._dim = dim
        self._name = name or f"hashing-{dim}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        tokens = list(words)
        # Add word bigrams
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Hash a token into a bucket index and a sign (+1/-1) using Murmur/MD5."""
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self._dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)

        for row, text in enumerate(texts):
            tokens = self._tokenize(text)
            if not tokens:
                continue

            # Term frequency counting
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for token, count in tf.items():
                idx, sign = self._hash_token(token)
                weight = (1.0 + math.log(count)) * sign
                vectors[row, idx] += weight

            # L2 normalise
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 1e-12:
                vectors[row] /= norm

        return vectors.astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    """SentenceTransformer embedding model with automatic offline fallback."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        model_str = model_name or os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        self._model_name = model_str
        self._model = None
        self._fallback: Optional[HashingEmbedder] = None
        self._dim = 384  # default for bge-small-en-v1.5, updated once loaded

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # Load local model
            self._model = SentenceTransformer(self._model_name)
            # Detect actual dim
            test_emb = self._model.encode(["test"])
            self._dim = int(test_emb.shape[1])
        except Exception as exc:
            warnings.warn(
                f"[WARN] sentence-transformers not available or failed to load '{self._model_name}': {exc}. "
                f"Falling back automatically to deterministic HashingEmbedder(dim=512).",
                UserWarning,
                stacklevel=2,
            )
            self._fallback = HashingEmbedder(dim=512, name=f"fallback-hashing-512")
            self._dim = self._fallback.dim
            self._model_name = self._fallback.name

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.embed(texts)

        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        # SentenceTransformer encode with normalize_embeddings=True
        embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)


def get_embedder(kind: str = "hashing", model_name: Optional[str] = None, dim: int = 512) -> Embedder:
    """Factory for embedders.

    Args:
        kind: 'hashing' (default, offline) or 'st' / 'sentence-transformers'.
        model_name: Model identifier if using sentence-transformers.
        dim: Vector dimension if using HashingEmbedder.
    """
    kind_lower = kind.lower().strip()
    if kind_lower in ("st", "sentence-transformers", "sentence_transformers"):
        return SentenceTransformerEmbedder(model_name=model_name)
    return HashingEmbedder(dim=dim)
