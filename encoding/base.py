"""Base classes and shared helpers for byte encoders.

A `ByteEncoder` maps a 1-D float vector to a byte stream that downstream LSH
backends (TLSH today, possibly others later) consume. Encoders vary in how
many bytes per scalar they emit and what each byte encodes; state that
depends on data (e.g. bucket edges, projection matrices) is supplied at
construction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]

N_BUCKETS = 256
N_EDGES = N_BUCKETS + 1  # 257


def _as_float32_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(x, dtype=np.float32)


def _validate_edges(edges: ArrayLike) -> np.ndarray:
    e = np.asarray(edges, dtype=np.float64)
    if e.shape != (N_EDGES,):
        raise ValueError(f"edges must have length {N_EDGES}, got shape {e.shape}")
    if not np.all(np.diff(e) > 0):
        raise ValueError("edges must be strictly monotonically increasing")
    return e


class ByteEncoder(ABC):
    """Maps a 1-D float vector to a byte stream for fuzzy hashing.

    Subclasses set any data-dependent state (e.g. bucket edges) at
    construction; `encode_vector` is then stateless and deterministic.
    """

    @abstractmethod
    def encode_vector(self, vector: ArrayLike) -> bytes:
        """Encode a 1-D vector into a byte stream."""

    def encode_layers(self, tensor: ArrayLike) -> list[bytes]:
        """Encode each row of a 2-D `[layers, hidden_dim]` tensor independently.

        Override if your encoder does joint cross-layer encoding."""
        t = _as_float32_numpy(tensor)
        if t.ndim != 2:
            raise ValueError(f"expected 2-D tensor, got shape {t.shape}")
        return [self.encode_vector(t[i]) for i in range(t.shape[0])]


class BucketEdgesEncoder(ByteEncoder):
    """ByteEncoder parameterized by 257 monotonically increasing bucket edges.

    All bucket-based byte encodings live under this base class so they can
    share edge validation and the per-layer edges API in `save_lsh_sidecar`.
    """

    def __init__(self, edges: ArrayLike):
        self.edges = _validate_edges(edges)
