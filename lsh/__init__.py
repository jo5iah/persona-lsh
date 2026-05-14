"""LSH backends + backward-compat module-level functions.

Public API:

  - `LSHBackend`                 — abstract base class for any LSH scheme
  - `TLSHBackend`                — TLSH over a `ByteEncoder` byte stream
  - `hash_vector`, `hash_layers`, `hash_layers_per_layer_edges`,
    `hash_persona_file`, `diff`, `pairwise_diff`
                                  — pre-refactor module-level helpers that
                                    use TLSHBackend + BucketFractionalEncoder
                                    by default. New code should construct an
                                    explicit backend instead.

New backends (e.g. Random-Projection LSH) should subclass `LSHBackend` and
be exported here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from encoding import ArrayLike, BucketFractionalEncoder

from .base import LSHBackend
from .tlsh_backend import TLSHBackend, tlsh_module as _tlsh

__all__ = [
    "LSHBackend",
    "TLSHBackend",
    "diff",
    "hash_layers",
    "hash_layers_per_layer_edges",
    "hash_persona_file",
    "hash_vector",
    "pairwise_diff",
]


def _tlsh_for_edges(edges: ArrayLike) -> TLSHBackend:
    return TLSHBackend(BucketFractionalEncoder(edges))


def hash_vector(vector: ArrayLike, edges: ArrayLike) -> str:
    """TLSH digest of a single 1-D activation vector under the given edges."""
    return _tlsh_for_edges(edges).hash_vector(vector)


def hash_layers(vectors: ArrayLike, edges: ArrayLike) -> list[str]:
    """One TLSH digest per layer of a `[layers, hidden_dim]` tensor.

    A single `edges` array is used for every layer. For per-layer edges,
    see `hash_layers_per_layer_edges`.
    """
    backend = _tlsh_for_edges(edges)
    return [backend.hash_vector(vectors[i]) for i in range(vectors.shape[0])]


def hash_layers_per_layer_edges(
    vectors: ArrayLike, edges_per_layer: list[ArrayLike]
) -> list[str]:
    """One TLSH digest per layer, with its own bucket edges per layer."""
    if len(edges_per_layer) != vectors.shape[0]:
        raise ValueError(
            f"edges_per_layer has length {len(edges_per_layer)} but the tensor "
            f"has {vectors.shape[0]} layers"
        )
    backends = [_tlsh_for_edges(e) for e in edges_per_layer]
    return [backends[i].hash_vector(vectors[i]) for i in range(vectors.shape[0])]


def hash_persona_file(path, edges: ArrayLike) -> list[str]:
    """Load a saved persona vector (.pt) and TLSH-hash each layer."""
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path}: expected torch.Tensor, got {type(tensor).__name__}")
    return hash_layers(tensor, edges)


def diff(digest_a: str, digest_b: str) -> int:
    """TLSH distance between two digests (0 = identical, larger = more different).

    Returned as `int` for backward compatibility with the pre-refactor API
    (the underlying tlsh.diff is integer-valued).
    """
    return _tlsh.diff(digest_a, digest_b)


def pairwise_diff(digests: Iterable[str]) -> np.ndarray:
    """Symmetric NxN matrix of pairwise TLSH distances (int32 for back-compat)."""
    ds = list(digests)
    n = len(ds)
    matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _tlsh.diff(ds[i], ds[j])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix
