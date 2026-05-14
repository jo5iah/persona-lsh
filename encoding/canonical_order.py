"""Canonical-order encoder: reorder coords by calibration-derived variance."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class CanonicalOrderEncoder(BucketEdgesEncoder):
    """`BucketFractional`-style encoding, but coordinates are reordered first.

    The reordering permutation is supplied at construction (typically derived
    from a calibration corpus via `variance_permutation_per_layer`). The
    intent: put the coords that vary most across the corpus -- which is where
    the persona-direction signal lives -- at the **start** of the byte stream,
    so TLSH n-grams over the early bytes are dense in trait signal.

    Stability vs. `MagnitudeSorted`: the permutation is fixed at construction
    time and shared across all vectors that will be compared, so a small
    perturbation to one vector does not shuffle its bytes. Within each
    permuted position the encoding is just `(bucket_idx, frac)`, which is
    itself smooth under perturbation.
    """

    def __init__(self, edges: ArrayLike, permutation: ArrayLike):
        super().__init__(edges)
        p = np.asarray(permutation, dtype=np.int64)
        if p.ndim != 1:
            raise ValueError(f"permutation must be 1-D, got shape {p.shape}")
        n = p.size
        sorted_p = np.sort(p)
        if not np.array_equal(sorted_p, np.arange(n)):
            raise ValueError(
                "permutation must be a permutation of [0, len(permutation))"
            )
        self.permutation = p

    def encode_vector(self, vector: ArrayLike) -> bytes:
        v = _as_float32_numpy(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        if v.size != self.permutation.size:
            raise ValueError(
                f"vector size {v.size} != permutation size {self.permutation.size}"
            )
        v_perm = v[self.permutation]
        e = self.edges

        b1 = np.clip(np.searchsorted(e, v_perm, side="right") - 1, 0, N_BUCKETS - 1).astype(np.int32)
        lo = e[b1]
        hi = e[b1 + 1]
        width = hi - lo
        frac = (v_perm.astype(np.float64) - lo) / width
        b2 = np.clip(np.rint(255.0 * frac), 0, N_BUCKETS - 1).astype(np.int32)

        pairs = np.empty(v_perm.size * 2, dtype=np.uint8)
        pairs[0::2] = b1.astype(np.uint8)
        pairs[1::2] = b2.astype(np.uint8)
        return pairs.tobytes()


def variance_permutation_per_layer(tensors: Sequence[ArrayLike]) -> list[np.ndarray]:
    """For each layer, return indices that sort coords by descending variance.

    `tensors` is a list of `[num_layers, hidden_dim]` calibration vectors
    (e.g. several persona-direction tensors). The variance is computed across
    that corpus per coord per layer; coords with high variance are precisely
    the dimensions where the trait conditioning matters most -- those go first
    in the permutation, so `CanonicalOrderEncoder` puts them at the head of
    the byte stream.
    """
    if not tensors:
        raise ValueError("need at least one tensor to derive a permutation")
    np_tensors = [_as_float32_numpy(t) for t in tensors]
    n_layers = np_tensors[0].shape[0]
    for t in np_tensors:
        if t.ndim != 2:
            raise ValueError(f"expected 2-D tensors, got shape {t.shape}")
        if t.shape[0] != n_layers:
            raise ValueError(
                f"all tensors must share num_layers; got {t.shape[0]} vs {n_layers}"
            )

    perms: list[np.ndarray] = []
    for layer in range(n_layers):
        stacked = np.stack([t[layer] for t in np_tensors], axis=0)  # [N, hidden]
        variances = stacked.var(axis=0)
        # Descending: argsort of -variances. Stable so ties keep coord order.
        perms.append(np.argsort(-variances, kind="stable"))
    return perms
