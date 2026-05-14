"""Bucket-edge generators for `BucketEdgesEncoder` subclasses.

These helpers don't know anything about which encoder will use them; they
produce 257-element monotonic arrays suitable for any `BucketEdgesEncoder`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import ArrayLike, N_EDGES, _as_float32_numpy

__all__ = [
    "linear_edges",
    "quantile_edges",
    "symlog_edges",
    "calibrate_edges_per_layer",
]


def linear_edges(lo: float, hi: float) -> np.ndarray:
    """`N_EDGES` evenly spaced edges spanning `[lo, hi]`."""
    if not hi > lo:
        raise ValueError(f"hi ({hi}) must be > lo ({lo})")
    return np.linspace(lo, hi, N_EDGES, dtype=np.float64)


def quantile_edges(samples: ArrayLike) -> np.ndarray:
    """Edges placed at the empirical quantiles of `samples`.

    Equal-mass bucketing — robust to heavy-tailed activation distributions.
    Duplicate quantiles are perturbed by `nextafter` so strict monotonicity
    holds.
    """
    s = _as_float32_numpy(samples).reshape(-1).astype(np.float64)
    if s.size < N_EDGES:
        raise ValueError(f"need at least {N_EDGES} samples for quantile edges, got {s.size}")
    qs = np.linspace(0.0, 1.0, N_EDGES)
    edges = np.quantile(s, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def symlog_edges(lo: float, hi: float, linthresh: float = 1.0) -> np.ndarray:
    """Symmetric log-spaced edges — denser near zero, coarser in the tails.

    `linthresh` is the linear-region half-width around zero (mirrors
    matplotlib's `symlog`). Useful when activations are roughly bell-shaped
    around 0 but with heavy tails.
    """
    if not hi > lo:
        raise ValueError(f"hi ({hi}) must be > lo ({lo})")
    if linthresh <= 0:
        raise ValueError("linthresh must be > 0")

    def fwd(x: np.ndarray) -> np.ndarray:
        sign = np.sign(x)
        absx = np.abs(x)
        linear = absx <= linthresh
        out = np.where(linear, absx / linthresh, 1.0 + np.log10(absx / linthresh))
        return sign * out

    a, b = fwd(np.array([lo, hi], dtype=np.float64))
    u = np.linspace(a, b, N_EDGES)

    def inv(u: np.ndarray) -> np.ndarray:
        sign = np.sign(u)
        absu = np.abs(u)
        linear = absu <= 1.0
        out = np.where(linear, absu * linthresh, linthresh * 10.0 ** (absu - 1.0))
        return sign * out

    edges = inv(u)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def calibrate_edges_per_layer(
    tensors: Sequence[ArrayLike],
    edges_method: str,
) -> list[np.ndarray]:
    """Pool many `[layers, hidden_dim]` tensors and compute per-layer edges.

    Use this when you intend to LSH-compare multiple persona vectors with
    each other: every vector must be encoded in the *same* bucket frame per
    layer, otherwise comparison picks up bucket-frame jitter on top of the
    real activation difference.
    """
    if not tensors:
        raise ValueError("need at least one tensor to calibrate from")
    np_tensors = [_as_float32_numpy(t) for t in tensors]
    n_layers = np_tensors[0].shape[0]
    for t in np_tensors:
        if t.ndim != 2:
            raise ValueError(f"expected 2-D tensors, got shape {t.shape}")
        if t.shape[0] != n_layers:
            raise ValueError(
                f"all tensors must share num_layers; got {t.shape[0]} vs {n_layers}"
            )

    edges_per_layer: list[np.ndarray] = []
    for layer in range(n_layers):
        pooled = np.concatenate([t[layer] for t in np_tensors], axis=0)
        if edges_method == "linear":
            lo = float(pooled.min())
            hi = float(pooled.max())
            if hi <= lo:
                hi = lo + 1e-6
            edges_per_layer.append(linear_edges(lo, hi))
        elif edges_method == "quantile":
            edges_per_layer.append(quantile_edges(pooled))
        elif edges_method == "symlog":
            absmax = float(np.abs(pooled).max())
            if absmax == 0:
                absmax = 1.0
            edges_per_layer.append(symlog_edges(-absmax, absmax))
        else:
            raise ValueError(f"unknown edges_method: {edges_method!r}")
    return edges_per_layer
