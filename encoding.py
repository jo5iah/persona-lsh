"""Distance-preserving 2-byte encoding for activation values.

Each scalar `v` is encoded as a pair of bytes `(b1, b2)`:

* `b1` is the bucket index in `[0, 255]`, found by locating `v` inside a
  user-supplied array of 257 monotonically increasing edges.
* `b2` is the *nearest neighboring bucket index*:
    - if `v` is in the lower quartile of its bucket  -> `b1 - 1`
    - if `v` is in the upper quartile of its bucket  -> `b1 + 1`
    - otherwise (middle 50% of the bucket)           -> `b1`
  Clamped to `[0, 255]` at the extremes.

Two nearby floats therefore produce nearby `(b1, b2)` pairs, so the TLSH
n-gram statistics over the concatenated stream respect activation distance.

Bucket edges are passed in as an array so the caller controls the binning
(linear, quantile, symlog, ...). Helpers for common shapes are provided.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]

N_BUCKETS = 256
N_EDGES = N_BUCKETS + 1  # 257


def _as_float32_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
    return x


def _validate_edges(edges: ArrayLike) -> np.ndarray:
    e = np.asarray(edges, dtype=np.float64)
    if e.shape != (N_EDGES,):
        raise ValueError(f"edges must have length {N_EDGES}, got shape {e.shape}")
    if not np.all(np.diff(e) > 0):
        raise ValueError("edges must be strictly monotonically increasing")
    return e


def encode_vector(vector: ArrayLike, edges: ArrayLike) -> bytes:
    """Encode a 1-D vector to a `2 * len(vector)` byte stream.

    Bytes are interleaved: `b1_0, b2_0, b1_1, b2_1, ...`.
    """
    v = _as_float32_numpy(vector)
    if v.ndim != 1:
        raise ValueError(f"expected 1-D vector, got shape {v.shape}")
    e = _validate_edges(edges)

    # Bucket index in [0, 255]: searchsorted with side='right' returns the
    # insertion point in [0, 257]; subtracting 1 gives [-1, 256], clipped.
    b1 = np.clip(np.searchsorted(e, v, side="right") - 1, 0, N_BUCKETS - 1).astype(np.int32)

    lo = e[b1]
    hi = e[b1 + 1]
    width = hi - lo
    frac = (v.astype(np.float64) - lo) / width  # in [0, 1) for in-range values

    # Lower-quartile -> lean to b1-1; upper-quartile -> b1+1; middle -> b1.
    offset = np.where(frac < 0.25, -1, np.where(frac >= 0.75, 1, 0))
    b2 = np.clip(b1 + offset, 0, N_BUCKETS - 1).astype(np.int32)

    pairs = np.empty(v.size * 2, dtype=np.uint8)
    pairs[0::2] = b1.astype(np.uint8)
    pairs[1::2] = b2.astype(np.uint8)
    return pairs.tobytes()


def encode_layers(tensor: ArrayLike, edges: ArrayLike) -> list[bytes]:
    """Encode each row of a `[layers, hidden_dim]` tensor independently."""
    t = _as_float32_numpy(tensor)
    if t.ndim != 2:
        raise ValueError(f"expected 2-D [layers, hidden_dim] tensor, got shape {t.shape}")
    return [encode_vector(t[i], edges) for i in range(t.shape[0])]


# --- Edge-generation helpers -------------------------------------------------


def linear_edges(lo: float, hi: float) -> np.ndarray:
    """`N_EDGES` evenly spaced edges spanning `[lo, hi]`."""
    if not hi > lo:
        raise ValueError(f"hi ({hi}) must be > lo ({lo})")
    return np.linspace(lo, hi, N_EDGES, dtype=np.float64)


def quantile_edges(samples: ArrayLike) -> np.ndarray:
    """Edges placed at the empirical quantiles of `samples`.

    Equal-mass bucketing — robust to heavy-tailed activation distributions.
    Duplicate quantiles (e.g. from large constant regions) are perturbed by a
    tiny epsilon so the strict-monotonic invariant holds.
    """
    s = _as_float32_numpy(samples).reshape(-1).astype(np.float64)
    if s.size < N_EDGES:
        raise ValueError(f"need at least {N_EDGES} samples for quantile edges, got {s.size}")
    qs = np.linspace(0.0, 1.0, N_EDGES)
    edges = np.quantile(s, qs)
    # Break ties so strict monotonicity holds.
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

    # Edges uniform in symlog space, then mapped back.
    a, b = fwd(np.array([lo, hi], dtype=np.float64))
    u = np.linspace(a, b, N_EDGES)

    def inv(u: np.ndarray) -> np.ndarray:
        sign = np.sign(u)
        absu = np.abs(u)
        linear = absu <= 1.0
        out = np.where(linear, absu * linthresh, linthresh * 10.0 ** (absu - 1.0))
        return sign * out

    edges = inv(u)
    # Same tie-break as quantile_edges, in case of float collisions near 0.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges
