"""Top-k masked encoder: zero out coords below the top-k by magnitude."""
from __future__ import annotations

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class TopKMaskedEncoder(BucketEdgesEncoder):
    """2 bytes per scalar, but coords whose `|v|` is below the top-k are zeroed.

    For each scalar, if `|v|` ranks in the top `top_frac` of the vector by
    magnitude, emit `(bucket_idx, round(255 * frac))` as `BucketFractional`
    would. Otherwise emit `(0, 0)`.

    Targets the noise-dilution failure mode: when only ~3% of activation dims
    carry the persona signal, the 97% of "noise" dims overwhelm TLSH's n-gram
    statistics. Zeroing them concentrates TLSH's attention on the signal-
    bearing positions.

    Note: a stream of mostly-zero bytes has reduced byte-value diversity;
    TLSH may emit `TNULL` if the vector is too small or too sparse. Sensible
    defaults: `top_frac` in [0.05, 0.20] for thousand-dim activations.
    """

    def __init__(self, edges: ArrayLike, top_frac: float = 0.10):
        super().__init__(edges)
        if not (0.0 < top_frac <= 1.0):
            raise ValueError(f"top_frac must be in (0, 1], got {top_frac}")
        self.top_frac = float(top_frac)

    def encode_vector(self, vector: ArrayLike) -> bytes:
        v = _as_float32_numpy(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        e = self.edges
        n = v.size
        k = max(1, int(round(self.top_frac * n)))

        abs_v = np.abs(v)
        if k >= n:
            keep_mask = np.ones(n, dtype=bool)
        else:
            # The kth largest |v| is at the (n - k)th position after partition.
            threshold = np.partition(abs_v, n - k)[n - k]
            keep_mask = abs_v >= threshold

        b1 = np.clip(np.searchsorted(e, v, side="right") - 1, 0, N_BUCKETS - 1)
        lo = e[b1]
        hi = e[b1 + 1]
        width = hi - lo
        frac = (v.astype(np.float64) - lo) / width
        b2 = np.clip(np.rint(255.0 * frac), 0, N_BUCKETS - 1)

        b1 = np.where(keep_mask, b1, 0).astype(np.uint8)
        b2 = np.where(keep_mask, b2, 0).astype(np.uint8)

        pairs = np.empty(n * 2, dtype=np.uint8)
        pairs[0::2] = b1
        pairs[1::2] = b2
        return pairs.tobytes()
