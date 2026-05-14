"""Single-byte bucket encoder (one byte per scalar = bucket index only)."""
from __future__ import annotations

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class BucketSingleByteEncoder(BucketEdgesEncoder):
    """One byte per scalar: just the bucket index `[0, 255]`.

    Drops the within-bucket fractional position that `BucketFractionalEncoder`
    emits as byte-2. The hope is that halving the byte stream length makes the
    signal-bearing bytes proportionally more visible to TLSH's sliding-window
    n-gram statistics, when most of the dimensions are noise.
    """

    def encode_vector(self, vector: ArrayLike) -> bytes:
        v = _as_float32_numpy(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        b1 = np.clip(
            np.searchsorted(self.edges, v, side="right") - 1, 0, N_BUCKETS - 1
        ).astype(np.uint8)
        return b1.tobytes()
