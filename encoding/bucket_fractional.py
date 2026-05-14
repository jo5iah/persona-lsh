"""Bucket-index + fractional-position byte encoder (the current default).

For each scalar `v`:
- byte-1 = bucket index in `[0, 255]`
- byte-2 = round(255 * (v - lo) / (hi - lo))   (fractional position in bucket)

See `encoding/base.py` for the ByteEncoder ABC and `encoding/edges.py` for
edge generators.
"""
from __future__ import annotations

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class BucketFractionalEncoder(BucketEdgesEncoder):
    """byte-1 = bucket index, byte-2 = fractional position within the bucket.

    Two nearby floats yield nearby `(b1, b2)` byte pairs, so TLSH n-gram
    statistics over the concatenated stream respect float distance.
    """

    def encode_vector(self, vector: ArrayLike) -> bytes:
        v = _as_float32_numpy(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        e = self.edges

        # Bucket index in [0, 255].
        b1 = np.clip(
            np.searchsorted(e, v, side="right") - 1, 0, N_BUCKETS - 1
        ).astype(np.int32)

        lo = e[b1]
        hi = e[b1 + 1]
        width = hi - lo
        frac = (v.astype(np.float64) - lo) / width
        b2 = np.clip(np.rint(255.0 * frac), 0, N_BUCKETS - 1).astype(np.int32)

        pairs = np.empty(v.size * 2, dtype=np.uint8)
        pairs[0::2] = b1.astype(np.uint8)
        pairs[1::2] = b2.astype(np.uint8)
        return pairs.tobytes()
