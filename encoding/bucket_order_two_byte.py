"""BucketOrderTwoByte encoder: 256-ile bucket + neighbor-lean byte-2."""
from __future__ import annotations

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class BucketOrderTwoByteEncoder(BucketEdgesEncoder):
    """2 bytes per scalar: bucket index + nearest-edge neighbor index.

    For each scalar `v` in bucket `b`:
    - middle ~67% of the bucket           -> byte-2 = b      (no lean)
    - lower ~16.5% (closer to left edge)  -> byte-2 = b - 1
    - upper ~16.5% (closer to right edge) -> byte-2 = b + 1
    Clamped to `[0, 255]` at the extremes.

    Designed for 256-ile (`quantile_edges`) bucketing, where every bucket
    holds equal probability mass -- so the middle/edge distinction has
    consistent probabilistic meaning across all buckets. With other edge
    schemes the encoder still works but the 67% threshold no longer
    corresponds to 67% of the value mass per bucket.

    Compared to the original 25/50/25 neighbor-index scheme (the pre-fractional
    encoder), this widens the "middle, no neighbor" band to 67% so that small
    in-bucket perturbations are less likely to flip byte-2 to a neighboring
    bucket index.
    """

    MIDDLE_FRAC = 0.67

    def encode_vector(self, vector: ArrayLike) -> bytes:
        v = _as_float32_numpy(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        e = self.edges

        b1 = np.clip(
            np.searchsorted(e, v, side="right") - 1, 0, N_BUCKETS - 1
        ).astype(np.int32)
        lo = e[b1]
        hi = e[b1 + 1]
        width = hi - lo
        frac = (v.astype(np.float64) - lo) / width

        lower_thresh = (1.0 - self.MIDDLE_FRAC) / 2.0  # ~0.165
        upper_thresh = 1.0 - lower_thresh              # ~0.835
        offset = np.where(frac < lower_thresh, -1, np.where(frac >= upper_thresh, 1, 0))
        b2 = np.clip(b1 + offset, 0, N_BUCKETS - 1).astype(np.int32)

        pairs = np.empty(v.size * 2, dtype=np.uint8)
        pairs[0::2] = b1.astype(np.uint8)
        pairs[1::2] = b2.astype(np.uint8)
        return pairs.tobytes()
