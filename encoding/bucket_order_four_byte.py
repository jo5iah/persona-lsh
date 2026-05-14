"""BucketOrderFourByte encoder: 4 bytes per scalar with edge-lean markers.

Same `(67% middle / 16.5% / 16.5%)` thresholding as
`BucketOrderTwoByteEncoder`, but emits a four-byte fingerprint per scalar.
For each scalar `v` in bucket `B`:

  - middle ~67% of `B`              -> `(B, B, B, B)`
  - lower ~16.5% (close to `A=B-1`) -> `(A, B, B, B)`
  - upper ~16.5% (close to `C=B+1`) -> `(B, B, B, C)`

The motivation: TLSH n-gram statistics weigh every byte uniformly. By
repeating the bucket index four times we increase the signal weight per
scalar relative to the noise weight of other scalars in the n-gram window;
the edge-lean information appears at fixed positions (byte 0 for lower
lean, byte 3 for upper lean) so n-gram windows that cross a leaning value
see a stereotyped pattern they can match across vectors.
"""
from __future__ import annotations

import numpy as np

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    N_BUCKETS,
    _as_float32_numpy,
)


class BucketOrderFourByteEncoder(BucketEdgesEncoder):
    """4 bytes per scalar: bucket index repeated, neighbor lean at endpoints."""

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

        b1_u8 = b1.astype(np.uint8)
        a = np.clip(b1 - 1, 0, N_BUCKETS - 1).astype(np.uint8)  # left neighbor
        c = np.clip(b1 + 1, 0, N_BUCKETS - 1).astype(np.uint8)  # right neighbor

        is_lower = frac < lower_thresh
        is_upper = frac >= upper_thresh

        pos0 = np.where(is_lower, a, b1_u8)
        pos3 = np.where(is_upper, c, b1_u8)

        out = np.empty(v.size * 4, dtype=np.uint8)
        out[0::4] = pos0
        out[1::4] = b1_u8
        out[2::4] = b1_u8
        out[3::4] = pos3
        return out.tobytes()
