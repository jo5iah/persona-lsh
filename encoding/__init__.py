"""Byte encoders for converting float vectors to byte streams for fuzzy hashing.

Public API (backward-compatible with the pre-refactor flat `encoding.py`):

  - `ByteEncoder`, `BucketEdgesEncoder`             (abstract base classes)
  - `BucketFractionalEncoder`                       (current default encoder)
  - `linear_edges`, `quantile_edges`, `symlog_edges` (edge generators)
  - `calibrate_edges_per_layer`                     (per-layer edge calibration)
  - `encode_vector`, `encode_layers`                (module-level helpers
                                                     using BucketFractional)
  - `N_BUCKETS`, `N_EDGES`, `ArrayLike`             (constants and type alias)

New encoders should subclass `ByteEncoder` (or `BucketEdgesEncoder` if they
need 257 bucket edges) and be exported here so the pluggable LSH backends
can compose them.
"""
from __future__ import annotations

from .base import (
    ArrayLike,
    BucketEdgesEncoder,
    ByteEncoder,
    N_BUCKETS,
    N_EDGES,
)
from .bucket_fractional import BucketFractionalEncoder
from .bucket_order_two_byte import BucketOrderTwoByteEncoder
from .bucket_single import BucketSingleByteEncoder
from .canonical_order import CanonicalOrderEncoder, variance_permutation_per_layer
from .edges import (
    calibrate_edges_per_layer,
    linear_edges,
    quantile_edges,
    symlog_edges,
)
from .topk_masked import TopKMaskedEncoder


def encode_vector(vector: ArrayLike, edges: ArrayLike) -> bytes:
    """Default `BucketFractionalEncoder` encoding of a single vector."""
    return BucketFractionalEncoder(edges).encode_vector(vector)


def encode_layers(tensor: ArrayLike, edges: ArrayLike) -> list[bytes]:
    """Default `BucketFractionalEncoder` encoding of each row of a 2-D tensor."""
    return BucketFractionalEncoder(edges).encode_layers(tensor)


__all__ = [
    "ArrayLike",
    "BucketEdgesEncoder",
    "BucketFractionalEncoder",
    "BucketOrderTwoByteEncoder",
    "BucketSingleByteEncoder",
    "ByteEncoder",
    "CanonicalOrderEncoder",
    "N_BUCKETS",
    "N_EDGES",
    "TopKMaskedEncoder",
    "calibrate_edges_per_layer",
    "encode_layers",
    "encode_vector",
    "linear_edges",
    "quantile_edges",
    "symlog_edges",
    "variance_permutation_per_layer",
]
