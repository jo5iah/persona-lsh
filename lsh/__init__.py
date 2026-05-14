"""LSH backends.

Currently exports:
  - `LSHBackend`              : abstract base class
  - `RandomProjectionBackend` : sign-bit / SimHash LSH
  - `ArrayLike`               : type alias

The package is shaped to support additional LSH variants behind the same
ABC -- a new backend subclasses `LSHBackend`, implements `hash_vector` and
`distance`, and slots into `demo/bench.py` by being added to its
`backend_specs` list. No callers need to change.
"""
from __future__ import annotations

from .base import ArrayLike, LSHBackend
from .rp_backend import RandomProjectionBackend

__all__ = ["ArrayLike", "LSHBackend", "RandomProjectionBackend"]
