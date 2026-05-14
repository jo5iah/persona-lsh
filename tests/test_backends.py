"""Tests for the post-refactor pluggable interfaces: ByteEncoder + LSHBackend.

These don't replace the existing encoding/LSH tests — they validate the
abstract interfaces and that the default TLSH + BucketFractional pipeline
composes correctly. Future encoders (alternative TLSH byte schemes) and
backends (Random-Projection LSH, ...) should add their own tests but rely on
the interfaces these tests exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoding import (  # noqa: E402
    BucketEdgesEncoder,
    BucketFractionalEncoder,
    ByteEncoder,
    linear_edges,
    quantile_edges,
)
from lsh import LSHBackend, TLSHBackend  # noqa: E402


# --- ByteEncoder interface ---------------------------------------------------


def test_bucket_fractional_is_a_byte_encoder():
    enc = BucketFractionalEncoder(linear_edges(-1.0, 1.0))
    assert isinstance(enc, ByteEncoder)
    assert isinstance(enc, BucketEdgesEncoder)


def test_bucket_fractional_encode_vector_returns_bytes_of_expected_length():
    enc = BucketFractionalEncoder(linear_edges(-3.0, 3.0))
    out = enc.encode_vector(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert isinstance(out, bytes)
    assert len(out) == 6  # 2 bytes per scalar


def test_bucket_fractional_encode_layers_handles_2d_tensor():
    enc = BucketFractionalEncoder(linear_edges(-3.0, 3.0))
    tensor = torch.randn(4, 1024)
    layers = enc.encode_layers(tensor)
    assert len(layers) == 4
    assert all(isinstance(b, bytes) and len(b) == 2048 for b in layers)


def test_bucket_fractional_round_trips_through_module_level_helper():
    """The module-level `encode_vector` must produce identical bytes to a
    directly-constructed BucketFractionalEncoder.encode_vector(). This guards
    the backward-compat shim."""
    from encoding import encode_vector as module_level_encode

    edges = linear_edges(-2.0, 2.0)
    v = np.array([0.1, -0.7, 1.3], dtype=np.float32)
    direct = BucketFractionalEncoder(edges).encode_vector(v)
    via_module = module_level_encode(v, edges)
    assert direct == via_module


def test_encoder_rejects_non_1d_vector_input():
    enc = BucketFractionalEncoder(linear_edges(-1.0, 1.0))
    with pytest.raises(ValueError, match="1-D"):
        enc.encode_vector(np.zeros((2, 3), dtype=np.float32))


def test_bucket_edges_encoder_validates_edges_at_construction():
    bad = np.linspace(0, 1, 100)  # wrong length
    with pytest.raises(ValueError, match="length 257"):
        BucketFractionalEncoder(bad)

    nonmonotonic = np.linspace(0, 1, 257)
    nonmonotonic[100] = nonmonotonic[99]
    with pytest.raises(ValueError, match="monotonically"):
        BucketFractionalEncoder(nonmonotonic)


# --- LSHBackend interface ----------------------------------------------------


def test_tlsh_backend_is_an_lsh_backend():
    backend = TLSHBackend(BucketFractionalEncoder(linear_edges(-3.0, 3.0)))
    assert isinstance(backend, LSHBackend)


def test_tlsh_backend_hashes_vector_to_t2_digest():
    edges = linear_edges(-3.0, 3.0)
    backend = TLSHBackend(BucketFractionalEncoder(edges))
    v = torch.randn(2048, generator=torch.Generator().manual_seed(0)).float()
    digest = backend.hash_vector(v)
    assert digest.startswith("T2"), f"expected T2 prefix, got {digest[:4]!r}"
    assert len(digest) == 72


def test_tlsh_backend_self_distance_is_zero():
    edges = linear_edges(-3.0, 3.0)
    backend = TLSHBackend(BucketFractionalEncoder(edges))
    v = torch.randn(2048, generator=torch.Generator().manual_seed(1)).float()
    d = backend.hash_vector(v)
    assert backend.distance(d, d) == 0


def test_tlsh_backend_hash_layers_returns_one_digest_per_row():
    edges = linear_edges(-3.0, 3.0)
    backend = TLSHBackend(BucketFractionalEncoder(edges))
    t = torch.randn(5, 2048, generator=torch.Generator().manual_seed(2)).float()
    digests = backend.hash_layers(t)
    assert len(digests) == 5
    assert all(d.startswith("T2") for d in digests)


def test_tlsh_backend_pairwise_distance_is_symmetric_and_zero_diagonal():
    edges = linear_edges(-3.0, 3.0)
    backend = TLSHBackend(BucketFractionalEncoder(edges))
    g = torch.Generator().manual_seed(3)
    digests = [backend.hash_vector(torch.randn(2048, generator=g).float()) for _ in range(4)]
    m = backend.pairwise_distance(digests)
    assert m.shape == (4, 4)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_backend_composes_with_calibration_edges():
    """End-to-end smoke: build edges once, build a backend per layer, hash a
    multi-layer tensor's rows -- the production pattern in save_lsh_sidecar."""
    from encoding import calibrate_edges_per_layer

    g = torch.Generator().manual_seed(4)
    tensors = [torch.randn(6, 2048, generator=g).float() for _ in range(3)]
    edges_per_layer = calibrate_edges_per_layer(tensors, "quantile")
    encoders = [BucketFractionalEncoder(e) for e in edges_per_layer]
    backends = [TLSHBackend(enc) for enc in encoders]

    digests = [backends[i].hash_vector(tensors[0][i]) for i in range(6)]
    assert all(d.startswith("T2") for d in digests)
    assert all(len(d) == 72 for d in digests)


# --- Module-level backward-compat helpers ------------------------------------


def test_module_level_hash_vector_matches_explicit_backend():
    """`lsh.hash_vector(v, edges)` must return the same digest as
    `TLSHBackend(BucketFractionalEncoder(edges)).hash_vector(v)`. Otherwise the
    backward-compat shim is silently wrong."""
    from lsh import hash_vector as module_level_hash

    edges = linear_edges(-3.0, 3.0)
    v = torch.randn(2048, generator=torch.Generator().manual_seed(5)).float()
    backend_direct = TLSHBackend(BucketFractionalEncoder(edges)).hash_vector(v)
    via_module = module_level_hash(v, edges)
    assert backend_direct == via_module


def test_module_level_diff_matches_backend_distance():
    """`lsh.diff(a, b)` must equal `TLSHBackend(...).distance(a, b)` (int vs
    float aside) for any two digests."""
    from lsh import diff as module_level_diff

    edges = linear_edges(-3.0, 3.0)
    backend = TLSHBackend(BucketFractionalEncoder(edges))
    g = torch.Generator().manual_seed(6)
    da = backend.hash_vector(torch.randn(2048, generator=g).float())
    db = backend.hash_vector(torch.randn(2048, generator=g).float())
    assert module_level_diff(da, db) == int(backend.distance(da, db))
