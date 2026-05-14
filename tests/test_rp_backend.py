"""Tests for the Random-Projection LSH backend.

These pin both the basic interface contract (it's an LSHBackend) and the
statistical properties that make sign-bit LSH direction-preserving:
self-distance is 0, antiparallel vectors are at distance 1, scale-invariance,
and orthogonal vectors are near 0.5 with concentration that tightens as
n_bits grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lsh import LSHBackend, RandomProjectionBackend  # noqa: E402


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# --- Interface conformance ---------------------------------------------------


def test_rp_is_an_lsh_backend():
    rp = RandomProjectionBackend(dim=32, n_bits=64, seed=0)
    assert isinstance(rp, LSHBackend)


def test_rp_rejects_invalid_n_bits():
    with pytest.raises(ValueError, match="multiple of 8"):
        RandomProjectionBackend(dim=64, n_bits=7)
    with pytest.raises(ValueError, match="multiple of 8"):
        RandomProjectionBackend(dim=64, n_bits=0)


def test_rp_rejects_invalid_dim():
    with pytest.raises(ValueError, match="dim"):
        RandomProjectionBackend(dim=0, n_bits=64)


def test_rp_rejects_vector_with_wrong_dim():
    rp = RandomProjectionBackend(dim=64, n_bits=64, seed=0)
    with pytest.raises(ValueError, match="dims"):
        rp.hash_vector(np.zeros(32))


def test_rp_rejects_non_1d_input():
    rp = RandomProjectionBackend(dim=64, n_bits=64, seed=0)
    with pytest.raises(ValueError, match="1-D"):
        rp.hash_vector(np.zeros((2, 32)))


def test_rp_digest_is_expected_hex_length():
    rp = RandomProjectionBackend(dim=64, n_bits=256, seed=0)
    d = rp.hash_vector(torch.randn(64, generator=_gen(1)).float())
    assert len(d) == 64  # 256 bits / 4 hex per byte = 64 hex chars
    # All hex chars.
    int(d, 16)  # raises if not hex


# --- Statistical properties --------------------------------------------------


def test_rp_self_distance_is_zero():
    rp = RandomProjectionBackend(dim=128, n_bits=64, seed=0)
    v = torch.randn(128, generator=_gen(2)).float()
    d = rp.hash_vector(v)
    assert rp.distance(d, d) == 0.0


def test_rp_distance_in_unit_interval():
    rp = RandomProjectionBackend(dim=128, n_bits=128, seed=0)
    g = _gen(3)
    digests = [rp.hash_vector(torch.randn(128, generator=g).float()) for _ in range(5)]
    for i in range(5):
        for j in range(5):
            assert 0.0 <= rp.distance(digests[i], digests[j]) <= 1.0


def test_rp_negative_vector_distance_is_one():
    """sign(-v @ r) = -sign(v @ r) for every r, so all n_bits flip."""
    rp = RandomProjectionBackend(dim=128, n_bits=256, seed=0)
    v = torch.randn(128, generator=_gen(4)).float()
    assert rp.distance(rp.hash_vector(v), rp.hash_vector(-v)) == 1.0


def test_rp_positive_scaling_invariant():
    """sign(alpha v @ r) = sign(v @ r) for alpha > 0 — digest must not change."""
    rp = RandomProjectionBackend(dim=128, n_bits=64, seed=0)
    v = torch.randn(128, generator=_gen(5)).float()
    assert rp.hash_vector(v) == rp.hash_vector(2.5 * v)
    assert rp.hash_vector(v) == rp.hash_vector(0.01 * v)
    assert rp.hash_vector(v) == rp.hash_vector(100.0 * v)


def test_rp_orthogonal_random_vectors_distance_near_half():
    """Two independent standard-Gaussian vectors are near-orthogonal in high
    dimension. Their normalized Hamming distance is ~0.5 by the angular
    equivalence; with 1024 bits we expect tight concentration (std ~ 0.016)."""
    rp = RandomProjectionBackend(dim=512, n_bits=1024, seed=0)
    g = _gen(6)
    a = torch.randn(512, generator=g).float()
    b = torch.randn(512, generator=g).float()
    dist = rp.distance(rp.hash_vector(a), rp.hash_vector(b))
    assert 0.45 < dist < 0.55, f"expected ~0.5, got {dist}"


def test_rp_near_duplicate_smaller_distance_than_random_pair():
    rp = RandomProjectionBackend(dim=512, n_bits=512, seed=0)
    g = _gen(7)
    v = torch.randn(512, generator=g).float()
    v_close = v + 0.001 * torch.randn(512, generator=g).float()
    v_far = torch.randn(512, generator=g).float()
    d_close = rp.distance(rp.hash_vector(v), rp.hash_vector(v_close))
    d_far = rp.distance(rp.hash_vector(v), rp.hash_vector(v_far))
    assert d_close < d_far


def test_rp_cosine_angular_relationship():
    """Hamming-distance / n_bits estimates theta / pi where theta is the
    angle. So for a known cosine, the distance should land near the
    predicted value. We check this with a controlled-angle pair."""
    rp = RandomProjectionBackend(dim=256, n_bits=4096, seed=0)
    g = _gen(8)
    # Construct two unit vectors with cosine = 0.5 -> theta = pi/3 -> dist ~ 1/3.
    a = torch.randn(256, generator=g).float()
    a = a / a.norm()
    # Orthogonal direction in the same space.
    perp = torch.randn(256, generator=g).float()
    perp = perp - (perp @ a) * a
    perp = perp / perp.norm()
    b = 0.5 * a + (3.0 ** 0.5 / 2.0) * perp  # cos(60deg) = 0.5
    assert abs((a @ b).item() - 0.5) < 1e-5

    dist = rp.distance(rp.hash_vector(a), rp.hash_vector(b))
    # Expected ~ 1/3 = 0.333; with 4096 bits the std is ~0.014, so a generous
    # tolerance.
    assert 0.28 < dist < 0.39, f"expected ~0.333, got {dist}"


# --- Backend determinism + composition --------------------------------------


def test_rp_seed_determines_digest():
    """Same (dim, n_bits, seed) yields identical projection matrices and so
    identical digests; different seeds usually give different digests."""
    v = torch.randn(64, generator=_gen(9)).float()
    rp_a = RandomProjectionBackend(dim=64, n_bits=64, seed=42)
    rp_b = RandomProjectionBackend(dim=64, n_bits=64, seed=42)
    rp_c = RandomProjectionBackend(dim=64, n_bits=64, seed=99)
    assert rp_a.hash_vector(v) == rp_b.hash_vector(v)
    assert rp_a.hash_vector(v) != rp_c.hash_vector(v)


def test_rp_hash_layers_returns_per_row_digests():
    rp = RandomProjectionBackend(dim=128, n_bits=64, seed=0)
    t = torch.randn(5, 128, generator=_gen(10)).float()
    digests = rp.hash_layers(t)
    assert len(digests) == 5
    assert all(isinstance(d, str) and len(d) == 16 for d in digests)


def test_rp_pairwise_distance_symmetric_zero_diagonal():
    rp = RandomProjectionBackend(dim=64, n_bits=64, seed=0)
    g = _gen(11)
    digests = [rp.hash_vector(torch.randn(64, generator=g).float()) for _ in range(4)]
    m = rp.pairwise_distance(digests)
    assert m.shape == (4, 4)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_rp_distance_rejects_mismatched_digest_lengths():
    rp = RandomProjectionBackend(dim=64, n_bits=64, seed=0)
    good = rp.hash_vector(torch.randn(64, generator=_gen(12)).float())
    with pytest.raises(ValueError, match="hex chars"):
        rp.distance(good, "deadbeef")
