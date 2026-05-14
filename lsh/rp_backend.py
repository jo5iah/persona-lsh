"""Sign-bit Random Projection LSH (the SimHash construction).

For each random direction `r` drawn from a standard Gaussian in `R^d`, the
sign of `v @ r` gives one bit of the hash. With `n_bits` independent
directions, the Hamming distance between two hashes is an unbiased
estimator of the angle between the original vectors:

    E[hamming(hash(a), hash(b))] = n_bits * theta(a, b) / pi

So normalized Hamming distance is a direction-preserving surrogate for
angular distance: vectors with cosine ~+1 hash to near-identical bit
patterns (small Hamming), orthogonal vectors hash to ~50% differing bits
(Hamming ~ n_bits/2), and antiparallel vectors hash to fully inverted
patterns (Hamming ~ n_bits).

This is the natural LSH for the persona-vectors task because the persona
direction is what cosine recovers, and each sign bit is a Bernoulli draw
whose probability is set by that angle -- contributions from every input
dimension are weighted by the random projection rather than uniformly.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import ArrayLike, LSHBackend


def _to_float32_1d(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(x, dtype=np.float32)


class RandomProjectionBackend(LSHBackend):
    """Sign-bit LSH over a fixed random Gaussian projection.

    Args:
        dim: Input vector dimensionality. Every vector passed to
            `hash_vector` must have exactly `dim` elements.
        n_bits: Output digest length in bits; must be a multiple of 8.
            Default 256 gives angular resolution ~pi / 256 ~= 0.7 deg per
            Hamming unit, which is plenty for activation-space comparisons.
        seed: Determines the random projection matrix. Two backends with
            the same `(dim, n_bits, seed)` produce comparable digests; with
            different seeds they do not.

    Digest format: lowercase hex string of length `n_bits / 4`. No prefix —
    digests are not meant to be parsed or compared as strings; use
    `distance(a, b)` for comparison.
    """

    def __init__(self, dim: int, n_bits: int = 256, seed: int = 42):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_bits < 8 or n_bits % 8 != 0:
            raise ValueError(f"n_bits must be a positive multiple of 8, got {n_bits}")
        self.dim = int(dim)
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        # Gaussian entries are standard for sign-LSH — gives the angular
        # equivalence above. Casting to float32 keeps memory tight and is
        # adequate since we only need the sign of the projection.
        self.projection = rng.standard_normal((self.dim, self.n_bits)).astype(np.float32)

    def hash_vector(self, vector: ArrayLike) -> str:
        v = _to_float32_1d(vector)
        if v.ndim != 1:
            raise ValueError(f"expected 1-D vector, got shape {v.shape}")
        if v.size != self.dim:
            raise ValueError(
                f"vector has {v.size} dims, backend expects {self.dim}"
            )
        projected = v @ self.projection  # [n_bits]
        bits = (projected > 0).astype(np.uint8)
        packed = np.packbits(bits, bitorder="big")
        return packed.tobytes().hex()

    def distance(self, digest_a: str, digest_b: str) -> float:
        """Normalized Hamming distance, in `[0, 1]`."""
        n_hex_expected = self.n_bits // 4
        if len(digest_a) != n_hex_expected or len(digest_b) != n_hex_expected:
            raise ValueError(
                f"digests must be {n_hex_expected} hex chars (for {self.n_bits} bits); "
                f"got {len(digest_a)} and {len(digest_b)}"
            )
        bytes_a = np.frombuffer(bytes.fromhex(digest_a), dtype=np.uint8)
        bytes_b = np.frombuffer(bytes.fromhex(digest_b), dtype=np.uint8)
        # XOR per-byte, popcount via unpackbits + sum.
        xor = np.bitwise_xor(bytes_a, bytes_b)
        unpacked = np.unpackbits(xor, bitorder="big")
        hamming = int(unpacked.sum())
        return hamming / self.n_bits
