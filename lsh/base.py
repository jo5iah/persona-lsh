"""Abstract base class for LSH backends.

An `LSHBackend` maps a 1-D float vector to a string digest and implements a
distance function over digests. Concrete implementations decide how the
hashing is done (random-projection signs today; future MinHash, LSH-Forest,
or other variants subclass the same ABC so they slot into the bench harness).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]


class LSHBackend(ABC):
    """Hash 1-D float vectors to digest strings and compare them."""

    @abstractmethod
    def hash_vector(self, vector: ArrayLike) -> str:
        """Hash a 1-D vector to a digest string."""

    @abstractmethod
    def distance(self, digest_a: str, digest_b: str) -> float:
        """Distance between two digests (0 = identical, larger = more different)."""

    def hash_layers(self, tensor) -> list[str]:
        """Default: hash each row of a 2-D `[layers, hidden_dim]` tensor.

        Override only if the backend does cross-layer hashing."""
        return [self.hash_vector(tensor[i]) for i in range(tensor.shape[0])]

    def pairwise_distance(self, digests: Iterable[str]) -> np.ndarray:
        """Symmetric NxN matrix of pairwise digest distances."""
        ds = list(digests)
        n = len(ds)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(ds[i], ds[j])
                matrix[i, j] = d
                matrix[j, i] = d
        return matrix
