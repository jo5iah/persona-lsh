"""TLSH backend: byte-encode a vector then run it through the vendored TLSH.

The vendored TLSH lives under `persona_vectors/vendor/tlsh`. It is a fork:
T2 digests are produced (the upstream T1 prefix is rejected on read), and
the threaded bucket-merge bug in `fast_update5` has been fixed. See the
project memory `project-persona-lsh` for details.
"""
from __future__ import annotations

from encoding import ArrayLike, ByteEncoder

try:
    import tlsh as _tlsh
except ImportError as e:
    raise ImportError(
        "The `tlsh` extension is not installed. Build the vendored copy with: "
        "`cd persona_vectors/vendor/tlsh/py_ext && pip install .`"
    ) from e

from .base import LSHBackend


# Re-export for backward-compat module-level helpers in `lsh/__init__.py`.
tlsh_module = _tlsh


class TLSHBackend(LSHBackend):
    """Run TLSH over the byte stream produced by a `ByteEncoder`."""

    def __init__(self, encoder: ByteEncoder):
        self.encoder = encoder

    def hash_vector(self, vector: ArrayLike) -> str:
        return _tlsh.hash(self.encoder.encode_vector(vector))

    def distance(self, digest_a: str, digest_b: str) -> float:
        return float(_tlsh.diff(digest_a, digest_b))
