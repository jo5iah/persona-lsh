"""Projection seeds derived from the model: a base seed from its Hugging Face metadata, plus a random offset.

RP-LSH digests are comparable only when they come from the same projection matrix, which
`RandomProjectionBackend` builds from `(dim, n_bits, seed)`. A fixed `seed=42` works inside one notebook but
ties every model and every experiment to one arbitrary number. This module ties the seed to the model instead:

    fingerprint = SHA-256 of the model's full HF metadata (every file in its snapshot, by name and content id)
    base        = SHA-256(salt, fingerprint) mod RANGE        (no salt: the fingerprint itself, mod RANGE)
    seed        = (base + plus_rand) mod RANGE

Three layers, each with its own job:
- **fingerprint**: binds the seed to the exact checkpoint. Same files, same fingerprint, on any machine.
- **salt**: one secret shared by every model and experiment in a project or deployment. Seeds stay fixed per
  model (so experiments line up), but without the salt nobody can compute them from public metadata. An
  empty salt gives the public canonical seeds, for an interoperable standard. The salt lives outside the repo
  (`PERSONA_LSH_SALT` or `~/.config/persona-lsh/salt`); results record only `salt_id`.
- **plus_rand**: a per-experiment or per-replicate offset. `plus_rand = 0` is the model's seed under the salt.
  Consecutive offsets (`replicate_seeds`) give independent projections, to measure how much an accuracy
  depends on the hyperplanes drawn. On its own, plus_rand is a weak key: the offset is linear, so anyone who
  knows the base and sees one seed recovers it. Use the salt for secrecy and plus_rand for variation.

The fingerprint doubles as a check that two experiments ran the same weights: any change to any file in the
repo (weights, config, tokenizer, chat template, even the README) gives a new fingerprint and a new base seed.
Use `scope="weights"` for a seed that survives documentation-only commits.

Offline by design (the rig loads with `HF_HUB_OFFLINE=1`): it reads the Hugging Face cache layout, where each
snapshot file is a link to `blobs/<id>`, and the id is the file's SHA-256 for LFS files (the weights) and its
git blob SHA-1 for small files. No weights are read. A plain directory (not a cache snapshot) is hashed by
content, which reads every file once.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

RANGE = 2**32   # numpy, torch.manual_seed and most other RNGs take a 32-bit seed
VERSION = 1     # bump if the fingerprint recipe changes, so old and new seeds can't be confused

WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
WEIGHT_NAMES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json", "chat_template.json", "chat_template.jinja",
                "model.safetensors.index.json", "pytorch_model.bin.index.json"}


def _repo_id(snapshot: Path) -> str | None:
    # .../models--org--name/snapshots/<commit> -> "org/name"
    parts = snapshot.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return part[len("models--"):].replace("--", "/")
    return None


def _content_id(path: Path) -> str:
    # The cache blob's name when `path` links into blobs/, else the SHA-256 of the content.
    if path.is_symlink():
        target = Path(os.readlink(path))
        if target.parent.name == "blobs":
            return target.name
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _in_scope(name: str, scope: str) -> bool:
    if scope == "full":
        return True
    if scope == "weights":
        base = name.rsplit("/", 1)[-1]
        return base in WEIGHT_NAMES or base.endswith(WEIGHT_FILES)
    raise ValueError(f"scope must be 'full' or 'weights', got {scope!r}")


def model_fingerprint(snapshot, repo_id: str | None = None, scope: str = "full") -> dict:
    """The model's metadata as a canonical record, and its SHA-256.

    `snapshot` is a local snapshot directory (`huggingface_hub.snapshot_download(repo, local_files_only=True)`).
    Returns {"repo", "scope", "version", "files": {relative path: content id}, "sha256"}. The commit hash is
    left out on purpose: the content ids already pin every file, and a commit that changes nothing in scope
    shouldn't change the seed."""
    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"not a directory: {snapshot}")
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        dangling = path.is_symlink() and not path.exists()
        if dangling and Path(os.readlink(path)).parent.name != "blobs":
            continue   # a broken link outside the cache layout: nothing to identify it by
        rel = path.relative_to(snapshot).as_posix()
        if rel.startswith(".") or not _in_scope(rel, scope):
            continue
        files[rel] = _content_id(path)
    if not files:
        raise ValueError(f"no files in scope {scope!r} under {snapshot}")
    record = {"repo": repo_id or _repo_id(snapshot), "scope": scope, "version": VERSION, "files": files}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return record | {"sha256": hashlib.sha256(canonical).hexdigest()}


SALT_ENV = "PERSONA_LSH_SALT"
SALT_FILE = Path.home() / ".config" / "persona-lsh" / "salt"


def load_salt(salt: str | bytes | None = None) -> bytes:
    """The common salt, shared by every model's seed in a project or deployment.

    In order: `salt` if given; the `PERSONA_LSH_SALT` environment variable; `~/.config/persona-lsh/salt`;
    else empty (the public, unsalted seeds). Keep the salt out of every repo and every results file: record
    `salt_id(salt)` instead."""
    if salt is None:
        salt = os.environ.get(SALT_ENV)
    if salt is None and SALT_FILE.is_file():
        salt = SALT_FILE.read_text().strip()
    if salt is None:
        return b""
    return salt if isinstance(salt, bytes) else str(salt).encode()


def new_salt() -> str:
    """A fresh 256-bit salt as hex, for `~/.config/persona-lsh/salt` (chmod 600)."""
    return secrets.token_hex(32)


def salt_id(salt: bytes) -> str:
    """A short public label for a salt, safe to record with results: "none" if unsalted."""
    return hashlib.sha256(b"persona-lsh salt id\0" + salt).hexdigest()[:12] if salt else "none"


def base_seed(fingerprint, mod: int = RANGE, salt: bytes = b"") -> int:
    """The model's seed before any offset: SHA-256 of (salt, the fingerprint's SHA-256), read as an integer,
    mod `mod`. `fingerprint` is a record from `model_fingerprint` or its hex string.

    With an empty salt this is the public canonical seed. With the project's salt, every model's seed is still
    fixed by its metadata (the same across experiments), but nobody without the salt can compute it."""
    digest = fingerprint["sha256"] if isinstance(fingerprint, dict) else str(fingerprint)
    if not salt:
        return int(digest, 16) % mod
    salted = hashlib.sha256(b"persona-lsh seed v%d\0" % VERSION + salt + b"\0" + bytes.fromhex(digest))
    return int(salted.hexdigest(), 16) % mod


def draw_plus_rand(mod: int = RANGE) -> int:
    """A fresh random offset in [1, mod): the key of a keyed projection. Record it with the results."""
    return 1 + secrets.randbelow(mod - 1)


def projection_seed(base: int, plus_rand: int = 0, mod: int = RANGE) -> int:
    """(base + plus_rand) mod `mod`. plus_rand = 0 is the canonical seed."""
    if mod < 2:
        raise ValueError(f"mod must be at least 2, got {mod}")
    return (int(base) + int(plus_rand)) % mod


def replicate_seeds(base: int, n: int, plus_rand: int = 0, mod: int = RANGE) -> list[int]:
    """n projection seeds for one model: plus_rand, plus_rand + 1, ... added to base, mod `mod`.

    The first is `projection_seed(base, plus_rand)`. Use them to repeat an RP-LSH result over independent
    hyperplanes."""
    return [projection_seed(base, plus_rand + k, mod) for k in range(n)]


def model_seed(snapshot, plus_rand: int = 0, scope: str = "full", mod: int = RANGE, repo_id: str | None = None,
               salt: str | bytes | None = None) -> dict:
    """Everything to record for one model's projection: {"repo", "fingerprint", "scope", "salt id", "base",
    "plus_rand", "mod", "seed", "files"}. `seed` goes to `RandomProjectionBackend(dim, n_bits, seed=...)`.

    `salt` as in `load_salt` (None reads the environment or the salt file). The record holds the salt's id,
    never the salt. With a salt, `base` is secret too: record it only where the salt could be."""
    fp = model_fingerprint(snapshot, repo_id=repo_id, scope=scope)
    salt = load_salt(salt)
    base = base_seed(fp, mod, salt)
    return {"repo": fp["repo"], "fingerprint": fp["sha256"], "scope": scope, "version": VERSION,
            "salt id": salt_id(salt), "base": base, "plus_rand": int(plus_rand), "mod": mod,
            "seed": projection_seed(base, plus_rand, mod), "files": len(fp["files"])}
