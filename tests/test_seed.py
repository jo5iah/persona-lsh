"""Offline tests of lsh/seed.py: model-derived projection seeds, on a fake Hugging Face cache."""
import os

import pytest

from lsh import RandomProjectionBackend
from lsh.seed import (RANGE, base_seed, draw_plus_rand, load_salt, model_fingerprint, model_seed, new_salt,
                      projection_seed,
                      replicate_seeds)


def fake_snapshot(root, files, repo="org/model", commit="abc123"):
    # The HF cache layout: snapshot files link to ../../blobs/<content id>.
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    blobs, snap = repo_dir / "blobs", repo_dir / "snapshots" / commit
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    for name, (blob_id, content) in files.items():
        (blobs / blob_id).write_bytes(content)
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blobs / blob_id, target.parent), target)
    return snap


FILES = {"config.json": ("sha1cfg", b"{}"), "model-00001-of-00002.safetensors": ("a" * 64, b"w1"),
         "model-00002-of-00002.safetensors": ("b" * 64, b"w2"), "README.md": ("sha1readme", b"# hi")}


def test_fingerprint_reads_blob_ids_and_repo(tmp_path):
    fp = model_fingerprint(fake_snapshot(tmp_path, FILES))
    assert fp["repo"] == "org/model"
    assert fp["files"]["model-00001-of-00002.safetensors"] == "a" * 64
    assert len(fp["sha256"]) == 64


def test_same_files_same_seed_across_commits(tmp_path):
    a = model_seed(fake_snapshot(tmp_path / "a", FILES, commit="c1"))
    b = model_seed(fake_snapshot(tmp_path / "b", FILES, commit="c2"))
    assert a["seed"] == b["seed"] and a["fingerprint"] == b["fingerprint"]


def test_any_changed_file_changes_full_seed_but_weights_scope_ignores_readme(tmp_path):
    edited = FILES | {"README.md": ("sha1readme2", b"# edited")}
    a, b = fake_snapshot(tmp_path / "a", FILES), fake_snapshot(tmp_path / "b", edited)
    assert model_seed(a)["seed"] != model_seed(b)["seed"]
    assert model_seed(a, scope="weights")["seed"] == model_seed(b, scope="weights")["seed"]
    reweighted = FILES | {"model-00002-of-00002.safetensors": ("c" * 64, b"w3")}
    c = fake_snapshot(tmp_path / "c", reweighted)
    assert model_seed(a, scope="weights")["seed"] != model_seed(c, scope="weights")["seed"]


def test_repo_id_is_part_of_the_fingerprint(tmp_path):
    a = model_seed(fake_snapshot(tmp_path, FILES, repo="org/one"))
    b = model_seed(fake_snapshot(tmp_path, FILES, repo="org/two"))
    assert a["seed"] != b["seed"]


def test_plain_directory_hashes_content(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    fp1 = model_fingerprint(tmp_path, repo_id="local/x")
    (tmp_path / "config.json").write_text('{"a": 1}')
    assert model_fingerprint(tmp_path, repo_id="local/x")["sha256"] != fp1["sha256"]


def test_plus_rand_mod_range():
    assert projection_seed(10, 0) == 10
    assert projection_seed(RANGE - 1, 2) == 1
    assert projection_seed(7, 5, mod=10) == 2
    assert replicate_seeds(RANGE - 1, 3) == [RANGE - 1, 0, 1]
    assert all(1 <= draw_plus_rand(100) < 100 for _ in range(200))
    assert 0 <= base_seed("f" * 64) < RANGE
    with pytest.raises(ValueError):
        projection_seed(1, 1, mod=1)


def test_salt_changes_every_seed_but_keeps_them_fixed_per_model(tmp_path, monkeypatch):
    monkeypatch.delenv("PERSONA_LSH_SALT", raising=False)
    monkeypatch.setattr("lsh.seed.SALT_FILE", tmp_path / "no-salt-file")
    snap = fake_snapshot(tmp_path, FILES)
    public = model_seed(snap)
    salted = model_seed(snap, salt="project-salt")
    assert public["salt id"] == "none" and salted["salt id"] != "none"
    assert salted["seed"] != public["seed"]
    assert model_seed(snap, salt="project-salt")["seed"] == salted["seed"]   # fixed per model
    assert model_seed(snap, salt="other-salt")["seed"] != salted["seed"]
    assert "project-salt" not in str(salted)                                  # never recorded
    monkeypatch.setenv("PERSONA_LSH_SALT", "project-salt")
    assert model_seed(snap)["seed"] == salted["seed"]                         # read from the environment


def test_salt_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PERSONA_LSH_SALT", raising=False)
    salt_file = tmp_path / "salt"
    salt_file.write_text(new_salt() + "\n")
    monkeypatch.setattr("lsh.seed.SALT_FILE", salt_file)
    assert load_salt() == salt_file.read_text().strip().encode()


def test_seed_drives_backend(tmp_path):
    info = model_seed(fake_snapshot(tmp_path, FILES), salt="")
    a = RandomProjectionBackend(dim=16, n_bits=64, seed=info["seed"])
    b = RandomProjectionBackend(dim=16, n_bits=64, seed=projection_seed(info["base"], 0))
    keyed = RandomProjectionBackend(dim=16, n_bits=64, seed=projection_seed(info["base"], 12345))
    assert (a.projection == b.projection).all()
    assert not (a.projection == keyed.projection).all()
