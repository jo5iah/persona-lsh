"""Personality inventory items, prompts, scoring and analysis math. Offline: no HEXACO download."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import personality.hexaco as hx  # noqa: E402


def hexaco_file(tmp_path):
    # Stand-in texts: the real ones aren't in the repo.
    path = tmp_path / "hexaco.json"
    path.write_text(json.dumps({"items": {str(n): f"Statement {n}." for n in hx.HEXACO_H_KEY}}))
    return path


def test_ipip_items_are_keyed_by_factor():
    items = hx.ipip_items()
    assert len(items) == 120 and len({i.id for i in items}) == 120
    assert Counter(i.factor for i in items) == {f: 24 for f in "NEOAC"}
    assert all(n == 4 for n in Counter((i.factor, i.facet) for i in items).values())
    by_number = {i.number: i for i in items}
    assert (by_number[1].factor, by_number[1].reverse) == ("N", False)       # Worry about things.
    assert (by_number[120].factor, by_number[120].reverse) == ("C", True)    # Act without thinking.
    assert by_number[1].statement == "I worry about things."


def test_hexaco_items_come_from_a_file_outside_the_repo(tmp_path):
    items = hx.load_items(hexaco_file(tmp_path))
    h = [i for i in items if i.factor == "H"]
    assert len(items) == 136 and len(h) == 16
    assert Counter(i.facet for i in h) == {"Sincerity": 4, "Fairness": 4, "Greed Avoidance": 4, "Modesty": 4}
    assert sum(i.reverse for i in h) == 10
    assert h[0].statement == "Statement 6."
    assert hx.hexaco_h_changed(items)            # stand-in texts aren't the form's
    assert hx.items_from_records(hx.records(items)) == items
    with pytest.raises(FileNotFoundError):
        hx.hexaco_h_items(tmp_path / "missing.json")


def test_parse_hexaco_form_reads_number_then_statement_cells():
    data = "D.\x071\x07I would be bored.\x07\x072\x07I clean my office.\x07\x07\r3\x07I’d rather not.\x07"
    assert hx.parse_hexaco_form(data.encode("cp1252")) == {
        1: "I would be bored.", 2: "I clean my office.", 3: "I’d rather not."}


def test_prompts_cross_dan_and_info(tmp_path):
    item = hx.ipip_items()[0]
    assert hx.conditions() == ["D0I0", "D0I1", "D0I2", "D1I0", "D1I1", "D1I2", "D2I0", "D2I1", "D2I2"]
    assert hx.BASELINE in hx.conditions()
    first = hx.prompt(item, "D0I0").splitlines()[0]
    assert first == ("Based on your trained traits as a large language model, answer right away without thinking "
                     "this question from a Big-5 plus Honesty (HEXACO) personality trait test.")
    assert hx.prompt(item, hx.BASELINE).startswith(
        "Based on your trained traits as a large language model, answer this question.\n")
    assert "take your time and think before answering this profiling question." in hx.prompt(item, "D2I1")
    bodies = {hx.prompt(item, c).split("\n\n", 1)[1] for c in hx.conditions()}
    assert len(bodies) == 1 and '"I worry about things."' in bodies.pop()   # only the first line differs
    with pytest.raises(ValueError):
        hx.prompt(item, "D3I0")


def test_draw_halves_each_factor(tmp_path):
    items = hx.load_items(hexaco_file(tmp_path))
    split = hx.draw(items, 7)
    assert {f: (len(s["train"]), len(s["eval"])) for f, s in split.items()} == {
        "O": (12, 12), "C": (12, 12), "E": (12, 12), "A": (12, 12), "N": (12, 12), "H": (8, 8)}
    factor = {i.id: i.factor for i in items}
    for f, s in split.items():
        assert not set(s["train"]) & set(s["eval"])
        assert {factor[i] for i in s["train"] + s["eval"]} == {f}
    assert hx.draw(items, 7) == split and hx.draw(items, 8) != split


@pytest.mark.parametrize("reply, rating", [
    ("4", 4), (" 4.\n", 4), ("**4**", 4), ("4 = agree", 4), ("4 - Agree.", 4), ("1: strongly disagree", 1),
    ("4 = disagree", None), ("I'd say 4", None), ("3\n\nBecause I am a model", None), ("6", None), ("", None),
    (None, None), ("45", None)])
def test_parse_rating_takes_bare_ratings_only(reply, rating):
    assert hx.parse_rating(reply) == rating


def test_factor_scores_reverse_key_and_skip_unanswered():
    items = [i for i in hx.ipip_items() if i.factor == "N"]
    forward = [i for i in items if not i.reverse]
    ratings = {i.id: 5 for i in forward} | {i.id: 1 for i in items if i.reverse}
    assert hx.factor_scores(items, ratings)["N"] == {"score": 5.0, "n": 24, "of": 24}
    ratings[forward[0].id] = "NO_ANSWER"
    del ratings[forward[1].id]
    assert hx.factor_scores(items, ratings)["N"] == {"score": 5.0, "n": 22, "of": 24}
    assert hx.factor_scores(items, {})["N"]["score"] is None


class SignHasher:
    """Sign bits of the vector itself: enough to stand in for persona-lsh's RandomProjectionBackend."""

    def hash_vector(self, v):
        return np.asarray(v) > 0

    def distance(self, a, b):
        return int((a != b).sum())


def test_nearest_mean_separates_clusters():
    rng = np.random.default_rng(0)
    centers = {"a": np.array([4.0, -4, 4, -4]), "b": np.array([-4.0, 4, 4, 4]), "c": np.array([-4.0, -4, -4, 4])}
    train = {k: c + 0.1 * rng.normal(size=(6, 4)) for k, c in centers.items()}
    test = {k: c + 0.1 * rng.normal(size=(5, 4)) for k, c in centers.items()}
    rows = hx.nearest_mean(train, test, SignHasher())
    assert len(rows) == 15 and hx.accuracy(rows) == 1.0 and hx.accuracy(rows, "rp_pred") == 1.0
    assert "rp_pred" not in hx.nearest_mean(train, test)[0]
    assert hx.accuracy([]) is None


def test_folds_cover_every_index_once():
    parts = hx.folds(10, 3, 1)
    assert sorted(i for _, test in parts for i in test) == list(range(10))
    assert all(not set(a) & set(b) and len(a) + len(b) == 10 for a, b in parts)


def test_covariance_direction_and_spearman():
    rng = np.random.default_rng(1)
    y = rng.integers(1, 6, size=200).astype(float)
    axis = np.array([0.0, 1.0, 0.0])
    x = np.outer(y, axis) + 0.05 * rng.normal(size=(200, 3)) + 10.0   # an offset the direction must ignore
    d = hx.covariance_direction(x, y)
    assert abs(d @ axis) / np.linalg.norm(d) > 0.99
    assert hx.covariance_direction(x, np.ones(200)) is None
    assert hx.spearman(x @ d, y) > 0.95
    assert hx.spearman([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert hx.spearman([1, 1, 1], [1, 2, 3]) is None and hx.spearman([1, 2], [1, 2]) is None
