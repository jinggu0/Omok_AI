"""Ladder rating fit and adaptive opponent selection (EVALUATION §14, §16); conformance traps."""

import math

import numpy as np
import pytest

from dojozero.evaluation.ladder import fit_ratings, select_opponents


def simulate(true: dict[str, float], pairs: list[tuple[str, str]], n: int, seed: int):
    rng = np.random.default_rng(seed)
    games = []
    for a, b in pairs:
        p = 1 / (1 + 10 ** ((true[b] - true[a]) / 400))
        for _ in range(n):
            games.append((a, b, float(rng.random() < p)))
    return games


def test_fit_recovers_known_ratings_relative_to_the_anchor():
    true = {"anchor": 0.0, "x": 150.0, "y": -200.0, "dojo": 60.0}
    pairs = [("dojo", "anchor"), ("dojo", "x"), ("dojo", "y"), ("x", "y")]
    r = fit_ratings(simulate(true, pairs, 2000, 1), "anchor", prior_sd=5000)
    assert r["anchor"]["rating"] == 0.0 and r["anchor"].get("anchor")
    for p in ("x", "y", "dojo"):
        assert abs(r[p]["rating"] - true[p]) < 3 * r[p]["se"] + 5, (p, r[p])
        assert r[p]["lo"] < r[p]["rating"] < r[p]["hi"]


def test_fit_is_finite_for_perfect_records_and_symmetric():
    games = [("dojo", "weak", 1.0)] * 20 + [("dojo", "strong", 0.0)] * 20
    r = fit_ratings(games, "weak", prior_sd=600)
    assert all(math.isfinite(v["rating"]) for v in r.values())
    assert r["dojo"]["rating"] > 0 and r["strong"]["rating"] > r["dojo"]["rating"]
    even = fit_ratings([("a", "b", 0.5)] * 10 + [("a", "b", 1.0), ("a", "b", 0.0)] * 5, "b")
    assert abs(even["a"]["rating"]) < 1e-6
    with pytest.raises(ValueError):
        fit_ratings(games, "nobody")


def test_selection_prefers_informative_opponents():
    s = select_opponents({"pela": 1.0, "embryo": 0.6, "jax": 0.3, "rapfi": 0.0}, max_primary=3)
    assert s["primary"] == ["embryo", "jax"] and "within" in s["reason"]
    s = select_opponents({"pela": 1.0, "yixin": 0.9, "rapfi": 0.05, "kata": 0.15})
    assert s["primary"] == ["yixin", "kata"] and "above and below" in s["reason"]  # nearest rungs both sides
    s = select_opponents({"pela": 0.9, "yixin": 0.8})
    assert s["primary"] == ["yixin"] and "beats every" in s["reason"]
    s = select_opponents({"rapfi": 0.0, "kata": 0.1})
    assert s["primary"] == ["kata"] and "loses to every" in s["reason"]


def test_conformance_traps_are_consistent_with_the_rules():
    from dojozero.evaluation.conformance import board_moves, traps
    ts = traps()  # asserts every trap against the canonical C++ rules
    assert {t["name"] for t in ts} >= {"win_in_1", "black_overline", "black_double_four", "black_double_three",
                                       "white_overline"}
    for t in ts:
        mv = board_moves(t["black"], t["white"])
        assert len(mv) == len(t["black"]) + len(t["white"])
