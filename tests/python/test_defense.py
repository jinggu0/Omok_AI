"""Root defense filter (search driver): skips a move that gives the opponent a proven forced win
when an unrefuted candidate exists; otherwise leaves the search's choice alone."""

import time

from dojozero import _core
from dojozero.features import FeatureSpec
from dojozero.search import UniformNumpyEvaluator
from dojozero.search.driver import SearchSettings, defend, search_many

S = SearchSettings(defense_k=4, defense_ms=1000)


def open_three_position():
    # Black H8 I8 J8 is an open three; White to move must block it.
    g = _core.GameState("RENJU_GOMOCUP")
    for m in ["H8", "A1", "I8", "A3", "J8"]:
        assert g.apply(m)
    return g


def fake(action, *others, proven=2):
    kids = [{"action": action, "visits": 10}] + [{"action": a, "visits": 5 - k} for k, a in enumerate(others)]
    return {"action": action, "proven": proven, "children": kids}


def test_refuted_choice_is_replaced_by_the_next_unrefuted_candidate():
    r = defend(open_three_position(), fake("A5", "O15", "K8"), S)
    assert r["action"] == "K8" and r["defense"]["changed"] and r["defense"]["original"] == "A5"
    assert set(r["defense"]["refuted"]) == {"A5", "O15"}


def test_choice_stands_when_it_is_not_refuted_or_everything_is():
    r = defend(open_three_position(), fake("G8", "A5"), S)
    assert r["action"] == "G8" and not r["defense"]["changed"]
    r = defend(open_three_position(), fake("A5", "O15", "B9"), S)  # all ignore the three
    assert r["action"] == "A5" and not r["defense"]["changed"] and len(r["defense"]["refuted"]) == 3


def test_proven_win_and_opening_phases_are_untouched():
    r = defend(open_three_position(), fake("A5", "K8", proven=1), S)
    assert r["action"] == "A5" and r["defense"]["checked"] == 0
    g = _core.GameState("OMOKDOJANG_DAN")
    r = defend(g, fake("H8"), S)
    assert r["defense"]["checked"] == 0


def test_filter_inside_a_time_limited_search():
    g = open_three_position()
    s = SearchSettings(simulations=100000, root="PUCT", max_ms=400, root_tactics_nodes=0, defense_k=8)
    t0 = time.perf_counter()
    r = search_many([g], UniformNumpyEvaluator(), s, FeatureSpec().core())[0]
    ms = (time.perf_counter() - t0) * 1000
    assert g.copy().apply(r["action"]) and r["defense"]["checked"] >= 1
    assert ms < 400 + 400, ms
    after = g.copy()
    after.apply(r["action"])
    # whatever the uniform search picked, the filter left a move the solver does not refute
    if len(r["defense"]["refuted"]) < r["defense"]["checked"]:
        assert _core.analyze_tactics(after, 100000, 50000, 0, 1000)["result"] != "WIN"
