"""Python batched search driver (C++ trees + Python evaluator)."""

import json
import random

import numpy as np
import pytest
import torch

from dojozero import _core
from dojozero.features import FeatureSpec
from dojozero.model import ModelConfig, PolicyValueNet
from dojozero.search import NetEvaluator, UniformNumpyEvaluator, search_many
from dojozero.search.driver import SearchSettings


def random_states(n, profile="OMOKDOJANG_DAN", seed=1):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = _core.GameState(profile)
        for _ in range(rng.randint(0, 40)):
            acts = g.legal_actions()
            if not acts:
                break
            g.apply(rng.choice(acts))
            if g.is_terminal:
                g.undo()
                break
        if not g.is_terminal:
            out.append(g)
    return out


@pytest.mark.parametrize("root", ["PUCT", "GUMBEL"])
def test_search_many_returns_legal_actions(root):
    states = random_states(12)
    res = search_many(states, UniformNumpyEvaluator(), SearchSettings(simulations=48, root=root, max_inflight=4),
                      FeatureSpec().core(), batch_size=64)
    for g, r in zip(states, res):
        assert r["action"] in g.legal_actions(), (g.phase, r["action"])
        if r["children"]:
            assert abs(sum(c["target"] for c in r["children"]) - 1.0) < 1e-3


def test_batch_size_one_matches_large_batches():
    """In-flight batching changes visit order but never legality; with max_inflight=1 and a
    deterministic evaluator, results do not depend on how many trees share a batch."""
    states = random_states(6, "RENJU_RIF", seed=3)
    s = SearchSettings(simulations=40, max_inflight=1)
    a = search_many(states, UniformNumpyEvaluator(), s, FeatureSpec().core(), batch_size=1)
    b = search_many(states, UniformNumpyEvaluator(), s, FeatureSpec().core(), batch_size=256)
    for ra, rb in zip(a, b):
        assert ra["action"] == rb["action"]
        assert [c["visits"] for c in ra["children"]] == [c["visits"] for c in rb["children"]]


def test_network_search_takes_immediate_win():
    spec = FeatureSpec()
    torch.manual_seed(0)
    model = PolicyValueNet(ModelConfig(spec.num_planes, blocks=1, channels=8, head_hidden=16))
    g = _core.GameState.from_spec(json.dumps({"black": ["F8", "G8", "H8", "I8"],
                                              "white": ["B15", "E15", "H15", "K15"], "side": "BLACK"}))
    (r,) = search_many([g], NetEvaluator(model, "cpu"), SearchSettings(simulations=16), spec.core())
    assert r["action"] in ("E8", "J8") and r["proven"] == 1


def test_heuristic_and_random_bots_are_legal():
    for g in random_states(10, seed=7):
        assert _core.heuristic_action(g, 1) in g.legal_actions()
        assert _core.random_action(g, 1) in g.legal_actions()


def test_proof_aware_search_reports_proofs_and_avoids_proven_losses():
    # Black H8 I8 J8 open three, White to move (Phase 8 E3d proof-aware MCTS): moves that ignore the
    # three let Black win by force; the tree marks them proven losses and never plays one.
    g = _core.GameState("RENJU_GOMOCUP")
    for m in ["H8", "A1", "I8", "A3", "J8"]:
        assert g.apply(m)
    s = SearchSettings(simulations=400, root="PUCT", root_tactics_nodes=0, proof_threads=4, proof_min_visits=1)
    r = search_many([g], UniformNumpyEvaluator(), s, FeatureSpec().core())[0]
    assert r["proof_jobs"] >= 1 and r["proof_found"] >= 1
    chosen = next(c for c in r["children"] if c["action"] == r["action"])
    assert chosen["proven"] != -1
    after = g.copy()
    assert after.apply(r["action"])
    assert _core.analyze_tactics(after, 100000, 50000, 0, 0.0)["result"] != "WIN"
