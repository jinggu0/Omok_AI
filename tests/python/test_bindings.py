"""C++ binding consistency: features, legal masks, aux targets and D4 augmentation."""

import json
import random

import numpy as np
import pytest

from dojozero import _core
from dojozero.features import GATHER, NUM_POINTS, FeatureSpec, transform_points, transform_policy


def random_state(rng: random.Random, profile: str, plies: int) -> "_core.GameState":
    g = _core.GameState(profile)
    for _ in range(plies):
        actions = g.legal_actions()
        if not actions or g.is_terminal:
            break
        g.apply(rng.choice(actions))
        if g.is_terminal:
            g.undo()
            break
    return g


def test_points_and_table():
    assert _core.point_to_str(112) == "H8"
    assert _core.str_to_point("A1") == 0
    t = np.asarray(_core.transform_table())
    assert t.shape == (8, 225)
    for row in t:
        assert sorted(row.tolist()) == list(range(225))
    assert (t[0] == np.arange(225)).all()


def test_plane_names_and_hash():
    a = FeatureSpec()
    b = FeatureSpec(tactical_planes=True)
    assert a.num_planes == len(a.core().plane_names)
    assert b.num_planes == a.num_planes + 6
    assert a.hash() != b.hash()


@pytest.mark.parametrize("profile", ["RENJU_RIF", "OMOKDOJANG_DAN"])
def test_encode_commutes_with_d4(profile):
    rng = random.Random(7)
    spec = FeatureSpec(history_planes=2, tactical_planes=True)
    cfg = spec.core()
    for trial in range(25):
        g = random_state(rng, profile, rng.randint(0, 30))
        base = np.asarray(_core.encode(g, cfg)).reshape(spec.num_planes, NUM_POINTS)
        legal = np.asarray(_core.legal_mask(g))
        aux = np.asarray(_core.aux_targets(g)).reshape(-1, NUM_POINTS)
        for t in range(8):
            gt = g.transformed(t)
            enc_t = np.asarray(_core.encode(gt, cfg)).reshape(spec.num_planes, NUM_POINTS)
            assert (transform_points(base, t) == enc_t).all(), (trial, t)
            assert (transform_policy(legal, t) == np.asarray(_core.legal_mask(gt))).all()
            aux_t = np.asarray(_core.aux_targets(gt)).reshape(-1, NUM_POINTS)
            assert (transform_points(aux, t) == aux_t).all()


def test_legal_mask_matches_legal_actions():
    rng = random.Random(3)
    for _ in range(20):
        g = random_state(rng, "OMOKDOJANG_DAN", rng.randint(0, 25))
        mask = np.asarray(_core.legal_mask(g))
        points = {a.split(":")[-1] for a in g.legal_actions() if a not in ("swap", "keep", "direct5", "offer10", "pass")}
        assert {_core.point_to_str(i) for i in np.flatnonzero(mask[:225])} == points
        assert bool(mask[225]) == ("pass" in g.legal_actions())


def test_encode_boards_matches_single_encode():
    rng = random.Random(5)
    states, boards, sides = [], [], []
    while len(states) < 30:
        g = random_state(rng, "RENJU_RIF", rng.randint(2, 40))
        if g.is_terminal:
            continue
        states.append(g)
        boards.append(np.asarray(g.board()))
        sides.append(0 if g.side_to_move == "BLACK" else 1)
    spec = FeatureSpec()
    f, l, a = _core.encode_boards(np.stack(boards), np.array(sides, np.uint8), np.zeros(len(states), np.uint8),
                                  spec.core(), 4)
    for i, g in enumerate(states):
        # from_board drops history (last move), so compare against a board-only reconstruction
        gb = _core.GameState.from_board(boards[i], sides[i], "RENJU_RIF")
        assert (np.asarray(_core.encode(gb, spec.core())) == f[i]).all()
        assert (np.asarray(_core.legal_mask(gb)) == l[i]).all()
        assert (np.asarray(_core.aux_targets(gb)) == a[i]).all()
        assert (np.asarray(_core.legal_mask(g)) == l[i]).all()


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        _core.GameState.from_spec(json.dumps({"black": ["H8", "H8"]}))
    with pytest.raises(ValueError):
        _core.GameState("NO_SUCH_PROFILE")
    g = _core.GameState("TARAGUCHI10")
    assert g.apply("H8") is True
    assert g.apply("A1") is False  # swap decision phase


def test_aux_targets_semantics():
    g = _core.GameState.from_spec(json.dumps({"black": ["F8", "G8", "H8", "I8"], "white": ["B15", "E15", "H15", "K15"],
                                              "side": "BLACK"}))
    aux = np.asarray(_core.aux_targets(g)).reshape(5, 225)
    assert {_core.point_to_str(i) for i in np.flatnonzero(aux[0])} == {"E8", "J8"}
    w = _core.GameState.from_spec(json.dumps({"black": ["F8", "G8", "H8", "I8"], "white": ["B15", "E15", "H15", "K15", "N15"],
                                              "side": "WHITE"}))
    aux_w = np.asarray(_core.aux_targets(w)).reshape(5, 225)
    assert {_core.point_to_str(i) for i in np.flatnonzero(aux_w[1])} == {"E8", "J8"}  # must block
    f = _core.GameState.from_spec(json.dumps({"black": ["F8", "G8", "H6", "H7"], "white": ["B15", "E15", "H15", "K15"],
                                              "side": "BLACK"}))
    aux_f = np.asarray(_core.aux_targets(f)).reshape(5, 225)
    assert aux_f[4][_core.str_to_point("H8")] == 1  # 3-3 forbidden


def test_gather_is_inverse_of_table():
    t = np.asarray(_core.transform_table())
    for k in range(8):
        x = np.arange(225)
        y = x[GATHER[k]]
        assert (y[t[k]] == x).all()


def test_power_throttling_opt_out_is_harmless():
    import sys
    from dojozero.evaluation.tournament import disable_power_throttling
    assert disable_power_throttling() is (sys.platform == "win32")
