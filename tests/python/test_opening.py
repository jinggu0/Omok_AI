"""Phase 7: Taraguchi-10 opening targets (record -> replay -> loss), league games, PFSP, prefix
openings and opening diagnostics."""

import numpy as np
import pytest
import torch

from dojozero import _core
from dojozero.evaluation.match import make_prefix_openings
from dojozero.evaluation.opening import OpeningDB, opening_stats, proposal_regret
from dojozero.features import FeatureSpec
from dojozero.model.net import ModelConfig, PolicyValueNet
from dojozero.replay import ReplayBuffer
from dojozero.replay.buffer import HEAD_BRANCH, HEAD_POLICY, HEAD_PROPOSE, HEAD_SELECT, HEAD_SWAP
from dojozero.rl.loop import league_members, league_results, pfsp_weights
from dojozero.search import UniformNumpyEvaluator
from dojozero.selfplay.generate import Opponent, SelfPlaySettings, play_games
from dojozero.selfplay.records import write_chunk
from dojozero.trainer.losses import LossWeights, total_loss

PROFILE = "OMOKDOJANG_DAN"
OPENING = SelfPlaySettings(games=6, sims_full=8, sims_fast=4, p_full=1.0, root_tactics_nodes=100, max_inflight=2,
                           start="empty", record_opening=True)


@pytest.fixture(scope="module")
def opening_games():
    return play_games(UniformNumpyEvaluator(), FeatureSpec().core(), OPENING, seed=11)


def test_empty_start_records_every_opening_decision(opening_games):
    for game in opening_games:
        assert game["meta"]["opening_len"] == 0
        g = _core.GameState(PROFILE)
        recorded = {p["i"]: p for p in game["positions"]}
        for i, a in enumerate(game["actions"]):
            assert i in recorded, "p_full=1: every learner move has a target"
            assert recorded[i]["phase"] == g.phase and recorded[i]["actor"] == g.actor
            g.apply(a)
    phases = {p["phase"] for game in opening_games for p in game["positions"]}
    assert {"SWAP_AFTER_1", "CHOICE_AFTER_4", "NORMAL_RENJU"} <= phases


def test_opening_targets_reach_their_heads(opening_games, tmp_path):
    path = tmp_path / "op.jsonl.gz"
    write_chunk(path, opening_games)
    buf = ReplayBuffer(FeatureSpec())
    n = buf.ingest(path, 1)
    assert n == sum(len(g["positions"]) for g in opening_games)
    counts = buf.head_counts()
    assert counts["swap"] > 0 and counts["branch"] > 0 and counts["policy"] > 0
    assert buf.stats["opening_targets"] > 0
    data = buf._concat()
    two = np.isin(data["head"], (HEAD_SWAP, HEAD_BRANCH))
    assert np.allclose(data["tgt2"][two].sum(1), 1.0) and np.allclose(data["tgt2"][~two], 0.0)
    pts = np.isin(data["head"], (HEAD_POLICY, HEAD_PROPOSE, HEAD_SELECT))
    assert np.allclose(data["policy"][pts].sum(1), 1.0, atol=1e-4)
    # augmented point targets stay inside the legal mask
    b = buf.sample(64, seed=3, step=0, device="cpu")
    pm = (b["head"] != HEAD_SWAP) & (b["head"] != HEAD_BRANCH)
    support = (b["policy"] > 0) & pm[:, None]
    assert bool((b["legal"][support] > 0).all())
    # the loss uses the opening heads and stays finite
    torch.manual_seed(0)
    net = PolicyValueNet(ModelConfig(FeatureSpec().num_planes, blocks=1, channels=8, head_hidden=16))
    out = {k: v.float() for k, v in net(b["x"]).items()}
    loss, parts = total_loss(out, b, LossWeights())
    assert torch.isfinite(loss) and float(parts["opening"]) > 0
    loss.backward()


def test_point_targets_of_decision_phases_follow_the_phase():
    """A proposal target in a NORMAL position (wrong head) is rejected by replay verification."""
    from dojozero.replay import RecordError
    g = _core.GameState(PROFILE)
    actions = []
    while g.phase != "NORMAL_RENJU":
        a = _core.heuristic_action(g, len(actions))
        g.apply(a)
        actions.append(a)
    legal = g.legal_actions()
    game = {"format_version": 1,
            "meta": {"game_id": "x", "profile": PROFILE, "result": g.result, "winner": g.winner},
            "actions": actions,
            "positions": [{"i": len(actions), "actor": g.actor, "phase": g.phase,
                           "policy": {"propose:" + legal[0]: 1.0}, "value": 0.0, "proven": 0}]}
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "bad.jsonl.gz"
        write_chunk(path, [game])
        with pytest.raises(RecordError):
            ReplayBuffer(FeatureSpec()).ingest(path, 1)


def test_league_games_record_only_the_learner():
    s = SelfPlaySettings(games=6, sims_full=6, sims_fast=3, p_full=1.0, root_tactics_nodes=50, max_inflight=2,
                         league_fraction=1.0)
    opps = [Opponent("heur", "heuristic"), Opponent("rand", "random")]
    games = play_games(UniformNumpyEvaluator(), FeatureSpec().core(), s, seed=4, opponents=opps,
                       opponent_weights=[1.0, 1.0])
    assert all(g["meta"]["opponent"] in ("heur", "rand") for g in games)
    for game in games:
        assert game["positions"] and all(p["actor"] == game["meta"]["main_player"] for p in game["positions"])
    res = league_results(games)
    assert sum(e["games"] for e in res.values()) == 6
    assert all(0 <= e["score"] <= e["games"] for e in res.values())


def test_pfsp_prefers_hard_opponents():
    stats = {"easy": {"score": 9.0, "games": 10}, "hard": {"score": 1.0, "games": 10}}
    w = pfsp_weights(stats, ["easy", "hard", "unseen"])
    assert w[1] > w[2] > w[0] > 0
    hist = [{"version": v, "checkpoint": f"m{v}.pt"} for v in range(4)]
    cfg = {"league": {"fixed": [{"name": "heuristic", "spec": "heuristic"}], "past_best": 2}}
    names = [m["name"] for m in league_members(cfg, hist)]
    assert names == ["heuristic", "best_v1", "best_v2"]  # the current best (v3) is the learner


def test_prefix_openings_are_distinct_and_unfinished():
    ops = make_prefix_openings(200, 5)
    keys = set()
    for op in ops:
        assert len(op) >= 1
        g = _core.GameState(PROFILE)
        for a in op:
            assert g.apply(a)
        assert g.phase != "NORMAL_RENJU" and not g.is_terminal
        keys.add((g.canonical_hash(), g.phase))
    assert len(keys) == 200
    with pytest.raises(RuntimeError):
        make_prefix_openings(5, 1, max_attempts_per_opening=0)


def test_opening_stats_db_and_regret(opening_games, tmp_path):
    st = opening_stats(opening_games)
    assert st["games"] == 6 and 0 <= st["p0_score"] <= 1
    assert "decisions/SWAP_AFTER_1" in st and st["distinct_normal_starts"] >= 1
    db = OpeningDB(tmp_path / "db.json")
    assert db.add_games(opening_games, model_version=0) > 0
    db.save()
    again = OpeningDB(tmp_path / "db.json")
    assert again.summary()["entries"] == db.summary()["entries"] > 0
    r = proposal_regret(UniformNumpyEvaluator(), FeatureSpec().core(), opening_games)
    if r["positions"]:
        assert r["mean_regret"] == pytest.approx(0.0, abs=1e-9)  # uniform evaluator: all values equal
