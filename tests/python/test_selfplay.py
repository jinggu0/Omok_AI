"""Self-play records, replay buffer verification, SPRT and a one-generation loop smoke test."""

import copy
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from dojozero.evaluation.sprt import sprt
from dojozero.features import NUM_POINTS, FeatureSpec
from dojozero.replay import RecordError, ReplayBuffer
from dojozero.search import UniformNumpyEvaluator
from dojozero.selfplay.generate import SelfPlaySettings, play_games
from dojozero.selfplay.records import read_chunk, write_chunk, z_for

REPO = Path(__file__).resolve().parents[2]
DATAGEN = REPO / "build" / "msvc" / "cpp" / "Release" / "dojozero_datagen.exe"
TINY = SelfPlaySettings(games=4, sims_full=8, sims_fast=4, p_full=0.6, root_tactics_nodes=200, max_inflight=2)


@pytest.fixture(scope="module")
def games():
    return play_games(UniformNumpyEvaluator(), FeatureSpec().core(), TINY, seed=5)


def test_records_round_trip_and_replay(games, tmp_path):
    path = tmp_path / "gen_001.jsonl.gz"
    write_chunk(path, games)
    assert read_chunk(path) == games
    buf = ReplayBuffer(FeatureSpec(), window=2)
    n = buf.ingest(path, 1)
    assert n == sum(len(g["positions"]) for g in games)
    assert n > 0
    b = buf.sample(32, seed=1, step=0, device="cpu")
    support = b["policy"] > 0
    assert bool((b["legal"][support] > 0).all()), "policy targets must stay legal after D4 augmentation"
    assert torch.allclose(b["policy"].sum(1), torch.ones(32), atol=1e-4)


def test_value_targets_follow_winner(games, tmp_path):
    path = tmp_path / "c.jsonl.gz"
    write_chunk(path, games)
    buf = ReplayBuffer(FeatureSpec())
    buf.ingest(path, 1)
    expected = [z_for(p["actor"], g["meta"]["winner"]) for g in games for p in g["positions"]]
    assert list(buf._concat()["value"]) == expected


@pytest.mark.parametrize("corruption", ["action", "policy", "result", "actor"])
def test_corrupted_records_are_rejected(games, tmp_path, corruption):
    bad = copy.deepcopy(games[0])
    if corruption == "action":
        # the last move (NORMAL_RENJU) onto move 1's point, which is occupied: illegal in any game. (A
        # replaced opening action can be another LEGAL game, e.g. "A1" parses as "propose:A1" in
        # OFFER_TEN_5TH_MOVES, which replay correctly accepts.)
        bad["actions"][-1] = bad["actions"][0]
    elif corruption == "policy":
        p = bad["positions"][0]
        occupied = bad["actions"][0]  # move 1 (H8) is occupied in every later position
        p["policy"] = {occupied: 1.0}
    elif corruption == "result":
        bad["meta"]["result"] = "DRAW" if bad["meta"]["result"] != "DRAW" else "BLACK_WIN"
    else:
        p = bad["positions"][0]
        p["actor"] = "P1" if p["actor"] == "P0" else "P0"
    path = tmp_path / "bad.jsonl.gz"
    write_chunk(path, [bad])
    with pytest.raises(RecordError):
        ReplayBuffer(FeatureSpec()).ingest(path, 1)


def test_window_evicts_old_generations(games, tmp_path):
    buf = ReplayBuffer(FeatureSpec(), window=2)
    for gen in (1, 2, 3):
        p = tmp_path / f"g{gen}.jsonl.gz"
        write_chunk(p, games)
        buf.ingest(p, gen)
    assert sorted({b.generation for b in buf.blocks}) == [2, 3]


def test_sprt_decisions():
    assert sprt([1.0, 0.75, 1.0, 1.0, 0.75] * 8, 0, 30).decision == "H1"
    assert sprt([0.0, 0.25, 0.0, 0.5, 0.25] * 8, 0, 30).decision == "H0"
    r = sprt([0.5, 0.5], 0, 30)
    assert r.decision in ("continue", "H0") and r.lower < 0 < r.upper


@pytest.fixture(scope="module")
def tiny_init(tmp_path_factory):
    if not DATAGEN.exists():
        pytest.skip("dojozero_datagen not built")
    root = tmp_path_factory.mktemp("rl")
    data = root / "data"
    subprocess.run([str(DATAGEN), "--out", str(data), "--samples", "300", "--seed", "4", "--threads", "4",
                    "--vcf-nodes", "1000", "--vct-nodes", "300", "--loss-nodes", "200"], check=True, capture_output=True)
    from dojozero.trainer import train
    cfg = {"run_name": "init", "run_dir": str(root), "seed": 1, "device": "cpu", "deterministic": False,
           "data": {"dir": str(data), "holdout_pct": 10.0, "augment": True, "exclude_benchmarks": False},
           "features": {"history_planes": 0, "tactical_planes": False, "forbidden_plane": True},
           "model": {"blocks": 1, "channels": 8, "head_hidden": 16}, "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5},
           "optim": {"lr": 0.01, "weight_decay": 0.0001, "betas": [0.9, 0.999], "warmup_steps": 1, "total_steps": 2,
                     "batch_size": 16, "grad_clip": 5.0},
           "log_every": 1, "eval_every": 100, "checkpoint_every": 2}
    train.run(cfg, quiet=True)
    return root, data, root / "init" / "checkpoints" / "step_2.pt"


def test_one_generation_loop(tiny_init):
    root, data, ck = tiny_init
    from dojozero.rl import loop
    cfg = {"run_name": "loop", "run_dir": str(root), "seed": 2, "device": "cpu", "amp_bf16": False,
           "init_checkpoint": str(ck), "generations": 1,
           "selfplay": {"games": 3, "sims_full": 6, "sims_fast": 3, "p_full": 0.6, "root": "GUMBEL", "gumbel_m": 8,
                        "root_tactics_nodes": 100, "max_inflight": 2, "max_plies": 225, "profile": "OMOKDOJANG_DAN"},
           "replay": {"window": 2, "capacity": 10000},
           "train": {"steps_per_generation": 3, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001,
                     "betas": [0.9, 0.999], "grad_clip": 5.0, "tactical_mix": 0.25, "tactical_data": str(data),
                     "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5}},
           "eval": {"sims": 4, "pairs_per_round": 2, "min_pairs": 2, "max_pairs": 2, "elo0": 0, "elo1": 30,
                    "alpha": 0.05, "beta": 0.05, "anchor_pairs": 1}}
    reg = loop.run(cfg, quiet=True)
    g = reg.data["generations"][0]
    assert g["games"] == 3 and g["positions"] > 0
    assert (root / "loop" / "selfplay" / "gen_001.jsonl.gz").exists()
    assert (root / "loop" / "models" / "gen_001.pt").exists()
    assert g["eval"]["pairs"] == 2 and g["anchor"] is not None
    # resume continues numbering and refuses to overwrite without --resume
    with pytest.raises(RuntimeError):
        loop.run(cfg, quiet=True)
    reg2 = loop.run(cfg, resume=True, quiet=True)
    assert [x["generation"] for x in reg2.data["generations"]] == [1, 2]


def test_one_generation_opening_league_loop(tiny_init):
    """Phase 7 settings: empty/prefix starts with opening targets, a league pool, prefix-opening eval."""
    root, data, ck = tiny_init
    from dojozero.rl import loop
    cfg = {"run_name": "loop7", "run_dir": str(root), "seed": 3, "device": "cpu", "amp_bf16": False,
           "init_checkpoint": str(ck), "generations": 1, "regret_positions": 2,
           "selfplay": {"games": 4, "sims_full": 6, "sims_fast": 3, "p_full": 0.8, "root": "GUMBEL", "gumbel_m": 8,
                        "root_tactics_nodes": 100, "max_inflight": 2, "max_plies": 225, "profile": "OMOKDOJANG_DAN",
                        "start": "mixed", "prefix_prob": 0.5, "record_opening": True, "league_fraction": 0.5,
                        "opponent_sims": 4},
           "league": {"pfsp_eps": 0.05, "past_best": 1,
                      "fixed": [{"name": "heuristic", "spec": "heuristic"}, {"name": "init", "spec": f"engine:{ck}"}]},
           "replay": {"window": 2, "capacity": 10000},
           "train": {"steps_per_generation": 3, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001,
                     "betas": [0.9, 0.999], "grad_clip": 5.0, "tactical_mix": 0.25, "tactical_data": str(data),
                     "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5, "opening": 1.0}},
           "eval": {"openings": "prefix", "sims": 4, "pairs_per_round": 2, "min_pairs": 2, "max_pairs": 2, "elo0": 0,
                    "elo1": 30, "alpha": 0.05, "beta": 0.05, "anchor_pairs": 1}}
    reg = loop.run(cfg, quiet=True)
    g = reg.data["generations"][0]
    assert g["games"] == 4 and g["positions"] > 0
    assert "opening" in g["train"] and g["opening"]["games"] >= 0
    assert (root / "loop7" / "opening_db.json").exists()
    assert reg.data["best_history"][0]["version"] == 0
    assert sum(e["games"] for e in reg.data["league"].values()) == sum(e["games"] for e in g["league"].values())


def test_openings_are_distinct_and_terminate():
    """Regression: the old centre-greedy generator looped forever when asked for 256 openings."""
    from dojozero import _core
    from dojozero.evaluation.match import make_openings
    ops = make_openings(300, 3)
    hashes = set()
    for op in ops:
        g = _core.GameState("OMOKDOJANG_DAN")
        for a in op:
            assert g.apply(a)
        assert g.phase == "NORMAL_RENJU" and not g.is_terminal
        hashes.add(g.canonical_hash())
    assert len(hashes) == 300
    with pytest.raises(RuntimeError):
        make_openings(10, 1, max_attempts_per_opening=0)


def test_run_lock_refuses_second_owner(tmp_path):
    import subprocess as sp
    import sys as _sys
    from dojozero.rl.loop import RunLock, pid_alive
    holder = sp.Popen([_sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(holder.pid)
        (tmp_path / "LOCK").write_text(f"{holder.pid} x", encoding="utf-8")
        with pytest.raises(RuntimeError):
            RunLock(tmp_path)
    finally:
        holder.kill()
        holder.wait()
    assert not pid_alive(holder.pid)
    lock = RunLock(tmp_path)  # stale lock is taken over
    assert (tmp_path / "LOCK").exists()
    lock.release()
    assert not (tmp_path / "LOCK").exists()
