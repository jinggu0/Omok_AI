"""Phase 8 E5: tactical value auxiliary in the RL loop - solver labels aligned with the replay
buffer, proof value override, zero-weight unlabelled positions, and a 2-generation loop with the
labelling stage and the value-calibration gate."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from dojozero import _core
from dojozero.features import NUM_POINTS, FeatureSpec
from dojozero.replay import ReplayBuffer
from dojozero.rl import tactical
from dojozero.search import UniformNumpyEvaluator
from dojozero.selfplay.generate import SelfPlaySettings, play_games
from dojozero.selfplay.records import write_chunk
from dojozero.tools import reanalysis_data

REPO = Path(__file__).resolve().parents[2]
DATAGEN = REPO / "build" / "msvc" / "cpp" / "Release" / "dojozero_datagen.exe"
PHASE_NORMAL = "phase_NORMAL_RENJU"


@pytest.fixture
def small_solver(monkeypatch):
    monkeypatch.setitem(reanalysis_data.SOLVER, "vcf_nodes", 3000)
    monkeypatch.setitem(reanalysis_data.SOLVER, "vct_nodes", 1500)
    monkeypatch.setitem(reanalysis_data.SOLVER, "loss_nodes", 1000)


def test_labels_align_with_replay_and_override_values(tmp_path, small_solver):
    spec = FeatureSpec()
    s = SelfPlaySettings(games=3, sims_full=8, sims_fast=4, p_full=0.5, root="GUMBEL", gumbel_m=8,
                         root_tactics_nodes=50, max_inflight=2, profile="OMOKDOJANG_DAN")
    games = play_games(UniformNumpyEvaluator(), spec.core(), s, seed=5)
    chunk = tmp_path / "gen_001.jsonl.gz"
    write_chunk(chunk, games)
    rb = ReplayBuffer(spec, window=2)
    n = rb.ingest(chunk, 1)
    path = tactical.label_chunk(chunk, 1, fraction=1.0, seed=3, top_k=4, threads=2)
    z = np.load(path)
    assert len(z["idx"]) > 0 and z["idx"].max() < n
    # every labelled index is a NORMAL_RENJU position of the replay block (ingest order)
    plane = list(spec.core().plane_names).index(PHASE_NORMAL)
    feats = rb.blocks[0].feats
    assert all(feats[i, plane].max() == 1 for i in z["idx"])
    before = rb.blocks[0].value.copy()
    att = rb.attach_labels(1, path, override_value=True)
    proven = z["proof"] != 0
    assert att["attached"] == len(z["idx"])
    assert (rb.blocks[0].value[z["idx"][proven]] == z["proof"][proven]).all()
    unl = np.setdiff1d(np.arange(n), z["idx"][proven])
    assert (rb.blocks[0].value[unl] == before[unl]).all()  # UNKNOWN keeps the game result
    b = rb.sample(16, seed=1, step=0, device="cpu")
    assert {"danger_t", "danger_w", "refute_t", "refute_w"} <= set(b)
    lab = np.zeros(n, bool)
    lab[z["idx"]] = True
    assert rb.blocks[0].danger_w[~lab].sum() == 0 and rb.blocks[0].refute_w[~lab].sum() == 0


def test_unlabelled_blocks_carry_zero_weights():
    spec = FeatureSpec()
    s = SelfPlaySettings(games=2, sims_full=4, sims_fast=2, p_full=0.5, gumbel_m=4, root_tactics_nodes=0,
                         max_inflight=1, profile="OMOKDOJANG_DAN")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        chunk = Path(d) / "gen_001.jsonl.gz"
        write_chunk(chunk, play_games(UniformNumpyEvaluator(), spec.core(), s, seed=1))
        rb = ReplayBuffer(spec, window=2)
        rb.ingest(chunk, 1)
        b = rb.sample(8, seed=0, step=0, device="cpu")
        assert float(b["danger_w"].sum()) == 0 and float(b["refute_w"].sum()) == 0


@pytest.fixture(scope="module")
def tiny_tactical(tmp_path_factory):
    """Tiny init checkpoint with tactical heads + a tiny gate array file."""
    if not DATAGEN.exists():
        pytest.skip("dojozero_datagen not built")
    import torch
    from dojozero.model.load import load_model
    from dojozero.trainer import checkpoint, train
    root = tmp_path_factory.mktemp("rl_tac")
    data = root / "data"
    subprocess.run([str(DATAGEN), "--out", str(data), "--samples", "300", "--seed", "4", "--threads", "4",
                    "--vcf-nodes", "1000", "--vct-nodes", "300", "--loss-nodes", "200"], check=True, capture_output=True)
    cfg = {"run_name": "init", "run_dir": str(root), "seed": 1, "device": "cpu", "deterministic": False,
           "data": {"dir": str(data), "holdout_pct": 10.0, "augment": True, "exclude_benchmarks": False},
           "features": {"history_planes": 0, "tactical_planes": False, "forbidden_plane": True},
           "model": {"blocks": 1, "channels": 8, "head_hidden": 16}, "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5},
           "optim": {"lr": 0.01, "weight_decay": 0.0001, "betas": [0.9, 0.999], "warmup_steps": 1, "total_steps": 2,
                     "batch_size": 16, "grad_clip": 5.0},
           "log_every": 1, "eval_every": 100, "checkpoint_every": 2}
    train.run(cfg, quiet=True)
    model, spec, ck = load_model(str(root / "init" / "checkpoints" / "step_2.pt"), "cpu", tactical_heads=True)
    tck = root / "init_tactical.pt"
    opt = torch.optim.AdamW(model.parameters())
    checkpoint.save(tck, model=model, optimizer=opt, scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
                    step=0, config={}, feature_spec=spec.to_dict(), model_config=model.cfg.to_dict(), env={}, metrics={})
    # gate arrays: a few RENJU positions with proven results (labels are arbitrary but well-formed)
    g = _core.GameState("RENJU_GOMOCUP")
    feats, res, side = [], [], []
    for k, m in enumerate(["H8", "H9", "I9", "G7", "J10", "I8", "J8", "K11"]):
        assert g.apply(m)
        feats.append(np.asarray(_core.encode(g, spec.core())).reshape(-1, NUM_POINTS))
        res.append(1 if k % 2 else -1)
        side.append(0 if g.side_to_move == "BLACK" else 1)
    arr = root / "gate.npz"
    np.savez(arr, feats=np.stack(feats), pos_result=np.array(res, np.int8), side=np.array(side, np.int8),
             split=np.full(len(res), 2, np.int8))
    return root, data, tck, arr


def test_two_generation_loop_with_labels_and_gate(tiny_tactical, small_solver):
    root, data, ck, arr = tiny_tactical
    from dojozero.rl import loop
    cfg = {"run_name": "looptac", "run_dir": str(root), "seed": 2, "device": "cpu", "amp_bf16": False,
           "init_checkpoint": str(ck), "generations": 2,
           "selfplay": {"games": 3, "sims_full": 6, "sims_fast": 3, "p_full": 0.6, "root": "GUMBEL", "gumbel_m": 8,
                        "root_tactics_nodes": 100, "max_inflight": 2, "max_plies": 225, "profile": "OMOKDOJANG_DAN"},
           "replay": {"window": 2, "capacity": 10000},
           "train": {"steps_per_generation": 3, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001,
                     "betas": [0.9, 0.999], "grad_clip": 5.0, "tactical_mix": 0.25, "tactical_data": str(data),
                     "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5, "danger": 1.0, "refute": 1.0}},
           "eval": {"sims": 4, "pairs_per_round": 2, "min_pairs": 2, "max_pairs": 2, "elo0": 0, "elo1": 30,
                    "alpha": 0.05, "beta": 0.05, "anchor_pairs": 1},
           "tactical_aux": {"enabled": True, "label_fraction": 1.0, "top_k": 4, "threads": 2, "override_value": True},
           "gate": {"arrays": str(arr), "ece_margin": 0.03}}
    reg = loop.run(cfg, quiet=True)
    gens = reg.data["generations"]
    assert [g["generation"] for g in gens] == [1, 2]
    for g in gens:
        assert g["labels"]["labelled"] > 0 and Path(root / "looptac" / "selfplay").exists()
        assert g["gate"] is not None and "ece5" in g["gate"]["candidate"]
        assert g["promoted"] == (g["eval"]["decision"] == "H1" and g["gate"]["pass"])
    assert "danger" in gens[1]["train"] and "refute" in gens[1]["train"]
