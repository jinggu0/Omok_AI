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
        res.append(1 if (k // 2) % 2 else -1)  # both results for each colour
        side.append(0 if g.side_to_move == "BLACK" else 1)
    arr = root / "gate.npz"
    np.savez(arr, feats=np.stack(feats), pos_result=np.array(res, np.int8), side=np.array(side, np.int8),
             split=np.full(len(res), 2, np.int8))
    return root, data, tck, arr


@pytest.mark.parametrize("mode", ["ece", "balanced"])
def test_two_generation_loop_with_labels_and_gate(tiny_tactical, small_solver, mode):
    root, data, ck, arr = tiny_tactical
    from dojozero.rl import loop
    cfg = {"run_name": f"looptac_{mode}", "run_dir": str(root), "seed": 2, "device": "cpu", "amp_bf16": False,
           "init_checkpoint": str(ck), "generations": 2,
           "selfplay": {"games": 3, "sims_full": 6, "sims_fast": 3, "p_full": 0.6, "root": "GUMBEL", "gumbel_m": 8,
                        "root_tactics_nodes": 100, "max_inflight": 2, "max_plies": 225, "profile": "OMOKDOJANG_DAN"},
           "replay": {"window": 2, "capacity": 10000},
           "train": {"steps_per_generation": 3, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001,
                     "betas": [0.9, 0.999], "grad_clip": 5.0, "tactical_mix": 0.25, "tactical_data": str(data),
                     "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5, "danger": 1.0, "refute": 1.0},
                     **({"value_weighting": {"balance": True}} if mode == "balanced" else {})},
           "eval": {"sims": 4, "pairs_per_round": 2, "min_pairs": 2, "max_pairs": 2, "elo0": 0, "elo1": 30,
                    "alpha": 0.05, "beta": 0.05, "anchor_pairs": 1},
           "tactical_aux": {"enabled": True, "label_fraction": 1.0, "top_k": 4, "threads": 2, "override_value": True},
           "gate": {"arrays": str(arr), "ece_margin": 0.03, "mode": mode}}
    reg = loop.run(cfg, quiet=True)
    gens = reg.data["generations"]
    assert [g["generation"] for g in gens] == [1, 2]
    for g in gens:
        assert g["labels"]["labelled"] > 0 and Path(root / f"looptac_{mode}" / "selfplay").exists()
        assert g["gate"] is not None and "ece5" in g["gate"]["candidate"]
        if mode == "balanced":
            assert {"black", "white", "auroc"} <= set(g["gate"]["candidate"]) and isinstance(g["gate"]["fails"], list)
        assert g["promoted"] == (g["eval"]["decision"] == "H1" and g["gate"]["pass"])
    assert "danger" in gens[1]["train"] and "refute" in gens[1]["train"]


def _bal(bias_b, sep_b, bias_w, sep_w, auroc):
    return {"auroc": auroc, "black": {"bias": bias_b, "separation": sep_b}, "white": {"bias": bias_w, "separation": sep_w}}


def test_balanced_gate_rule():
    best = _bal(0.02, 0.65, -0.05, 0.66, 0.96)
    assert tactical.balanced_gate_pass(_bal(0.0, 0.63, 0.0, 0.64, 0.955), best) == (True, [])
    ok, fails = tactical.balanced_gate_pass(_bal(0.0, 0.61, -0.11, 0.66, 0.96), best)
    assert not ok and len(fails) == 2 and fails[0].startswith("black separation") and "white |bias|" in fails[1]
    ok, fails = tactical.balanced_gate_pass(_bal(0.0, 0.70, 0.0, 0.70, 0.94), best)
    assert not ok and fails[0].startswith("auroc")
    # pessimism is no longer rewarded: a symmetric shift of every value fails the bias bound
    ok, fails = tactical.balanced_gate_pass(_bal(-0.2, 0.65, -0.2, 0.66, 0.96), best)
    assert not ok and len(fails) == 2
    ok, fails = tactical.balanced_gate_pass({"black": {}, "white": best["white"]}, best)
    assert not ok and fails == ["black: a win or loss cell is empty", "auroc: a win or loss cell is empty"]


def test_balanced_gate_set_is_stratified(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    arr = tmp_path / "a.npz"
    np.savez(arr, feats=rng.integers(0, 2, (n, 3, NUM_POINTS)).astype(np.uint8), pos_result=rng.choice([1, -1, 0], n).astype(np.int8),
             side=rng.integers(0, 2, n).astype(np.int8), ids=np.arange(n), split=rng.integers(0, 3, n).astype(np.int8))
    m = tactical.build_balanced_gate_set(arr, tmp_path / "g.npz", split="val", seed=1)
    g = np.load(tmp_path / "g.npz")
    z = np.load(arr)
    assert m["n"] == 4 * m["per_cell"] == len(g["pos_result"]) and m["per_cell"] == min(m["cells_available"].values())
    assert (z["split"][g["source_index"]] == 1).all() and (g["pos_result"] != 0).all()
    for c in (0, 1):
        for r in (1, -1):
            assert ((g["side"] == c) & (g["pos_result"] == r)).sum() == m["per_cell"]
    assert m == tactical.build_balanced_gate_set(arr, tmp_path / "g2.npz", split="val", seed=1) | {"sha256": m["sha256"]}


def test_value_cell_weights_and_weighted_loss(tmp_path, small_solver):
    import torch
    from dojozero.trainer.losses import value_loss
    spec = FeatureSpec()
    s = SelfPlaySettings(games=4, sims_full=6, sims_fast=3, p_full=0.5, gumbel_m=4, root_tactics_nodes=50,
                         max_inflight=2, profile="OMOKDOJANG_DAN")
    chunk = tmp_path / "gen_001.jsonl.gz"
    write_chunk(chunk, play_games(UniformNumpyEvaluator(), spec.core(), s, seed=3))
    rb = ReplayBuffer(spec, window=2)
    rb.ingest(chunk, 1)
    assert (rb._concat()["value_w"] == 1).all()  # default: unweighted (E5 behaviour)
    rb.value_weighting = {"balance": True, "proof": 1.0, "clip": (0.0, 100.0)}
    d = rb._concat()
    cells = [(d["black"] == c) & (d["value"] == r) for c in (True, False) for r in (1, -1)]
    totals = [d["value_w"][m].sum() for m in cells if m.any()]
    assert len(totals) >= 2 and np.allclose(totals, totals[0], rtol=1e-4)  # equal total weight per cell
    assert np.isclose(d["value_w"][d["value"] != 0].mean(), 1.0, atol=1e-4) or len(totals) < 4
    rb.value_weighting = {"balance": True, "proof": 1.0, "clip": (0.9, 1.1)}
    w = rb._concat()["value_w"]
    assert w.min() >= 0.9 - 1e-6 and w.max() <= 1.1 + 1e-6
    path = tactical.label_chunk(chunk, 1, fraction=1.0, seed=3, top_k=4, threads=2)
    rb.attach_labels(1, path, override_value=True)
    rb.value_weighting = {"balance": False, "proof": 3.0, "clip": (0.5, 2.0)}
    d = rb._concat()
    assert d["proven"].sum() == (np.load(path)["proof"] != 0).sum()
    assert (d["value_w"][d["proven"]] == 3).all() and (d["value_w"][~d["proven"]] == 1).all()
    assert "value_w" in rb.sample(8, seed=0, step=0, device="cpu") and rb.value_cells()["proven"] == d["proven"].sum()
    # weighted value loss: weights rescale rows; uniform weights give the unweighted loss
    logits = torch.randn(6, 3)
    v = torch.tensor([1, -1, 1, -1, 1, 0])
    has = torch.ones(6, dtype=torch.bool)
    assert torch.isclose(value_loss(logits, v, has), value_loss(logits, v, has, torch.full((6,), 2.0)))
    w2 = torch.tensor([1.0, 0, 0, 0, 0, 0])
    assert torch.isclose(value_loss(logits, v, has, w2), value_loss(logits[:1], v[:1], has[:1]))


def test_weight_ema():
    import torch
    from dojozero.rl.loop import WeightEMA
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = WeightEMA(m, decay=0.5)
    w0 = m[0].weight.detach().clone()
    with torch.no_grad():
        m[0].weight.add_(2.0)
    m.train()
    m(torch.randn(4, 2))  # updates BN running stats and num_batches_tracked
    ema.update(m)
    a = ema.averaged(m)
    assert torch.allclose(a[0].weight, w0 + 1.0)
    assert torch.allclose(a[1].running_mean, m[1].running_mean * 0.5)
    assert int(a[1].num_batches_tracked) == int(m[1].num_batches_tracked) == 1
    assert torch.allclose(m[0].weight, w0 + 2.0)  # the learner itself is untouched
