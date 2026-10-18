"""End-to-end training smoke test and exact resume (TRAINING.md §9)."""

import json
import subprocess
from pathlib import Path

import pytest
import torch

from dojozero.trainer import train

REPO = Path(__file__).resolve().parents[2]
DATAGEN = REPO / "build" / "msvc" / "cpp" / "Release" / "dojozero_datagen.exe"


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    if not DATAGEN.exists():
        pytest.skip("dojozero_datagen not built")
    out = tmp_path_factory.mktemp("stage_a_tiny")
    subprocess.run([str(DATAGEN), "--out", str(out), "--samples", "400", "--seed", "5", "--threads", "4",
                    "--vcf-nodes", "2000", "--vct-nodes", "500", "--loss-nodes", "300"], check=True,
                   capture_output=True)
    return out


def config(data_dir: Path, run_dir: Path, name: str, steps: int) -> dict:
    return {
        "run_name": name, "run_dir": str(run_dir), "seed": 3, "device": "cpu", "deterministic": True,
        "data": {"dir": str(data_dir), "holdout_pct": 10.0, "augment": True, "exclude_benchmarks": False},
        "features": {"history_planes": 0, "tactical_planes": False, "forbidden_plane": True},
        "model": {"blocks": 1, "channels": 8, "head_hidden": 16},
        "loss": {"policy": 1.0, "value": 1.0, "aux": 0.5},
        "optim": {"lr": 0.01, "weight_decay": 0.0001, "betas": [0.9, 0.999], "warmup_steps": 2,
                  "total_steps": steps, "batch_size": 16, "grad_clip": 5.0},
        "log_every": 1, "eval_every": 1000, "checkpoint_every": 3,
    }


def train_losses(run_dir: Path, name: str) -> list[float]:
    recs = [json.loads(line) for line in (run_dir / name / "metrics.jsonl").read_text().splitlines()]
    return [r["train_loss"] for r in recs if "train_loss" in r]


def test_training_runs_and_resume_is_exact(tiny_dataset, tmp_path):
    # Uninterrupted 6-step run (checkpoint at step 3) ...
    train.run(config(tiny_dataset, tmp_path, "full", 6), quiet=True)
    a = train_losses(tmp_path, "full")
    assert len(a) == 6 and all(torch.isfinite(torch.tensor(a)))
    # ... must match a run resumed from that step-3 checkpoint, loss for loss.
    ck = tmp_path / "full" / "checkpoints" / "step_3.pt"
    assert ck.exists()
    train.run(config(tiny_dataset, tmp_path, "resumed", 6), resume=str(ck), quiet=True)
    b = train_losses(tmp_path, "resumed")
    assert len(b) == 3
    assert b == pytest.approx(a[3:], rel=0, abs=1e-6), (a, b)


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="no XPU")
def test_xpu_avoids_foreach_kernels(tiny_dataset, tmp_path):
    """XPU foreach kernels corrupted AdamW updates and crashed in clip_grad_norm_ (PROGRESS Phase 4)."""
    cfg = config(tiny_dataset, tmp_path, "xpu", 3)
    cfg["device"] = "xpu"
    cfg["deterministic"] = False
    _, _, _, opt, _ = train.build(cfg, torch.device("xpu"))
    assert opt.defaults["foreach"] is False
    src = Path(train.__file__).read_text(encoding="utf-8")
    assert 'foreach=False if device.type == "xpu"' in src.split("clip_grad_norm_")[1][:200]
    train.run(cfg, quiet=True)  # 3 real XPU steps with bf16 off must stay finite


def test_checkpoint_rejects_feature_mismatch(tiny_dataset, tmp_path):
    cfg = config(tiny_dataset, tmp_path, "fm", 3)
    train.run(cfg, quiet=True)
    other = config(tiny_dataset, tmp_path, "fm2", 3)
    other["features"]["tactical_planes"] = True
    with pytest.raises(ValueError):
        train.run(other, resume=str(tmp_path / "fm" / "checkpoints" / "step_3.pt"), quiet=True)
