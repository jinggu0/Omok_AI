"""Supervised tactical pretraining (Stage B, TRAINING.md §2.2).

    python -m dojozero.trainer.train --config configs/training/stage_b_tactical.yaml [--resume CKPT]

Run directory runs/<run_name>/: config.merged.yaml, env.json, git_diff.patch (if dirty),
metrics.jsonl, checkpoints/step_<n>.pt, command.txt (exact command line, CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .. import __version__
from ..evaluation.supervised import evaluate
from ..features import FeatureSpec
from ..model import ModelConfig, PolicyValueNet
from ..model.net import count_parameters
from . import checkpoint
from .data import TacticalDataset, benchmark_hashes, step_batch_indices
from .losses import LossWeights, total_loss

REPO = Path(__file__).resolve().parents[3]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    return torch.device(name)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def environment(device: torch.device) -> dict:
    env = {
        "dojozero_version": __version__,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(device),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }
    if device.type == "xpu":
        env["device_name"] = torch.xpu.get_device_name(0)
    return env


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def build(cfg: dict, device: torch.device):
    spec = FeatureSpec(**cfg["features"])
    mcfg = ModelConfig(in_planes=spec.num_planes, **cfg["model"])
    torch.manual_seed(cfg["seed"])
    model = PolicyValueNet(mcfg).to(device)
    o = cfg["optim"]
    # foreach=False on XPU: the multi-tensor AdamW kernel intermittently produced corrupted updates
    # (weights jumping from |w|<=1 to ~30 in one step, then NaN) with torch 2.14.1+xpu; the
    # per-parameter path is stable (docs/PROGRESS.md Phase 4).
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]),
                            foreach=False if device.type == "xpu" else None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(o["warmup_steps"], o["total_steps"]))
    return spec, mcfg, model, opt, sched


def run(cfg: dict, resume: str | None = None, quiet: bool = False) -> dict:
    device = pick_device(cfg.get("device", "auto"))
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
    run_dir = Path(cfg["run_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    spec, mcfg, model, opt, sched = build(cfg, device)
    weights = LossWeights(**cfg["loss"])
    o = cfg["optim"]
    start = 0
    if resume:
        ck = checkpoint.load(resume)
        if ck["feature_spec"]["hash"] != spec.hash():
            raise ValueError("checkpoint feature spec does not match the config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["torch_rng"])
        start = ck["step"]

    ds = TacticalDataset(REPO / cfg["data"]["dir"], spec, verbose=not quiet)
    exclude = benchmark_hashes(REPO / "benchmarks") if cfg["data"].get("exclude_benchmarks", True) else None
    split = ds.split(cfg["data"]["holdout_pct"], exclude)
    env = environment(device)
    env.update({"dataset_manifest": ds.manifest, "train_samples": int(len(split.train)),
                "holdout_samples": int(len(split.holdout)), "parameters": count_parameters(model)})
    if start == 0:
        (run_dir / "config.merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
        if env["git_dirty"]:
            (run_dir / "git_diff.patch").write_text(git("diff"), encoding="utf-8")
    (run_dir / "command.txt").open("a", encoding="utf-8").write(" ".join(sys.argv) + "\n")
    metrics_path = run_dir / "metrics.jsonl"

    def log(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if not quiet:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}), flush=True)

    def save(step: int, metrics: dict) -> Path:
        path = run_dir / "checkpoints" / f"step_{step}.pt"
        checkpoint.save(path, model=model, optimizer=opt, scheduler=sched, step=step, config=cfg,
                        feature_spec=spec.to_dict(), model_config=mcfg.to_dict(),
                        env={k: str(v) for k, v in env.items() if k != "dataset_manifest"}, metrics=metrics)
        return path

    # ADR-04: bf16 autocast on the XPU (measured ~2-4x faster than fp32); losses stay fp32.
    amp = bool(cfg.get("amp_bf16", False)) and device.type == "xpu"
    autocast = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    model.train()
    t_last, last_metrics = time.time(), {}
    losses_window: list[dict] = []
    for step in range(start, o["total_steps"]):
        idx, t = step_batch_indices(split.train, o["batch_size"], cfg["seed"], step, cfg["data"].get("augment", True))
        batch = ds.batch(idx, t, device)
        with autocast():
            out = model(batch["x"])
        out = {k: v.float() for k, v in out.items()}
        loss, parts = total_loss(out, batch, weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # foreach=False on XPU: foreach kernels crashed (native assert in _foreach_mul_, torch 2.14.1+xpu).
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"],
                                               foreach=False if device.type == "xpu" else None)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        sched.step()
        losses_window.append({k: float(v) for k, v in parts.items()} | {"grad_norm": float(gnorm)})
        done = step + 1
        if done % cfg["log_every"] == 0 or done == o["total_steps"]:
            dt = time.time() - t_last
            t_last = time.time()
            rec = {"step": done, "lr": sched.get_last_lr()[0],
                   "samples_per_s": len(losses_window) * o["batch_size"] / max(dt, 1e-9),
                   "weight_norm": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())))}
            for k in losses_window[0]:
                rec[f"train_{k}"] = float(np.mean([w[k] for w in losses_window]))
            losses_window.clear()
            log(rec)
        if done % cfg["eval_every"] == 0 or done == o["total_steps"]:
            last_metrics = {"step": done} | evaluate(model, ds, split.holdout, device, weights)
            log(last_metrics)
        if done % cfg["checkpoint_every"] == 0 or done == o["total_steps"]:
            save(done, last_metrics)
    return last_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--device")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--run-name")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    if args.steps:
        cfg["optim"]["total_steps"] = args.steps
    if args.run_name:
        cfg["run_name"] = args.run_name
    run(cfg, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
