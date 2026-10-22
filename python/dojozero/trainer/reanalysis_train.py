"""Small solver-supervised reanalysis fine-tune (PROGRESS Phase 8 E4, Case B).

    python -m dojozero.trainer.reanalysis_train --config configs/training/e4_reanalysis.yaml

Starts from the current best checkpoint with the tactical heads added (fresh), and trains on a mix
of (a) reanalysis positions (train split only), (b) the self-play replay of the starting run and
(c) Stage A tactical data, so the existing policy/value/opening behaviour is kept. Each step sums
the three batch losses (one loss per source; the sources have different label sets). Batches are
stateless functions of (seed, step). The final step is the candidate (no checkpoint selection on
validation or test metrics); validation losses are logged to detect overfitting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..features import FeatureSpec
from ..model.load import load_model, model_sha
from ..replay.buffer import ReplayBuffer
from . import checkpoint
from .data import TacticalDataset, benchmark_hashes
from .losses import LossWeights, total_loss
from .reanalysis import ReanalysisData
from .train import environment, git, lr_lambda, pick_device

REPO = Path(__file__).resolve().parents[3]


def run(cfg: dict, quiet: bool = False) -> dict:
    device = pick_device(cfg.get("device", "auto"))
    torch.manual_seed(cfg["seed"])
    model, spec, ck0 = load_model(str(REPO / cfg["init_checkpoint"]), device, tactical_heads=True)
    model.train()
    run_dir = REPO / cfg["run_dir"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]),
                            foreach=False if device.type == "xpu" else None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(o["warmup_steps"], o["total_steps"]))
    w = LossWeights(**cfg["loss"])
    w_plain = LossWeights(**{**cfg["loss"], "danger": 0.0, "refute": 0.0})
    re = ReanalysisData(REPO / cfg["data"]["reanalysis"])
    tr_idx, va_idx = re.indices("train"), re.indices("val")
    rb = ReplayBuffer(spec, window=len(cfg["data"]["replay_chunks"]))
    for k, c in enumerate(cfg["data"]["replay_chunks"]):
        rb.ingest(REPO / c, k + 1)
    sa = TacticalDataset(REPO / cfg["data"]["tactical"], spec, verbose=False)
    sa_split = sa.split(cfg["data"].get("tactical_holdout_pct", 2.0), benchmark_hashes(REPO / "benchmarks"))
    env = environment(device) | {
        "init_checkpoint": cfg["init_checkpoint"], "init_model_sha": ck0.get("env", {}).get("model_sha", "see registry"),
        "reanalysis_train": int(len(tr_idx)), "reanalysis_val": int(len(va_idx)), "replay_positions": len(rb),
        "tactical_train": int(len(sa_split.train))}
    (run_dir / "config.merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    if env["git_dirty"]:
        (run_dir / "git_diff.patch").write_text(git("diff"), encoding="utf-8")
    (run_dir / "command.txt").open("a", encoding="utf-8").write(" ".join(sys.argv) + "\n")
    metrics_path = run_dir / "metrics.jsonl"
    amp = bool(cfg.get("amp_bf16", False)) and device.type == "xpu"
    autocast = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    b = cfg["batch"]

    def log(rec: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if not quiet:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    @torch.no_grad()
    def val_losses() -> dict:
        model.eval()
        parts_sum: dict = {}
        for s in range(0, len(va_idx), 256):
            idx = va_idx[s:s + 256]
            bt = re.batch(idx, None, device)
            with autocast():
                out = model(bt["x"])
            out = {k: v.float() for k, v in out.items()}
            _, parts = total_loss(out, bt, w)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v) * len(idx)
        model.train()
        return {f"val_{k}": v / max(1, len(va_idx)) for k, v in parts_sum.items()}

    t0, window = time.time(), []
    log({"step": 0} | val_losses())
    for step in range(o["total_steps"]):
        rng = np.random.default_rng([cfg["seed"], step])
        i_re = tr_idx[rng.integers(0, len(tr_idx), size=b["reanalysis"])]
        t_re = rng.integers(0, 8, size=len(i_re)) if cfg["data"].get("augment", True) else None
        batches = [(re.batch(i_re, t_re, device), w, "re")]
        if b.get("replay", 0):
            batches.append((rb.sample(b["replay"], cfg["seed"], step, device), w_plain, "rp"))
        if b.get("tactical", 0):
            i_sa = sa_split.train[rng.integers(0, len(sa_split.train), size=b["tactical"])]
            t_sa = rng.integers(0, 8, size=len(i_sa))
            batches.append((sa.batch(i_sa, t_sa, device), w_plain, "sa"))
        opt.zero_grad(set_to_none=True)
        rec = {}
        for bt, ww, tag in batches:
            with autocast():
                out = model(bt["x"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = total_loss(out, bt, ww)
            (loss * cfg["source_weight"][tag]).backward()
            rec |= {f"{tag}_{k}": float(v) for k, v in parts.items()}
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"],
                                               foreach=False if device.type == "xpu" else None)
        if not torch.isfinite(gnorm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        sched.step()
        window.append(rec | {"grad_norm": float(gnorm)})
        done = step + 1
        if done % cfg["log_every"] == 0 or done == o["total_steps"]:
            avg = {k: float(np.mean([r[k] for r in window])) for k in window[0]}
            window.clear()
            log({"step": done, "lr": sched.get_last_lr()[0], "s": time.time() - t0} | avg)
        if done % cfg["eval_every"] == 0 or done == o["total_steps"]:
            log({"step": done} | val_losses())
    path = run_dir / "checkpoints" / f"step_{o['total_steps']}.pt"
    model.eval()
    sha = model_sha(model)
    checkpoint.save(path, model=model, optimizer=opt, scheduler=sched, step=o["total_steps"], config=cfg,
                    feature_spec=spec.to_dict(), model_config=model.cfg.to_dict(),
                    env={k: str(v) for k, v in env.items()} | {"model_sha": sha}, metrics={})
    log({"final_checkpoint": str(path.relative_to(REPO)), "model_sha": sha})
    return {"checkpoint": str(path), "model_sha": sha}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--run-name")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.steps:
        cfg["optim"]["total_steps"] = args.steps
    if args.run_name:
        cfg["run_name"] = args.run_name
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
