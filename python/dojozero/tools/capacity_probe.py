"""Network capacity probe (PROGRESS Phase 8 E5r; pre-registration benchmarks/reanalysis/e5r/preregistration.json).

Two questions, both diagnostic (no strength claim):
  1. throughput - how many simulations per 500 ms move does a larger trunk leave (search cost)?
  2. fit        - trained from scratch on the SAME fixed data with the SAME recipe, does a larger trunk reach
                  clearly lower held-out losses than the current 6 x 64 trunk (is the current size the limit
                  on what the data can teach)?

    python -m dojozero.tools.capacity_probe init       --blocks 10 --channels 128 --out runs/capacity_probe/init_b10c128.pt
    python -m dojozero.tools.capacity_probe throughput --ckpts <a.pt> <b.pt> ... --positions 40 --out <json>
    python -m dojozero.tools.capacity_probe fit        --blocks 6 --channels 64 --seed 1 --out-dir runs/capacity_probe/b6c64_s1
    python -m dojozero.tools.capacity_probe val        --ckpt runs/e5e_rl_T/models/gen_045.pt --out <json>

Fixed data (identical for every arm): replay = T'' self-play generations 46-62 (train) and 63 (held out,
never trained on); reanalysis = data/reanalysis_v1 train split (val split for held-out losses; the test split
is never read here). Batches are stateless functions of (seed, step).
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

from ..features import FeatureSpec
from ..model.load import load_model, model_sha
from ..model.net import ModelConfig, PolicyValueNet
from ..replay.buffer import ReplayBuffer
from ..trainer import checkpoint
from ..trainer.losses import LossWeights, total_loss
from ..trainer.reanalysis import ReanalysisData
from ..trainer.train import environment, lr_lambda, pick_device

REPO = Path(__file__).resolve().parents[3]
REF_CKPT = "runs/e5e_rl_T/models/gen_045.pt"  # feature spec source and reference model
SELFPLAY = "runs/e5e_rl_T/selfplay/gen_{:03d}.jsonl.gz"
TRAIN_GENS = list(range(46, 63))
VAL_GEN = 63
REANALYSIS = "data/reanalysis_v1/arrays.npz"
LOSS = {"policy": 1.0, "value": 1.0, "aux": 0.5, "opening": 1.0, "danger": 1.0, "refute": 1.0}
VAL_SEED, VAL_BATCHES, VAL_BS = 999, 24, 256
POSITIONS_FROM = "runs/evaluation/e5o_traingames_vs_pela_2023.json"  # gen_045 vs Pela, training-only openings
PLIES_AFTER_OPENING = (4, 12)


def ref_spec() -> tuple[FeatureSpec, dict]:
    ck = checkpoint.load(str(REPO / REF_CKPT))
    fs = ck["feature_spec"]
    return FeatureSpec(fs["history_planes"], fs["tactical_planes"], fs["forbidden_plane"]), ck


def fresh_model(blocks: int, channels: int, seed: int) -> tuple[PolicyValueNet, FeatureSpec]:
    spec, ck = ref_spec()
    torch.manual_seed(seed)
    mc = dict(ck["model_config"]) | {"blocks": blocks, "channels": channels, "tactical_heads": True}
    return PolicyValueNet(ModelConfig(**mc)), spec


def save_ckpt(path: Path, model, spec: FeatureSpec, step: int, config: dict, env: dict) -> str:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    sha = model_sha(model)
    checkpoint.save(path, model=model, optimizer=opt, scheduler=sched, step=step, config=config,
                    feature_spec=spec.to_dict(), model_config=model.cfg.to_dict(),
                    env={k: str(v) for k, v in env.items()} | {"model_sha": sha}, metrics={})
    return sha


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------------------------
class Data:
    def __init__(self, spec: FeatureSpec, train: bool = True):
        self.re = ReanalysisData(REPO / REANALYSIS)
        self.tr_idx, self.va_idx = self.re.indices("train"), self.re.indices("val")
        self.rb_val = ReplayBuffer(spec, window=1)
        self.rb_val.ingest(REPO / SELFPLAY.format(VAL_GEN), VAL_GEN)
        self.rb = None
        if train:
            self.rb = ReplayBuffer(spec, window=len(TRAIN_GENS))
            for g in TRAIN_GENS:
                self.rb.ingest(REPO / SELFPLAY.format(g), g)


@torch.no_grad()
def held_out(model, data: Data, device, autocast) -> dict:
    """Held-out losses: replay generation 63 (VAL_BATCHES x VAL_BS fixed-seed positions, no augmentation;
    danger / refute weights 0 as in training) and the reanalysis val split (all positions)."""
    model.eval()
    w = LossWeights(**LOSS)
    w_plain = LossWeights(**(LOSS | {"danger": 0.0, "refute": 0.0}))
    out: dict = {}
    for tag, batches, ww in (
            ("rp", [data.rb_val.sample(VAL_BS, VAL_SEED, k, device, augment=False) for k in range(VAL_BATCHES)], w_plain),
            ("re", [data.re.batch(data.va_idx[s:s + 256], None, device) for s in range(0, len(data.va_idx), 256)], w)):
        acc, n = {}, 0
        for bt in batches:
            with autocast():
                o = model(bt["x"])
            o = {k: v.float() for k, v in o.items()}
            _, parts = total_loss(o, bt, ww)
            m = len(bt["x"])
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v) * m
            n += m
        out |= {f"{tag}_{k}": v / n for k, v in acc.items()}
    model.train()
    return out


def amp_ctx(device):
    if device.type == "xpu":
        return lambda: torch.autocast(device_type="xpu", dtype=torch.bfloat16)
    return contextlib.nullcontext


# ---------------------------------------------------------------------------------------------
def cmd_init(a) -> int:
    model, spec = fresh_model(a.blocks, a.channels, a.seed)
    sha = save_ckpt(Path(a.out), model, spec, 0, {"capacity_probe": "random init", "seed": a.seed}, {})
    print(json.dumps({"out": a.out, "params": n_params(model), "model_sha": sha}))
    return 0


def cmd_throughput(a) -> int:
    from ..evaluation import corpus as corpus_mod
    from ..evaluation.tournament import make_tournament_player
    s = corpus_mod.load(str(REPO / "benchmarks/openings/renju_train_reanalysis_v1.json"))
    ops = {o["opening_id"]: o for o in s.openings}
    games = json.loads((REPO / POSITIONS_FROM).read_text())["games"]
    positions = []  # (opening, moves after the opening): mid-game states of real 500 ms games, non-terminal
    for gm in games:
        op = ops[gm["opening_id"]]
        rest = gm["moves"][len(op["moves"]):]
        for k in PLIES_AFTER_OPENING:
            if len(rest) > k + 2:
                positions.append((op, rest[:k]))
        if len(positions) >= a.positions:
            break
    # Players are interleaved per position (every checkpoint searches position k before any searches k+1),
    # so background load changes over time hit all checkpoints alike; ratios to the reference are the result.
    players = {ck: make_tournament_player("dojo:" + ck, a.device, {}, a.time_ms, 1) for ck in a.ckpts}
    raw = {ck: {"sims": [], "evals": [], "ms": []} for ck in a.ckpts}
    for op, rest in positions:
        for ck, p in players.items():
            g = s.start_state(op)
            for m in rest:
                assert g.apply(m)
            r = p.play(g, [])
            raw[ck]["sims"].append(r["stats"]["sims"])
            raw[ck]["evals"].append(r["stats"]["nn_evals"])
            raw[ck]["ms"].append(r["elapsed_ms"])
    res = {}
    ref = a.ckpts[0]
    for ck in a.ckpts:
        model, _, _ = load_model(str(REPO / ck), "cpu")
        d = raw[ck]
        ratio = [x / max(1, y) for x, y in zip(d["sims"], raw[ref]["sims"])]
        res[ck] = {"params": n_params(model), "model_config": model.cfg.to_dict(), "positions": len(positions),
                   "sims_per_move": float(np.mean(d["sims"])), "sims_median": float(np.median(d["sims"])),
                   "nn_evals_per_move": float(np.mean(d["evals"])), "ms_per_move": float(np.mean(d["ms"])),
                   "nn_evals_per_s": float(np.sum(d["evals"]) / (np.sum(d["ms"]) / 1000.0)),
                   "sims_ratio_to_first_median": float(np.median(ratio))}
        print(ck, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[ck].items()}), flush=True)
    Path(a.out).write_text(json.dumps({"time_ms": a.time_ms, "positions_from": POSITIONS_FROM,
                                       "plies_after_opening": PLIES_AFTER_OPENING, "rows": res},
                                      indent=1), encoding="utf-8")
    return 0


def cmd_fit(a) -> int:
    device = pick_device(a.device)
    model, spec = fresh_model(a.blocks, a.channels, a.seed)
    model.to(device).train()
    data = Data(spec)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4, betas=(0.9, 0.999),
                            foreach=False if device.type == "xpu" else None)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps))
    w = LossWeights(**LOSS)
    w_plain = LossWeights(**(LOSS | {"danger": 0.0, "refute": 0.0}))
    autocast = amp_ctx(device)
    cfg = vars(a) | {"train_gens": TRAIN_GENS, "val_gen": VAL_GEN, "reanalysis": REANALYSIS, "loss": LOSS}
    env = environment(device) | {"params": n_params(model), "replay_train": len(data.rb), "replay_val": len(data.rb_val),
                                 "reanalysis_train": int(len(data.tr_idx)), "reanalysis_val": int(len(data.va_idx))}
    (out_dir / "env.json").write_text(json.dumps(env | {"config": cfg}, indent=1, default=str), encoding="utf-8")
    metrics = out_dir / "metrics.jsonl"
    metrics.write_text("", encoding="utf-8")

    def log(rec: dict) -> None:
        with metrics.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    t0, window = time.time(), []
    log({"step": 0} | held_out(model, data, device, autocast))
    for step in range(a.steps):
        rng = np.random.default_rng([a.seed, step])
        i_re = data.tr_idx[rng.integers(0, len(data.tr_idx), size=a.batch_re)]
        batches = [(data.re.batch(i_re, rng.integers(0, 8, size=len(i_re)), device), w, "re"),
                   (data.rb.sample(a.batch_rp, a.seed, step, device), w_plain, "rp")]
        opt.zero_grad(set_to_none=True)
        rec = {}
        for bt, ww, tag in batches:
            with autocast():
                o = model(bt["x"])
            o = {k: v.float() for k, v in o.items()}
            loss, parts = total_loss(o, bt, ww)
            loss.backward()
            rec |= {f"train_{tag}_{k}": float(v) for k, v in parts.items()}
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False if device.type == "xpu" else None)
        if not torch.isfinite(g):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        opt.step()
        sched.step()
        window.append(rec)
        done = step + 1
        if done % 250 == 0 or done == a.steps:
            log({"step": done, "s": time.time() - t0, "lr": sched.get_last_lr()[0]}
                | {k: float(np.mean([r[k] for r in window])) for k in window[0]})
            window.clear()
        if done % a.eval_every == 0 or done == a.steps:
            log({"step": done, "s": time.time() - t0} | held_out(model, data, device, autocast))
    sha = save_ckpt(out_dir / f"step_{a.steps}.pt", model.eval(), spec, a.steps, cfg, env)
    log({"final_checkpoint": str((out_dir / f"step_{a.steps}.pt").as_posix()), "model_sha": sha,
         "params": n_params(model), "wall_s": time.time() - t0})
    return 0


def cmd_val(a) -> int:
    device = pick_device(a.device)
    model, spec, _ = load_model(str(REPO / a.ckpt), device, tactical_heads=True)
    data = Data(spec, train=False)
    r = {"ckpt": a.ckpt, "params": n_params(model)} | held_out(model, data, device, amp_ctx(device))
    Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps(r))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--blocks", type=int, required=True)
    i.add_argument("--channels", type=int, required=True)
    i.add_argument("--seed", type=int, default=1)
    i.add_argument("--out", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--ckpts", nargs="+", required=True)
    t.add_argument("--positions", type=int, default=40)
    t.add_argument("--time-ms", type=float, default=500)
    t.add_argument("--device", default="xpu")
    t.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--blocks", type=int, required=True)
    f.add_argument("--channels", type=int, required=True)
    f.add_argument("--seed", type=int, default=1)
    f.add_argument("--steps", type=int, default=6000)
    f.add_argument("--lr", type=float, default=2e-3)
    f.add_argument("--warmup", type=int, default=300)
    f.add_argument("--batch-re", type=int, default=64)
    f.add_argument("--batch-rp", type=int, default=192)
    f.add_argument("--eval-every", type=int, default=1000)
    f.add_argument("--device", default="auto")
    f.add_argument("--out-dir", required=True)
    v = sub.add_parser("val")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--device", default="auto")
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"init": cmd_init, "throughput": cmd_throughput, "fit": cmd_fit, "val": cmd_val}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
