"""Fixed-data screen of a value-loss change (PROGRESS Phase 8 E5c).

    python -m dojozero.tools.fixed_replay_screen --config configs/training/e5b_rl_T.yaml --arm base_s7 \
        --seed 7 --out benchmarks/reanalysis/e5c/screen_base_s7.json
    python -m dojozero.tools.fixed_replay_screen ... --balance        # colour x result balanced value weights
    python -m dojozero.tools.fixed_replay_screen ... --proof 3        # proven positions x3 in the value loss
    python -m dojozero.tools.fixed_replay_screen judge benchmarks/reanalysis/e5c/screen_*.json

Replays the learner side of the RL loop (rl.loop.train_generation, same replay window, Stage A mix,
reanalysis buffer and lag-1 label attachment) on the E5 T self-play chunks and labels, from the same
init checkpoint, without self-play or SPRT. After each generation the learner is checked with the
balanced gate against the init model. Proxy only: the self-play data do not follow the model, and
strength is not measured.

Pre-registered decision (PROGRESS E5c): proceed to the T'' run only if the balanced arm
(1) passes the gate in >= 3 of 6 generations, (2) passes >= 2 more than each baseline arm, and
(3) has a smaller mean |bias_black - bias_white| than each baseline arm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.load import load_model
from ..replay import ReplayBuffer
from ..rl import tactical as tacmod
from ..rl.loop import WeightEMA, save_model, train_generation
from ..trainer.data import TacticalDataset
from ..trainer.losses import LossWeights
from ..trainer.train import REPO, pick_device


def screen(args) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = pick_device(cfg.get("device", "auto"))
    o = cfg["train"]
    if args.lr is not None:
        o["lr"] = args.lr
    gate = REPO / cfg["gate"]["arrays"]
    learner, spec, ck = load_model(str(REPO / cfg["init_checkpoint"]), device)
    mcfg = ck["model_config"]
    opt = torch.optim.AdamW(learner.parameters(), lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]),
                            foreach=False if device.type == "xpu" else None)
    init = tacmod.value_balance(learner, gate, device)
    replay = ReplayBuffer(spec, window=cfg["replay"]["window"], capacity=cfg["replay"]["capacity"])
    replay.value_weighting = {"balance": args.balance, "proof": args.proof, "clip": (0.5, 2.0)}
    tac = cfg["tactical_aux"]
    tactical = None
    if float(o.get("tactical_mix", 0)) > 0:
        ds = TacticalDataset(REPO / o["tactical_data"], spec, verbose=False)
        tactical = (ds, ds.split(5.0).train, float(o["tactical_mix"]))
    reana = None
    if tac.get("enabled") and tac.get("reanalysis"):
        from ..trainer.reanalysis import ReanalysisData
        rd = ReanalysisData(REPO / tac["reanalysis"])
        reana = (rd, rd.indices("train"), int(round(o["batch_size"] * float(tac.get("reanalysis_mix", 0.15)))))
    weights = LossWeights(**o["loss"])
    ema = WeightEMA(learner, args.ema) if args.ema else None
    amp = bool(cfg.get("amp_bf16", True)) and device.type == "xpu"
    src = json.loads((REPO / args.source_registry).read_text(encoding="utf-8"))
    out_dir = REPO / "runs" / "e5c_screen" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, step = [], 0
    for rec in src["generations"]:
        g = rec["generation"]
        t0 = time.time()
        replay.ingest(REPO / rec["chunk"], g)
        cells = replay.value_cells()
        step, metrics = train_generation(learner, opt, replay, o, args.seed, step, weights, amp, device,
                                         tactical=tactical, reana=reana, ema=ema)
        cand = ema.averaged(learner) if ema is not None else learner  # the model that would be gated / promoted
        save_model(out_dir / f"gen_{g:03d}.pt", cand, opt, spec, mcfg, cfg, step, {}, metrics)
        m = tacmod.value_balance(cand, gate, device)
        test = tacmod.value_calibration(cand, REPO / "data/reanalysis_v1/arrays.npz", device)
        learner.train()
        ok, fails = tacmod.balanced_gate_pass(m, init)
        rows.append({"generation": g, "pass": ok, "fails": fails, "train": metrics, "replay_cells": cells,
                     "gate": m, "colour_gap": m["black"]["bias"] - m["white"]["bias"],
                     "test": {k: test[k] for k in ("ece5", "proven_win_value", "proven_loss_value")},
                     "seconds": time.time() - t0})
        print(f"{args.arm} gen {g}: pass={ok} sep B {m['black']['separation']:.3f} W {m['white']['separation']:.3f} "
              f"bias B {m['black']['bias']:+.3f} W {m['white']['bias']:+.3f} auroc {m['auroc']:.3f} {fails} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if tac.get("enabled"):
            lp = (REPO / rec["chunk"]).with_suffix("").with_suffix(".labels.npz")
            replay.attach_labels(g, lp, bool(tac.get("override_value", True)))
    rep = {"arm": args.arm, "seed": args.seed, "value_weighting": {"balance": args.balance, "proof": args.proof},
           "ema": args.ema, "lr": o["lr"],
           "config": args.config, "source_registry": args.source_registry, "init": init, "rows": rows,
           "passes": sum(r["pass"] for r in rows),
           "mean_abs_colour_gap": float(np.mean([abs(r["colour_gap"]) for r in rows]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("arm", "passes", "mean_abs_colour_gap")}))
    return 0


def judge(paths: list[str]) -> int:
    reps = {r["arm"]: r for r in (json.loads(Path(p).read_text(encoding="utf-8")) for p in paths)}
    base = [r for r in reps.values() if not r["value_weighting"]["balance"] and r["value_weighting"]["proof"] == 1.0]
    bal = [r for r in reps.values() if r["value_weighting"]["balance"]]
    for r in reps.values():
        print(f"{r['arm']}: passes {r['passes']}/6, mean |colour gap| {r['mean_abs_colour_gap']:.3f}")
    if len(base) < 2 or len(bal) != 1:
        print("need >= 2 baseline arms and 1 balanced arm")
        return 2
    b = bal[0]
    c1 = b["passes"] >= 3
    c2 = all(b["passes"] >= x["passes"] + 2 for x in base)
    c3 = all(b["mean_abs_colour_gap"] < x["mean_abs_colour_gap"] for x in base)
    print(json.dumps({"passes>=3": c1, "passes>=base+2": c2, "colour_gap<base": c3,
                      "decision": "PROCEED" if c1 and c2 and c3 else "STOP"}))
    return 0


def _group_stats(reps: list[dict]) -> dict:
    later = [r for rep in reps for r in rep["rows"] if r["generation"] >= 2]
    return {"seeds": [rep["seed"] for rep in reps], "passes_g2_6": int(sum(r["pass"] for r in later)),
            "n": len(later),
            "mean_max_abs_bias_g2_6": float(np.mean([max(abs(r["gate"]["black"]["bias"]), abs(r["gate"]["white"]["bias"]))
                                                      for r in later]))}


def judge_groups(argv: list[str]) -> int:
    """E5d pre-registered rule. argv: NAME=file,file,file ... ; the group named `base` is the reference.
    A group proceeds if over generations 2-6 of its seeds (1) passes >= 8 of 15, (2) passes >= base + 4,
    (3) mean max(|bias_black|, |bias_white|) < base. Several proceed -> more passes, then smaller bias."""
    groups = {}
    for a in argv:
        name, files = a.split("=", 1)
        groups[name] = _group_stats([json.loads(Path(f).read_text(encoding="utf-8")) for f in files.split(",")])
    base = groups["base"]
    out = {"groups": groups, "verdicts": {}}
    for name, gs in groups.items():
        if name == "base":
            continue
        c = {"passes>=8": gs["passes_g2_6"] >= 8, "passes>=base+4": gs["passes_g2_6"] >= base["passes_g2_6"] + 4,
             "bias<base": gs["mean_max_abs_bias_g2_6"] < base["mean_max_abs_bias_g2_6"]}
        out["verdicts"][name] = c | {"proceed": all(c.values())}
    ok = [n for n, v in out["verdicts"].items() if v["proceed"]]
    ok.sort(key=lambda n: (-groups[n]["passes_g2_6"], groups[n]["mean_max_abs_bias_g2_6"]))
    out["decision"] = ok[0] if ok else "STOP"
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "judge":
        return judge(argv[1:])
    if argv and argv[0] == "judge-groups":
        return judge_groups(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-registry", default="benchmarks/reanalysis/e5/registry_T.json")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--proof", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=None, help="EMA decay of the learner weights (candidate = EMA)")
    ap.add_argument("--lr", type=float, default=None, help="override train.lr")
    ap.add_argument("--out", required=True)
    return screen(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
