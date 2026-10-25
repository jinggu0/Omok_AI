"""Why did the value of proven-WIN positions drop? (PROGRESS Phase 8 E5 follow-up)

    python -m dojozero.tools.value_drift --out benchmarks/reanalysis/e5/value_drift.json

Network-only value of many checkpoints on solver-proven positions from three sources:
  test      E4 held-out test (RIF, DojoZero-to-move positions of the diagnosis games)
  val       E4 val split (RIF, other openings)
  selfplay  T-arm self-play positions with solver proofs (OMOKDOJANG_DAN + Taraguchi, gens 1-6)
For each: mean value on proven wins / losses, their midpoint (bias: 0 = symmetric), half the gap
(separation) and AUROC (win vs loss = discrimination), by colour and, on test/val, by proof type
and proof length. Also the replay value targets (game results) by colour of the side to move.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from ..replay import ReplayBuffer
from ..tools.reanalysis_eval import auroc
from ..trainer.reanalysis import SPLITS

REPO = Path(__file__).resolve().parents[3]


@torch.no_grad()
def values(model, feats: np.ndarray, device) -> np.ndarray:
    out = []
    for s in range(0, len(feats), 512):
        x = torch.from_numpy(feats[s:s + 512].reshape(-1, feats.shape[1], 15, 15).astype(np.float32)).to(device)
        w = torch.softmax(model(x)["wdl"].float(), dim=1).cpu().numpy()
        out.append(w[:, 0] - w[:, 2])
    return np.concatenate(out) if out else np.zeros(0)


def stats(v: np.ndarray, res: np.ndarray) -> dict:
    w, l = v[res == 1], v[res == -1]
    if not len(w) or not len(l):
        return {"n_win": int(len(w)), "n_loss": int(len(l))}
    return {"n_win": int(len(w)), "n_loss": int(len(l)), "win": float(w.mean()), "loss": float(l.mean()),
            "bias": float((w.mean() + l.mean()) / 2), "separation": float((w.mean() - l.mean()) / 2),
            "auroc": auroc(v, (res == 1).astype(int))}


def main(argv: list[str] | None = None) -> int:
    from ..model.load import load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    models = {"E4_base": "runs/taraguchi_v1/models/gen_001.pt", "E4_cand": "runs/e4_reanalysis_v1/checkpoints/step_1500.pt"}
    for arm in ("T", "C"):
        for g in range(1, 7):
            models[f"{arm}{g}"] = f"runs/e5_rl_{arm}/models/gen_{g:03d}.pt"
    # RIF positions (E4 arrays) with proof type / length from the labels file.
    z = np.load(REPO / "data/reanalysis_v1/arrays.npz", allow_pickle=False)
    lab = {}
    for line in gzip.open(REPO / "data/reanalysis_v1/labels.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line)
        lab[r["id"]] = r["pos"]
    sets = {}
    for name in ("test", "val"):
        sel = np.flatnonzero((z["split"] == SPLITS[name]) & (z["pos_result"] != 0))
        reason = np.array([lab[str(i)]["reason"] for i in z["ids"][sel]])
        depth = np.array([lab[str(i)]["depth"] or 0 for i in z["ids"][sel]])
        sets[name] = {"feats": z["feats"][sel], "res": z["pos_result"][sel].astype(int), "side": z["side"][sel],
                      "reason": reason, "depth": depth}
    # T-arm self-play proven positions (labels attached in ingest order).
    feats, res, side = [], [], []
    spec = None
    for g in range(1, 7):
        chunk = REPO / f"runs/e5_rl_T/selfplay/gen_{g:03d}.jsonl.gz"
        lp = chunk.with_suffix("").with_suffix(".labels.npz")
        if not lp.exists():
            continue
        if spec is None:
            _, spec, _ = load_model(str(REPO / models["T1"]), "cpu")
        rb = ReplayBuffer(spec, window=1)
        rb.ingest(chunk, g)
        lz = np.load(lp)
        keep = lz["proof"] != 0
        idx = lz["idx"][keep]
        blk = rb.blocks[0]
        feats.append(blk.feats[idx])
        res.append(lz["proof"][keep].astype(int))
        plane = list(spec.core().plane_names).index("actor_is_black")
        side.append(np.where(blk.feats[idx, plane].max(axis=1) > 0, 0, 1))
    sets["selfplay"] = {"feats": np.concatenate(feats), "res": np.concatenate(res), "side": np.concatenate(side)}
    report: dict = {"sets": {k: {"n": int(len(v["res"])), "wins": int((v["res"] == 1).sum())} for k, v in sets.items()},
                    "models": {}}
    for name, ck in models.items():
        model, _, _ = load_model(str(REPO / ck), args.device)
        r = {}
        for sname, s in sets.items():
            v = values(model, s["feats"], args.device)
            r[sname] = stats(v, s["res"]) | {"black": stats(v[s["side"] == 0], s["res"][s["side"] == 0]),
                                            "white": stats(v[s["side"] == 1], s["res"][s["side"] == 1])}
            if "reason" in s:
                w = s["res"] == 1
                r[sname]["win_by_reason"] = {rs: float(v[w & (s["reason"] == rs)].mean()) for rs in ("IMMEDIATE_WIN", "VCF", "VCT")}
                r[sname]["win_by_depth"] = {f"{a}-{b}": float(v[w & (s["depth"] >= a) & (s["depth"] <= b)].mean())
                                            for a, b in ((1, 3), (4, 9), (10, 19), (20, 80)) if (w & (s["depth"] >= a) & (s["depth"] <= b)).any()}
        report["models"][name] = r
        print(name, {k: {kk: round(vv, 3) for kk, vv in r[k].items() if isinstance(vv, float)} for k in sets}, flush=True)
        del model
    # Replay value targets by colour of the side to move (game results, before any proof override).
    tg = {}
    for arm in ("T", "C"):
        zs = {"black": [], "white": []}
        for g in range(1, 7):
            chunk = REPO / f"runs/e5_rl_{arm}/selfplay/gen_{g:03d}.jsonl.gz"
            if not chunk.exists():
                continue
            rb = ReplayBuffer(spec, window=1)
            rb.ingest(chunk, g)
            blk = rb.blocks[0]
            plane = list(spec.core().plane_names).index("actor_is_black")
            black = blk.feats[:, plane].max(axis=1) > 0
            zs["black"].append(blk.value[black])
            zs["white"].append(blk.value[~black])
        tg[arm] = {c: {"n": int(sum(len(x) for x in v)), "mean_z": float(np.concatenate(v).mean())} for c, v in zs.items()}
    report["replay_targets"] = tg
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["sets"]), json.dumps(tg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
