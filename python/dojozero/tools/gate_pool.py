"""Balanced value gate pool from fresh external games (PROGRESS Phase 8 E5j).

    python -m dojozero.tools.gate_pool collect --set renju_gate_e5_v1 --out data/gate_e5_v1
    python -m dojozero.tools.gate_pool label   --out data/gate_e5_v1
    python -m dojozero.tools.gate_pool arrays  --out data/gate_e5_v1 --ckpt runs/e5e_rl_T/models/gen_045.pt

collect: DojoZero-to-move positions (ply >= 5, NORMAL_RENJU, non-terminal) of the games recorded on one
opening set, unique by D4 canonical hash, dropping every hash in the E4 reanalysis positions.
label:   position result for the side to move with the E4 budget. UNKNOWN stays UNKNOWN (pos_result 0).
arrays:  `arrays.npz` with the fields tactical.build_balanced_gate_set reads; every row is split val and
         DojoZero-to-move, so `balanced_gate build --arrays <out>/arrays.npz --dojo-only` samples the gate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _core
from ..features import NUM_POINTS
from ..trainer.reanalysis import SIDES, SPLITS
from .reanalysis_data import PROFILE, SOLVER, sha256_file

REPO = Path(__file__).resolve().parents[3]
RESULT = {"WIN": 1, "LOSS": -1, "UNKNOWN": 0}


def game_positions(game_id: str, moves: list[str], a_color: str, engine_a: str, engine_b: str) -> list[dict]:
    """DojoZero-to-move positions of one recorded game, in move order."""
    if engine_a.startswith("dojo"):
        dojo_side, arm = a_color, engine_a
    elif engine_b.startswith("dojo"):
        dojo_side, arm = ("WHITE" if a_color == "BLACK" else "BLACK"), engine_b
    else:
        return []
    g = _core.GameState(PROFILE)
    out = []
    for ply, m in enumerate(moves):
        if g.phase == "NORMAL_RENJU" and not g.is_terminal and ply >= 5 and g.side_to_move == dojo_side:
            out.append({"id": f"{game_id}@{ply}", "game_id": game_id, "ply": ply, "moves": moves[:ply],
                        "side": g.side_to_move, "arm": arm, "canonical_hash": int(g.canonical_hash())})
        if not g.apply(m):
            raise ValueError(f"{game_id}: illegal recorded move {m} at ply {ply}")
    return out


def collect(db: str, set_id: str, out: Path, exclude: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("""select g.game_id, g.moves, g.a_color, m.engine_a, m.engine_b from game g join match m
                          using(match_id) where m.opening_set_id = ? and m.rule_profile = ? order by g.game_id""",
                       (set_id, PROFILE)).fetchall()
    with gzip.open(exclude, "rt", encoding="utf-8") as fh:
        seen = {json.loads(line)["canonical_hash"] for line in fh}
    stats, positions = Counter(), []
    for gid, mv, a_color, ea, eb in rows:
        stats["games"] += 1
        for p in game_positions(gid, json.loads(mv), a_color, ea, eb):
            if p["canonical_hash"] in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(p["canonical_hash"])
            positions.append(p)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "positions.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in positions:
            fh.write(json.dumps(p) + "\n")
    return {"set": set_id, **stats, "positions": len(positions), "by_side": dict(Counter(p["side"] for p in positions)),
            "by_arm": dict(Counter(p["arm"] for p in positions)), "sha256": sha256_file(out / "positions.jsonl.gz")}


def _state(moves: list[str]):
    g = _core.GameState(PROFILE)
    for m in moves:
        if not g.apply(m):
            raise ValueError(f"illegal move {m}")
    return g


def label(out: Path, threads: int) -> dict:
    positions = [json.loads(line) for line in gzip.open(out / "positions.jsonl.gz", "rt", encoding="utf-8")]

    def work(p: dict) -> dict:
        d = _core.analyze_tactics(_state(p["moves"]), SOLVER["vcf_nodes"], SOLVER["vct_nodes"], SOLVER["loss_nodes"], 0.0)
        return {"id": p["id"], "result": d["result"], "reason": d["reason"], "nodes": d["nodes"]}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = []
        for r in ex.map(work, positions):
            results.append(r)
            if len(results) % 1000 == 0:
                print(f"labelled {len(results)}/{len(positions)}", flush=True)
    with gzip.open(out / "labels.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    cells = Counter(f"{p['side']}_{r['result']}" for p, r in zip(positions, results))
    return {"labelled": len(results), "cells": dict(cells),
            "solver": {k: SOLVER[k] for k in ("vcf_nodes", "vct_nodes", "loss_nodes")},
            "sha256": sha256_file(out / "labels.jsonl.gz")}


def arrays(out: Path, ckpt: str) -> dict:
    from ..model.load import load_model
    _, spec, _ = load_model(ckpt, "cpu")
    cfg = spec.core()
    positions = [json.loads(line) for line in gzip.open(out / "positions.jsonl.gz", "rt", encoding="utf-8")]
    labels = {r["id"]: r for r in (json.loads(line) for line in gzip.open(out / "labels.jsonl.gz", "rt", encoding="utf-8"))}
    n = len(positions)
    feats = np.zeros((n, spec.num_planes, NUM_POINTS), np.uint8)
    for i, p in enumerate(positions):
        feats[i] = np.asarray(_core.encode(_state(p["moves"]), cfg)).reshape(-1, NUM_POINTS)
    np.savez_compressed(out / "arrays.npz", feats=feats, ids=np.array([p["id"] for p in positions]),
                        arm=np.array([p["arm"] for p in positions]),
                        side=np.array([SIDES[p["side"]] for p in positions], np.int8),
                        pos_result=np.array([RESULT[labels[p["id"]]["result"]] for p in positions], np.int8),
                        split=np.full(n, SPLITS["val"], np.int8), dojo_to_move=np.ones(n, np.int8))
    return {"positions": n, "feature_hash": spec.hash(), "sha256": sha256_file(out / "arrays.npz")}


def main(argv: list[str] | None = None) -> int:
    from ..evaluation.tournament import keep_awake
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "label", "arrays"])
    ap.add_argument("--set", default="renju_gate_e5_v1")
    ap.add_argument("--out", default=str(REPO / "data" / "gate_e5_v1"))
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--exclude", default=str(REPO / "data" / "reanalysis_v1" / "positions.jsonl.gz"))
    ap.add_argument("--ckpt", default="runs/e5e_rl_T/models/gen_045.pt")
    ap.add_argument("--threads", type=int, default=7)
    args = ap.parse_args(argv)
    out = Path(args.out)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    with keep_awake():
        if args.cmd == "collect":
            man["collect"] = collect(args.db, args.set, out, Path(args.exclude))
        elif args.cmd == "label":
            man["label"] = label(out, args.threads)
        else:
            man["arrays"] = arrays(out, args.ckpt)
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
