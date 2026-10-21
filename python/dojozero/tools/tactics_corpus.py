"""Export tactical-solver performance corpora (PROGRESS Phase 8 E3b).

    python -m dojozero.tools.tactics_corpus --ckpt runs/taraguchi_v1/models/gen_001.pt

Writes benchmarks/perf/<name>.json, each {"schema_version", "name", "source", "positions": [{"id",
"profile", "black", "white", "side", ...}]} plus a sha256 of the canonical positions list:

  bench_problems   all fixed benchmark problems (benchmarks/{tactical,vcf,vct})
  after_blunder    position right after each classified DojoZero tactical blunder (opponent to move;
                   the opponent's forced win was proven by the 1 s solver check)
  filter_workload  for each blunder position, the position after each of the network's top-8 policy
                   moves (what the root defense filter checks; policy order stands in for search visits,
                   which are not reproducible)

Positions are stone lists (no move history): the solvers only see the board and the side to move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

from .. import _core

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "benchmarks" / "perf"


def spec_of(g, pid: str, profile: str, **extra) -> dict:
    b = np.asarray(g.board())
    return {"id": pid, "profile": profile, "black": [_core.point_to_str(int(p)) for p in np.flatnonzero(b == 1)],
            "white": [_core.point_to_str(int(p)) for p in np.flatnonzero(b == 2)], "side": g.side_to_move} | extra


def write(name: str, source: str, positions: list[dict]) -> str:
    blob = json.dumps(positions, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(blob.encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"schema_version": 1, "name": name, "source": source, "sha256": h,
                                                  "positions": positions}, indent=0), encoding="utf-8")
    print(f"{name}: {len(positions)} positions, sha256 {h[:16]}")
    return h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args(argv)

    probs = []
    for f in sorted((REPO / "benchmarks").glob("*/*.json")):
        if f.parent.name in ("openings", "perf"):
            continue
        for p in json.loads(f.read_text(encoding="utf-8")).get("problems", []):
            pos = p["position"]
            probs.append({"id": p["id"], "profile": p.get("profile", "RENJU_RIF"), "black": pos.get("black", []),
                          "white": pos.get("white", []), "side": pos.get("side"),
                          "expected": p["expected"]["result"], "file": f"{f.parent.name}/{f.name}"})
    write("bench_problems", "benchmarks/{tactical,vcf,vct}/*.json", probs)

    from ..evaluation.match import make_player
    player = make_player(f"engine:{args.ckpt}", "cpu", {})
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute("""select a.result_json, g.moves, g.game_id, g.a_color from analysis a join game g using(game_id)
                          where a.kind='loss_category' order by g.game_id""").fetchall()
    after, work = [], []
    for rj, mv, gid, dcol in rows:
        r = json.loads(rj)
        if r["category"] != "tactical_blunder":
            continue
        ply = r["primary"]["ply"]
        moves = json.loads(mv)
        g = _core.GameState("RENJU_GOMOCUP")
        assert all(g.apply(m) for m in moves[:ply])
        h = g.copy()
        assert h.apply(moves[ply])
        after.append(spec_of(h, f"{gid}@{ply}", "RENJU_GOMOCUP", dojo_color=dcol, blunder=moves[ply]))
        feats = np.asarray(_core.encode(g, player.feature_config))[None]
        logits = np.where(np.asarray(_core.legal_mask(g)).astype(bool), player.evaluator(feats)["policy"][0], -np.inf)
        for rank, p in enumerate(np.argsort(-logits)[: args.top_k]):
            if not np.isfinite(logits[p]) or int(p) == 225:
                continue
            c = g.copy()
            if c.apply(_core.point_to_str(int(p))) and not c.is_terminal:
                work.append(spec_of(c, f"{gid}@{ply}#{rank}", "RENJU_GOMOCUP", dojo_color=dcol,
                                    candidate=_core.point_to_str(int(p)), policy_rank=rank,
                                    is_blunder=_core.point_to_str(int(p)) == moves[ply]))
    write("after_blunder", "DB analysis kind=loss_category, category=tactical_blunder (Phase 8 E2c)", after)
    write("filter_workload", f"top-{args.top_k} policy moves of {args.ckpt} at each blunder position", work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
