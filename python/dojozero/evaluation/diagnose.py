"""Weakness diagnosis experiments (EVALUATION §18, §20, RQ-01/RQ-05).

    python -m dojozero.evaluation.diagnose scaling --dojo dojo:<ckpt> --opponents pela_2023 yixin_2018 \
        --times 500 1000 2000 5000 --pairs 12
    python -m dojozero.evaluation.diagnose visits  --dojo dojo:<ckpt> --opponents pela_2023 yixin_2018 \
        --sims 64 256 1024 --opponent-ms 500 --pairs 12

scaling: both sides get the same time per move (score(t) curve). visits: DojoZero searches a fixed
number of simulations while the opponent keeps a fixed time per move — if DojoZero's score rises
with its own compute, the search budget limits it; if it stays flat, evaluation quality does.
All conditions use the same openings (the first `pairs` of the set), so conditions are paired.
Every match goes to the Strength DB; a JSON summary lists the match ids for the loss analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import corpus as corpus_mod
from .engine_games import external_player
from .strength_db import StrengthDB
from .tournament import keep_awake, make_tournament_player, run_match

REPO = Path(__file__).resolve().parents[3]


def _row(kind: str, opp: str, value, res: dict) -> dict:
    s = res["summary"]
    return {"kind": kind, "opponent": opp, "value": value, "match_id": res["match_id"], "pairs": s["pairs"],
            "score": s["score"], "score_ci95": s["score_ci95"], "elo": s["elo"], "elo_ci95": s["elo_ci95"],
            "failures": res["failures"], "dojo_ms_per_move": res["compute"]["mean_ms_per_move_a"],
            "dojo_search": res["compute"]["search_a"], "opp_ms_per_move": res["compute"]["mean_ms_per_move_b"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("scaling", "visits"))
    ap.add_argument("--dojo", required=True)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--openings", default="renju_balanced_v1")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--times", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    ap.add_argument("--sims", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--opponent-ms", type=int, default=500)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    s = corpus_mod.load(args.openings)
    db = StrengthDB(args.db)
    cache: dict = {}
    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with keep_awake():
        for opp in args.opponents:
            values = args.times if args.mode == "scaling" else args.sims
            for v in values:
                t0 = time.time()
                if args.mode == "scaling":
                    dojo = make_tournament_player(args.dojo, args.device, cache, float(v), 1)
                    ext = external_player(opp, v)
                    res = run_match(dojo, ext, s, args.pairs, db, tier="A", time_ms=float(v),
                                    command=f"diagnose scaling t={v} {args.dojo} vs {opp}", verbose=False)
                else:
                    dojo = make_tournament_player(f"{args.dojo}:sims={v}", args.device, cache, None, 1)
                    ext = external_player(opp, args.opponent_ms)
                    res = run_match(dojo, ext, s, args.pairs, db, tier="A", time_ms=None,
                                    command=f"diagnose visits sims={v} vs {opp}@{args.opponent_ms}ms",
                                    verbose=False)
                rows.append(_row(args.mode, opp, v, res))
                r = rows[-1]
                print(f"{args.mode} {opp} {v}: score {r['score']:.3f} [{r['score_ci95'][0]:.3f}, "
                      f"{r['score_ci95'][1]:.3f}] elo {r['elo']:+.0f} [{r['elo_ci95'][0]:+.0f}, {r['elo_ci95'][1]:+.0f}] "
                      f"dojo {r['dojo_ms_per_move']:.0f} ms/move sims {r['dojo_search'].get('sims', 0):.0f} "
                      f"failures {r['failures']} ({time.time() - t0:.0f}s)", flush=True)
                out.write_text(json.dumps({"mode": args.mode, "dojo": args.dojo, "openings": s.set_id,
                                           "hash": s.set_hash, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
