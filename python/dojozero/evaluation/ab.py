"""Paired A/B feature test against external engines (EVALUATION §19).

    python -m dojozero.evaluation.ab --baseline dojo:<ckpt> --candidate dojo:<ckpt>:defense=8 \
        --opponents pela_2023 yixin_2018 --pairs 40 --time-ms 500 --feature "root defense filter" ...

Both arms play the same openings against the same opponents with the same time per move. The arms
are interleaved per opening (baseline pair, candidate pair, next opening ...) so drift in machine
state hits both equally. Unit = (opponent, opening) pair; the paired difference
d = pair score(candidate) - pair score(baseline) is pooled over opponents. Decision (pre-registered):
ACCEPT if the 95% interval of mean(d) is above 0 and the candidate has no failures; REJECT if it is
below 0 or the candidate fails; otherwise INCONCLUSIVE (not a success). The result goes to the
Strength DB `experiment` table.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import _core
from . import corpus as corpus_mod
from .engine_games import external_player
from .match import elo
from .strength_db import StrengthDB
from .tournament import PROFILE, git_commit, hardware_id, keep_awake, make_tournament_player, play_game

REPO = Path(__file__).resolve().parents[3]


def paired_stats(diffs: list[float], seed: int = 0) -> dict:
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    boot = [float(rng.choice(d, n).mean()) for _ in range(5000)]
    lo_b, hi_b = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    lo, hi = min(mean - 1.96 * se, lo_b), max(mean + 1.96 * se, hi_b)  # the wider interval
    return {"pairs": n, "mean_diff": mean, "ci95": [lo, hi], "se": se}


def decide(stats: dict, candidate_failures: int) -> str:
    if candidate_failures > 0 or stats["ci95"][1] < 0:
        return "REJECT"
    if stats["ci95"][0] > 0:
        return "ACCEPT"
    return "INCONCLUSIVE"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--openings", default="renju_balanced_v1")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--time-ms", type=float, default=500)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--feature", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    s = corpus_mod.load(args.openings)
    ops = s.openings[args.offset:args.offset + args.pairs]
    db = StrengthDB(args.db)
    cache: dict = {}
    arms = {"baseline": make_tournament_player(args.baseline, args.device, cache, args.time_ms, 1),
            "candidate": make_tournament_player(args.candidate, args.device, cache, args.time_ms, 1)}
    rows, matches = [], []
    t0 = time.time()
    with keep_awake():
        for opp in args.opponents:
            ext = external_player(opp, int(args.time_ms))
            ext.reuse_process = True
            mids = {}
            for name, pl in arms.items():
                for p in (pl, ext):
                    db.register_engine(p.engine_id, p.engine_kind, name=p.engine_id, config=p.describe(),
                                       model_hash=getattr(p, "model_hash", ""), git_commit=git_commit())
                mids[name] = db.start_match(pl.engine_id, ext.engine_id, tier="B", rule_profile=PROFILE,
                                            profile_hash=str(_core.profile_hash(PROFILE)), opening_set_id=s.set_id,
                                            opening_set_hash=s.set_hash, compute_mode="fixed_time",
                                            budget={"time_ms_per_move": args.time_ms, "a": pl.describe(),
                                                    "b": ext.describe()},
                                            hardware_id=hardware_id(), concurrent_games=1, git_commit=git_commit(),
                                            command=f"ab {name} {args.feature} vs {opp}")
                matches.append(mids[name])
            for k, op in enumerate(ops):
                pair = {}
                for name, pl in arms.items():
                    recs = [play_game(pl, ext, op, s, a_first, args.offset + k) for a_first in (True, False)]
                    for r in recs:
                        db.add_game(mids[name], r)
                    pair[name] = {"score": float(np.mean([r["score_a"] for r in recs])),
                                  "failures": [r["termination_reason"] for r in recs
                                               if r["failed_side"] == "A"]}
                rows.append({"opponent": opp, "opening": op["opening_id"], **{n: pair[n] for n in pair},
                             "diff": pair["candidate"]["score"] - pair["baseline"]["score"]})
                if (k + 1) % 5 == 0:
                    st = paired_stats([r["diff"] for r in rows])
                    print(f"  {opp} {k + 1}/{len(ops)}: mean diff {st['mean_diff']:+.3f} "
                          f"[{st['ci95'][0]:+.3f}, {st['ci95'][1]:+.3f}] ({time.time() - t0:.0f}s)", flush=True)
            ext.end_game(final=True)
            for mid in mids.values():
                db.finish_match(mid, {"concurrent_games": 1})
    st = paired_stats([r["diff"] for r in rows])
    cand_fail = sum(len(r["candidate"]["failures"]) for r in rows)
    per_opp = {}
    for opp in args.opponents:
        rr = [r for r in rows if r["opponent"] == opp]
        b = float(np.mean([r["baseline"]["score"] for r in rr]))
        c = float(np.mean([r["candidate"]["score"] for r in rr]))
        per_opp[opp] = {"baseline": b, "candidate": c, "paired": paired_stats([r["diff"] for r in rr]),
                        "elo_baseline": elo(b), "elo_candidate": elo(c)}
    decision = decide(st, cand_fail)
    result = {"pooled": st, "per_opponent": per_opp, "candidate_failures": cand_fail, "decision": decision,
              "time_ms": args.time_ms, "openings": {"id": s.set_id, "hash": s.set_hash, "offset": args.offset},
              "baseline": arms["baseline"].describe(), "candidate": arms["candidate"].describe()}
    db.add_experiment(args.baseline, args.candidate, args.feature, args.hypothesis,
                      "paired pair-score difference vs external opponents (pooled)", "B", matches, result, decision)
    print(json.dumps(result, indent=1, default=str))
    Path(args.out).write_text(json.dumps(result | {"rows": rows}, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
