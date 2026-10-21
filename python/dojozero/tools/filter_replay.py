"""Replay the classified tactical-blunder positions through the root defense filter (Phase 8 E3b).

    python -m dojozero.tools.filter_replay --arms none,A,B --out benchmarks/perf/results/replay.json

Each arm searches every position (move time --ms, PUCT, no root noise) and plays its choice; the
choice is then judged by a solver. Arms (name=spec, or a preset):
  none   filter off
  A      filter with the frozen ee1d011 solver (defimpl=ref)
  B      filter with the optimized solver (defimpl=live, unused VCF time carried to VCT)
  B0     optimized solver with the ee1d011 40/40/20 time split (defcarry=0)
  P6     proof-aware MCTS (6 solver threads, depth 3), no post-search filter
  custom: name=k/share/threads/impl/carry[/proof_threads/proof_depth]   e.g.  B30=8/0.3/8/live/1

Judges (the opponent, to move after the choice, has a proven forced win = "still a blunder"):
  judge_ms   ee1d011 solver, VCF 200k / VCT 100k nodes, 1000 ms wall clock: the judge used for the
             E3a replay numbers (187 / 149); time-limited, so it varies slightly with machine load
  judge_nodes  VCF 200k / VCT 100k nodes, no clock: deterministic (live solver; node-limited results
             are bit-identical to the ee1d011 solver, see benchmarks/perf differential runs)
UNKNOWN is never counted as a blunder or as safe-by-proof; it is reported separately.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

from .. import _core
from ..evaluation.match import make_player
from ..evaluation.tournament import keep_awake
from ..search.driver import search_many

REPO = Path(__file__).resolve().parents[3]
# k, share, threads, impl, carry, proof_threads, proof_max_depth
PRESETS = {"none": (0, 0.5, 8, "live", True, 0, 3), "A": (8, 0.5, 8, "ref", True, 0, 3),
           "B": (8, 0.5, 8, "live", True, 0, 3), "B0": (8, 0.5, 8, "live", False, 0, 3),
           "P6": (0, 0.5, 8, "live", True, 6, 3)}  # P6: proof-aware MCTS, 6 solver threads, no post filter


def parse_arm(spec: str) -> tuple:
    p = spec.split("/")
    p += ["0", "3"][len(p) - 5:] if len(p) < 7 else []
    return int(p[0]), float(p[1]), int(p[2]), p[3], p[4] not in ("0", "false"), int(p[5]), int(p[6])


def load_positions(db: str, mode: str = "blunder", coverage: str | None = None) -> list[tuple[str, object]]:
    """blunder: the position before each classified tactical blunder (E2c). decision: the position
    before the earliest DojoZero move of the run of proven-lost moves ending at the blunder (E3c
    backtrack; the blunder itself when no earlier move was proven lost) = where the game was lost."""
    first = {}
    if mode == "decision":
        cov = json.loads(Path(coverage).read_text(encoding="utf-8"))
        for r in cov["rows"]:
            ply = r["ply"]
            for b in r.get("backtrack", []):  # newest first
                if b["code"] != "REFUTED":
                    break
                ply = b["ply"]
            first[r["id"].split("@")[0]] = ply
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for rj, mv, gid in con.execute("""select a.result_json, g.moves, g.game_id from analysis a join game g
                                       using(game_id) where a.kind='loss_category' order by g.game_id"""):
        r = json.loads(rj)
        if r["category"] != "tactical_blunder":
            continue
        ply = r["primary"]["ply"]
        if mode == "decision":
            if gid not in first:
                continue
            ply = first[gid]
        g = _core.GameState("RENJU_GOMOCUP")
        assert all(g.apply(m) for m in json.loads(mv)[:ply])
        out.append((f"{gid}@{ply}", g))
    return out


def pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def main(argv: list[str] | None = None) -> int:
    with keep_awake():  # also opts out of Windows power throttling (both arms alike)
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/taraguchi_v1/models/gen_001.pt")
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ms", type=float, default=500.0)
    ap.add_argument("--arms", default="none,A,B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--positions", choices=["blunder", "decision"], default="blunder")
    ap.add_argument("--coverage", default=str(REPO / "benchmarks" / "perf" / "results" / "coverage_247.json"))
    ap.add_argument("--judge-ms", type=int, default=1, help="0 = skip the time-limited E3a judge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    positions = load_positions(args.db, args.positions, args.coverage)
    if args.limit:
        positions = positions[: args.limit]
    pl = make_player(f"engine:{args.ckpt}", args.device, {})
    s = pl.settings
    s.root, s.gumbel_noise, s.simulations, s.max_ms = "PUCT", False, 100000, args.ms
    judged: dict[tuple[str, str], dict] = {}

    def judge(pid: str, g, a: str) -> dict:
        if (pid, a) not in judged:
            h = g.copy()
            assert h.apply(a)
            if h.is_terminal:
                j = {"ms": "TERMINAL", "nodes": "TERMINAL"}
            else:
                j = {"ms": _core.analyze_tactics(h, 200000, 100000, 0, 1000.0, "ref", False)["result"] if args.judge_ms else "SKIPPED",
                     "nodes": _core.analyze_tactics(h, 200000, 100000, 0, 0.0, "live", False)["result"]}
            judged[(pid, a)] = j
        return judged[(pid, a)]

    report = {"ckpt": args.ckpt, "move_ms": args.ms, "positions": len(positions), "position_mode": args.positions,
              "arms": {}}
    for arm in args.arms.split(","):
        name, _, spec = arm.partition("=")
        k, share, threads, impl, carry, pthreads, pdepth = PRESETS[name] if not spec else parse_arm(spec)
        s.defense_k, s.defense_share, s.defense_threads, s.defense_impl, s.defense_carry = k, share, threads, impl, carry
        s.proof_threads, s.proof_max_depth = pthreads, pdepth
        rows = []
        t0 = time.perf_counter()
        for pid, g in positions:
            r = search_many([g], pl.evaluator, s, pl.feature_config)[0]
            d = r.get("defense", {})
            j = judge(pid, g, r["action"])
            rows.append({"id": pid, "color": g.side_to_move, "action": r["action"], "sims": r["simulations"],
                         "judge_ms": j["ms"], "judge_nodes": j["nodes"], "checked": d.get("checked", 0),
                         "refuted": len(d.get("refuted", [])), "changed": d.get("changed", False),
                         "defense_ms": d.get("ms", 0.0), "budget_ms": d.get("budget_ms"), "checks": d.get("checks", []),
                         "proof_jobs": r.get("proof_jobs", 0), "proof_found": r.get("proof_found", 0),
                         "proof_nodes": r.get("proof_nodes", 0), "proven": r.get("proven", 2),
                         "children_proven_loss": sum(c["proven"] == -1 for c in r["children"])})
        wall = time.perf_counter() - t0
        summ = {"k": k, "share": share, "threads": threads, "impl": impl, "carry": carry,
                "proof_threads": pthreads, "proof_max_depth": pdepth, "wall_s": wall,
                "proof_jobs_mean": statistics.fmean([x["proof_jobs"] for x in rows]),
                "proof_found_mean": statistics.fmean([x["proof_found"] for x in rows]),
                "proof_nodes_mean": statistics.fmean([x["proof_nodes"] for x in rows]),
                "root_children_proven_loss_mean": statistics.fmean([x["children_proven_loss"] for x in rows]),
                "root_proven": dict(Counter(x["proven"] for x in rows))}
        for jk in ("judge_ms", "judge_nodes"):
            for col in ("ALL", "BLACK", "WHITE"):
                sel = [x for x in rows if col == "ALL" or x["color"] == col]
                summ[f"{jk}_{col}"] = {"n": len(sel), "blunder": sum(x[jk] == "WIN" for x in sel),
                                       "unknown": sum(x[jk] == "UNKNOWN" for x in sel)}
        why = Counter()
        for x in rows:
            if x["judge_nodes"] != "WIN":
                why["safe" if x["judge_nodes"] != "UNKNOWN" else "judge_unknown"] += 1
            elif x["checked"] and x["refuted"] == x["checked"]:
                why["all_k_refuted"] += 1
            else:
                why["budget_miss"] += 1
        checks = [c for x in rows for c in x["checks"] if c.get("result") not in ("ILLEGAL", "TERMINAL")]
        cms = [c["ms"] for c in checks]
        summ.update({
            "why_judge_nodes": dict(why),
            "checks": len(checks), "check_results": dict(Counter(c["result"] for c in checks)),
            "check_unknown_rate": (sum(c["result"] == "UNKNOWN" for c in checks) / len(checks)) if checks else None,
            "check_ms_mean": statistics.fmean(cms) if cms else None, "check_ms_p95": pct(cms, 0.95),
            "check_nodes_mean": statistics.fmean([c["nodes"] for c in checks]) if checks else None,
            "candidates_checked_mean": statistics.fmean([x["checked"] for x in rows]),
            "unsafe_candidates_mean": statistics.fmean([x["refuted"] for x in rows]),
            "override_rate": sum(x["changed"] for x in rows) / len(rows),
            "sims_mean": statistics.fmean([x["sims"] for x in rows]), "sims_median": pct([x["sims"] for x in rows], 0.5),
            "defense_ms_mean": statistics.fmean([x["defense_ms"] for x in rows]),
            "defense_ms_p95": pct([x["defense_ms"] for x in rows], 0.95),
        })
        report["arms"][name] = {"summary": summ, "rows": rows}
        print(name, json.dumps({k2: v for k2, v in summ.items() if not k2.startswith("judge_ms_")}), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
