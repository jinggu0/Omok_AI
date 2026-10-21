"""Policy candidate coverage at the classified tactical-blunder positions (Phase 8 E3c, "Case C").

    python -m dojozero.tools.coverage_replay --out benchmarks/perf/results/coverage_247.json

For each of the 247 positions BEFORE the blunder (DojoZero to move; rebuilt as filter_replay.load_positions
does) every legal move is judged by the C++ solver on the position after it (opponent to move):
analyze_tactics(h, 200000, 100000, 0, 0.0) — node budgets only, no clock, deterministic.
  REFUTED   opponent has a proven forced win (WIN)
  WINNING   the move wins for DojoZero (terminal win, or opponent to move has no defense = LOSS): proven safe
  UNK_DONE  UNKNOWN, total nodes < VCT budget: VCF and threat-space df-pn both ended without a proof
            (not a proof of safety: threat-space search is incomplete and depth-capped)
  UNK_BUDGET UNKNOWN with a node budget (possibly) exhausted
UNKNOWN is never counted as a loss or as proven safe. "Not refuted" = WINNING + UNK_DONE + UNK_BUDGET.
Ranks are 1-based over legal point moves: policy rank by raw network logit (gen_001.pt), visit rank by
root visits of a 500 ms PUCT search without noise and with the defense filter off (ties, e.g. unvisited
moves, broken by prior; visit_rank of an unvisited move is reported but flagged).

--backtrack N: for each position, DojoZero's earlier moves (ply-2, ply-4, ...) are judged the same way
(the move actually played), up to N of them, to find where the game was already proven lost.
--earlier: for every position whose moves are all refuted, the full coverage analysis is repeated at
the earliest DojoZero decision of the trailing run of refuted played moves (the decision that let the
loss in, as far as the solver can prove).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _core
from ..evaluation.match import make_player
from ..evaluation.tournament import keep_awake
from ..search.driver import search_many
from .filter_replay import load_positions

REPO = Path(__file__).resolve().parents[3]
KS = (8, 16, 32)
CODES = ("REFUTED", "WINNING", "UNK_DONE", "UNK_BUDGET")


def load_games(db: str) -> list[tuple[str, list[str], int]]:
    """(id, moves, ply) in load_positions order; ply = index of the blunder move."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for rj, mv, gid in con.execute("""select a.result_json, g.moves, g.game_id from analysis a join game g
                                       using(game_id) where a.kind='loss_category' order by g.game_id"""):
        r = json.loads(rj)
        if r["category"] == "tactical_blunder":
            out.append((f"{gid}@{r['primary']['ply']}", json.loads(mv), r["primary"]["ply"]))
    return out


def state_at(moves: list[str], ply: int):
    g = _core.GameState("RENJU_GOMOCUP")
    assert all(g.apply(m) for m in moves[:ply])
    return g


def judge(g, a: str, vcf: int, vct: int) -> tuple[str, int, str]:
    h = g.copy()
    if not h.apply(a):
        return "ILLEGAL", 0, ""
    if h.is_terminal:
        return ("WINNING" if h.winner is not None and h.winner == g.actor else "REFUTED"), 0, "TERMINAL"
    d = _core.analyze_tactics(h, vcf, vct, 0, 0.0, "live", False)
    if d["result"] == "WIN":
        return "REFUTED", d["nodes"], d["reason"]
    if d["result"] == "LOSS":
        return "WINNING", d["nodes"], d["reason"]
    return ("UNK_DONE" if d["nodes"] < vct else "UNK_BUDGET"), d["nodes"], d["reason"]


def net_and_search(pl, g) -> dict:
    """Policy order and MCTS visit order over the legal point moves of g."""
    legal = [a for a in g.legal_actions() if a != "pass"]
    feats = np.asarray(_core.encode(g, pl.feature_config))[None]
    logits = pl.evaluator(feats)["policy"][0]
    lg = {a: float(logits[_core.str_to_point(a)]) for a in legal}
    pol = sorted(legal, key=lambda a: -lg[a])
    r = search_many([g], pl.evaluator, pl.settings, pl.feature_config)[0]
    kids = {c["action"]: c for c in r["children"]}
    vis = sorted(legal, key=lambda a: (-kids.get(a, {}).get("visits", 0), -kids.get(a, {}).get("prior", 0.0), -lg[a]))
    return {"policy_order": pol, "visit_order": vis, "visits": {a: kids[a]["visits"] for a in kids if kids[a]["visits"]},
            "action": r["action"], "sims": r["simulations"], "root_proven": r["proven"], "value": r["value"],
            "search_ms": r["elapsed_ms"]}


def summarize_position(moves: dict, ns: dict, extra: dict) -> dict:
    """moves: action -> (code, nodes, reason)."""
    pol, vis = ns["policy_order"], ns["visit_order"]
    prank = {a: i + 1 for i, a in enumerate(pol)}
    vrank = {a: i + 1 for i, a in enumerate(vis)}
    cnt = {c: sum(v[0] == c for v in moves.values()) for c in CODES}
    nonref = [a for a, v in moves.items() if v[0] != "REFUTED"]
    strong = [a for a, v in moves.items() if v[0] in ("WINNING", "UNK_DONE")]
    safe = [a for a, v in moves.items() if v[0] == "WINNING"]

    def best(sel, rank):
        return min((rank[a] for a in sel), default=None)

    bp = min(nonref, key=lambda a: prank[a], default=None)
    bv = min(nonref, key=lambda a: vrank[a], default=None)
    out = dict(extra)
    out.update({
        "n_legal": len(moves), "counts": cnt,
        "status": "ALL_REFUTED" if not nonref else ("HAS_PROVEN_SAFE" if safe else "ONLY_UNKNOWN"),
        "best_nonref_policy_rank": best(nonref, prank), "best_nonref_policy_move": bp,
        "best_nonref_policy_move_code": moves[bp][0] if bp else None,
        "best_nonref_visit_rank": best(nonref, vrank), "best_nonref_visit_move": bv,
        "best_nonref_visit_move_visits": ns["visits"].get(bv, 0) if bv else None,
        "best_strong_policy_rank": best(strong, prank), "best_strong_visit_rank": best(strong, vrank),
        "best_safe_policy_rank": best(safe, prank),
        "nonref_in_policy_top": {k: sum(prank[a] <= k for a in nonref) for k in KS},
        "nonref_in_visit_top": {k: sum(vrank[a] <= k and ns["visits"].get(a, 0) > 0 for a in nonref) for k in KS},
        "refuted_policy_top": {k: sum(moves[a][0] == "REFUTED" for a in pol[:k]) for k in KS},
        "visited_moves": len(ns["visits"]),
        "search_action": ns["action"], "search_action_code": moves.get(ns["action"], ("?",))[0],
        "search_action_policy_rank": prank.get(ns["action"]), "sims": ns["sims"], "root_proven": ns["root_proven"],
        "root_value": ns["value"], "search_ms": ns["search_ms"], "forced_root": ns["sims"] <= 1,
        "nonref_policy_ranks": sorted(prank[a] for a in nonref),
        "moves": {a: [v[0], v[1]] for a, v in sorted(moves.items(), key=lambda kv: prank[kv[0]])},
    })
    if "blunder" in extra:
        out["blunder_code"] = moves.get(extra["blunder"], ("?",))[0]
        out["blunder_policy_rank"] = prank.get(extra["blunder"])
        out["blunder_visit_rank"] = vrank.get(extra["blunder"])
    return out


def analyze_many(items: list[tuple[str, object, dict]], pl, ex, args) -> list[dict]:
    """items: (id, state, extra). Network/MCTS first for all (sequential, no solver load), then the solver."""
    nss = []
    for pid, g, _ in items:
        nss.append(net_and_search(pl, g))
    rows = []
    for (pid, g, extra), ns in zip(items, nss):
        legal = ns["policy_order"]
        res = list(ex.map(lambda a: judge(g, a, args.vcf, args.vct), legal))
        rows.append(summarize_position(dict(zip(legal, res)), ns, {"id": pid, "color": g.side_to_move, **extra}))
    return rows


def pctl(v: list, q: float):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for col in ("ALL", "BLACK", "WHITE"):
        sel = [r for r in rows if col == "ALL" or r["color"] == col]
        n = len(sel)
        if not n:
            continue
        has = [r for r in sel if r["status"] != "ALL_REFUTED"]
        st = {s: sum(r["status"] == s for r in sel) for s in ("ALL_REFUTED", "HAS_PROVEN_SAFE", "ONLY_UNKNOWN")}
        a = {"n": n, "status": st,
             "all_refuted_share": st["ALL_REFUTED"] / n,
             "has_nonrefuted": len(has),
             "has_strong": sum(r["best_strong_policy_rank"] is not None for r in sel),
             "nonref_moves_per_position_median": pctl([r["n_legal"] - r["counts"]["REFUTED"] for r in has], 0.5),
             "legal_median": pctl([r["n_legal"] for r in sel], 0.5),
             "codes_total": {c: sum(r["counts"][c] for r in sel) for c in CODES}}
        for key, rk in (("policy", "best_nonref_policy_rank"), ("visit", "best_nonref_visit_rank"),
                        ("strong_policy", "best_strong_policy_rank"), ("strong_visit", "best_strong_visit_rank")):
            v = [r[rk] for r in sel if r[rk] is not None]
            a[f"best_{key}_rank"] = {"n": len(v), "median": pctl(v, 0.5), "p75": pctl(v, 0.75), "p90": pctl(v, 0.9),
                                     "max": max(v) if v else None,
                                     "eq1": sum(x == 1 for x in v),
                                     **{f"le{k}": sum(x <= k for x in v) for k in KS},
                                     **{f"gt{k}": sum(x > k for x in v) for k in KS}}
        # positions whose ONLY non-refuted moves are outside the top-K (of all positions in the colour)
        a["only_outside_policy_top"] = {k: sum(r["best_nonref_policy_rank"] is not None and r["best_nonref_policy_rank"] > k
                                               for r in sel) for k in KS}
        a["only_outside_visit_top"] = {k: sum(r["best_nonref_visit_rank"] is not None and r["best_nonref_visit_rank"] > k
                                              for r in sel) for k in KS}
        a["only_outside_policy_top_share_of_all"] = {k: a["only_outside_policy_top"][k] / n for k in KS}
        a["only_outside_policy_top_share_of_has_nonref"] = {k: (a["only_outside_policy_top"][k] / len(has)) if has else None
                                                             for k in KS}
        a["blunder_code"] = {c: sum(r.get("blunder_code") == c for r in sel) for c in CODES + ("?",)}
        a["search_action_code"] = {c: sum(r["search_action_code"] == c for r in sel) for c in CODES + ("?",)}
        a["sims_median"] = pctl([r["sims"] for r in sel], 0.5)
        out[col] = a
    return out


def backtrack(moves: list[str], ply: int, n: int, args) -> list[dict]:
    """Judge DojoZero's own earlier moves (the ones played) at ply-2, ply-4, ...; newest first."""
    out = []
    for p in range(ply - 2, -1, -2)[:n]:
        g = state_at(moves, p)
        if g.phase != "NORMAL_RENJU":
            break
        code, nodes, reason = judge(g, moves[p], args.vcf, args.vct)
        out.append({"ply": p, "move": moves[p], "code": code, "nodes": nodes, "reason": reason})
    return out


def main(argv: list[str] | None = None) -> int:
    with keep_awake():  # also opts out of Windows power throttling
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/taraguchi_v1/models/gen_001.pt")
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--after", default=str(REPO / "benchmarks" / "perf" / "after_blunder.json"))
    ap.add_argument("--replay", default=str(REPO / "benchmarks" / "perf" / "results" / "replay_247_k8_t8.json"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ms", type=float, default=500.0)
    ap.add_argument("--vcf", type=int, default=200000)
    ap.add_argument("--vct", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--backtrack", type=int, default=10)
    ap.add_argument("--earlier", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    games = load_games(args.db)
    positions = load_positions(args.db)
    assert [p for p, _ in positions] == [g[0] for g in games]
    after = {p["id"]: p for p in json.loads(Path(args.after).read_text(encoding="utf-8"))["positions"]}
    assert set(after) == {g[0] for g in games}
    replay = {}
    if Path(args.replay).exists():
        rep = json.loads(Path(args.replay).read_text(encoding="utf-8"))["arms"]
        if "B30" in rep:
            for x in rep["B30"]["rows"]:
                why = ("safe" if x["judge_nodes"] not in ("WIN", "UNKNOWN") else "judge_unknown" if x["judge_nodes"] == "UNKNOWN"
                       else "all_k_refuted" if x["checked"] and x["refuted"] == x["checked"] else "budget_miss")
                replay[x["id"]] = why
    if args.limit:
        games, positions = games[: args.limit], positions[: args.limit]

    pl = make_player(f"engine:{args.ckpt}", args.device, {})
    s = pl.settings
    s.root, s.gumbel_noise, s.simulations, s.max_ms, s.defense_k = "PUCT", False, 100000, args.ms, 0
    search_many([positions[0][1]], pl.evaluator, s, pl.feature_config)  # device warm-up (not recorded)

    items = []
    for (pid, moves, ply), (pid2, g) in zip(games, positions):
        a = after[pid]
        h = g.copy()
        assert h.apply(moves[ply]) and moves[ply] == a["blunder"] and h.side_to_move == a["side"]
        assert g.side_to_move == a["dojo_color"]
        items.append((pid, g, {"ply": ply, "blunder": moves[ply], "e3b_b30_why": replay.get(pid)}))

    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        t0 = time.perf_counter()
        for i in range(0, len(items), 16):
            rows += analyze_many(items[i: i + 16], pl, ex, args)
            print(f"{len(rows)}/{len(items)} {time.perf_counter() - t0:.0f}s", flush=True)
        main_wall = time.perf_counter() - t0
        for (pid, moves, ply), r in zip(games, rows):
            r["backtrack"] = backtrack(moves, ply, args.backtrack, args)
            run = 0
            for b in r["backtrack"]:
                if b["code"] != "REFUTED":
                    break
                run += 1
            r["refuted_played_run"] = run  # DojoZero moves before the blunder already proven lost, in a row
        earlier = []
        if args.earlier:
            t1 = time.perf_counter()
            sel = []
            for (pid, moves, ply), r in zip(games, rows):
                if r["status"] == "ALL_REFUTED" and r["refuted_played_run"] > 0:
                    p = r["backtrack"][r["refuted_played_run"] - 1]["ply"]
                    sel.append((f"{pid}<{p}", state_at(moves, p), {"of": pid, "ply": p, "blunder": moves[p],
                                                                  "plies_before": ply - p}))
            for i in range(0, len(sel), 16):
                earlier += analyze_many(sel[i: i + 16], pl, ex, args)
                print(f"earlier {len(earlier)}/{len(sel)} {time.perf_counter() - t1:.0f}s", flush=True)

    bt = {}
    for col in ("ALL", "BLACK", "WHITE"):
        sel = [r for r in rows if r["status"] == "ALL_REFUTED" and (col == "ALL" or r["color"] == col)]
        runs = [r["refuted_played_run"] for r in sel]
        bt[col] = {"n_all_refuted": len(sel), "run_distribution": {str(k): runs.count(k) for k in sorted(set(runs))},
                   "run0_all_refuted": runs.count(0)}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    report = {
        "schema_version": 1, "tool": "python -m dojozero.tools.coverage_replay", "argv": argv,
        "commit": commit, "ckpt": args.ckpt, "device": args.device, "move_ms": args.ms,
        "solver": {"call": "analyze_tactics(h, vcf, vct, 0, 0.0, 'live', False)", "vcf_nodes": args.vcf,
                   "vct_nodes": args.vct, "deterministic": True},
        "codes": {"REFUTED": "opponent proven forced win (WIN)", "WINNING": "DojoZero proven win (terminal or no defense)",
                  "UNK_DONE": "UNKNOWN, nodes < vct budget (threat-space search ended without proof; not a safety proof)",
                  "UNK_BUDGET": "UNKNOWN, budget possibly exhausted"},
        "rank_note": "1-based over legal point moves; policy = raw logit; visit = 500 ms PUCT root visits, ties by prior",
        "positions": len(rows), "after_blunder_sha256": json.loads(Path(args.after).read_text(encoding="utf-8"))["sha256"],
        "wall_s": {"main": main_wall, "total": time.perf_counter() - t_start}, "threads": args.threads,
        "cpu_count": os.cpu_count(),
        "summary": aggregate(rows), "summary_by_e3b_b30_why": {
            w: aggregate([r for r in rows if r["e3b_b30_why"] == w]).get("ALL")
            for w in sorted({r["e3b_b30_why"] for r in rows if r["e3b_b30_why"]})},
        "backtrack_summary": bt,
        "earlier_summary": aggregate(earlier) if earlier else None,
        "rows": rows, "earlier_rows": earlier,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"summary": report["summary"]["ALL"], "backtrack": bt["ALL"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
