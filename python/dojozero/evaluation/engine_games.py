"""Engine-vs-engine games between external engines (EVALUATION §12, §14, §15).

    python -m dojozero.evaluation.engine_games balance --candidates 300 --target 120
    python -m dojozero.evaluation.engine_games round-robin --openings renju_balanced_v1 --pairs 4

* Balanced corpus (OQ-14, method chosen 2026-10-19): candidates are 5-stone positions (white to
  move) from random Taraguchi-10 openings, distinct by canonical hash. Each candidate is played by
  two strong external engines with colours swapped (2 games). It is kept when the colours do not
  decide both games (black and white each won once); positions where one colour wins twice, or
  with any failure, are dropped. This is an approximate filter (2 games, one time control), so the
  per-opening record is stored as `approx_balance`.
* Round robin: every pair of eligible ladder engines plays the same openings with colours swapped,
  so external engines are connected to each other and not only through DojoZero (calibration v0
  had a star graph, PROGRESS Phase 8 E2).

Games between external engines run in parallel threads (each engine is its own process and the
protocol calls release the GIL); the number of concurrent games is recorded with every match.
DojoZero never plays here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import _core
from . import corpus as corpus_mod
from .external import ENGINE_DIR, engine_config, load_meta, local_path
from .strength_db import StrengthDB
from .tournament import PROFILE, ProcessPlayer, git_commit, hardware_id, keep_awake, play_game

REPO = Path(__file__).resolve().parents[3]
DECIDED = ("NORMAL_WIN", "DRAW", "FORBIDDEN_MOVE")


def external_player(engine_id: str, time_ms: int, engine_dir: Path = ENGINE_DIR) -> ProcessPlayer:
    meta = load_meta(engine_id, engine_dir=engine_dir)
    return ProcessPlayer(engine_id, engine_config(meta, local_path(engine_id, engine_dir=engine_dir),
                                                  timeout_turn_ms=time_ms), meta)


def eligible_engines(engine_dir: Path = ENGINE_DIR) -> list[str]:
    out = []
    for p in sorted(engine_dir.glob("*.json")):
        if p.stem.startswith("_") or p.stem == "local_paths":
            continue
        if json.loads(p.read_text(encoding="utf-8")).get("conformance", {}).get("eligible"):
            out.append(p.stem)
    return out


def play_jobs(jobs: list[dict], time_ms: int, concurrency: int, log=None) -> list[tuple[dict, dict]]:
    """jobs: {a, b (engine ids), opening, set, a_first, pair_id}. Returns (job, record) pairs in
    completion order. A fresh process per engine and game."""
    def run(job):
        a = external_player(job["a"], time_ms)
        b = external_player(job["b"], time_ms)
        return job, play_game(a, b, job["opening"], job["set"], job["a_first"], job["pair_id"])

    out = []
    t0 = time.time()
    with keep_awake(), ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if log and (k % 20 == 0 or k == len(jobs)):
                log(f"  {k}/{len(jobs)} games ({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------------------------
def five_stone_candidates(n: int, seed: int, exclude: set[int] = frozenset()) -> list[list[str]]:
    from .match import make_openings
    seen, out = set(exclude), []
    k = 0
    while len(out) < n:
        batch = make_openings(n * 2, seed * 1000 + k)
        k += 1
        for acts in batch:
            stones = corpus_mod.stones_of_taraguchi(acts)[:5]
            g = _core.GameState(PROFILE)
            if not all(g.apply(s) for s in stones) or g.is_terminal or g.side_to_move != "WHITE":
                continue
            h = g.canonical_hash()
            if h in seen:
                continue
            seen.add(h)
            out.append(stones)
            if len(out) == n:
                break
        if k > 50:
            raise RuntimeError(f"only {len(out)} distinct 5-stone candidates")
    return out


def build_balanced(candidates: int, seed: int, engines: tuple[str, str], time_ms: int, concurrency: int,
                   target: int, set_id: str, log=print, exclude_sets: tuple[str, ...] = ()) -> dict:
    exclude: set[int] = set()
    for sid in exclude_sets:  # never reuse a start position of an existing set (fresh holdout)
        exclude |= corpus_mod.load(sid).canonical_hashes()
    cand = five_stone_candidates(candidates, seed, exclude)
    ops = [{"opening_id": f"cand-{k:04d}", "moves": m} for k, m in enumerate(cand)]
    cset = corpus_mod.OpeningSet("candidates", PROFILE, ops, "", "candidates")
    jobs = [{"a": engines[0], "b": engines[1], "opening": o, "set": cset, "a_first": af, "pair_id": k}
            for k, o in enumerate(ops) for af in (True, False)]
    log(f"balance filter: {len(cand)} candidates x 2 games ({engines[0]} vs {engines[1]}, {time_ms} ms/move, "
        f"{concurrency} concurrent)")
    by: dict[int, list[dict]] = {}
    for job, rec in play_jobs(jobs, time_ms, concurrency, log):
        by.setdefault(job["pair_id"], []).append(rec)
    kept, stats = [], {"black_twice": 0, "white_twice": 0, "draw_involved": 0, "failure": 0, "balanced": 0}
    for k in sorted(by):
        recs = by[k]
        # A black forbidden move is a rule-defined loss (RIF), i.e. a decided game; process / protocol
        # failures say nothing about the position and disqualify the candidate.
        if any(r["termination_reason"] not in DECIDED for r in recs):
            stats["failure"] += 1
            continue
        winners = [r["winner"] for r in recs]
        if None in winners:
            stats["draw_involved"] += 1
            balanced = winners.count("BLACK") <= 1 and winners.count("WHITE") <= 1
        else:
            balanced = winners.count("BLACK") == 1
            if not balanced:
                stats["black_twice" if winners[0] == "BLACK" else "white_twice"] += 1
        if not balanced:
            continue
        stats["balanced"] += 1
        bw = winners.count("BLACK") - winners.count("WHITE")
        kept.append({
            "opening_id": f"bal1-{len(kept):04d}", "moves": cand[k], "rule_profile": PROFILE,
            "source": "generated: first 5 stones of a random Taraguchi-10 opening (white to move)",
            "source_version": f"seed={seed}, candidate {k}", "expected_legal": True,
            "approx_balance": {"value_black": bw / 2, "method": f"{engines[0]} vs {engines[1]}, colours swapped, "
                                                                f"{time_ms} ms/move, 2 games",
                               "games": [{"black": r["a_color"] == "BLACK" and engines[0] or engines[1],
                                          "winner": r["winner"], "plies": r["plies"],
                                          "termination": r["termination_reason"]} for r in recs],
                               "uncertainty": "2 games: approximate"},
            "creation_method": "five_stone_candidates + build_balanced filter (neither colour wins both games)"})
        if len(kept) >= target:
            break
    log(f"balance filter result: {stats} -> kept {len(kept)}")
    return {"schema_version": 1, "set_id": set_id, "profile": PROFILE,
            "status": (f"v1 - approximate balance filter ({engines[0]} vs {engines[1]}, {time_ms} ms, 2 games per "
                       f"opening); Tier A/B. Filter stats: {stats}"),
            "openings": kept}


def round_robin(engines: list[str], opening_set, pairs: int, time_ms: int, concurrency: int, db: StrengthDB,
                log=print, offset: int = 0) -> list[str]:
    ops = opening_set.openings[offset:offset + pairs]
    if len(ops) < pairs:
        raise ValueError(f"{opening_set.set_id}: not enough openings for {pairs} pairs from offset {offset}")
    matches = {}
    for a, b in itertools.combinations(engines, 2):
        pa, pb = external_player(a, time_ms), external_player(b, time_ms)
        for pl in (pa, pb):
            db.register_engine(pl.engine_id, "external", name=pl.meta.get("engine_name", ""),
                               version=pl.meta.get("version", ""), executable_sha256=pl.meta["executable_sha256"],
                               config=pl.describe(), git_commit=git_commit())
        matches[(a, b)] = db.start_match(a, b, tier="A", rule_profile=PROFILE, profile_hash=str(_core.profile_hash(PROFILE)),
                                         opening_set_id=opening_set.set_id, opening_set_hash=opening_set.set_hash,
                                         compute_mode="fixed_time", budget={"time_ms_per_move": time_ms},
                                         hardware_id=hardware_id(), concurrent_games=concurrency,
                                         git_commit=git_commit(), command=f"round robin {a} vs {b}")
    jobs = [{"a": a, "b": b, "opening": o, "set": opening_set, "a_first": af, "pair_id": offset + k}
            for (a, b) in matches for k, o in enumerate(ops) for af in (True, False)]
    log(f"round robin: {len(engines)} engines, {len(matches)} pairings x {pairs} pairs = {len(jobs)} games")
    fails = {}
    for job, rec in play_jobs(jobs, time_ms, concurrency, log):
        db.add_game(matches[(job["a"], job["b"])], rec)
        if rec["termination_reason"] not in ("NORMAL_WIN", "DRAW"):
            key = f"{job['a'] if rec['failed_side'] == 'A' else job['b']}:{rec['termination_reason']}"
            fails[key] = fails.get(key, 0) + 1
    for mid in matches.values():
        db.finish_match(mid, {"concurrent_games": concurrency})
    log(f"round robin failures: {fails or 'none'}")
    return list(matches.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("balance")
    b.add_argument("--candidates", type=int, default=300)
    b.add_argument("--target", type=int, default=120)
    b.add_argument("--seed", type=int, default=5)
    b.add_argument("--engines", nargs=2, default=["rapfi_2025", "alphagomoku_2026"])
    b.add_argument("--time-ms", type=int, default=200)
    b.add_argument("--concurrency", type=int, default=3)
    b.add_argument("--set-id", default="renju_balanced_v1")
    b.add_argument("--exclude-sets", nargs="*", default=[], help="opening sets whose start positions are excluded")
    r = sub.add_parser("round-robin")
    r.add_argument("--openings", required=True)
    r.add_argument("--pairs", type=int, default=4)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--time-ms", type=int, default=500)
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--engines", nargs="*")
    r.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.cmd == "balance":
        d = build_balanced(args.candidates, args.seed, tuple(args.engines), args.time_ms, args.concurrency,
                           args.target, args.set_id, log, tuple(args.exclude_sets))
        out = corpus_mod.CORPUS_DIR / f"{args.set_id}.json"
        out.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        s = corpus_mod.load(out)
        log(f"{s.set_id}: {len(s.openings)} openings, hash {s.set_hash[:16]}")
    else:
        s = corpus_mod.load(args.openings)
        engines = args.engines or eligible_engines()
        mids = round_robin(engines, s, args.pairs, args.time_ms, args.concurrency, StrengthDB(args.db), log,
                           args.offset)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"match_ids": mids, "engines": engines, "openings": s.set_id,
                                              "hash": s.set_hash, "pairs": args.pairs, "offset": args.offset,
                                              "time_ms": args.time_ms, "concurrency": args.concurrency}, indent=1),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
