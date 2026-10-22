"""Solver-labelled tactical reanalysis dataset (PROGRESS Phase 8 E4, Case B).

    python -m dojozero.tools.reanalysis_data collect --out data/reanalysis_v1
    python -m dojozero.tools.reanalysis_data label   --out data/reanalysis_v1 [--limit N]

collect: every position (opening stones placed .. last move) of recorded RENJU_GOMOCUP games from the
Strength DB, except games started from renju_balanced_v1 openings 24-103 (external A/B openings,
never used for training). Groups = (opening set, opening id): games from one opening share their
early positions, so a whole opening goes to one split. Split (fixed, seed-free):
  test  = renju_balanced_v1 openings 0-11 (the games of the 247 classified blunders, E2c)
  val   = every 7th remaining group by a CRC32 order (about 15%)
  train = the rest
Exact duplicates are removed by D4 canonical hash; a hash seen in test is removed from train/val,
and a hash seen in val is removed from train.

label: for each position, with the baseline network's top-16 policy moves + the move actually
played as candidates (all legal moves at the 247 blunder positions):
  position  analyze_tactics(g, VCF, VCT, LOSS) for the side to move: WIN = side has a forced win,
            LOSS = side is proven lost (opponent forced win against every reply), UNKNOWN otherwise
  move      analyze_tactics(after m, VCF, VCT, 0) for the opponent:
              PROVEN_UNSAFE        opponent has a forced win (exact)
              PROVEN_WINNING       m wins (five) or the opponent cannot stop an immediate win (exact)
              UNREFUTED_EXHAUSTED  no proof and the search ended below its budget (threat-space search
                                   exhausted: NOT a safety proof; heuristic negative only)
              UNKNOWN_BUDGET       no proof, budget exhausted (no label)
UNKNOWN is never stored as LOSS or as safe.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _core

REPO = Path(__file__).resolve().parents[3]
PROFILE = "RENJU_GOMOCUP"
SOLVER = {"vcf_nodes": 50000, "vct_nodes": 30000, "loss_nodes": 30000, "top_k": {"train": 8, "val": 16, "test": 16}}


def selected(p: dict) -> bool:
    """Positions that get solver labels: all train/val positions; in test only DojoZero-to-move
    positions of DojoZero games (the diagnosis target; engine-only test games are not labelled)."""
    return p["split"] != "test" or (p["source"] == "dojo_game" and p["dojo_to_move"])
CODES = ["PROVEN_UNSAFE", "PROVEN_WINNING", "UNREFUTED_EXHAUSTED", "UNKNOWN_BUDGET"]


def _opening_index() -> dict:
    from ..evaluation import corpus
    idx = {}
    for sid in ("renju_balanced_v1", "renju_provisional_v0", "renju_smoke_v0", TRAIN_SET):
        try:
            s = corpus.load(sid)
        except FileNotFoundError:
            continue
        for i, o in enumerate(s.openings):
            idx[(sid, o["opening_id"])] = i
    return idx


def split_for(set_id: str, opening_id: str, index: int) -> str:
    """Fixed split by opening group: balanced_v1 openings 0-11 (the 247 blunder games) = test;
    otherwise every group whose CRC32 is 0 mod 7 = val, the rest = train."""
    if set_id == "renju_balanced_v1" and index < 12:
        return "test"
    return "val" if zlib.crc32(f"{set_id}/{opening_id}".encode()) % 7 == 0 else "train"


def collect(db: str, out: Path) -> dict:
    idx = _opening_index()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("""select g.game_id, g.moves, g.a_color, m.engine_a, m.engine_b, m.opening_set_id, g.opening_id,
                                 m.rule_profile from game g join match m using(match_id) order by g.game_id""").fetchall()
    blunder_games = {gid for rj, gid in con.execute(
        "select a.result_json, a.game_id from analysis a where a.kind='loss_category'")
        if json.loads(rj)["category"] == "tactical_blunder"}
    groups: dict[tuple, list] = {}
    skipped = Counter()
    for gid, mv, a_color, ea, eb, sid, op, prof in rows:
        if prof != PROFILE:
            skipped["profile"] += 1
            continue
        i = idx.get((sid, op))
        if i is None:
            skipped["unknown_opening"] += 1
            continue
        if sid == "renju_balanced_v1" and 24 <= i < 104:
            skipped["external_ab_openings"] += 1
            continue
        groups.setdefault((sid, op, i), []).append((gid, json.loads(mv), a_color, ea, eb))

    def split_of(key) -> str:
        return split_for(*key)

    order = {"test": 0, "val": 1, "train": 2}
    seen: dict[int, str] = {}
    positions = []
    dup = Counter()
    for key in sorted(groups, key=lambda k: (order[split_of(k)], k)):
        sp = split_of(key)
        for gid, moves, a_color, ea, eb in groups[key]:
            dojo_side = {"BLACK": "BLACK", "WHITE": "WHITE"}.get(a_color) if ea.startswith("dojo") else (
                ("WHITE" if a_color == "BLACK" else "BLACK") if eb.startswith("dojo") else None)
            g = _core.GameState(PROFILE)
            for ply, m in enumerate(moves):
                if g.phase == "NORMAL_RENJU" and not g.is_terminal and ply >= 5:
                    h = int(g.canonical_hash())
                    if h in seen:
                        dup["same_split" if seen[h] == sp else "cross_split"] += 1
                    else:
                        seen[h] = sp
                        positions.append({
                            "id": f"{gid}@{ply}", "game_id": gid, "ply": ply, "split": sp,
                            "group": f"{key[0]}/{key[1]}", "opening_index": key[2], "moves": moves[:ply],
                            "side": g.side_to_move, "played": m, "dojo_side": dojo_side,
                            "dojo_to_move": dojo_side == g.side_to_move,
                            "source": "dojo_game" if dojo_side else "engine_game",
                            "blunder_game": gid in blunder_games, "canonical_hash": h})
                if not g.apply(m):
                    raise ValueError(f"{gid}: illegal recorded move {m} at ply {ply}")
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "positions.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in positions:
            fh.write(json.dumps(p) + "\n")
    stats = {
        "positions": len(positions), "skipped_games": dict(skipped), "duplicates_removed": dict(dup),
        "by_split": dict(Counter(p["split"] for p in positions)),
        "by_split_source": {f"{k[0]}/{k[1]}": v for k, v in Counter((p["split"], p["source"]) for p in positions).items()},
        "by_split_side": {f"{k[0]}/{k[1]}": v for k, v in Counter((p["split"], p["side"]) for p in positions).items()},
        "groups": {sp: sum(1 for k in groups if split_of(k) == sp) for sp in ("train", "val", "test")},
        "games": {sp: sum(len(groups[k]) for k in groups if split_of(k) == sp) for sp in ("train", "val", "test")},
    }
    return stats


def _analyze_move(g, m: str) -> dict:
    h = g.copy()
    if not h.apply(m):
        return {"m": m, "code": "ILLEGAL"}
    if h.is_terminal:
        w = h.winner
        return {"m": m, "code": "PROVEN_WINNING" if w is not None and w == g.actor else "TERMINAL_OTHER", "nodes": 0}
    d = _core.analyze_tactics(h, SOLVER["vcf_nodes"], SOLVER["vct_nodes"], 0, 0.0)
    if d["result"] == "WIN":
        code = "PROVEN_UNSAFE"
    elif d["result"] == "LOSS":
        code = "PROVEN_WINNING"
    else:
        code = "UNREFUTED_EXHAUSTED" if d["nodes"] < SOLVER["vct_nodes"] else "UNKNOWN_BUDGET"
    return {"m": m, "code": code, "nodes": d["nodes"], "reason": d["reason"],
            "depth": len(d["pv"]) if d["result"] == "WIN" else None}


def label(out: Path, ckpt: str, device: str, limit: int, threads: int, all_legal_ids: set[str]) -> dict:
    from ..evaluation.match import make_player
    pl = make_player(f"engine:{ckpt}", device, {})
    positions = [json.loads(l) for l in gzip.open(out / "positions.jsonl.gz", "rt", encoding="utf-8")]
    positions = [p for p in positions if selected(p)]
    if limit:
        positions = positions[:limit]
    states = []
    for p in positions:
        g = _core.GameState(PROFILE)
        for m in p["moves"]:
            assert g.apply(m)
        states.append(g)
    # Baseline network candidates (top-16 policy over legal moves), batched.
    cands = []
    for s in range(0, len(states), 256):
        chunk = states[s:s + 256]
        feats = np.stack([np.asarray(_core.encode(g, pl.feature_config)) for g in chunk])
        logits = pl.evaluator(feats)["policy"]
        for p, g, lg in zip(positions[s:s + 256], chunk, logits):
            legal = np.asarray(_core.legal_mask(g)).astype(bool)
            lg = np.where(legal, lg, -np.inf)[:225]
            k = SOLVER["top_k"][p["split"]]
            cands.append([_core.point_to_str(int(i)) for i in np.argsort(-lg)[:k] if np.isfinite(lg[i])])

    def work(i: int) -> dict:
        p, g = positions[i], states[i]
        t0 = time.perf_counter()
        d = _core.analyze_tactics(g, SOLVER["vcf_nodes"], SOLVER["vct_nodes"], SOLVER["loss_nodes"], 0.0)
        pos = {"result": d["result"], "reason": d["reason"], "nodes": d["nodes"],
               "move": _core.point_to_str(d["move"]) if d["move"] is not None and d["result"] == "WIN" else None,
               "depth": len(d["pv"]) if d["result"] != "UNKNOWN" else None,
               "ms": (time.perf_counter() - t0) * 1000}
        moves = list(cands[i])
        if p["played"] not in moves:
            moves.append(p["played"])
        if p["id"] in all_legal_ids:
            legal = np.flatnonzero(np.asarray(_core.legal_mask(g))[:225])
            moves += [m for m in (_core.point_to_str(int(q)) for q in legal) if m not in moves]
        lab = [_analyze_move(g, m) for m in moves]
        return {"id": p["id"], "pos": pos, "candidates": cands[i], "moves": lab,
                "ms": (time.perf_counter() - t0) * 1000}

    t0 = time.time()
    results = [None] * len(positions)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for k, r in enumerate(ex.map(work, range(len(positions)))):
            results[k] = r
            if (k + 1) % 1000 == 0:
                print(f"labelled {k + 1}/{len(positions)} ({time.time() - t0:.0f}s)", flush=True)
    name = "labels.jsonl.gz" if not limit else f"labels_sample{limit}.jsonl.gz"
    with gzip.open(out / name, "wt", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    codes = Counter(m["code"] for r in results for m in r["moves"])
    return {"labelled": len(results), "wall_s": time.time() - t0, "file": name, "move_codes": dict(codes),
            "position_results": dict(Counter(r["pos"]["result"] for r in results)),
            "ms_per_position_mean": float(np.mean([r["ms"] for r in results])),
            "ms_per_position_p95": float(np.percentile([r["ms"] for r in results], 95))}


TRAIN_SET = "renju_train_reanalysis_v1"
TRAIN_SEED = 101  # balanced_v1 used seed 5; a future external holdout must use a seed >= 201


def build_train_openings(n: int) -> Path:
    """Training-only openings (never for evaluation): 5-stone positions, white to move, from random
    Taraguchi-10 openings (the balanced_v1 candidate method) with seed 101, excluding every start
    position of the existing opening sets by canonical hash. No balance filter (not needed for
    training data)."""
    from ..evaluation import corpus
    from ..evaluation.engine_games import five_stone_candidates
    exclude: set[int] = set()
    for sid in ("renju_balanced_v1", "renju_provisional_v0", "renju_smoke_v0"):
        exclude |= corpus.load(sid).canonical_hashes()
    cands = five_stone_candidates(n, TRAIN_SEED, exclude)
    openings = [{"opening_id": f"trainre1-{k:04d}", "moves": m, "rule_profile": PROFILE,
                 "source": "generated: first 5 stones of random Taraguchi-10 openings (engine_games.five_stone_candidates)",
                 "source_version": f"seed={TRAIN_SEED}, candidate {k}", "expected_legal": True,
                 "approx_balance": {"value_black": None, "method": "not assessed (training data only)", "uncertainty": None},
                 "creation_method": "five_stone_candidates(n, 101, exclude=existing sets)"} for k, m in enumerate(cands)]
    doc = {"schema_version": 1, "set_id": TRAIN_SET, "profile": PROFILE,
           "status": "TRAINING-ONLY - reanalysis game generation (Phase 8 E4); never use for evaluation",
           "openings": openings}
    out = corpus.CORPUS_DIR / f"{TRAIN_SET}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    s = corpus.load(out)
    print(f"{s.set_id}: {len(s.openings)} openings, hash {s.set_hash}")
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    from ..evaluation.tournament import keep_awake
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["openings", "collect", "label"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(REPO / "data" / "reanalysis_v1"))
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--ckpt", default="runs/taraguchi_v1/models/gen_001.pt")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--all-legal-247", action="store_true",
                    help="also label every legal move at the 247 blunder positions (E3c already did, other budget)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    with keep_awake():
        if args.cmd == "openings":
            build_train_openings(args.n)
            return 0
        if args.cmd == "collect":
            man["collect"] = collect(args.db, out)
            man["positions_sha256"] = sha256_file(out / "positions.jsonl.gz")
        else:
            blunder_ids = set()
            cov = REPO / "benchmarks" / "perf" / "results" / "coverage_247.json"
            if args.all_legal_247 and cov.exists():  # all legal moves at the 247 blunder positions (~3 h more)
                blunder_ids = {r["id"] for r in json.loads(cov.read_text(encoding="utf-8"))["rows"]}
            info = label(out, args.ckpt, args.device, args.limit, args.threads, blunder_ids)
            info |= {"solver": SOLVER, "ckpt": args.ckpt, "codes": CODES,
                     "sha256": sha256_file(out / info["file"])}
            man["label" if not args.limit else f"label_sample{args.limit}"] = info
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items()}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
