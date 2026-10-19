"""Rule conformance pre-check for external engines (EVALUATION §12.2, ASSUMPTIONS A-P1..A-P3).

    python -m dojozero.evaluation.conformance --engines all --update-meta

Each engine is started over the Gomocup protocol (INFO rule 4) and given trap positions with
`BOARD` (the engine is the side to move):
  * win_in_1          black has exactly one five-making point -> must play it (BOARD + colours understood)
  * black_overline    black's only "five" is a six (forbidden) -> must NOT play it
  * black_double_four a point making two fours (forbidden)    -> must NOT play it
  * black_double_three a point making two open threes (forbidden) -> must NOT play it
  * white_overline    white completes six = a win in Renju     -> should play it
Then a few short games against the in-process heuristic check crash / timeout / protocol errors.
Every trap position and verdict is produced by the canonical C++ rules (no rule logic here).

Result: forbidden_violations (black traps played) and failures; stored in runs/evaluation/conformance/
and, with --update-meta, in external_engines/<id>.json "conformance". An engine with forbidden
violations or failures is not eligible for Tier B/C ladder use (threshold 0 by default).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import _core
from . import corpus as corpus_mod
from .external import ENGINE_DIR, engine_config, load_meta, local_path
from .tournament import InProcessPlayer, ProcessPlayer, run_match

PROFILE = "RENJU_GOMOCUP"
REPO = Path(__file__).resolve().parents[3]
FAR_WHITE = ["A1", "A3", "A5", "A7", "A9", "A11", "A13"]
FAR_BLACK = ["O1", "O3", "O5", "O7", "O9", "O11", "O13", "O15"]


def _state(black: list[str], white: list[str], side: str):
    return _core.GameState.from_spec(json.dumps({"black": black, "white": white, "side": side}), PROFILE)


def traps() -> list[dict]:
    """(name, black, white, side, target, expect) with target verified by the C++ rules."""
    t = [
        {"name": "win_in_1", "black": ["E8", "F8", "G8", "H8"], "white": ["D8"] + FAR_WHITE[:3], "side": "BLACK",
         "target": "I8", "expect": "play"},
        {"name": "black_overline", "black": ["E8", "F8", "G8", "I8", "J8"], "white": FAR_WHITE[:5], "side": "BLACK",
         "target": "H8", "expect": "avoid"},
        {"name": "black_double_four", "black": ["E8", "F8", "G8", "H5", "H6", "H7"], "white": FAR_WHITE[:6],
         "side": "BLACK", "target": "H8", "expect": "avoid"},
        {"name": "black_double_three", "black": ["F8", "G8", "H6", "H7"], "white": FAR_WHITE[:4], "side": "BLACK",
         "target": "H8", "expect": "avoid"},
        {"name": "white_overline", "black": FAR_BLACK[:6], "white": ["E8", "F8", "G8", "I8", "J8"], "side": "WHITE",
         "target": "H8", "expect": "play"},
    ]
    for x in t:  # self-check of the trap definitions against the canonical rules
        g = _state(x["black"], x["white"], x["side"])
        p = _core.str_to_point(x["target"])
        if x["expect"] == "avoid":
            assert g.black_verdict(p).startswith("FORBIDDEN"), (x["name"], g.black_verdict(p))
        else:
            g2 = g.copy()
            assert g2.apply(x["target"]) and g2.is_terminal and g2.winner is not None, x["name"]
    return t


def board_moves(black: list[str], white: list[str]) -> list[int]:
    """Interleave stones black-first (BOARD order; only colours matter for the engine)."""
    out = []
    for i in range(max(len(black), len(white))):
        if i < len(black):
            out.append(_core.str_to_point(black[i]))
        if i < len(white):
            out.append(_core.str_to_point(white[i]))
    return out


def check_traps(cfg) -> list[dict]:
    results = []
    for x in traps():
        e = _core.ExternalEngine(cfg)
        failure, detail = e.start()
        row = {"trap": x["name"], "expect": x["expect"], "target": x["target"]}
        if failure != "NONE":
            row |= {"failure": failure, "detail": detail}
            results.append(row)
            e.end()
            continue
        r = e.board(board_moves(x["black"], x["white"]))
        e.end()
        row["failure"] = r["failure"]
        row["elapsed_ms"] = round(r["elapsed_ms"], 1)
        if r["failure"] != "NONE":
            row["detail"] = r["detail"]
        else:
            mv = _core.point_to_str(r["move"])
            row["move"] = mv
            hit = mv == x["target"]
            row["ok"] = hit if x["expect"] == "play" else not hit
            if x["expect"] == "avoid" and hit:
                row["violation"] = "FORBIDDEN_MOVE"
        results.append(row)
    return results


def check_engine(engine_id: str, games_pairs: int = 2, timeout_ms: int = 1000, engine_dir: Path = ENGINE_DIR) -> dict:
    meta = load_meta(engine_id, engine_dir=engine_dir)
    cfg = engine_config(meta, local_path(engine_id, engine_dir=engine_dir), timeout_turn_ms=timeout_ms)
    t0 = time.time()
    about = ""
    e = _core.ExternalEngine(cfg)
    start = e.start()
    if start[0] == "NONE":
        about = e.about()
    e.end()
    trap_rows = check_traps(cfg) if start[0] == "NONE" else []
    games = None
    if start[0] == "NONE" and games_pairs > 0:
        smoke = corpus_mod.load("renju_smoke_v0")
        res = run_match(InProcessPlayer("heuristic", "cpu", {}, None, 11), ProcessPlayer(engine_id, cfg, meta),
                        smoke, games_pairs, None, tier="A", time_ms=None, verbose=False)
        games = {"games": res["summary"]["games"], "engine_score": 1 - res["summary"]["score"],
                 "failures": res["failures"]}
    violations = sum(1 for r in trap_rows if r.get("violation"))
    failures = sum(1 for r in trap_rows if r.get("failure", "NONE") != "NONE")
    failures += sum((games or {}).get("failures", {}).values())
    misses = [r["trap"] for r in trap_rows if r.get("failure") == "NONE" and not r.get("ok") and not r.get("violation")]
    return {"engine_id": engine_id, "date": time.strftime("%Y-%m-%d"), "start": start[0], "start_detail": start[1],
            "about": about, "traps": trap_rows, "games": games, "forbidden_violations": violations,
            "failures": failures, "missed_wins": misses,
            "eligible": start[0] == "NONE" and violations == 0 and failures == 0, "seconds": round(time.time() - t0, 1)}


def update_meta(result: dict, engine_dir: Path = ENGINE_DIR) -> None:
    p = engine_dir / f"{result['engine_id']}.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["conformance"] = {"checked": True, "date": result["date"], "forbidden_violations": result["forbidden_violations"],
                           "failures": result["failures"], "missed_wins": result["missed_wins"],
                           "eligible": result["eligible"], "about": result["about"]}
    p.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["all"])
    ap.add_argument("--pairs", type=int, default=2, help="short games vs the heuristic (pairs)")
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--update-meta", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO / "runs" / "evaluation" / "conformance"))
    args = ap.parse_args(argv)
    ids = args.engines
    if ids == ["all"]:
        ids = sorted(p.stem for p in ENGINE_DIR.glob("*.json") if not p.stem.startswith("_") and p.stem != "local_paths")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for eid in ids:
        r = check_engine(eid, args.pairs, args.timeout_ms)
        (out / f"{eid}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        if args.update_meta:
            update_meta(r)
        traps_s = " ".join(f"{t['trap']}={'ok' if t.get('ok') else t.get('violation') or t.get('failure') or 'miss'}"
                           for t in r["traps"])
        print(f"{eid:18s} start={r['start']:8s} eligible={r['eligible']!s:5s} viol={r['forbidden_violations']} "
              f"fail={r['failures']} games={r['games']} | {traps_s} ({r['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
