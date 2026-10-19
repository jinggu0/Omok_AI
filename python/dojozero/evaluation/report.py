"""Strength Report v0 from the Strength DB (PRD FR-26, EVALUATION §22).

    python -m dojozero.evaluation.report --db runs/evaluation/results.sqlite --out runs/evaluation/report.md

v0 covers the match sections (internal / external results, failures, compute). Sections that
need data this version does not collect are printed as UNKNOWN rather than guessed; estimates
are never shown as measurements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .match import summarize
from .strength_db import StrengthDB

UNKNOWN = "UNKNOWN (not measured by report v0)"


def match_block(db: StrengthDB, m: dict) -> list[str]:
    games = db.games(m["match_id"])
    if not games:
        return [f"- match `{m['match_id']}`: no games"]
    s = summarize([{"opening": g["pair_id"], "score_a": g["score_a"], "a_color": g["a_color"], "plies": g["plies"]}
                   for g in games])
    fails: dict[str, int] = {}
    for g in games:
        if g["termination_reason"] not in ("NORMAL_WIN", "DRAW"):
            k = f"{g['failed_side']}:{g['termination_reason']}"
            fails[k] = fails.get(k, 0) + 1
    ea, eb = db.engine(m["engine_a"]) or {}, db.engine(m["engine_b"]) or {}
    compute = json.loads(m["compute_json"] or "{}")
    budget = json.loads(m["budget_json"] or "{}")
    lo, hi = s["elo_ci95"]
    rows = [
        f"### {m['engine_a']} vs {m['engine_b']}  (tier {m['tier']}, {m['compute_mode']})",
        f"- match `{m['match_id']}` · {m['start_time']} → {m['end_time']} · commit `{m['git_commit']}`",
        f"- opening set `{m['opening_set_id']}` hash `{m['opening_set_hash'][:16]}` · profile {m['rule_profile']} · "
        f"hardware `{m['hardware_id']}` · concurrent games {m['concurrent_games']}",
        f"- budget: time/move {budget.get('time_ms_per_move')} ms; A {json.dumps(budget.get('a', {}))}; "
        f"B {json.dumps(budget.get('b', {}))}",
        f"- kinds: A = {ea.get('kind', '?')}, B = {eb.get('kind', '?')} "
        f"(rating name: {'dojo_external_elo' if 'external' in (ea.get('kind'), eb.get('kind')) else 'dojo_internal_elo'})",
        "",
        "| games | pairs | A W/D/L | score A | 95% CI | ΔElo (A−B) | 95% CI | pentanomial | A as black / white |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {s['games']} | {s['pairs']} | {s['a_wins']}/{s['draws']}/{s['a_losses']} | {s['score']:.3f} | "
        f"[{s['score_ci95'][0]:.3f}, {s['score_ci95'][1]:.3f}] | {s['elo']:+.0f} | [{lo:+.0f}, {hi:+.0f}] | "
        f"{s['pentanomial']} | {_f(s['a_score_as_black'])} / {_f(s['a_score_as_white'])} |",
        "",
        f"- failures (side:reason → games): {fails or 'none'}",
        f"- compute: wall {compute.get('wall_s', 0):.0f} s; ms/move A {compute.get('mean_ms_per_move_a', 0):.0f}, "
        f"B {compute.get('mean_ms_per_move_b', 0):.0f}; search A {compute.get('search_a', {})}",
        "",
    ]
    return rows


def _f(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def render(db: StrengthDB, match_ids: list[str] | None = None) -> str:
    ms = [m for m in db.matches() if not match_ids or m["match_id"] in match_ids]
    out = ["# DojoZero Strength Report (v0)", "",
           "Rating names follow EVALUATION §16; Standard Renju (RENJU_GOMOCUP) only — Taraguchi-10 results are "
           "reported separately.", "", "## Matches", ""]
    for m in ms:
        out += match_block(db, m)
    out += ["## Tactical", UNKNOWN, "", "## Performance", UNKNOWN, "", "## Opening", UNKNOWN, "",
            "## Diagnosis", UNKNOWN + " — needs loss analysis (EVALUATION §21, E3)", "",
            "## Recommendation", UNKNOWN + " — no recommendation without measurements", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--match", action="append")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    text = render(StrengthDB(args.db), args.match)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
