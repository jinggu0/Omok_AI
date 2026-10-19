"""Post-game loss classification (EVALUATION §21.1, PRD FR-27) — first, rule-of-thumb version.

    python -m dojozero.evaluation.loss_analysis --matches runs/evaluation/diagnose_scaling.json \
        --ckpt runs/taraguchi_v1/models/gen_001.pt --out runs/evaluation/loss_analysis.json

Every lost (or drawn) DojoZero game is replayed by the C++ engine. At each DojoZero move the
tactical solver (sound: WIN / LOSS are proofs, independently verified in Phase 3) analyses the
position before the move (DojoZero to move) and after it (opponent to move). The first decisive
event, in game order, gives the primary category:

  missed_win        DojoZero had a proven win and its move left no proven win
  tactical_blunder  before the move the solver did not prove a loss; after it the opponent has a
                    proven win (a forced win the solver sees within its budget was allowed)
  strategic_loss    at DojoZero's turn the position is already proven lost although no earlier move
                    of DojoZero allowed a solver-visible win: the game was lost beyond the solver's
                    horizon (positional / long-horizon value), `lost_from_start` if this is DojoZero's
                    first move after the opening
  unclassified      none of the above within the budget

For tactical blunders, the top-k network policy moves are tried: if one of them leaves the opponent
without a proven win, a non-refuted alternative existed (`alternative_rank`). Network values (W - L
for DojoZero) are recorded at every DojoZero move: `overconfident` marks a decisive position that
the network still valued >= +0.2 for DojoZero; `first_swing` is the first drop >= 0.5 between
consecutive DojoZero moves. Results are re-analysis data and are stored apart from match results.
The categories depend on the solver budget, which is recorded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .. import _core
from . import corpus as corpus_mod
from .strength_db import StrengthDB

PROFILE = "RENJU_GOMOCUP"
REPO = Path(__file__).resolve().parents[3]


class NetProbe:
    """Network value / policy for the side to move (wdl and policy logits from the checkpoint)."""

    def __init__(self, ckpt: str | None, device: str = "cpu"):
        self.player = None
        if ckpt:
            from .match import make_player
            self.player = make_player(f"engine:{ckpt}", device, {})

    def value_policy(self, g) -> tuple[float, np.ndarray] | tuple[None, None]:
        if self.player is None:
            return None, None
        feats = np.asarray(_core.encode(g, self.player.feature_config))[None]
        out = self.player.evaluator(feats)
        wdl = out["wdl"][0].astype(np.float64)
        wdl = np.exp(wdl - wdl.max())
        wdl /= wdl.sum()
        logits = np.where(np.asarray(_core.legal_mask(g)).astype(bool), out["policy"][0], -np.inf)
        return float(wdl[0] - wdl[2]), logits


def _tactics(g, budget: dict) -> dict:
    return _core.analyze_tactics(g, budget["vcf"], budget["vct"], budget["loss"], budget["ms"])


def classify_game(moves: list[str], opening_len: int, dojo_color: str, probe: NetProbe, budget: dict,
                  top_k: int = 10) -> dict:
    # Pass 1: per DojoZero move, solver verdicts before / after the move and the network value.
    g = _core.GameState(PROFILE)
    for m in moves[:opening_len]:
        assert g.apply(m)
    recs = []
    for i in range(opening_len, len(moves)):
        mv = moves[i]
        if g.side_to_move == dojo_color:
            v, logits = probe.value_policy(g)
            b = _tactics(g, budget)
            after_state = g.copy()
            assert after_state.apply(mv)
            a = {"result": "TERMINAL"} if after_state.is_terminal else _tactics(after_state, budget)
            recs.append({"ply": i, "move": mv, "before": b["result"],
                         "solver_move": None if b.get("move") is None else _core.point_to_str(b["move"]),
                         "after": a["result"], "value": v, "state": g.copy(), "logits": logits})
        assert g.apply(mv)
    # Pass 2: events in game order.
    events = []
    for k, r in enumerate(recs):
        ev = None
        if r["before"] == "LOSS":
            ev = "lost_from_start" if k == 0 else "strategic_loss"
        elif r["before"] == "WIN" and r["after"] not in ("LOSS", "TERMINAL") and r["move"] != r["solver_move"]:
            still_winning = k + 1 < len(recs) and recs[k + 1]["before"] == "WIN"
            if not still_winning:
                ev = "missed_win"
        elif r["after"] == "WIN":
            ev = "tactical_blunder"
        if not ev:
            continue
        e = {"ply": r["ply"], "event": ev, "move": r["move"], "before": r["before"], "after": r["after"],
             "value_before": r["value"], "solver_move": r["solver_move"]}
        if ev == "tactical_blunder" and r["logits"] is not None:
            order = [int(p) for p in np.argsort(-r["logits"])[:top_k] if np.isfinite(r["logits"][p])]
            e["played_rank"] = next((q for q, p in enumerate(order) if _core.point_to_str(p) == r["move"]), None)
            e["alternative_rank"] = None
            for q, p in enumerate(order):
                alt = _core.point_to_str(p)
                if alt == r["move"]:
                    continue
                s2 = r["state"].copy()
                if s2.apply(alt) and (s2.is_terminal or _tactics(s2, budget)["result"] != "WIN"):
                    e["alternative_rank"], e["alternative"] = q, alt
                    break
        events.append(e)
    # Primary = the first event that is not "lost while already proven lost" after an earlier event.
    primary = events[0] if events else None
    values = [{"ply": r["ply"], "value": r["value"]} for r in recs]
    vals = [x["value"] for x in values]
    swing = None
    if vals and vals[0] is not None:
        swing = next((values[k]["ply"] for k in range(1, len(vals)) if vals[k - 1] - vals[k] >= 0.5), None)
    cat = primary["event"] if primary else "unclassified"
    if cat == "lost_from_start":
        cat = "strategic_loss"
        primary = dict(primary, lost_from_start=True)
    overconf = bool(primary and primary.get("value_before") is not None and primary["value_before"] >= 0.2)
    return {"category": cat, "primary": primary, "events": events, "values": values, "first_swing_ply": swing,
            "overconfident": overconf, "dojo_moves": len(recs)}


def analyse_matches(db: StrengthDB, match_ids: list[str], probe: NetProbe, budget: dict, dojo_prefix: str = "dojo-",
                    log=print) -> list[dict]:
    out = []
    for mid in match_ids:
        m = db.match(mid)
        if not m["engine_a"].startswith(dojo_prefix):
            continue
        s = corpus_mod.load(m["opening_set_id"])
        op_len = {o["opening_id"]: len(o["moves"]) for o in s.openings}
        for g in db.games(mid):
            if g["score_a"] >= 1.0:
                continue
            r = classify_game(g["moves"], op_len[g["opening_id"]], g["a_color"], probe, budget)
            r |= {"game_id": g["game_id"], "match_id": mid, "command": m["command"], "opponent": m["engine_b"],
                  "dojo_color": g["a_color"], "score_a": g["score_a"], "termination": g["termination_reason"],
                  "plies": g["plies"]}
            db.add_analysis(g["game_id"], (r["primary"] or {}).get("ply", -1), "loss_category", budget,
                            {k: r[k] for k in ("category", "primary", "first_swing_ply", "overconfident")})
            out.append(r)
        log(f"  {m['command']}: {sum(1 for r in out if r['match_id'] == mid)} non-wins analysed")
    return out


def summarize(rows: list[dict]) -> dict:
    cats = Counter(r["category"] for r in rows)
    tb = [r["primary"] for r in rows if r["category"] == "tactical_blunder"]
    return {
        "games": len(rows), "categories": dict(cats),
        "lost_from_start": sum(1 for r in rows if (r["primary"] or {}).get("lost_from_start")),
        "overconfident": sum(r["overconfident"] for r in rows),
        "tactical_blunder_alternative_found": sum(1 for p in tb if p.get("alternative_rank") is not None),
        "tactical_blunder_played_rank": dict(Counter(str(p.get("played_rank")) for p in tb)),
        "decisive_dojo_move_index": float(np.mean([sum(1 for v in r["values"] if v["ply"] <= r["primary"]["ply"])
                                                   for r in rows if r["primary"]])) if rows else None,
        "by_color": dict(Counter(f"{r['dojo_color']}:{r['category']}" for r in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", required=True, help="JSON files with match ids (rows[].match_id or matches)")
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vcf", type=int, default=200000)
    ap.add_argument("--vct", type=int, default=100000)
    ap.add_argument("--loss", type=int, default=50000)
    ap.add_argument("--ms", type=float, default=1000)
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    mids = []
    for f in args.matches:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        mids += [r["match_id"] for r in d.get("rows", [])] + d.get("matches", [])
    budget = {"vcf": args.vcf, "vct": args.vct, "loss": args.loss, "ms": args.ms}
    db = StrengthDB(args.db)
    from .tournament import keep_awake
    with keep_awake():
        rows = analyse_matches(db, mids, NetProbe(args.ckpt, args.device), budget)
    summ = summarize(rows)
    print(json.dumps(summ, indent=1, default=str))
    Path(args.out).write_text(json.dumps({"budget": budget, "summary": summ, "games": rows}, indent=1, default=str),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
