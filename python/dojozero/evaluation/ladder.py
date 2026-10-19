"""External Strength Ladder: rating fit and adaptive opponent selection (EVALUATION §12, §14, §16).

    python -m dojozero.evaluation.ladder calibrate --dojo dojo:runs/.../gen_001.pt --config configs/evaluation/ladder.yaml

Rating model: Elo / Bradley-Terry, P(i scores against j) = 1 / (1 + 10^((R_j - R_i)/400)), draws
count as half points, fitted by maximum likelihood over all games in the given matches with one
anchor engine fixed at 0 (the anchor is always named in the report). A weak Gaussian prior
(sd `prior_sd`, around 0) keeps all-win / all-loss records finite; it is reported, and ratings
with few games are dominated by it. 95% intervals come from the inverse Hessian (game level;
pair correlation is ignored, so they are approximate). The result is `dojo_external_elo` when
external engines are in the pool, never a Gomocup rating.

Adaptive selection (§14): probe each eligible ladder engine with a few pairs; engines whose
probe score lies in [lo, hi] are primary calibration opponents. If none does, add the nearest
rungs above / below the score band (by probe score) so the main match is informative.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .strength_db import StrengthDB

LN10_400 = math.log(10) / 400.0


def fit_ratings(games: list[tuple[str, str, float]], anchor: str, prior_sd: float = 600.0,
                iters: int = 100) -> dict[str, dict]:
    """games: (player_i, player_j, score_i). Returns {player: {rating, lo, hi, se, games}}."""
    players = sorted({p for g in games for p in g[:2]})
    if anchor not in players:
        raise ValueError(f"anchor {anchor} played no game")
    idx = {p: k for k, p in enumerate(players)}
    free = [p for p in players if p != anchor]
    fidx = {p: k for k, p in enumerate(free)}
    r = np.zeros(len(players))
    n_games = {p: 0 for p in players}
    for i, j, _ in games:
        n_games[i] += 1
        n_games[j] += 1
    lam = 1.0 / prior_sd ** 2
    for _ in range(iters):
        g = np.zeros(len(free))
        h = np.zeros((len(free), len(free)))
        for pi, pj, s in games:
            ri, rj = r[idx[pi]], r[idx[pj]]
            e = 1.0 / (1.0 + math.exp(-(ri - rj) * LN10_400))
            w = e * (1 - e) * LN10_400 ** 2
            d = (s - e) * LN10_400
            for p, sign in ((pi, 1), (pj, -1)):
                if p in fidx:
                    g[fidx[p]] += sign * d
            for p, q, sgn in ((pi, pi, 1), (pj, pj, 1), (pi, pj, -1), (pj, pi, -1)):
                if p in fidx and q in fidx:
                    h[fidx[p], fidx[q]] += sgn * w
        cur = np.array([r[idx[p]] for p in free])
        g -= lam * cur
        h += lam * np.eye(len(free))
        step = np.linalg.solve(h, g)
        for p in free:
            r[idx[p]] += step[fidx[p]]
        if np.max(np.abs(step)) < 1e-6:
            break
    cov = np.linalg.inv(h) if len(free) else np.zeros((0, 0))
    out = {}
    for p in players:
        if p == anchor:
            out[p] = {"rating": 0.0, "lo": 0.0, "hi": 0.0, "se": 0.0, "games": n_games[p], "anchor": True}
            continue
        se = float(math.sqrt(max(cov[fidx[p], fidx[p]], 0.0)))
        rv = float(r[idx[p]])
        out[p] = {"rating": rv, "lo": rv - 1.96 * se, "hi": rv + 1.96 * se, "se": se, "games": n_games[p]}
    return out


def select_opponents(probe: dict[str, float], lo: float = 0.25, hi: float = 0.75, max_primary: int = 3) -> dict:
    """probe: engine_id -> DojoZero score from probe pairs. Returns primaries and why (§14)."""
    inside = sorted((e for e, s in probe.items() if lo <= s <= hi), key=lambda e: abs(probe[e] - 0.5))
    primary = inside[:max_primary]
    reason = "score within [lo, hi]"
    if not primary and probe:
        above = [e for e, s in probe.items() if s > hi]   # DojoZero too strong for these
        below = [e for e, s in probe.items() if s < lo]   # too weak for these
        # nearest rungs on each side of the band: hardest of the easy ones, easiest of the hard ones
        cand = []
        if above:
            cand.append(min(above, key=lambda e: probe[e]))
        if below:
            cand.append(max(below, key=lambda e: probe[e]))
        primary = cand
        reason = ("no engine inside the band: nearest rung(s) "
                  + ("above and below" if above and below else "above only (DojoZero beats every probed engine)"
                     if above else "below only (DojoZero loses to every probed engine)"))
    return {"primary": primary, "reason": reason, "lo": lo, "hi": hi}


def games_from_db(db: StrengthDB, match_ids: list[str]) -> list[tuple[str, str, float]]:
    out = []
    for mid in match_ids:
        m = db.match(mid)
        for g in db.games(mid):
            out.append((m["engine_a"], m["engine_b"], g["score_a"]))
    return out


def calibrate(cfg: dict, dojo_spec: str, device: str, db_path: str, log=print) -> dict:
    from . import corpus as corpus_mod
    from .tournament import ProcessPlayer, make_tournament_player, run_match
    from .external import ENGINE_DIR, engine_config, load_meta, local_path
    s = corpus_mod.load(cfg["openings"])
    db = StrengthDB(db_path)
    tier = cfg.get("tier", "A")
    time_ms = float(cfg["time_ms"])
    cache: dict = {}
    dojo = make_tournament_player(dojo_spec, device, cache, time_ms, cfg.get("seed", 1))
    eligible = []
    for eid in cfg["ladder"]:
        meta = load_meta(eid)
        conf = meta.get("conformance", {})
        if not conf.get("eligible"):
            log(f"  skip {eid}: conformance {conf}")
            continue
        eligible.append((eid, meta))

    def player(eid, meta):
        return ProcessPlayer(eid, engine_config(meta, local_path(eid), timeout_turn_ms=int(time_ms)), meta,
                             reuse_process=True)

    probe, matches = {}, []
    # Probe pairs use the last openings of the set so they do not overlap the main match openings.
    probe_set = corpus_mod.OpeningSet(s.set_id, s.profile, s.openings[-cfg["probe_pairs"]:], s.set_hash, s.status, s.path)
    for eid, meta in eligible:
        res = run_match(dojo, player(eid, meta), probe_set, cfg["probe_pairs"], db, tier=tier, time_ms=time_ms,
                        command=f"ladder probe {dojo_spec} vs {eid}", verbose=False)
        probe[eid] = res["summary"]["score"]
        matches.append(res["match_id"])
        log(f"  probe {eid}: DojoZero score {probe[eid]:.3f} over {res['summary']['pairs']} pairs, "
            f"failures {res['failures']}")
    sel = select_opponents(probe, cfg.get("lo", 0.25), cfg.get("hi", 0.75), cfg.get("max_primary", 3))
    log(f"  selection: {sel}")
    main_set = corpus_mod.OpeningSet(s.set_id, s.profile, s.openings[:len(s.openings) - cfg["probe_pairs"]], s.set_hash,
                                     s.status, s.path)
    for eid in sel["primary"]:
        meta = dict(eligible)[eid]
        res = run_match(dojo, player(eid, meta), main_set, cfg["main_pairs"], db, tier=tier, time_ms=time_ms,
                        command=f"ladder main {dojo_spec} vs {eid}", verbose=False)
        matches.append(res["match_id"])
        su = res["summary"]
        log(f"  main {eid}: score {su['score']:.3f} [{su['score_ci95'][0]:.3f}, {su['score_ci95'][1]:.3f}] "
            f"Δelo {su['elo']:+.0f} [{su['elo_ci95'][0]:+.0f}, {su['elo_ci95'][1]:+.0f}] failures {res['failures']}")
    games = games_from_db(db, matches)
    anchor = cfg.get("anchor") if cfg.get("anchor") in probe else (sel["primary"] or list(probe))[0]
    ratings = fit_ratings(games, anchor, cfg.get("prior_sd", 600.0))
    for eid, r in ratings.items():
        db.add_rating("dojo_external_elo", eid, r["rating"], r["lo"], r["hi"], r["games"], anchor, matches)
    return {"dojo": dojo.engine_id, "anchor": anchor, "probe": probe, "selection": sel, "ratings": ratings,
            "matches": matches, "opening_set": {"id": s.set_id, "hash": s.set_hash, "status": s.status},
            "time_ms": time_ms, "tier": tier}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--dojo", required=True)
    c.add_argument("--config", required=True)
    c.add_argument("--device", default="xpu")
    c.add_argument("--db", default="runs/evaluation/results.sqlite")
    c.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    res = calibrate(cfg, args.dojo, args.device, args.db, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
