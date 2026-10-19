"""Self-play game generation (TRAINING.md §2.4).

Game-parallel: all live games are searched together each round (one tree per game, shared network
batches). Exploration comes from the Gumbel root with noise; the recorded policy target is the
Gumbel improved policy. Playout cap randomization: a move gets a full search (and a recorded
target) with probability p_full, otherwise a fast search whose result is only used to play on.
Start positions are random Taraguchi-10 openings (opening learning is Phase 7). No resignation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .. import _core
from ..evaluation.match import PROFILE, make_openings
from ..search.driver import SearchSettings, search_many
from .records import FORMAT_VERSION

POLICY_TOP_K = 48  # entries kept per recorded policy target (renormalized)


@dataclass
class SelfPlaySettings:
    games: int = 64
    sims_full: int = 64
    sims_fast: int = 16
    p_full: float = 0.35
    root: str = "GUMBEL"
    gumbel_m: int = 16
    root_tactics_nodes: int = 2000
    max_inflight: int = 8
    max_plies: int = 225
    profile: str = PROFILE


def _search_settings(s: SelfPlaySettings, sims: int, seed: int) -> SearchSettings:
    return SearchSettings(simulations=sims, root=s.root, gumbel_m=s.gumbel_m, gumbel_noise=True,
                          root_tactics_nodes=s.root_tactics_nodes, max_inflight=s.max_inflight, seed=seed)


def _policy_target(result: dict) -> dict[str, float]:
    if not result["children"]:  # root solved by tactics: the proven move
        return {result["action"]: 1.0}
    kids = sorted(result["children"], key=lambda c: c["target"], reverse=True)[:POLICY_TOP_K]
    kids = [c for c in kids if c["target"] > 1e-5] or kids[:1]
    total = sum(c["target"] for c in kids) or 1.0
    return {c["action"]: round(c["target"] / total, 6) for c in kids}


def play_games(evaluator, feature_config, s: SelfPlaySettings, seed: int, meta: dict | None = None,
               verbose: bool = False) -> list[dict]:
    rng = np.random.default_rng(seed)
    openings = make_openings(s.games, seed, s.profile)
    games = []
    for k, op in enumerate(openings):
        g = _core.GameState(s.profile)
        for a in op:
            assert g.apply(a)
        games.append({"state": g, "actions": list(op), "positions": [], "id": k})
    rnd, t0 = 0, time.time()
    while True:
        live = [x for x in games if not x["state"].is_terminal and len(x["actions"]) < s.max_plies]
        if not live:
            break
        rnd += 1
        full = rng.random(len(live)) < s.p_full
        for group, sims, record in ((full, s.sims_full, True), (~full, s.sims_fast, False)):
            todo = [x for x, f in zip(live, group) if f]
            if not todo:
                continue
            res = search_many([x["state"] for x in todo], evaluator,
                              _search_settings(s, sims, seed * 100_003 + rnd * 2 + int(record)), feature_config)
            for x, r in zip(todo, res):
                if record and x["state"].phase == "NORMAL_RENJU":
                    x["positions"].append({"i": len(x["actions"]), "actor": r["actor"], "policy": _policy_target(r),
                                           "value": round(float(r["value"]), 5), "proven": int(r["proven"])})
                if not x["state"].apply(r["action"]):
                    raise RuntimeError(f"search returned illegal action {r['action']}")
                x["actions"].append(r["action"])
        if verbose and rnd % 20 == 0:
            done = sum(x["state"].is_terminal for x in games)
            print(f"  self-play round {rnd}: {done}/{len(games)} finished ({time.time() - t0:.0f}s)", flush=True)
    out = []
    for x in games:
        st = x["state"]
        out.append({
            "format_version": FORMAT_VERSION,
            "meta": (meta or {}) | {"game_id": f"{seed}-{x['id']}", "profile": s.profile,
                                    "profile_hash": _core.profile_hash(s.profile), "seed": seed,
                                    "opening_len": len(openings[x["id"]]), "result": st.result, "winner": st.winner,
                                    "black_holder": st.black_holder, "plies": len(x["actions"]),
                                    "truncated": not st.is_terminal, "selfplay": asdict(s)},
            "actions": x["actions"],
            "positions": x["positions"],
        })
    return out
