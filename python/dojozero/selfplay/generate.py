"""Self-play / league game generation (TRAINING.md §2.4-§2.6).

Game-parallel: every round, live games are grouped by who controls the player to act (the learner
or a league opponent) and each group is searched together (shared network batches). Exploration:
Gumbel root with noise; recorded target = Gumbel improved policy. Playout cap randomization: the
learner's move gets a full search (and a recorded target) with probability p_full.

Phase 7: games may start from the empty board (every Taraguchi-10 decision is searched and recorded
for the swap / branch / propose / select heads), from a random opening prefix, or from a random
opening played to NORMAL_RENJU. League games pair the learner with an opponent from a pool; only
the learner's moves become training targets. No resignation.
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
    start: str = "random_opening"  # "random_opening" | "empty" | "mixed"
    prefix_prob: float = 0.5       # "mixed": probability of a random opening prefix instead of the empty board
    record_opening: bool = False   # record targets for Taraguchi-10 decisions (Phase 7)
    league_fraction: float = 0.0   # share of games against a league opponent
    opponent_sims: int = 32


@dataclass
class Opponent:
    name: str
    kind: str                      # "engine" | "heuristic" | "random"
    evaluator: object = None
    feature_config: object = None


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


def _start_actions(s: SelfPlaySettings, seed: int) -> list[list[str]]:
    if s.start == "random_opening":
        return make_openings(s.games, seed, s.profile)
    if s.start == "empty":
        return [[] for _ in range(s.games)]
    if s.start == "mixed":
        rng = np.random.default_rng([seed, 23])
        full = make_openings(s.games, seed, s.profile)
        out = []
        for op in full:
            if rng.random() < s.prefix_prob:
                out.append(op[: int(rng.integers(1, len(op)))])  # stop somewhere inside the opening
            else:
                out.append([])
        return out
    raise ValueError(f"unknown start mode {s.start}")


def play_games(evaluator, feature_config, s: SelfPlaySettings, seed: int, meta: dict | None = None,
               verbose: bool = False, opponents: list[Opponent] | None = None,
               opponent_weights: list[float] | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    starts = _start_actions(s, seed)
    opponents = opponents or []
    weights = np.asarray(opponent_weights or [1.0] * len(opponents), dtype=float)
    games = []
    for k, op in enumerate(starts):
        g = _core.GameState(s.profile)
        for a in op:
            assert g.apply(a)
        opp = None
        if opponents and rng.random() < s.league_fraction:
            opp = int(rng.choice(len(opponents), p=weights / weights.sum()))
        games.append({"state": g, "actions": list(op), "positions": [], "id": k, "opp": opp,
                      "main": "P0" if rng.random() < 0.5 else "P1", "start_len": len(op)})

    def controller(x) -> int | None:  # None = learner
        if x["opp"] is None or x["state"].actor == x["main"]:
            return None
        return x["opp"]

    rnd, t0 = 0, time.time()
    while True:
        live = [x for x in games if not x["state"].is_terminal and len(x["actions"]) < s.max_plies]
        if not live:
            break
        rnd += 1
        jobs = []  # (games, evaluator, feature_config, sims, record, kind)
        mine = [x for x in live if controller(x) is None]
        full = rng.random(len(mine)) < s.p_full
        jobs.append(([x for x, f in zip(mine, full) if f], evaluator, feature_config, s.sims_full, True, "engine"))
        jobs.append(([x for x, f in zip(mine, full) if not f], evaluator, feature_config, s.sims_fast, False, "engine"))
        for k, o in enumerate(opponents):
            jobs.append(([x for x in live if controller(x) == k], o.evaluator, o.feature_config, s.opponent_sims, False,
                         o.kind))
        for j, (todo, ev, fcfg, sims, record, kind) in enumerate(jobs):
            if not todo:
                continue
            if kind == "heuristic":
                moves = [{"action": _core.heuristic_action(x["state"], seed * 7919 + rnd * 131 + x["id"])} for x in todo]
            elif kind == "random":
                moves = [{"action": _core.random_action(x["state"], seed * 7919 + rnd * 131 + x["id"])} for x in todo]
            else:
                moves = search_many([x["state"] for x in todo], ev, _search_settings(s, sims, seed * 100_003 + rnd * 16 + j),
                                    fcfg)
            for x, r in zip(todo, moves):
                st = x["state"]
                if record and (s.record_opening or st.phase == "NORMAL_RENJU"):
                    x["positions"].append({"i": len(x["actions"]), "actor": r["actor"], "phase": st.phase,
                                           "policy": _policy_target(r), "value": round(float(r["value"]), 5),
                                           "proven": int(r["proven"])})
                if not st.apply(r["action"]):
                    raise RuntimeError(f"illegal action {r['action']} in phase {st.phase}")
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
                                    "opening_len": x["start_len"], "result": st.result, "winner": st.winner,
                                    "black_holder": st.black_holder, "plies": len(x["actions"]),
                                    "truncated": not st.is_terminal, "final_hash": st.hash(),
                                    "opponent": "self" if x["opp"] is None else opponents[x["opp"]].name,
                                    "main_player": None if x["opp"] is None else x["main"],
                                    "selfplay": asdict(s)},
            "actions": x["actions"],
            "positions": x["positions"],
        })
    return out
