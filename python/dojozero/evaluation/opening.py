"""Taraguchi-10 opening diagnostics (EVALUATION.md §7, TRAINING.md §2.5).

* opening_stats: decision frequencies per phase (only decisions made by players, not by a random
  start prefix), P0 (tentative black) and final black-holder win rates, opening diversity.
* OpeningDB: canonical opening state -> visits, mean outcome for the actor, mean root value,
  model version (JSON, merged across generations).
* proposal_regret: quality of ten-proposal sets. With the network value V_black(q) of each
  candidate fifth move q, the optimal set is the top-10 equivalence classes by V_black (white
  picks the minimum, so black maximises the 10th best: RULES.md §10.4). Regret = min V over that
  reference set - min V over the proposals actually offered (>= 0, smaller is better). It is a
  network-value diagnostic, not ground truth.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from .. import _core
from ..selfplay.records import z_for

DECISION_PHASES = ("SWAP_AFTER_1", "SWAP_AFTER_2", "SWAP_AFTER_3", "CHOICE_AFTER_4", "SWAP_BEFORE_DIRECT_5",
                   "SWAP_AFTER_DIRECT_5")


def opening_stats(games: list[dict]) -> dict:
    decisions: dict[str, Counter] = defaultdict(Counter)
    p0_score, black_score, normal_hashes = [], [], []
    for game in games:
        meta = game["meta"]
        g = _core.GameState(meta["profile"])
        start = meta.get("opening_len", 0)
        reached = False
        for i, a in enumerate(game["actions"]):
            if g.phase == "NORMAL_RENJU" and not reached:
                normal_hashes.append(g.canonical_hash())
                reached = True
            if i >= start and g.phase in DECISION_PHASES:
                decisions[g.phase][a] += 1
            g.apply(a)
        w = meta["winner"]
        p0_score.append(0.5 if w is None else float(w == "P0"))
        black_score.append(0.5 if w is None else float(w == meta["black_holder"]))
    out = {"games": len(games), "p0_score": float(np.mean(p0_score)) if games else None,
           "black_holder_score": float(np.mean(black_score)) if games else None}
    for ph in DECISION_PHASES:
        c = decisions.get(ph)
        if c:
            total = sum(c.values())
            out[f"decisions/{ph}"] = {k: round(v / total, 3) for k, v in sorted(c.items())} | {"n": total}
    cnt = Counter(normal_hashes)
    n = sum(cnt.values())
    out["distinct_normal_starts"] = len(cnt)
    out["normal_start_entropy_bits"] = -sum(v / n * math.log2(v / n) for v in cnt.values()) if n else 0.0
    return out


class OpeningDB:
    """canonical opening state (hash:phase) -> statistics, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def add_games(self, games: list[dict], model_version: int) -> int:
        added = 0
        for game in games:
            meta = game["meta"]
            g = _core.GameState(meta["profile"])
            targets = {p["i"]: p for p in game["positions"]}
            for i, a in enumerate(game["actions"]):
                if g.phase == "NORMAL_RENJU":
                    break
                if i in targets:
                    p = targets[i]
                    key = f"{g.canonical_hash():016x}:{g.phase}"
                    e = self.data.setdefault(key, {"phase": g.phase, "n": 0, "z_sum": 0.0, "v_sum": 0.0})
                    e["n"] += 1
                    e["z_sum"] += z_for(p["actor"], meta["winner"])
                    e["v_sum"] += p["value"]
                    e["model_version"] = model_version
                    best = max(p["policy"].items(), key=lambda kv: kv[1])
                    e["top_action"] = best[0]
                    added += 1
                g.apply(a)
        return added

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=0), encoding="utf-8")
        tmp.replace(self.path)

    def summary(self) -> dict:
        by_phase = Counter(e["phase"] for e in self.data.values())
        return {"entries": len(self.data), "by_phase": dict(by_phase)}


def _black_values(evaluator, feature_config, states: list) -> np.ndarray:
    """Network value (W - L) for the black holder of each state; wdl logits are for the actor."""
    if not states:
        return np.zeros(0)
    feats = np.stack([np.asarray(_core.encode(s, feature_config)) for s in states])
    wdl = evaluator(feats)["wdl"].astype(np.float64)
    wdl = np.exp(wdl - wdl.max(axis=1, keepdims=True))
    wdl /= wdl.sum(axis=1, keepdims=True)
    sign = np.array([1.0 if s.actor == s.black_holder else -1.0 for s in states])
    return (wdl[:, 0] - wdl[:, 2]) * sign


def _after_selecting(offer_state, proposal: str):
    """State after `proposal` was offered and selected: offer it first, fill the remaining slots
    with any admissible proposals, then select it (the engine checks every step)."""
    s = offer_state.copy()
    assert s.apply(proposal)
    while s.phase == "OFFER_TEN_5TH_MOVES":
        assert s.apply(next(a for a in s.legal_actions() if a.startswith("propose:")))
    assert s.phase == "SELECT_ONE_OF_TEN"
    assert s.apply("select:" + proposal.split(":", 1)[1])
    return s


def proposal_regret(evaluator, feature_config, games: list[dict], max_positions: int = 20) -> dict:
    """Regret of offered ten-proposal sets vs the value-greedy reference (see module docstring)."""
    regrets = []
    for game in games:
        if len(regrets) >= max_positions:
            break
        meta = game["meta"]
        g = _core.GameState(meta["profile"])
        offer_start, offered = None, []
        for i, a in enumerate(game["actions"]):
            if g.phase == "OFFER_TEN_5TH_MOVES":
                if offer_start is None:
                    offer_start = g.copy()
                    start_i = i
                offered.append(a)
            g.apply(a)
            if g.phase not in ("OFFER_TEN_5TH_MOVES",) and offer_start is not None:
                break
        # only sets proposed by a player (not by a random start prefix)
        if offer_start is None or len(offered) != 10 or start_i < meta.get("opening_len", 0):
            continue
        cands = [a for a in offer_start.legal_actions() if a.startswith("propose:")]
        vals = dict(zip(cands, _black_values(evaluator, feature_config, [_after_selecting(offer_start, a) for a in cands])))
        # Greedy reference: repeatedly take the highest-value admissible proposal (the engine's legal
        # PROPOSE actions enforce symmetry distinctness); greedy maximises the 10th best value.
        ref, ref_vals = offer_start.copy(), []
        while ref.phase == "OFFER_TEN_5TH_MOVES":
            legal = [a for a in ref.legal_actions() if a.startswith("propose:")]
            best = max(legal, key=lambda a: vals[a])
            ref_vals.append(vals[best])
            ref.apply(best)
        regrets.append(min(ref_vals) - min(vals[a] for a in offered))
    if not regrets:
        return {"positions": 0}
    return {"positions": len(regrets), "mean_regret": float(np.mean(regrets)), "max_regret": float(np.max(regrets))}
