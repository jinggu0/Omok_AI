"""Paired matches with confidence intervals (EVALUATION.md §4, §5).

Every opening is played twice with the roles swapped: game 1 has player A as P0, game 2 has
player B as P0. Openings are random Taraguchi-10 openings (heuristic placements near the centre,
random swap/branch decisions) played until NORMAL_RENJU, so both players face the same colours
and positions from each start. The unit of statistics is the pair (pentanomial).

    python -m dojozero.evaluation.match --a engine:runs/.../step_6000.pt:sims=64 --b heuristic --pairs 50
Player specs: "heuristic", "random", "policy:<ckpt>", "engine:<ckpt>[:sims=N][:root=PUCT|GUMBEL]".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..model.load import load_model
from ..search.driver import NetEvaluator, SearchSettings, search_many

PROFILE = "OMOKDOJANG_DAN"


def make_openings(n: int, seed: int, profile: str = PROFILE, max_attempts_per_opening: int = 50) -> list[list[str]]:
    """n distinct random Taraguchi-10 openings (distinct by canonical D4 hash), played up to
    NORMAL_RENJU. Placements are uniform over legal points within Chebyshev distance 3 of H8 when
    such points exist (else over all legal points); swap/branch/proposal/selection decisions are
    uniform. Raises instead of looping forever when distinct openings run out."""
    rng = np.random.default_rng([seed, 17])
    seen, out = set(), []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * max_attempts_per_opening:
            raise RuntimeError(f"could only build {len(out)} distinct openings out of {n}")
        g = _core.GameState(profile)
        actions = []
        while g.phase != "NORMAL_RENJU" and not g.is_terminal:
            legal = g.legal_actions()
            near = []
            for a in legal:
                pt = a.split(":")[-1]
                if pt[:1].isalpha() and pt[1:].isdigit():
                    p = _core.str_to_point(pt)
                    if max(abs(p // 15 - 7), abs(p % 15 - 7)) <= 3:
                        near.append(a)
            pool = near or legal
            a = pool[int(rng.integers(len(pool)))]
            g.apply(a)
            actions.append(a)
        if g.is_terminal or g.canonical_hash() in seen:
            continue
        seen.add(g.canonical_hash())
        out.append(actions)
    return out


@dataclass
class Player:
    spec: str
    kind: str
    evaluator: object = None
    feature_config: object = None
    settings: SearchSettings = field(default_factory=SearchSettings)

    def name(self) -> str:
        return self.spec


def make_player(spec: str, device: str, cache: dict) -> Player:
    kind, _, rest = spec.partition(":")
    if kind in ("heuristic", "random"):
        return Player(spec, kind)
    # Options are trailing ":key=value" segments; the checkpoint path may itself contain ':' (C:\...).
    segs = rest.split(":")
    opts = []
    while segs and "=" in segs[-1] and segs[-1].split("=")[0] in ("sims", "root", "tactics"):
        opts.insert(0, segs.pop())
    ckpt = ":".join(segs)
    parts = [kind, ckpt] + opts
    if ckpt not in cache:
        model, fspec, _ = load_model(ckpt, device)
        cache[ckpt] = (NetEvaluator(model, device), fspec.core())
    evaluator, fcfg = cache[ckpt]
    s = SearchSettings()
    for kv in parts[2:]:
        k, v = kv.split("=")
        if k == "sims":
            s.simulations = int(v)
        elif k == "root":
            s.root = v
        elif k == "tactics":
            s.root_tactics_nodes = int(v)
    return Player(spec, kind, evaluator, fcfg, s)


def policy_moves(player: Player, states: list) -> list[str]:
    feats = np.stack([np.asarray(_core.encode(g, player.feature_config)) for g in states])
    out = player.evaluator(feats)
    moves = []
    for g, logits in zip(states, out["policy"]):
        legal = np.asarray(_core.legal_mask(g)).astype(bool)
        if not legal.any():  # decision phase without point actions: fall back to the heuristic
            moves.append(_core.heuristic_action(g, 0))
            continue
        idx = int(np.argmax(np.where(legal, logits, -np.inf)))
        moves.append("pass" if idx == 225 else _core.point_to_str(idx))
    return moves


def choose_moves(player: Player, states: list, seed: int) -> list[str]:
    if player.kind == "heuristic":
        return [_core.heuristic_action(g, seed + i) for i, g in enumerate(states)]
    if player.kind == "random":
        return [_core.random_action(g, seed + i) for i, g in enumerate(states)]
    if player.kind == "policy":
        return policy_moves(player, states)
    player.settings.seed = seed
    return [r["action"] for r in search_many(states, player.evaluator, player.settings, player.feature_config)]


def play(a: Player, b: Player, openings: list[list[str]], seed: int, verbose: bool = True) -> list[dict]:
    games = []
    for k, op in enumerate(openings):
        for swap in (False, True):
            g = _core.GameState(PROFILE)
            for act in op:
                assert g.apply(act)
            games.append({"opening": k, "a_is_p0": not swap, "state": g, "moves": list(op)})
    rnd = 0
    t0 = time.time()
    while True:
        live = [x for x in games if not x["state"].is_terminal]
        if not live:
            break
        rnd += 1
        for player, is_a in ((a, True), (b, False)):
            todo = [x for x in live if not x["state"].is_terminal and
                    ((x["state"].actor == "P0") == (x["a_is_p0"] == is_a))]
            if not todo:
                continue
            moves = choose_moves(player, [x["state"] for x in todo], seed + rnd * 1000 + (0 if is_a else 500))
            for x, m in zip(todo, moves):
                if not x["state"].apply(m):
                    raise RuntimeError(f"illegal move {m} from {player.name()}")
                x["moves"].append(m)
        if verbose and rnd % 10 == 0:
            done = sum(x["state"].is_terminal for x in games)
            print(f"round {rnd}: {done}/{len(games)} games finished ({time.time() - t0:.0f}s)", flush=True)
    results = []
    for x in games:
        w = x["state"].winner
        a_player = "P0" if x["a_is_p0"] else "P1"
        score = 0.5 if w is None else (1.0 if w == a_player else 0.0)
        a_color = "BLACK" if x["state"].black_holder == a_player else "WHITE"  # fixed after the opening
        results.append({"opening": x["opening"], "a_is_p0": x["a_is_p0"], "a_color": a_color, "score_a": score,
                        "result": x["state"].result, "plies": len(x["moves"]), "moves": x["moves"]})
    return results


def elo(s: float) -> float:
    s = min(max(s, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; conservative when all pair scores are identical (normal CI collapses)."""
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(results: list[dict]) -> dict:
    by_open: dict[int, list[float]] = {}
    for r in results:
        by_open.setdefault(r["opening"], []).append(r["score_a"])
    pairs = np.array([np.mean(v) for v in by_open.values()])
    n = len(pairs)
    mean = float(pairs.mean())
    se = float(pairs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    lo_n, hi_n = mean - 1.96 * se, mean + 1.96 * se
    lo_w, hi_w = wilson(mean, n)
    lo, hi = min(lo_n, lo_w), max(hi_n, hi_w)  # report the wider (more conservative) interval
    penta = {f"{k:.2f}": int((pairs == k).sum()) for k in (0.0, 0.25, 0.5, 0.75, 1.0)}
    scores = np.array([r["score_a"] for r in results])
    as_black = [r["score_a"] for r in results if r["a_color"] == "BLACK"]
    as_white = [r["score_a"] for r in results if r["a_color"] == "WHITE"]
    return {
        "games": len(results), "pairs": n,
        "a_wins": int((scores == 1).sum()), "draws": int((scores == 0.5).sum()), "a_losses": int((scores == 0).sum()),
        "score": mean, "score_ci95": [lo, hi], "score_ci95_normal": [lo_n, hi_n], "score_ci95_wilson": [lo_w, hi_w],
        "elo": elo(mean), "elo_ci95": [elo(lo), elo(hi)],
        "pentanomial": penta,
        "a_score_as_black": float(np.mean(as_black)) if as_black else None,
        "a_score_as_white": float(np.mean(as_white)) if as_white else None,
        "mean_plies": float(np.mean([r["plies"] for r in results])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    ap.add_argument("--out")
    args = ap.parse_args()
    cache: dict = {}
    a = make_player(args.a, args.device, cache)
    b = make_player(args.b, args.device, cache)
    openings = make_openings(args.pairs, args.seed)
    t0 = time.time()
    results = play(a, b, openings, args.seed)
    summary = {"a": args.a, "b": args.b, "seed": args.seed, "profile": PROFILE, "seconds": time.time() - t0,
               "torch": torch.__version__, "device": args.device} | summarize(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary | {"games_detail": results}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
