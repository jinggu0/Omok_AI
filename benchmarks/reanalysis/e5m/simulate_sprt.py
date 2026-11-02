"""E5m m1: Monte Carlo of the RL loop's SPRT promotion (rl.loop.evaluate_pair: looks every 20 pairs,
min 20, max 80, elo0 0 / elo1 30, alpha = beta = 0.05). Pair scores come from the pentanomial of the
m2 replay, exponentially tilted to each true score.

    PYTHONPATH=python python benchmarks/reanalysis/e5m/simulate_sprt.py <replay.json> <out.json>
"""

import json
import sys

import numpy as np

from dojozero.evaluation.sprt import sprt

VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
TRUE = (0.50, 0.52, 0.55, 0.58, 0.60)
RUNS, LOOK, MIN, MAX = 20000, 20, 20, 80
E = dict(elo0=0.0, elo1=30.0, alpha=0.05, beta=0.05)


def pentanomial(replay: dict) -> np.ndarray:
    by_open: dict = {}
    for r in replay["games_detail"]:
        by_open.setdefault(r["opening"], []).append(r["score_a"])
    pairs = np.array([np.mean(v) for v in by_open.values()])
    return np.array([(pairs == v).sum() for v in VALUES], dtype=float) / len(pairs)


def tilt(p: np.ndarray, target: float) -> np.ndarray:
    def q(t):
        w = p * np.exp(t * VALUES)
        return w / w.sum()
    lo, hi = -50.0, 50.0  # the tilted mean is increasing in t: bisection
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if q(mid) @ VALUES < target else (lo, mid)
    return q((lo + hi) / 2)


def one(rng, q) -> tuple[str, int, float]:
    scores: list[float] = []
    while len(scores) < MAX:
        scores += list(rng.choice(VALUES, size=LOOK, p=q))
        res = sprt(scores, **E)
        if res.decision != "continue" and len(scores) >= MIN:
            break
    return res.decision, res.pairs, res.score


def main():
    replay = json.load(open(sys.argv[1], encoding="utf-8"))
    p = pentanomial(replay)
    rng = np.random.default_rng(2026_11_02)
    out = {"pentanomial_m2": dict(zip(map(str, VALUES), p.round(4).tolist())), "runs": RUNS, "sprt": E | {"look": LOOK, "min": MIN, "max": MAX}, "rows": {}}
    for t in TRUE:
        q = tilt(p, t)
        res = [one(rng, q) for _ in range(RUNS)]
        h1 = [s for d, _, s in res if d == "H1"]
        row = {"p_h1": len(h1) / RUNS, "p_h0": sum(d == "H0" for d, _, _ in res) / RUNS,
               "mean_score_shown_at_h1": float(np.mean(h1)) if h1 else None}
        out["rows"][f"{t:.2f}"] = row
        print(t, row, flush=True)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
