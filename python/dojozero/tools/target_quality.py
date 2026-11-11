"""Self-play target quality probe (PROGRESS Phase 8 E5s; pre-registration benchmarks/reanalysis/e5s/preregistration.json).

Question: how far are the recorded 64-simulation Gumbel targets of the RL loop from what the SAME network gives
with 16x more search (1,024 simulations)? Diagnostic only (no training, no games).

For N NORMAL_RENJU positions of T'' self-play generation 63 that were not solved at the root by tactics
(proven == 2, search.h kUnknown), the state is rebuilt from the game record and searched again with the generating network
(v6 = gen_058) and the self-play search settings (GUMBEL root, m 16, noise on, root tactics 2,000 nodes):
  rec   - the recorded target (64 sims, self-play seed)
  a64   - a fresh 64-sim search (new seed): the seed-to-seed noise floor of a 64-sim target
  m256  - a 256-sim search (middle point)
  r1024 - a 1,024-sim search (reference)
Measured: top-1 agreement between targets, KL(r1024 || x), root value differences, and the solver label
(E4 budgets: VCF 50k / VCT 30k) of each target's top-1 move: PROVEN_UNSAFE = the opponent then has a
proven forced win.

    python -m dojozero.tools.target_quality --n 800 --out benchmarks/reanalysis/e5s/target_quality.json
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .. import _core
from ..evaluation.match import make_player
from ..search.driver import SearchSettings, search_many
from ..selfplay.generate import _policy_target
from .reanalysis_data import _analyze_move

REPO = Path(__file__).resolve().parents[3]
CHUNK = "runs/e5e_rl_T/selfplay/gen_063.jsonl.gz"
MODEL = "runs/e5e_rl_T/models/gen_058.pt"
MODEL_SHA = "a3e5562aa6253629b16fbe7b2357c3ddef06c2a6c24a43b4a0a39efa1c11367b"
PROFILE = "OMOKDOJANG_DAN"
SEARCH = {"root": "GUMBEL", "gumbel_m": 16, "gumbel_noise": True, "root_tactics_nodes": 2000, "max_inflight": 8}


def _lit(x):
    return x if isinstance(x, (list, dict)) else ast.literal_eval(x)


def load_positions(n: int, seed: int) -> tuple[list[dict], dict]:
    pool, stats = [], {"games": 0, "normal": 0, "proven": 0}
    with gzip.open(REPO / CHUNK, "rt") as f:
        for line in f:
            d = json.loads(line)
            meta, actions = _lit(d["meta"]), _lit(d["actions"])
            stats["games"] += 1
            for p in _lit(d["positions"]):
                if p["phase"] != "NORMAL_RENJU":
                    continue
                stats["normal"] += 1
                if p.get("proven") != 2:  # search.h kUnknown = 2; 1 / -1 = root proven win / loss
                    stats["proven"] += 1
                    continue
                pool.append({"game": meta["game_id"], "i": p["i"], "actor": p["actor"], "actions": actions[:p["i"]],
                             "rec": p["policy"], "rec_value": p["value"], "winner": meta["winner"],
                             "opponent": meta["opponent"]})
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))
    stats["pool"] = len(pool)
    return [pool[k] for k in idx], stats


def state_of(pos: dict):
    g = _core.GameState(PROFILE)
    for a in pos["actions"]:
        assert g.apply(a), a
    assert g.phase == "NORMAL_RENJU" and g.actor == pos["actor"], (g.phase, g.actor, pos["actor"])
    return g


def run_search(states, ev, fcfg, sims: int, seed: int, chunk: int = 64) -> list[dict]:
    out = []
    for s in range(0, len(states), chunk):
        st = SearchSettings(simulations=sims, seed=seed + s, **SEARCH)
        rs = search_many(states[s:s + chunk], ev, st, fcfg)
        out += [{"policy": _policy_target(r), "value": float(r["value"]), "action": r["action"]} for r in rs]
    return out


def top1(p: dict) -> str:
    return max(p.items(), key=lambda kv: kv[1])[0]


def kl(p: dict, q: dict, eps: float = 1e-4) -> float:
    """KL(p || q) over p's support, q floored at eps and renormalised over the union support."""
    keys = set(p) | set(q)
    qq = {k: max(q.get(k, 0.0), eps) for k in keys}
    z = sum(qq.values())
    return float(sum(v * math.log(v / (qq[k] / z)) for k, v in p.items() if v > 0))


def ci(x) -> dict:
    x = np.asarray(x, dtype=float)
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"mean": m, "ci95": [m - 1.96 * se, m + 1.96 * se], "n": int(len(x))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--ref-sims", type=int, default=1024)
    ap.add_argument("--mid-sims", type=int, default=256)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    pos, stats = load_positions(a.n, a.seed)
    states = [state_of(p) for p in pos]
    pl = make_player("engine:" + str(REPO / MODEL), a.device, {})
    from ..model.load import model_sha
    assert model_sha(pl.evaluator.model) == MODEL_SHA, "not the generating network"
    a64 = run_search(states, pl.evaluator, pl.feature_config, 64, 7_000_000)
    t64 = time.time() - t0
    m256 = run_search(states, pl.evaluator, pl.feature_config, a.mid_sims, 8_000_000)
    ref = run_search(states, pl.evaluator, pl.feature_config, a.ref_sims, 9_000_000)
    tref = time.time() - t0 - t64
    # solver labels of every distinct top-1 move
    jobs = {}
    for k, p in enumerate(pos):
        for tag, pol in (("rec", p["rec"]), ("a64", a64[k]["policy"]), ("m256", m256[k]["policy"]), ("ref", ref[k]["policy"])):
            jobs.setdefault((k, top1(pol)), None)

    def lab(key):
        k, m = key
        return key, _analyze_move(states[k], m)["code"]

    with ThreadPoolExecutor(a.threads) as ex:
        labels = dict(ex.map(lab, list(jobs)))
    rows = []
    for k, p in enumerate(pos):
        t = {"rec": top1(p["rec"]), "a64": top1(a64[k]["policy"]), "m256": top1(m256[k]["policy"]),
             "ref": top1(ref[k]["policy"])}
        z = 1.0 if p["winner"] == p["actor"] else (-1.0 if p["winner"] is not None else 0.0)
        rows.append({"game": p["game"], "i": p["i"], "ply": len(p["actions"]), "opponent": p["opponent"],
                     "top1": t, "label": {x: labels[(k, m)] for x, m in t.items()},
                     "kl_ref_rec": kl(ref[k]["policy"], p["rec"]), "kl_ref_a64": kl(ref[k]["policy"], a64[k]["policy"]),
                     "kl_a64_rec": kl(a64[k]["policy"], p["rec"]), "kl_ref_m256": kl(ref[k]["policy"], m256[k]["policy"]),
                     "value": {"rec": p["rec_value"], "a64": a64[k]["value"], "m256": m256[k]["value"],
                               "ref": ref[k]["value"]}, "z": z})
    agree = lambda x, y: [float(r["top1"][x] == r["top1"][y]) for r in rows]  # noqa: E731
    unsafe = lambda x: [float(r["label"][x] == "PROVEN_UNSAFE") for r in rows]  # noqa: E731
    v = {x: np.array([r["value"][x] for r in rows]) for x in ("rec", "a64", "m256", "ref")}
    zz = np.array([r["z"] for r in rows])
    summary = {
        "positions": len(rows), "selection": stats,
        "top1_agreement": {"rec_vs_a64": ci(agree("rec", "a64")), "rec_vs_ref": ci(agree("rec", "ref")),
                           "a64_vs_ref": ci(agree("a64", "ref")), "m256_vs_ref": ci(agree("m256", "ref"))},
        "kl": {"ref_rec": ci([r["kl_ref_rec"] for r in rows]), "ref_a64": ci([r["kl_ref_a64"] for r in rows]),
               "a64_rec": ci([r["kl_a64_rec"] for r in rows]), "ref_m256": ci([r["kl_ref_m256"] for r in rows])},
        "top1_proven_unsafe": {x: ci(unsafe(x)) for x in ("rec", "a64", "m256", "ref")},
        "unsafe_rec_minus_ref": ci(np.array(unsafe("rec")) - np.array(unsafe("ref"))),
        "unsafe_a64_minus_ref": ci(np.array(unsafe("a64")) - np.array(unsafe("ref"))),
        "unsafe_m256_minus_ref": ci(np.array(unsafe("m256")) - np.array(unsafe("ref"))),
        "label_counts": {x: {c: sum(r["label"][x] == c for r in rows) for c in
                             ("PROVEN_UNSAFE", "PROVEN_WINNING", "UNREFUTED_EXHAUSTED", "UNKNOWN_BUDGET", "TERMINAL_OTHER")}
                         for x in ("rec", "a64", "m256", "ref")},
        "value": {"mean_abs_rec_minus_ref": float(np.mean(np.abs(v["rec"] - v["ref"]))),
                  "mean_abs_a64_minus_ref": float(np.mean(np.abs(v["a64"] - v["ref"]))),
                  "mean_abs_m256_minus_ref": float(np.mean(np.abs(v["m256"] - v["ref"]))),
                  "mse_vs_outcome": {x: float(np.mean((v[x] - zz) ** 2)) for x in v},
                  "corr_with_outcome": {x: float(np.corrcoef(v[x], zz)[0, 1]) for x in v}},
        "seconds": {"a64": t64, "ref": tref, "total": time.time() - t0},
    }
    out = {"chunk": CHUNK, "model": MODEL, "model_sha": MODEL_SHA, "search": SEARCH, "ref_sims": a.ref_sims,
           "solver": "reanalysis_data._analyze_move (VCF 50k / VCT 30k)", "seed": a.seed, "summary": summary, "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
