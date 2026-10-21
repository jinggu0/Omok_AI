"""Batched search driver: C++ trees + PyTorch network inference (ARCHITECTURE.md §3.10).

Python owns the network; C++ owns the trees. Each round, leaves from every tree are collected into
one batch, evaluated once on the device (XPU bf16 by default), and applied back. Nothing calls
Python from C++ threads.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from .. import _core
from ..features import NUM_POINTS, NUM_POLICY


@dataclass
class SearchSettings:
    simulations: int = 128
    root: str = "PUCT"            # or "GUMBEL"
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    gumbel_m: int = 16
    gumbel_noise: bool = False    # evaluation: deterministic; self-play: True
    leaf_tactics: bool = True
    leaf_vcf_nodes: int = 0
    root_tactics_nodes: int = 5000
    max_inflight: int = 8
    seed: int = 1
    max_ms: float = 0.0           # wall-clock limit per move (0 = simulations only; then a cap)
    # Root defense filter (PROGRESS Phase 8 E2c/E3): after the search, the defense_k most-visited
    # root moves are checked in order with the C++ solver; a move after which the opponent has a
    # proven forced win (VCF/VCT) is skipped when an unrefuted candidate exists. 0 = off.
    defense_k: int = 0
    defense_share: float = 0.3    # with max_ms: share of the move time reserved for the filter
    defense_ms: float = 300.0     # without max_ms: wall-clock cap per candidate check
    defense_vcf: int = 100000
    defense_vct: int = 50000
    defense_threads: int = 4
    defense_impl: str = "live"    # "ref" = frozen ee1d011 solver (same-binary A/B, Phase 8 E3b)
    defense_carry: bool = True    # VCT may use time a fast VCF left unused (TacticalLimits)
    # Proof-aware MCTS (PROGRESS Phase 8 E3d): solver threads prove the most-visited tree nodes while
    # the search runs (SearchConfig.proof_*). 0 = off.
    proof_threads: int = 0
    proof_vcf_nodes: int = 20000
    proof_vct_nodes: int = 4000
    proof_max_vct_nodes: int = 256000
    proof_min_visits: int = 4
    proof_max_depth: int = 3


def make_config(s: SearchSettings) -> "_core.SearchConfig":
    c = _core.SearchConfig()
    c.simulations = s.simulations
    c.root = _core.RootMode.GUMBEL if s.root.upper() == "GUMBEL" else _core.RootMode.PUCT
    c.c_puct = s.c_puct
    c.fpu_reduction = s.fpu_reduction
    c.gumbel_m = s.gumbel_m
    c.gumbel_noise = s.gumbel_noise
    c.leaf_tactics = s.leaf_tactics
    c.leaf_vcf_nodes = s.leaf_vcf_nodes
    c.root_tactics_nodes = s.root_tactics_nodes
    c.max_inflight = s.max_inflight
    c.seed = s.seed
    c.max_ms = s.max_ms * (1.0 - s.defense_share) if (s.defense_k > 0 and s.max_ms > 0) else s.max_ms
    c.proof_threads = s.proof_threads
    c.proof_vcf_nodes = s.proof_vcf_nodes
    c.proof_vct_nodes = s.proof_vct_nodes
    c.proof_max_vct_nodes = s.proof_max_vct_nodes
    c.proof_min_visits = s.proof_min_visits
    c.proof_max_depth = s.proof_max_depth
    return c


class NetEvaluator:
    """Callable: uint8 features [N, C, 15, 15] -> dict of float32 numpy heads."""

    def __init__(self, model: torch.nn.Module, device: torch.device | str, amp_bf16: bool = True):
        self.model = model.eval()
        self.device = torch.device(device)
        self.amp = amp_bf16 and self.device.type == "xpu"
        self.evaluations = 0
        self.batches = 0

    BUCKETS = (8, 16, 32, 64, 128, 256, 512, 1024)

    @torch.no_grad()
    def __call__(self, feats: np.ndarray) -> dict[str, np.ndarray]:
        # Pad to a fixed bucket size: every new input shape makes the XPU backend prepare new
        # kernels, which measurably dominated small searches (docs/PROGRESS.md Phase 5).
        n = len(feats)
        size = next((b for b in self.BUCKETS if b >= n), n)
        if size != n:
            feats = np.concatenate([feats, np.zeros((size - n,) + feats.shape[1:], feats.dtype)])
        x = torch.from_numpy(feats).to(self.device, dtype=torch.float32)
        ctx = torch.autocast(device_type="xpu", dtype=torch.bfloat16) if self.amp else contextlib.nullcontext()
        with ctx:
            out = self.model(x)
        self.evaluations += n
        self.batches += 1
        return {k: out[k][:n].float().cpu().numpy() for k in ("policy", "wdl", "swap", "branch", "propose", "select")}


class UniformNumpyEvaluator:
    """Network stand-in with all-zero logits (equal priors, value 0) for tests and timing."""

    def __call__(self, feats: np.ndarray) -> dict[str, np.ndarray]:
        n = len(feats)
        z = lambda k: np.zeros((n, k), np.float32)  # noqa: E731
        return {"policy": z(NUM_POLICY), "wdl": z(3), "swap": z(2), "branch": z(2), "propose": z(NUM_POINTS),
                "select": z(NUM_POINTS)}


def search_many(states: list, evaluator, settings: SearchSettings, feature_config: "_core.FeatureConfig",
                batch_size: int = 256) -> list[dict]:
    """Search every state (one tree each) with shared network batches; returns result dicts."""
    sb = _core.SearchBatch(feature_config)
    sb.add_many(states, make_config(settings))  # parallel tree construction (root tactics)
    idle = 0
    while not sb.done():
        feats = sb.collect(batch_size)
        if len(feats) == 0:
            idle += 1
            if idle > 3:
                raise RuntimeError("search stalled: no leaf could be selected")
            continue
        idle = 0
        out = evaluator(feats)
        sb.apply(out["policy"], out["wdl"], out["swap"], out["branch"], out["propose"], out["select"])
    results = [sb.result(i) for i in range(len(states))]
    if settings.defense_k > 0:
        results = [defend(g, r, settings) for g, r in zip(states, results)]
    return results


def defend(state, result: dict, s: SearchSettings) -> dict:
    """Root defense filter: keep the search's choice unless the solver proves that it gives the
    opponent a forced win while a lower-ranked candidate (by visits) is not refuted. Proofs come from
    the sound C++ solver (WIN = proven); UNKNOWN counts as not refuted. If every candidate is
    refuted, or the root is already a proven win, the search's choice stands."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    info = {"checked": 0, "refuted": [], "changed": False, "ms": 0.0, "checks": []}
    result["defense"] = info
    if state.phase != "NORMAL_RENJU" or result.get("proven") == 1 or not result.get("children"):
        return result
    t0 = time.perf_counter()
    kids = sorted(result["children"], key=lambda c: c["visits"], reverse=True)
    order = [result["action"]] + [c["action"] for c in kids if c["action"] != result["action"]]
    cands = [a for a in order if a != "pass"][: s.defense_k]
    threads = max(1, min(s.defense_threads, len(cands)))
    rounds = -(-len(cands) // threads)  # checks run in parallel rounds; split the reserved time over them
    ms = (s.max_ms * s.defense_share / rounds) if s.max_ms > 0 else s.defense_ms

    def check(a: str) -> tuple[bool, dict]:
        g = state.copy()
        if not g.apply(a):
            return True, {"a": a, "result": "ILLEGAL"}
        if g.is_terminal:
            return g.winner is not None and g.winner != state.actor, {"a": a, "result": "TERMINAL"}
        d = _core.analyze_tactics(g, s.defense_vcf, s.defense_vct, 0, ms, s.defense_impl, s.defense_carry)
        # Opponent to move: WIN = proven forced win (refuted). UNKNOWN is never treated as refuted.
        return d["result"] == "WIN", {"a": a, "result": d["result"], "reason": d["reason"], "nodes": d["nodes"],
                                      "ms": d["elapsed_ms"]}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        checked = list(ex.map(check, cands))
    verdicts = [v for v, _ in checked]
    info["checks"] = [c for _, c in checked]
    info["budget_ms"] = ms
    info["checked"] = len(cands)
    info["refuted"] = [a for a, v in zip(cands, verdicts) if v]
    pick = next((a for a, v in zip(cands, verdicts) if not v), None)
    if pick is not None and pick != result["action"]:
        info["changed"] = True
        info["original"] = result["action"]
        result["action"] = pick
    info["ms"] = (time.perf_counter() - t0) * 1000
    return result
