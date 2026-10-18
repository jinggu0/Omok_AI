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
    return [sb.result(i) for i in range(len(states))]
