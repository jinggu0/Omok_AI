"""Feature configuration and D4 augmentation.

Augmentation permutes cached C++ feature planes with the C++ transform table, so Python never
defines its own rotation convention (tested against encoding the transformed state in C++).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import _core

NUM_POINTS = 225
NUM_POLICY = _core.NUM_POLICY_ACTIONS  # 225 points + pass
NUM_AUX = _core.NUM_AUX_PLANES
AUX_NAMES = ["immediate_win", "forced_block", "four_threat", "three_threat", "forbidden"]


@dataclass(frozen=True)
class FeatureSpec:
    history_planes: int = 0
    tactical_planes: bool = False
    forbidden_plane: bool = True

    def core(self) -> "_core.FeatureConfig":
        return _core.FeatureConfig(self.history_planes, self.tactical_planes, self.forbidden_plane)

    @property
    def num_planes(self) -> int:
        return self.core().num_planes

    def hash(self) -> int:
        return self.core().hash()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hash"] = self.hash()
        d["plane_names"] = list(self.core().plane_names)
        return d


def _gather_indices() -> np.ndarray:
    """G[t][q] = source point p with T[t][p] = q, i.e. new[..., q] = old[..., G[t][q]]."""
    table = np.asarray(_core.transform_table(), dtype=np.int64)  # [8, 225]
    g = np.empty_like(table)
    for t in range(table.shape[0]):
        g[t, table[t]] = np.arange(NUM_POINTS)
    return g


GATHER = _gather_indices()  # [8, 225]


def transform_points(x: np.ndarray, t: int) -> np.ndarray:
    """Apply transform t to an array whose last axis is the 225 points."""
    return x[..., GATHER[t]]


def transform_policy(p: np.ndarray, t: int) -> np.ndarray:
    """Policy vectors [..., 226]: points permuted, pass index unchanged."""
    out = p.copy()
    out[..., :NUM_POINTS] = p[..., GATHER[t]]
    return out
