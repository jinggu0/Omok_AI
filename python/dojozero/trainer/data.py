"""Stage A tactical dataset loading (TRAINING.md §2.1-§2.2, §5, §6).

Raw samples (boards + solver labels) come from ``dojozero_datagen``. Features, legal masks and
auxiliary targets are computed by the C++ encoder (``_core.encode_boards``) and cached on disk under
``cache_<feature-hash>/`` so a feature change can never reuse stale planes.

Batches are stateless functions of (seed, step): resuming from a checkpoint reproduces the exact
batch sequence. D4 augmentation uses the C++ transform table (features.GATHER).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..features import GATHER, NUM_POINTS, FeatureSpec


@dataclass
class Split:
    train: np.ndarray
    holdout: np.ndarray


class TacticalDataset:
    FIELDS = ["boards", "sides", "profiles", "policy", "has_policy", "value", "has_value", "reason", "plies"]

    def __init__(self, root: str | Path, spec: FeatureSpec, threads: int = 0, verbose: bool = True):
        self.root = Path(root)
        self.spec = spec
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        raw = {k: np.load(self.root / f"{k}.npy") for k in self.FIELDS}
        self.raw = raw
        self.n = len(raw["sides"])
        cache = self.root / f"cache_{spec.hash():016x}"
        if (cache / "features.npy").exists():
            self.features = np.load(cache / "features.npy", mmap_mode="r")
            self.legal = np.load(cache / "legal.npy")
            self.aux = np.load(cache / "aux.npy", mmap_mode="r")
        else:
            cache.mkdir(parents=True, exist_ok=True)
            feats, legal, aux = [], [], []
            chunk = 10000
            for s in range(0, self.n, chunk):
                f, l, a = _core.encode_boards(raw["boards"][s : s + chunk], raw["sides"][s : s + chunk],
                                              raw["profiles"][s : s + chunk], spec.core(), threads)
                feats.append(f.reshape(len(f), f.shape[1], NUM_POINTS))
                legal.append(l)
                aux.append(a.reshape(len(a), a.shape[1], NUM_POINTS))
                if verbose:
                    print(f"encoded {min(s + chunk, self.n)}/{self.n}", flush=True)
            self.features = np.concatenate(feats)
            self.legal = np.concatenate(legal)
            self.aux = np.concatenate(aux)
            np.save(cache / "features.npy", self.features)
            np.save(cache / "legal.npy", self.legal)
            np.save(cache / "aux.npy", self.aux)
            (cache / "feature_spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        self._check_targets()

    def _check_targets(self) -> None:
        """Policy targets must be legal actions (TRAINING.md §4: illegal target mass = data error)."""
        bad = (self.raw["policy"] > 0) & (self.legal == 0)
        if bad.any():
            raise ValueError(f"{int(bad.any(axis=1).sum())} samples have policy targets on illegal actions")

    def split(self, holdout_pct: float, exclude_hashes: set[int] | None = None) -> Split:
        """Deterministic split by CRC32 of the board bytes (independent of Python's hash seed).
        Samples whose canonical hash is in exclude_hashes (benchmark positions) are dropped."""
        crc = np.array([zlib.crc32(self.raw["boards"][i].tobytes()) for i in range(self.n)], dtype=np.uint64)
        is_holdout = (crc % 10000) < int(holdout_pct * 100)
        keep = np.ones(self.n, dtype=bool)
        if exclude_hashes:
            for i in range(self.n):
                g = _core.GameState.from_board(self.raw["boards"][i], int(self.raw["sides"][i]),
                                               ["RENJU_RIF", "OMOKDOJANG_DAN"][int(self.raw["profiles"][i])])
                if g.canonical_hash() in exclude_hashes:
                    keep[i] = False
        idx = np.arange(self.n)
        return Split(train=idx[keep & ~is_holdout], holdout=idx[keep & is_holdout])

    def batch(self, indices: np.ndarray, transforms: np.ndarray | None, device: torch.device) -> dict:
        """Gather samples, apply per-sample D4 transforms (None = identity), move to device."""
        feats = np.asarray(self.features[indices])  # [B, C, 225]
        legal = self.legal[indices].copy()          # [B, 226]
        aux = np.asarray(self.aux[indices])         # [B, 5, 225]
        policy = self.raw["policy"][indices].copy()
        if transforms is not None:
            g = GATHER[transforms]  # [B, 225]
            feats = np.take_along_axis(feats, g[:, None, :], axis=2)
            aux = np.take_along_axis(aux, g[:, None, :], axis=2)
            legal[:, :NUM_POINTS] = np.take_along_axis(legal[:, :NUM_POINTS], g, axis=1)
            policy[:, :NUM_POINTS] = np.take_along_axis(policy[:, :NUM_POINTS], g, axis=1)
        b = len(indices)
        to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)  # noqa: E731
        return {
            "x": to(feats.reshape(b, feats.shape[1], 15, 15), torch.float32),
            "legal": to(legal, torch.uint8),
            "policy": to(policy, torch.float32),
            "has_policy": to(self.raw["has_policy"][indices], torch.bool),
            "value": to(self.raw["value"][indices], torch.int64),
            "has_value": to(self.raw["has_value"][indices], torch.bool),
            "aux": to(aux, torch.float32),
            "reason": self.raw["reason"][indices],
        }


def step_batch_indices(pool: np.ndarray, batch_size: int, seed: int, step: int, augment: bool):
    """Stateless sampling: identical for a given (seed, step) -> exact resume."""
    rng = np.random.default_rng([seed, step])
    idx = pool[rng.integers(0, len(pool), size=batch_size)]
    t = rng.integers(0, 8, size=batch_size) if augment else None
    return idx, t


def benchmark_hashes(bench_root: str | Path) -> set[int]:
    """Canonical hashes of all benchmark positions (leak check, EVALUATION.md §3)."""
    out: set[int] = set()
    for path in sorted(Path(bench_root).rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for p in doc.get("problems", []):
            try:
                g = _core.GameState.from_spec(json.dumps(p["position"]), p.get("profile", "RENJU_RIF"))
                out.add(g.canonical_hash())
            except ValueError:
                pass
    return out
