"""Replay buffer (TRAINING.md §5).

Ingestion replays each game's action sequence through the C++ GameState and, at every recorded
position, re-encodes features / legal mask / aux targets with the C++ encoder. While replaying it
verifies the record: the recorded actor must match the engine, every action must be legal, every
policy-target action must be legal at that position, and the final result must match the record
(a mismatch raises RecordError instead of training on corrupt data).

Window: only the most recent `window` generations are kept (older positions are dropped), with a
hard cap on positions. Sampling is a pure function of (seed, step) with per-sample D4 transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .. import _core
from ..features import GATHER, NUM_AUX, NUM_POINTS, NUM_POLICY, FeatureSpec
from ..selfplay.records import read_chunk, z_for


class RecordError(RuntimeError):
    pass


def _policy_index(action: str) -> int | None:
    if action == "pass":
        return NUM_POINTS
    if ":" in action or action in ("swap", "keep", "direct5", "offer10"):
        return None  # opening heads: trained from Phase 7
    return _core.str_to_point(action)


@dataclass
class _Block:
    generation: int
    feats: np.ndarray   # uint8 [n, C, 225]
    legal: np.ndarray   # uint8 [n, 226]
    aux: np.ndarray     # uint8 [n, 5, 225]
    policy: np.ndarray  # float32 [n, 226]
    value: np.ndarray   # int8 [n]
    root_value: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))


class ReplayBuffer:
    def __init__(self, spec: FeatureSpec, window: int = 4, capacity: int = 200_000):
        self.spec = spec
        self.cfg = spec.core()
        self.window = window
        self.capacity = capacity
        self.blocks: list[_Block] = []
        self.stats = {"games": 0, "positions": 0, "skipped_opening_targets": 0}

    def __len__(self) -> int:
        return sum(len(b.value) for b in self.blocks)

    def ingest(self, chunk_path, generation: int) -> int:
        games = read_chunk(chunk_path)
        feats, legal, aux, pol, val, rv = [], [], [], [], [], []
        for game in games:
            meta = game["meta"]
            g = _core.GameState(meta["profile"])
            targets = {p["i"]: p for p in game["positions"]}
            for i, a in enumerate(game["actions"] + [None]):
                if i in targets:
                    p = targets[i]
                    if g.actor != p["actor"]:
                        raise RecordError(f"{meta['game_id']}: actor mismatch at {i}")
                    mask = np.asarray(_core.legal_mask(g))
                    vec = np.zeros(NUM_POLICY, np.float32)
                    usable = True
                    for act, prob in p["policy"].items():
                        k = _policy_index(act)
                        if k is None:
                            usable = False
                            break
                        if not mask[k]:
                            raise RecordError(f"{meta['game_id']}: illegal policy target {act} at {i}")
                        vec[k] = prob
                    if not usable:
                        self.stats["skipped_opening_targets"] += 1
                    else:
                        feats.append(np.asarray(_core.encode(g, self.cfg)).reshape(-1, NUM_POINTS))
                        legal.append(mask)
                        aux.append(np.asarray(_core.aux_targets(g)).reshape(NUM_AUX, NUM_POINTS))
                        pol.append(vec / max(vec.sum(), 1e-12))
                        val.append(z_for(p["actor"], meta["winner"]))
                        rv.append(p["value"])
                if a is None:
                    break
                if not g.apply(a):
                    raise RecordError(f"{meta['game_id']}: illegal action {a} at {i}")
            if g.result != meta["result"] or g.winner != meta["winner"]:
                raise RecordError(f"{meta['game_id']}: replayed result {g.result} != recorded {meta['result']}")
            self.stats["games"] += 1
        if not val:
            return 0
        self.blocks.append(_Block(generation, np.stack(feats), np.stack(legal), np.stack(aux), np.stack(pol),
                                  np.asarray(val, np.int8), np.asarray(rv, np.float32)))
        self.stats["positions"] += len(val)
        self._evict()
        return len(val)

    def _evict(self) -> None:
        newest = max(b.generation for b in self.blocks)
        self.blocks = [b for b in self.blocks if b.generation > newest - self.window]
        while len(self) > self.capacity and len(self.blocks) > 1:
            self.blocks.pop(0)

    def _concat(self):
        if not hasattr(self, "_cache_key") or self._cache_key != tuple(id(b) for b in self.blocks):
            self._cat = {k: np.concatenate([getattr(b, k) for b in self.blocks])
                         for k in ("feats", "legal", "aux", "policy", "value")}
            self._cache_key = tuple(id(b) for b in self.blocks)
        return self._cat

    def sample(self, batch_size: int, seed: int, step: int, device, augment: bool = True) -> dict:
        data = self._concat()
        rng = np.random.default_rng([seed, step, 7])
        idx = rng.integers(0, len(data["value"]), size=batch_size)
        feats = data["feats"][idx]
        aux = data["aux"][idx]
        legal = data["legal"][idx].copy()
        policy = data["policy"][idx].copy()
        if augment:
            t = rng.integers(0, 8, size=batch_size)
            gi = GATHER[t]
            feats = np.take_along_axis(feats, gi[:, None, :], axis=2)
            aux = np.take_along_axis(aux, gi[:, None, :], axis=2)
            legal[:, :NUM_POINTS] = np.take_along_axis(legal[:, :NUM_POINTS], gi, axis=1)
            policy[:, :NUM_POINTS] = np.take_along_axis(policy[:, :NUM_POINTS], gi, axis=1)
        to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)  # noqa: E731
        b = len(idx)
        return {
            "x": to(feats.reshape(b, feats.shape[1], 15, 15), torch.float32),
            "legal": to(legal, torch.uint8),
            "policy": to(policy, torch.float32),
            "has_policy": torch.ones(b, dtype=torch.bool, device=device),
            "value": to(data["value"][idx], torch.int64),
            "has_value": torch.ones(b, dtype=torch.bool, device=device),
            "aux": to(aux, torch.float32),
        }

    def value_balance(self) -> dict:
        v = self._concat()["value"]
        return {"win": float((v > 0).mean()), "draw": float((v == 0).mean()), "loss": float((v < 0).mean())}
