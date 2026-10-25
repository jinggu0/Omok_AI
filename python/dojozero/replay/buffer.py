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
from ..trainer.losses import HEAD_BRANCH, HEAD_POLICY, HEAD_PROPOSE, HEAD_SELECT, HEAD_SWAP


class RecordError(RuntimeError):
    pass


DECISION_INDEX = {"swap": (HEAD_SWAP, 0), "keep": (HEAD_SWAP, 1), "direct5": (HEAD_BRANCH, 0),
                  "offer10": (HEAD_BRANCH, 1)}


def target_slot(action: str) -> tuple[int, int]:
    """(head, index) of an action string in the network's output heads."""
    if action in DECISION_INDEX:
        return DECISION_INDEX[action]
    if action == "pass":
        return HEAD_POLICY, NUM_POINTS
    if action.startswith("propose:"):
        return HEAD_PROPOSE, _core.str_to_point(action[8:])
    if action.startswith("select:"):
        return HEAD_SELECT, _core.str_to_point(action[7:])
    return HEAD_POLICY, _core.str_to_point(action)


@dataclass
class _Block:
    generation: int
    feats: np.ndarray   # uint8 [n, C, 225]
    legal: np.ndarray   # uint8 [n, 226]
    aux: np.ndarray     # uint8 [n, 5, 225]
    policy: np.ndarray  # float32 [n, 226]  point/pass targets (policy, propose, select heads)
    value: np.ndarray   # int8 [n]
    head: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    tgt2: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))  # swap / branch
    root_value: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    # Solver tactical labels (Phase 8 E5, rl.tactical.label_chunk); weight 0 = unlabelled.
    danger_t: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    danger_w: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    refute_t: np.ndarray = field(default_factory=lambda: np.zeros((0, 225), np.float16))
    refute_w: np.ndarray = field(default_factory=lambda: np.zeros((0, 225), np.float16))
    value_overridden: int = 0
    proven: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # value target is a solver proof


class ReplayBuffer:
    def __init__(self, spec: FeatureSpec, window: int = 4, capacity: int = 200_000):
        self.spec = spec
        self.cfg = spec.core()
        self.window = window
        self.capacity = capacity
        self.blocks: list[_Block] = []
        self.stats = {"games": 0, "positions": 0, "opening_targets": 0}
        # Value-loss weights (PROGRESS Phase 8 E5c): `balance` gives the four (colour to move) x (win / loss)
        # cells of the replay the same total weight (mean 1 over decisive positions, then clipped);
        # `proof` multiplies the weight of positions whose value target is a solver proof.
        self.value_weighting = {"balance": False, "proof": 1.0, "clip": (0.5, 2.0)}

    def __len__(self) -> int:
        return sum(len(b.value) for b in self.blocks)

    def ingest(self, chunk_path, generation: int) -> int:
        games = read_chunk(chunk_path)
        feats, legal, aux, pol, val, rv, heads, t2 = [], [], [], [], [], [], [], []
        for game in games:
            meta = game["meta"]
            g = _core.GameState(meta["profile"])
            targets = {p["i"]: p for p in game["positions"]}
            for i, a in enumerate(game["actions"] + [None]):
                if i in targets:
                    p = targets[i]
                    if g.actor != p["actor"]:
                        raise RecordError(f"{meta['game_id']}: actor mismatch at {i}")
                    if "phase" in p and g.phase != p["phase"]:
                        raise RecordError(f"{meta['game_id']}: phase mismatch at {i}")
                    mask = np.asarray(_core.legal_mask(g))
                    legal_actions = set(g.legal_actions())
                    vec = np.zeros(NUM_POLICY, np.float32)
                    vec2 = np.zeros(2, np.float32)
                    head = None
                    for act, prob in p["policy"].items():
                        if act not in legal_actions:
                            raise RecordError(f"{meta['game_id']}: illegal policy target {act} at {i}")
                        h, k = target_slot(act)
                        if head is not None and h != head:
                            raise RecordError(f"{meta['game_id']}: mixed heads in one target at {i}")
                        head = h
                        if h in (HEAD_SWAP, HEAD_BRANCH):
                            vec2[k] = prob
                        else:
                            if not mask[k]:
                                raise RecordError(f"{meta['game_id']}: target {act} outside legal mask at {i}")
                            vec[k] = prob
                    feats.append(np.asarray(_core.encode(g, self.cfg)).reshape(-1, NUM_POINTS))
                    legal.append(mask)
                    aux.append(np.asarray(_core.aux_targets(g)).reshape(NUM_AUX, NUM_POINTS))
                    pol.append(vec / max(vec.sum(), 1e-12))
                    t2.append(vec2 / max(vec2.sum(), 1e-12))
                    heads.append(head)
                    val.append(z_for(p["actor"], meta["winner"]))
                    rv.append(p["value"])
                    if head != HEAD_POLICY or g.phase != "NORMAL_RENJU":
                        self.stats["opening_targets"] += 1
                if a is None:
                    break
                if not g.apply(a):
                    raise RecordError(f"{meta['game_id']}: illegal action {a} at {i}")
            if g.result != meta["result"] or g.winner != meta["winner"]:
                raise RecordError(f"{meta['game_id']}: replayed result {g.result} != recorded {meta['result']}")
            if "final_hash" in meta and g.hash() != meta["final_hash"]:
                raise RecordError(f"{meta['game_id']}: replayed final position differs from the record")
            self.stats["games"] += 1
        if not val:
            return 0
        self.blocks.append(_Block(generation, np.stack(feats), np.stack(legal), np.stack(aux), np.stack(pol),
                                  np.asarray(val, np.int8), head=np.asarray(heads, np.int8), tgt2=np.stack(t2),
                                  root_value=np.asarray(rv, np.float32)))
        self.stats["positions"] += len(val)
        self._evict()
        return len(val)

    def attach_labels(self, generation: int, labels_path, override_value: bool) -> dict:
        """Attach a generation's solver labels (rl.tactical.label_chunk, ingest order). With
        override_value, proven positions take the proof (+1/-1 for the actor) as value target instead
        of the game result; UNKNOWN positions keep the game result."""
        blk = next((b for b in self.blocks if b.generation == generation), None)
        if blk is None:
            return {"attached": 0}
        z = np.load(labels_path)
        n = len(blk.value)
        if len(blk.danger_t) != n:
            blk.danger_t, blk.danger_w = np.zeros(n, np.float32), np.zeros(n, np.float32)
            blk.refute_t, blk.refute_w = np.zeros((n, NUM_POINTS), np.float16), np.zeros((n, NUM_POINTS), np.float16)
        idx = z["idx"]
        if len(idx) and idx.max() >= n:
            raise RecordError(f"labels for generation {generation} do not match the replay block")
        blk.danger_t[idx], blk.danger_w[idx] = z["danger_t"], z["danger_w"]
        blk.refute_t[idx], blk.refute_w[idx] = z["refute_t"], z["refute_w"]
        changed = 0
        if override_value:
            proven = z["proof"] != 0
            changed = int((blk.value[idx[proven]] != z["proof"][proven]).sum())
            blk.value[idx[proven]] = z["proof"][proven]
            blk.value_overridden = changed
            blk.proven = np.zeros(n, bool)
            blk.proven[idx[proven]] = True
        self._cache_key = None
        return {"attached": int(len(idx)), "proven": int((z["proof"] != 0).sum()), "value_changed": changed}

    def _evict(self) -> None:
        newest = max(b.generation for b in self.blocks)
        self.blocks = [b for b in self.blocks if b.generation > newest - self.window]
        while len(self) > self.capacity and len(self.blocks) > 1:
            self.blocks.pop(0)

    def _concat(self):
        if not hasattr(self, "_cache_key") or self._cache_key != (tuple(id(b) for b in self.blocks), repr(self.value_weighting)):
            for b in self.blocks:  # unlabelled blocks: zero targets with zero weight
                if len(b.danger_t) != len(b.value):
                    n = len(b.value)
                    b.danger_t, b.danger_w = np.zeros(n, np.float32), np.zeros(n, np.float32)
                    b.refute_t, b.refute_w = np.zeros((n, NUM_POINTS), np.float16), np.zeros((n, NUM_POINTS), np.float16)
                if len(b.proven) != len(b.value):
                    b.proven = np.zeros(len(b.value), bool)
            self._cat = {k: np.concatenate([getattr(b, k) for b in self.blocks])
                         for k in ("feats", "legal", "aux", "policy", "value", "head", "tgt2",
                                   "danger_t", "danger_w", "refute_t", "refute_w", "proven")}
            plane = list(self.cfg.plane_names).index("actor_is_black")
            self._cat["black"] = self._cat["feats"][:, plane].max(axis=1) > 0
            self._cat["value_w"] = self._value_weights(self._cat)
            self._cache_key = (tuple(id(b) for b in self.blocks), repr(self.value_weighting))
        return self._cat

    def _value_weights(self, d: dict) -> np.ndarray:
        vw = self.value_weighting
        w = np.ones(len(d["value"]), np.float32)
        if vw.get("balance"):
            n_dec = int((d["value"] != 0).sum())
            for c in (True, False):
                for r in (1, -1):
                    m = (d["black"] == c) & (d["value"] == r)
                    if m.any():
                        w[m] = n_dec / (4 * int(m.sum()))
            lo, hi = vw.get("clip", (0.5, 2.0))
            w = np.clip(w, lo, hi).astype(np.float32)
        if float(vw.get("proof", 1.0)) != 1.0:
            w[d["proven"]] *= float(vw["proof"])
        return w

    def value_cells(self) -> dict:
        """Replay value targets per (colour to move, result) cell: share of positions and mean weight."""
        d = self._concat()
        out = {}
        for c, cn in ((True, "black"), (False, "white")):
            for r, rn in ((1, "win"), (-1, "loss")):
                m = (d["black"] == c) & (d["value"] == r)
                out[f"{cn}_{rn}"] = {"share": round(float(m.mean()), 4),
                                     "weight": round(float(d["value_w"][m].mean()), 3) if m.any() else None}
        out["proven"] = int(d["proven"].sum())
        return out

    def sample(self, batch_size: int, seed: int, step: int, device, augment: bool = True) -> dict:
        data = self._concat()
        rng = np.random.default_rng([seed, step, 7])
        idx = rng.integers(0, len(data["value"]), size=batch_size)
        feats = data["feats"][idx]
        aux = data["aux"][idx]
        legal = data["legal"][idx].copy()
        policy = data["policy"][idx].copy()
        rt = data["refute_t"][idx].astype(np.float32)
        rw = data["refute_w"][idx].astype(np.float32)
        if augment:
            t = rng.integers(0, 8, size=batch_size)
            gi = GATHER[t]
            feats = np.take_along_axis(feats, gi[:, None, :], axis=2)
            aux = np.take_along_axis(aux, gi[:, None, :], axis=2)
            legal[:, :NUM_POINTS] = np.take_along_axis(legal[:, :NUM_POINTS], gi, axis=1)
            policy[:, :NUM_POINTS] = np.take_along_axis(policy[:, :NUM_POINTS], gi, axis=1)
            rt = np.take_along_axis(rt, gi, axis=1)
            rw = np.take_along_axis(rw, gi, axis=1)
        to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)  # noqa: E731
        b = len(idx)
        return {
            "x": to(feats.reshape(b, feats.shape[1], 15, 15), torch.float32),
            "legal": to(legal, torch.uint8),
            "policy": to(policy, torch.float32),
            "has_policy": torch.ones(b, dtype=torch.bool, device=device),
            "value": to(data["value"][idx], torch.int64),
            "has_value": torch.ones(b, dtype=torch.bool, device=device),
            "value_w": to(data["value_w"][idx], torch.float32),
            "aux": to(aux, torch.float32),
            "head": to(data["head"][idx], torch.int64),
            "tgt2": to(data["tgt2"][idx], torch.float32),
            "danger_t": to(data["danger_t"][idx], torch.float32),
            "danger_w": to(data["danger_w"][idx], torch.float32),
            "refute_t": to(rt, torch.float32),
            "refute_w": to(rw, torch.float32),
        }

    def head_counts(self) -> dict:
        h = self._concat()["head"]
        names = ["policy", "swap", "branch", "propose", "select"]
        return {names[k]: int((h == k).sum()) for k in range(5)}

    def value_balance(self) -> dict:
        v = self._concat()["value"]
        return {"win": float((v > 0).mean()), "draw": float((v == 0).mean()), "loss": float((v < 0).mean())}
