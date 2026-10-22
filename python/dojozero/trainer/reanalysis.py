"""Solver-supervised tactical reanalysis training data (PROGRESS Phase 8 E4, Case B).

    python -m dojozero.trainer.reanalysis build --root data/reanalysis_v1

Input: positions.jsonl.gz + labels.jsonl.gz from dojozero.tools.reanalysis_data. Every position is
replayed from its move list (the encoder uses the last move, so a bare board would not do).

Targets (only solver-proven labels are strong; UNKNOWN is never a target):
  refute  per labelled move: 1 = PROVEN_UNSAFE (weight 1), 0 = PROVEN_WINNING (weight 1),
          0 = UNREFUTED_EXHAUSTED (weight HEUR_W: threat-space search exhausted without a proof,
          NOT a safety proof), UNKNOWN_BUDGET / unlabelled -> weight 0
  danger  1 = position proven LOST for the side to move (weight 1), 0 = proven WIN (weight 1),
          0 = UNKNOWN with at least one UNREFUTED_EXHAUSTED candidate (weight HEUR_W), else weight 0
  value   +1 / -1 on proven WIN / LOSS positions (exact), otherwise no value target
  policy  proven WIN: the winning move(s); otherwise, when some labelled move is PROVEN_UNSAFE and
          mass remains: the baseline policy with the PROVEN_UNSAFE moves set to 0 and renormalised
          (unlabelled moves keep their baseline mass: nothing is assumed about them); proven LOSS: none
D4 augmentation is applied per batch (after the split), with the C++ transform table.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..features import GATHER, NUM_AUX, NUM_POINTS, NUM_POLICY

PROFILE = "RENJU_GOMOCUP"
CODE = {"PROVEN_UNSAFE": 0, "PROVEN_WINNING": 1, "UNREFUTED_EXHAUSTED": 2, "UNKNOWN_BUDGET": 3}
HEUR_W = 0.3
SPLITS = {"train": 0, "val": 1, "test": 2}
SIDES = {"BLACK": 0, "WHITE": 1}


def _state(moves: list[str]):
    g = _core.GameState(PROFILE)
    for m in moves:
        if not g.apply(m):
            raise ValueError(f"illegal move {m}")
    return g


@torch.no_grad()
def baseline_outputs(model, feats: np.ndarray, legal: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Baseline policy (softmax over legal actions) and value (P(win) - P(loss)), CPU fp32."""
    pol, val = [], []
    for s in range(0, len(feats), batch):
        x = torch.from_numpy(feats[s:s + batch].reshape(-1, feats.shape[1], 15, 15).astype(np.float32))
        out = model(x)
        lg = out["policy"].masked_fill(torch.from_numpy(legal[s:s + batch] == 0), float("-inf"))
        pol.append(torch.softmax(lg, dim=1).numpy())
        w = torch.softmax(out["wdl"], dim=1).numpy()
        val.append(w[:, 0] - w[:, 2])
    return np.concatenate(pol).astype(np.float32), np.concatenate(val).astype(np.float32)


def make_targets(code: np.ndarray, res: np.ndarray, win_move: np.ndarray, legal: np.ndarray,
                 p0: np.ndarray) -> dict:
    """code [N, 225] (CODE values, -1 unlabelled), res [N] (+1 WIN / -1 LOSS / 0 UNKNOWN for the
    side to move), win_move [N] (-1 none), legal [N, 226], p0 [N, 226] baseline policy."""
    n = len(res)
    unsafe, winning, unref = code == CODE["PROVEN_UNSAFE"], code == CODE["PROVEN_WINNING"], code == CODE["UNREFUTED_EXHAUSTED"]
    refute_t = unsafe.astype(np.float32)
    refute_w = np.where(unsafe | winning, 1.0, np.where(unref, HEUR_W, 0.0)).astype(np.float32)
    danger_t = (res == -1).astype(np.float32)
    danger_w = np.where(res != 0, 1.0, np.where(unref.any(axis=1), HEUR_W, 0.0)).astype(np.float32)
    policy = np.zeros((n, NUM_POLICY), np.float32)
    has_policy = np.zeros(n, bool)
    for i in range(n):
        if res[i] == 1:
            tgt = np.zeros(NUM_POLICY, np.float32)
            if win_move[i] >= 0:
                tgt[win_move[i]] = 1
            tgt[:NUM_POINTS][winning[i]] = 1
            tgt *= legal[i]
            if tgt.sum() > 0:
                policy[i], has_policy[i] = tgt / tgt.sum(), True
        elif res[i] == 0 and unsafe[i].any():
            tgt = p0[i].astype(np.float32).copy()
            tgt[:NUM_POINTS][unsafe[i]] = 0
            if tgt.sum() > 1e-3:
                policy[i], has_policy[i] = tgt / tgt.sum(), True
    return {"refute_t": refute_t, "refute_w": refute_w, "danger_t": danger_t, "danger_w": danger_w,
            "value": res.astype(np.int64), "has_value": res != 0, "policy": policy, "has_policy": has_policy}


def build(root: Path, ckpt: str) -> dict:
    from ..model.load import load_model, model_sha
    model, spec, _ = load_model(ckpt, "cpu")
    cfg = spec.core()
    pos = {json.loads(l)["id"]: json.loads(l) for l in gzip.open(root / "positions.jsonl.gz", "rt", encoding="utf-8")}
    labels = [json.loads(l) for l in gzip.open(root / "labels.jsonl.gz", "rt", encoding="utf-8")]
    n = len(labels)
    feats = np.zeros((n, spec.num_planes, NUM_POINTS), np.uint8)
    legal = np.zeros((n, NUM_POLICY), np.uint8)
    aux = np.zeros((n, NUM_AUX, NUM_POINTS), np.uint8)
    code = np.full((n, NUM_POINTS), -1, np.int8)
    meta = {k: np.zeros(n, np.int8) for k in ("split", "side", "dojo_game", "dojo_to_move", "blunder_game", "pos_result")}
    played = np.full(n, -1, np.int16)
    win_move = np.full(n, -1, np.int16)
    pos_nodes = np.zeros(n, np.int32)
    ids, groups = [], []
    for i, lab in enumerate(labels):
        p = pos[lab["id"]]
        g = _state(p["moves"])
        feats[i] = np.asarray(_core.encode(g, cfg)).reshape(-1, NUM_POINTS)
        legal[i] = np.asarray(_core.legal_mask(g))
        aux[i] = np.asarray(_core.aux_targets(g)).reshape(NUM_AUX, NUM_POINTS)
        for m in lab["moves"]:
            if m["code"] in CODE:
                code[i, _core.str_to_point(m["m"])] = CODE[m["code"]]
        meta["split"][i] = SPLITS[p["split"]]
        meta["side"][i] = SIDES[p["side"]]
        meta["dojo_game"][i] = p["source"] == "dojo_game"
        meta["dojo_to_move"][i] = bool(p["dojo_to_move"])
        meta["blunder_game"][i] = bool(p["blunder_game"])
        meta["pos_result"][i] = {"WIN": 1, "LOSS": -1}.get(lab["pos"]["result"], 0)
        pos_nodes[i] = lab["pos"]["nodes"]
        played[i] = _core.str_to_point(p["played"])
        if lab["pos"]["result"] == "WIN" and lab["pos"]["move"]:
            win_move[i] = _core.str_to_point(lab["pos"]["move"])
        ids.append(p["id"])
        groups.append(p["group"])
    p0, v0 = baseline_outputs(model, feats, legal)
    res = meta["pos_result"]
    t = make_targets(code, res, win_move, legal, p0)
    unsafe, unref = code == 0, code == 2
    refute_t, refute_w, danger_t, danger_w = t["refute_t"], t["refute_w"], t["danger_t"], t["danger_w"]
    value, has_value, policy, has_policy = t["value"], t["has_value"], t["policy"], t["has_policy"]
    arrays = dict(feats=feats, legal=legal, aux=aux, code=code, played=played, win_move=win_move, pos_nodes=pos_nodes,
                  p0=p0.astype(np.float16), v0=v0, refute_t=refute_t, refute_w=refute_w, danger_t=danger_t,
                  danger_w=danger_w, value=value, has_value=has_value, policy=policy, has_policy=has_policy,
                  ids=np.asarray(ids), groups=np.asarray(groups), **meta)
    out = root / "arrays.npz"
    np.savez_compressed(out, **arrays)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    split_counts = {s: int((meta["split"] == k).sum()) for s, k in SPLITS.items()}
    stats = {
        "file": out.name, "sha256": sha, "positions": n, "baseline_ckpt": ckpt, "baseline_model_sha": model_sha(model),
        "feature_hash": spec.hash(), "heuristic_weight": HEUR_W, "by_split": split_counts,
        "labelled_moves": {k: int((code == v).sum()) for k, v in CODE.items()},
        "positions_with_unsafe": int(unsafe.any(axis=1).sum()),
        "has_policy": int(has_policy.sum()), "has_value": int(has_value.sum()),
        "danger_pos": int((danger_t * (danger_w > 0)).sum()), "danger_labelled": int((danger_w > 0).sum()),
        "per_split": {s: {"positions": split_counts[s],
                          "pos_result": {r: int(((meta["split"] == k) & (res == v)).sum()) for r, v in (("WIN", 1), ("LOSS", -1), ("UNKNOWN", 0))},
                          "unsafe_moves": int(unsafe[meta["split"] == k].sum()),
                          "unrefuted_moves": int(unref[meta["split"] == k].sum()),
                          "black": int(((meta["split"] == k) & (meta["side"] == 0)).sum()),
                          "white": int(((meta["split"] == k) & (meta["side"] == 1)).sum())} for s, k in SPLITS.items()},
    }
    return stats


class ReanalysisData:
    """Arrays from build(); batches as dicts compatible with trainer.losses.total_loss."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=False)
        self.a = {k: z[k] for k in z.files}
        self.n = len(self.a["split"])

    def indices(self, split: str, where: np.ndarray | None = None) -> np.ndarray:
        m = self.a["split"] == SPLITS[split]
        if where is not None:
            m &= where
        return np.flatnonzero(m)

    def batch(self, idx: np.ndarray, transforms: np.ndarray | None, device) -> dict:
        a = self.a
        feats = a["feats"][idx]
        aux = a["aux"][idx]
        legal = a["legal"][idx].copy()
        policy = a["policy"][idx].copy()
        rt, rw = a["refute_t"][idx], a["refute_w"][idx]
        if transforms is not None:
            g = GATHER[transforms]
            feats = np.take_along_axis(feats, g[:, None, :], axis=2)
            aux = np.take_along_axis(aux, g[:, None, :], axis=2)
            legal[:, :NUM_POINTS] = np.take_along_axis(legal[:, :NUM_POINTS], g, axis=1)
            policy[:, :NUM_POINTS] = np.take_along_axis(policy[:, :NUM_POINTS], g, axis=1)
            rt = np.take_along_axis(rt, g, axis=1)
            rw = np.take_along_axis(rw, g, axis=1)
        b = len(idx)
        to = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=dt)  # noqa: E731
        return {
            "x": to(feats.reshape(b, feats.shape[1], 15, 15), torch.float32),
            "legal": to(legal, torch.uint8),
            "policy": to(policy, torch.float32),
            "has_policy": to(a["has_policy"][idx], torch.bool),
            "value": to(a["value"][idx], torch.int64),
            "has_value": to(a["has_value"][idx], torch.bool),
            "aux": to(aux, torch.float32),
            "danger_t": to(a["danger_t"][idx], torch.float32),
            "danger_w": to(a["danger_w"][idx], torch.float32),
            "refute_t": to(rt, torch.float32),
            "refute_w": to(rw, torch.float32),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--root", default="data/reanalysis_v1")
    ap.add_argument("--ckpt", default="runs/taraguchi_v1/models/gen_001.pt")
    args = ap.parse_args(argv)
    root = Path(args.root)
    stats = build(root, args.ckpt)
    man_path = root / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    man["arrays"] = stats
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
