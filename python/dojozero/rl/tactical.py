"""Tactical value auxiliary for the RL loop (PROGRESS Phase 8 E5; design: Phase 8 E4 / ADR-11).

label_chunk  solver labels for a share of a self-play chunk's NORMAL_RENJU target positions, in the
             replay buffer's ingest order. Candidates = the position's top-k self-play visit targets
             + the move played (no network call). Same budgets and codes as the E4 reanalysis
             (tools.reanalysis_data): PROVEN_UNSAFE / PROVEN_WINNING exact, UNREFUTED_EXHAUSTED a
             heuristic negative, UNKNOWN_BUDGET no label. Position: WIN / LOSS proven for the actor
             (side to move) or UNKNOWN. Saved next to the chunk as gen_NNN.labels.npz.
value_calibration  network-only value calibration on the E4 held-out test split (proven positions),
             used as the promotion gate: a candidate whose ECE is worse than the best's by more than
             a margin is not promoted, whatever the SPRT says.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..features import NUM_POINTS, NUM_POLICY
from ..replay.buffer import read_chunk
from ..tools.reanalysis_data import SOLVER, _analyze_move
from ..trainer.reanalysis import CODE, SPLITS, make_targets


def label_chunk(chunk_path: str | Path, generation: int, fraction: float, seed: int, top_k: int = 8,
                threads: int = 7) -> Path:
    """Label a deterministic `fraction` of the chunk's NORMAL_RENJU policy-target positions."""
    chunk_path = Path(chunk_path)
    out = chunk_path.with_suffix("").with_suffix(".labels.npz")
    games = read_chunk(chunk_path)
    jobs = []  # (ingest index, state, candidates)
    k = 0
    for game in games:
        g = _core.GameState(game["meta"]["profile"])
        targets = {p["i"]: p for p in game["positions"]}
        for i, a in enumerate(game["actions"] + [None]):
            if i in targets:
                p = targets[i]
                if g.phase == "NORMAL_RENJU" and not g.is_terminal:
                    rng = np.random.default_rng([seed, generation, k])
                    if rng.random() < fraction:
                        moves = [m for m, _ in sorted(p["policy"].items(), key=lambda kv: -kv[1])
                                 if ":" not in m and m not in ("pass", "swap", "keep")][:top_k]
                        if a is not None and ":" not in a and a not in moves and a not in ("pass", "swap", "keep"):
                            moves.append(a)
                        jobs.append((k, g.copy(), moves))
                k += 1
            if a is None:
                break
            assert g.apply(a)

    def work(job):
        idx, g, moves = job
        d = _core.analyze_tactics(g, SOLVER["vcf_nodes"], SOLVER["vct_nodes"], SOLVER["loss_nodes"], 0.0)
        code = np.full(NUM_POINTS, -1, np.int8)
        for m in moves:
            c = _analyze_move(g, m)["code"]
            if c in CODE:
                code[_core.str_to_point(m)] = CODE[c]
        res = {"WIN": 1, "LOSS": -1}.get(d["result"], 0)
        return idx, code, res

    with ThreadPoolExecutor(max_workers=threads) as ex:
        rows = list(ex.map(work, jobs))
    idx = np.array([r[0] for r in rows], np.int32)
    code = np.stack([r[1] for r in rows]) if rows else np.zeros((0, NUM_POINTS), np.int8)
    res = np.array([r[2] for r in rows], np.int8)
    t = make_targets(code, res, np.full(len(rows), -1), np.ones((len(rows), NUM_POLICY), np.uint8),
                     np.zeros((len(rows), NUM_POLICY), np.float32))
    np.savez_compressed(out, idx=idx, code=code, proof=res, refute_t=t["refute_t"].astype(np.float16),
                        refute_w=t["refute_w"].astype(np.float16), danger_t=t["danger_t"], danger_w=t["danger_w"],
                        meta=np.asarray(json.dumps({"generation": generation, "fraction": fraction, "seed": seed,
                                                    "top_k": top_k, "solver": SOLVER, "positions": int(k),
                                                    "labelled": len(rows)})))
    return out


def label_stats(path: str | Path) -> dict:
    z = np.load(path)
    code = z["code"]
    return {"labelled": int(len(z["idx"])), "proof_win": int((z["proof"] == 1).sum()),
            "proof_loss": int((z["proof"] == -1).sum()),
            "moves": {name: int((code == v).sum()) for name, v in CODE.items()},
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


@torch.no_grad()
def value_calibration(model, arrays_path: str | Path, device, split: str = "test") -> dict:
    """ECE (5 bins) / Brier / mean values on proven positions of the E4 arrays (network only)."""
    z = np.load(arrays_path, allow_pickle=False)
    sel = np.flatnonzero((z["split"] == SPLITS[split]) & (z["pos_result"] != 0))
    feats, res, side = z["feats"][sel], z["pos_result"][sel], z["side"][sel]
    model.eval()
    v = []
    for s in range(0, len(sel), 256):
        x = torch.from_numpy(feats[s:s + 256].reshape(-1, feats.shape[1], 15, 15).astype(np.float32)).to(device)
        w = torch.softmax(model(x)["wdl"].float(), dim=1).cpu().numpy()
        v.append(w[:, 0] - w[:, 2])
    v = np.concatenate(v)
    pw, y = (v + 1) / 2, (res == 1).astype(float)
    bins = np.clip((pw * 5).astype(int), 0, 4)
    ece = float(sum(abs(pw[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(5) if (bins == b).any()))
    mean = lambda x: float(x.mean()) if len(x) else None  # noqa: E731
    out = {"n": int(len(sel)), "ece5": ece, "brier": float(np.mean((pw - y) ** 2)),
           "proven_loss_value": mean(v[res == -1]), "proven_win_value": mean(v[res == 1]),
           "proven_loss_ge0": mean((v[res == -1] >= 0).astype(float))}
    for name, c in (("black", 0), ("white", 1)):
        out[f"{name}_proven_loss_value"] = mean(v[(side == c) & (res == -1)])
    return out
