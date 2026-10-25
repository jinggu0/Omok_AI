"""Tactical value auxiliary for the RL loop (PROGRESS Phase 8 E5; design: Phase 8 E4 / ADR-11).

label_chunk  solver labels for a share of a self-play chunk's NORMAL_RENJU target positions, in the
             replay buffer's ingest order. Candidates = the position's top-k self-play visit targets
             + the move played (no network call). Same budgets and codes as the E4 reanalysis
             (tools.reanalysis_data): PROVEN_UNSAFE / PROVEN_WINNING exact, UNREFUTED_EXHAUSTED a
             heuristic negative, UNKNOWN_BUDGET no label. Position: WIN / LOSS proven for the actor
             (side to move) or UNKNOWN. Saved next to the chunk as gen_NNN.labels.npz.
value_calibration  network-only value calibration on the E4 held-out test split (proven positions),
             used as the promotion gate: a candidate whose ECE is worse than the best's by more than
             a margin is not promoted, whatever the SPRT says (E5; biased towards pessimistic models
             because the test split is 86% proven losses).
value_balance / balanced_gate_pass  the E5b gate: separation, |bias| bound and AUROC, black and white
             separately, on a win / loss x colour balanced set from the E4 val split
             (build_balanced_gate_set). The E4 test split stays a final evaluation set only.
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


def build_balanced_gate_set(arrays_path: str | Path, out_path: str | Path, split: str = "val", seed: int = 0) -> dict:
    """Stratified gate set: the same number of proven positions in each (colour to move) x (win / loss)
    cell of one E4 split (the smallest cell's size), sampled without replacement with a fixed seed."""
    z = np.load(arrays_path, allow_pickle=False)
    rng = np.random.default_rng(seed)
    cells = {(c, r): np.flatnonzero((z["split"] == SPLITS[split]) & (z["side"] == c) & (z["pos_result"] == r))
             for c in (0, 1) for r in (1, -1)}
    n = min(len(ix) for ix in cells.values())
    sel = np.sort(np.concatenate([rng.choice(ix, n, replace=False) for ix in cells.values()]))
    out_path = Path(out_path)
    np.savez_compressed(out_path, feats=z["feats"][sel], pos_result=z["pos_result"][sel], side=z["side"][sel],
                        ids=z["ids"][sel], split=z["split"][sel], source_index=sel.astype(np.int32))
    return {"source": str(arrays_path).replace("\\", "/"), "split": split, "seed": seed, "per_cell": int(n),
            "cells_available": {f"{'black' if c == 0 else 'white'}_{'win' if r == 1 else 'loss'}": int(len(ix))
                                for (c, r), ix in cells.items()},
            "n": int(len(sel)), "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()}


def _balance(v: np.ndarray, res: np.ndarray) -> dict:
    from ..tools.reanalysis_eval import auroc
    w, l = v[res == 1], v[res == -1]
    if not len(w) or not len(l):
        return {"n_win": int(len(w)), "n_loss": int(len(l))}
    return {"n_win": int(len(w)), "n_loss": int(len(l)), "win": float(w.mean()), "loss": float(l.mean()),
            "bias": float((w.mean() + l.mean()) / 2), "separation": float((w.mean() - l.mean()) / 2),
            "auroc": float(auroc(v, (res == 1).astype(int)))}


@torch.no_grad()
def value_balance(model, gate_path: str | Path, device) -> dict:
    """Network-only value on a balanced gate set: win / loss means, bias (their midpoint, 0 = symmetric),
    separation (half the gap) and AUROC, overall and for black / white to move; ECE (5 bins) recorded too."""
    z = np.load(gate_path, allow_pickle=False)
    feats, res, side = z["feats"], z["pos_result"].astype(int), z["side"]
    model.eval()
    v = []
    for s in range(0, len(res), 256):
        x = torch.from_numpy(feats[s:s + 256].reshape(-1, feats.shape[1], 15, 15).astype(np.float32)).to(device)
        w = torch.softmax(model(x)["wdl"].float(), dim=1).cpu().numpy()
        v.append(w[:, 0] - w[:, 2])
    v = np.concatenate(v)
    pw, y = (v + 1) / 2, (res == 1).astype(float)
    bins = np.clip((pw * 5).astype(int), 0, 4)
    ece = float(sum(abs(pw[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(5) if (bins == b).any()))
    return {"n": int(len(res)), "ece5": ece, **_balance(v, res),
            "black": _balance(v[side == 0], res[side == 0]), "white": _balance(v[side == 1], res[side == 1])}


def balanced_gate_pass(cand: dict, best: dict, separation_margin: float = 0.03, bias_max: float = 0.10,
                       auroc_margin: float = 0.01) -> tuple[bool, list[str]]:
    """Balanced value gate (PROGRESS Phase 8 E5b): for black and for white, the candidate's separation may
    be at most `separation_margin` below the best's and |bias| at most `bias_max`; overall AUROC at most
    `auroc_margin` below the best's. Returns (pass, failed conditions)."""
    fails = []
    for c in ("black", "white"):
        if "separation" not in cand[c] or "separation" not in best[c]:
            fails.append(f"{c}: a win or loss cell is empty")
            continue
        if cand[c]["separation"] < best[c]["separation"] - separation_margin:
            fails.append(f"{c} separation {cand[c]['separation']:.3f} < {best[c]['separation']:.3f} - {separation_margin}")
        if abs(cand[c]["bias"]) > bias_max:
            fails.append(f"{c} |bias| {abs(cand[c]['bias']):.3f} > {bias_max}")
    if "auroc" not in cand or "auroc" not in best:
        fails.append("auroc: a win or loss cell is empty")
    elif cand["auroc"] < best["auroc"] - auroc_margin:
        fails.append(f"auroc {cand['auroc']:.3f} < {best['auroc']:.3f} - {auroc_margin}")
    return not fails, fails
