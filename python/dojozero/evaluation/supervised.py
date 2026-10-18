"""Holdout evaluation for supervised tactical pretraining (TRAINING.md §2.2).

Training loss is not a strength measure (EVALUATION.md §1); these are diagnostics: policy top-1
inside the solver's target set per label source, proven-value accuracy, and aux precision/recall,
each next to a trivial baseline so "the number went up" can be judged.
"""

from __future__ import annotations

import numpy as np
import torch

from ..features import AUX_NAMES, NUM_POINTS
from ..trainer.losses import LossWeights, total_loss

REASONS = ["IMMEDIATE_WIN", "NO_DEFENSE", "FORCED_BLOCK", "VCF", "VCT", "PROVEN_LOSS", "NONE"]


@torch.no_grad()
def evaluate(model, ds, indices: np.ndarray, device, weights: LossWeights, batch_size: int = 1024) -> dict:
    model.eval()
    sums = {"loss": 0.0, "policy": 0.0, "value": 0.0, "aux": 0.0}
    n_batches = 0
    pol_hit: dict[str, list[float]] = {}
    pol_base: dict[str, list[float]] = {}
    val_hit, val_n = 0, 0
    tp = np.zeros(len(AUX_NAMES))
    fp = np.zeros(len(AUX_NAMES))
    fn = np.zeros(len(AUX_NAMES))
    for s in range(0, len(indices), batch_size):
        idx = indices[s : s + batch_size]
        b = ds.batch(idx, None, device)
        out = {k: v.float().cpu() for k, v in model(b["x"]).items()}
        b = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()}  # metrics on CPU
        _, parts = total_loss(out, b, weights)
        for k in sums:
            sums[k] += float(parts[k])
        n_batches += 1
        legal = b["legal"] > 0
        logits = out["policy"].masked_fill(~legal, float("-inf"))
        pred = logits.argmax(dim=1)
        hit = b["policy"].gather(1, pred[:, None]).squeeze(1) > 0
        base = (b["policy"] > 0).sum(1).float() / legal.sum(1).clamp_min(1).float()
        for i, r in enumerate(b["reason"]):
            if not bool(b["has_policy"][i]):
                continue
            name = REASONS[int(r)]
            pol_hit.setdefault(name, []).append(float(hit[i]))
            pol_base.setdefault(name, []).append(float(base[i]))
        hv = b["has_value"]
        if bool(hv.any()):
            wl = out["wdl"][hv][:, [0, 2]].argmax(dim=1)  # 0 = win, 1 = loss
            val_hit += int(((wl == 0) == (b["value"][hv] > 0)).sum())
            val_n += int(hv.sum())
        pa = (out["aux"] > 0).cpu().numpy()
        ta = b["aux"].cpu().numpy() > 0.5
        tp += (pa & ta).sum(axis=(0, 2))
        fp += (pa & ~ta).sum(axis=(0, 2))
        fn += (~pa & ta).sum(axis=(0, 2))
    model.train()
    res = {f"holdout_{k}": v / max(1, n_batches) for k, v in sums.items()}
    for name, h in pol_hit.items():
        res[f"policy_top1/{name}"] = float(np.mean(h))
        res[f"policy_random_baseline/{name}"] = float(np.mean(pol_base[name]))
        res[f"policy_count/{name}"] = len(h)
    if val_n:
        res["value_accuracy"] = val_hit / val_n
        res["value_count"] = val_n
    for j, name in enumerate(AUX_NAMES):
        res[f"aux_precision/{name}"] = float(tp[j] / max(1.0, tp[j] + fp[j]))
        res[f"aux_recall/{name}"] = float(tp[j] / max(1.0, tp[j] + fn[j]))
    _ = NUM_POINTS
    return res
