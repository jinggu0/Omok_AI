"""E5j: colour bias of gate v2 split by the arm that played the game (selection effect check).

    PYTHONPATH=python python benchmarks/reanalysis/e5j/measure_by_arm.py <gate.npz> <arrays.npz> <out.json>

Same value and game-level bootstrap as measure_gate.py, on the pre-registered checkpoints.
"""

import json
import sys

import numpy as np

from dojozero.model.load import load_model
from dojozero.trainer.train import pick_device
from measure_gate import CI, PATHS, bias_ci, values


def main():
    gate, arrays, out_path = sys.argv[1:4]
    device = pick_device("auto")
    z = np.load(gate, allow_pickle=False)
    feats, res, side = z["feats"], z["pos_result"].astype(int), z["side"]
    arm = np.load(arrays, allow_pickle=False)["arm"][z["source_index"]]
    games = np.array([i.split("@")[0] for i in z["ids"]])
    report = {"gate": gate, "cells": {}, "rows": {}}
    for a in np.unique(arm):
        report["cells"][str(a)] = {c: {"win": int(((arm == a) & (side == i) & (res == 1)).sum()),
                                       "loss": int(((arm == a) & (side == i) & (res == -1)).sum())}
                                   for i, c in enumerate(("black", "white"))}
    for name in CI:
        m, _, _ = load_model(PATHS[name], device)
        v = values(m, feats, device)
        report["rows"][name] = {}
        for a in np.unique(arm):
            row = {}
            for i, c in enumerate(("black", "white")):
                mask = (arm == a) & (side == i)
                win, loss = v[mask & (res == 1)].mean(), v[mask & (res == -1)].mean()
                row[c] = {"win": float(win), "loss": float(loss), "bias": float((win + loss) / 2)} | bias_ci(v, res, games, mask)
            report["rows"][name][str(a)] = row
            print(name, a[:12], " ".join(f"{c} {row[c]['bias']:+.3f} [{row[c]['ci95'][0]:+.3f}, {row[c]['ci95'][1]:+.3f}]" for c in row), flush=True)
        del m
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["cells"]))


if __name__ == "__main__":
    main()
