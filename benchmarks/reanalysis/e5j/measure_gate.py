"""E5j: E4 candidate + T'' generations on one gate set, plus the promotion counterfactual.

    PYTHONPATH=python python benchmarks/reanalysis/e5j/measure_gate.py <gate.npz> <out.json>

Network-only value; game-level bootstrap CI of the colour bias for the pre-registered checkpoints.
The same by the arm that played the game is in measure_by_arm.py.
"""

import json
import sys

import numpy as np
import torch

from dojozero.model.load import load_model
from dojozero.rl.tactical import balanced_gate_pass, value_balance
from dojozero.trainer.train import pick_device

GENS = 52
PROMOTIONS = [(2, "E4"), (21, "g2"), (23, "g21"), (45, "g23")]
CI = ("E4", "g2", "g45", "g52")
PATHS = {"E4": "runs/e4_reanalysis_v1/checkpoints/step_1500.pt"} | {f"g{g}": f"runs/e5e_rl_T/models/gen_{g:03d}.pt" for g in range(1, GENS + 1)}


def values(model, feats, device):
    out = []
    with torch.no_grad():
        for s in range(0, len(feats), 512):
            x = torch.from_numpy(feats[s:s + 512].reshape(-1, feats.shape[1], 15, 15).astype(np.float32)).to(device)
            w = torch.softmax(model(x)["wdl"].float(), dim=1).cpu().numpy()
            out.append(w[:, 0] - w[:, 2])
    return np.concatenate(out)


def bias_ci(v, res, games, mask, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(games[mask])
    ix = {g: np.flatnonzero(mask & (games == g)) for g in uniq}
    out = []
    for _ in range(n):
        s = np.concatenate([ix[g] for g in rng.choice(uniq, len(uniq))])
        w, l = v[s][res[s] == 1], v[s][res[s] == -1]
        if len(w) and len(l):
            out.append((w.mean() + l.mean()) / 2)
    return {"games": int(len(uniq)), "ci95": [float(x) for x in np.percentile(out, [2.5, 97.5])]}


def main():
    gate, out_path = sys.argv[1], sys.argv[2]
    device = pick_device("auto")
    z = np.load(gate, allow_pickle=False)
    feats, res, side = z["feats"], z["pos_result"].astype(int), z["side"]
    games = np.array([i.split("@")[0] for i in z["ids"]])
    report = {"gate": gate, "n": int(len(res)), "rows": {}, "ci": {}}
    for name, path in PATHS.items():
        m, _, _ = load_model(path, device)
        d = value_balance(m, gate, device)
        report["rows"][name] = {c: {k: d[c][k] for k in ("win", "loss", "bias", "separation", "auroc")} for c in ("black", "white")} | {"auroc": d["auroc"]}
        if name in CI:
            v = values(m, feats, device)
            report["ci"][name] = {c: bias_ci(v, res, games, side == i) for i, c in enumerate(("black", "white"))}
        r = report["rows"][name]
        print(f"{name:4s} B {r['black']['bias']:+.3f} W {r['white']['bias']:+.3f} sep {r['black']['separation']:.3f}/{r['white']['separation']:.3f} auroc {r['auroc']:.3f}", flush=True)
        del m
    report["promotions"] = []
    for g, prev in PROMOTIONS:
        ok, fails = balanced_gate_pass(report["rows"][f"g{g}"], report["rows"][prev])
        report["promotions"].append({"generation": g, "against": prev, "pass": ok, "fails": fails})
    report["bias_over_threshold"] = {c: [k for k, r in report["rows"].items() if abs(r[c]["bias"]) > 0.10] for c in ("black", "white")}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["promotions"]), json.dumps(report["ci"]))
    print(json.dumps(report["bias_over_threshold"]))


if __name__ == "__main__":
    main()
