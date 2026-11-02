"""E5k: T'' generations 53-57 and gen_045 on gate v1 and v2 (network-only; v1 is a diagnostic that decides nothing).

    PYTHONPATH=python python benchmarks/reanalysis/e5k/measure_generations.py <out.json>
"""

import json
import sys

from dojozero.model.load import load_model
from dojozero.rl.tactical import balanced_gate_pass, value_balance
from dojozero.trainer.train import pick_device

GATES = {"v1": "data/reanalysis_v1/gate_balanced_v1.npz", "v2": "data/gate_e5_v1/gate_balanced_v2.npz"}
GENS = (45, 53, 54, 55, 56, 57)


def main():
    device = pick_device("auto")
    report = {"gates": GATES, "rows": {k: {} for k in GATES}}
    for g in GENS:
        m, _, _ = load_model(f"runs/e5e_rl_T/models/gen_{g:03d}.pt", device)
        for k, path in GATES.items():
            d = value_balance(m, path, device)
            report["rows"][k][f"g{g}"] = {c: {x: d[c][x] for x in ("win", "loss", "bias", "separation", "auroc")} for c in ("black", "white")} | {"auroc": d["auroc"]}
            r = report["rows"][k][f"g{g}"]
            print(f"{k} g{g} B {r['black']['bias']:+.3f} W {r['white']['bias']:+.3f} sep {r['black']['separation']:.3f}/{r['white']['separation']:.3f} auroc {r['auroc']:.3f}", flush=True)
        del m
    report["g55_vs_g45"] = {k: balanced_gate_pass(report["rows"][k]["g55"], report["rows"][k]["g45"]) for k in GATES}
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["g55_vs_g45"]))


if __name__ == "__main__":
    main()
