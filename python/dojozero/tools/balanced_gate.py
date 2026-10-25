"""Balanced value gate (PROGRESS Phase 8 E5b).

    python -m dojozero.tools.balanced_gate build --out data/reanalysis_v1/gate_balanced_v1.npz \
        --manifest benchmarks/reanalysis/e5b/gate_balanced_v1.json
    python -m dojozero.tools.balanced_gate replay --gate data/reanalysis_v1/gate_balanced_v1.npz \
        --out benchmarks/reanalysis/e5b/gate_counterfactual_e5.json

build   win / loss x colour balanced gate set from the E4 val split (rl.tactical.build_balanced_gate_set).
replay  counterfactual: the E5 generations of both arms re-judged with the balanced gate (recorded SPRT
        decisions, best = the last model that would have passed). Approximate: the E5 SPRT was played
        against the E5 best, and a different promotion history changes later self-play.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..rl import tactical

REPO = Path(__file__).resolve().parents[3]
INIT = "runs/e4_reanalysis_v1/checkpoints/step_1500.pt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arrays", default="data/reanalysis_v1/arrays.npz")
    b.add_argument("--split", default="val")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--manifest", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--gate", required=True)
    r.add_argument("--device", default="xpu")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        m = tactical.build_balanced_gate_set(REPO / args.arrays, REPO / args.out, args.split, args.seed)
        m["source"], m["out"] = args.arrays, args.out
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(json.dumps(m))
        return 0
    from ..model.load import load_model
    cache: dict = {}

    def metrics(ck: str) -> dict:
        if ck not in cache:
            model, _, _ = load_model(str(REPO / ck), args.device)
            cache[ck] = tactical.value_balance(model, REPO / args.gate, args.device)
            del model
        return cache[ck]

    report: dict = {"gate": args.gate, "init": {"checkpoint": INIT, **metrics(INIT)}, "arms": {}}
    for arm in ("T", "C"):
        reg = json.loads((REPO / f"benchmarks/reanalysis/e5/registry_{arm}.json").read_text(encoding="utf-8"))
        best, rows = INIT, []
        for g in reg["generations"]:
            cand = metrics(g["candidate"])
            ok, fails = tactical.balanced_gate_pass(cand, metrics(best))
            h1 = g["eval"]["decision"] == "H1"
            rows.append({"generation": g["generation"], "sprt": g["eval"]["decision"], "e5_promoted": g["promoted"],
                         "balanced_pass": ok, "fails": fails, "would_promote": h1 and ok, "best_before": best,
                         "candidate": {k: cand[k] for k in ("ece5", "auroc", "bias", "separation")} |
                         {c: {k: cand[c][k] for k in ("bias", "separation", "auroc")} for c in ("black", "white")}})
            if h1 and ok:
                best = g["candidate"]
            print(arm, rows[-1]["generation"], rows[-1]["sprt"], ok, fails, flush=True)
        report["arms"][arm] = {"rows": rows, "final_best": best}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
