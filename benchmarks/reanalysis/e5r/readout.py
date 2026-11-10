"""E5r pre-registered read-out of the capacity probe fits (benchmarks/reanalysis/e5r/preregistration.json).

    python benchmarks/reanalysis/e5r/readout.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
ARMS = ["b4c48_s1", "b6c64_s1", "b6c64_s2", "b10c96_s1", "b10c128_s1"]
METRICS = ["rp_policy", "rp_value", "re_value", "re_danger", "re_policy", "re_refute", "rp_opening"]
PRIMARY = ["rp_policy", "rp_value"]


def held(arm):
    rows = [json.loads(l) for l in (HERE / f"metrics_{arm}.jsonl").read_text().splitlines() if l.strip()]
    ev = {r["step"]: r for r in rows if "rp_policy" in r and "train_rp_policy" not in r}
    fin = [r for r in rows if "final_checkpoint" in r][0]
    tr = [r for r in rows if "train_rp_policy" in r][-1]
    return ev, fin, tr


out = {"arms": {}, "readout": {}}
for a in ARMS:
    ev, fin, tr = held(a)
    out["arms"][a] = {"params": fin["params"], "wall_s": fin["wall_s"], "final": {m: ev[6000][m] for m in METRICS},
                      "trajectory_rp_policy": {s: ev[s]["rp_policy"] for s in sorted(ev)},
                      "trajectory_rp_value": {s: ev[s]["rp_value"] for s in sorted(ev)},
                      "train_last_250": {"rp_policy": tr["train_rp_policy"], "rp_value": tr["train_rp_value"],
                                         "re_value": tr["train_re_value"]}}
ref = json.loads((HERE / "val_gen045.json").read_text())
out["gen_045_reference"] = {m: ref[m] for m in METRICS}
f = {a: out["arms"][a]["final"] for a in ARMS}
verdict = {}
for m in METRICS:
    s1, s2 = f["b6c64_s1"][m], f["b6c64_s2"][m]
    mean, d = (s1 + s2) / 2, abs(s1 - s2)
    band = max(2 * d, 0.01 * mean)
    imp = mean - f["b10c128_s1"][m]
    out["readout"][m] = {"b6c64_mean": mean, "seed_diff": d, "band": band, "b10c128_improvement": imp,
                         "b10c96_improvement": mean - f["b10c96_s1"][m], "b4c48_improvement": mean - f["b4c48_s1"][m],
                         "beyond_band": imp > band}
    if m in PRIMARY:
        verdict[m] = "better" if imp > band else ("worse" if -imp > band else "within")
if all(v == "better" for v in verdict.values()):
    out["verdict"] = "capacity signal"
elif all(v in ("within", "worse") for v in verdict.values()):
    out["verdict"] = "no capacity signal"
else:
    out["verdict"] = "mixed"
out["primary_by_metric"] = verdict
(HERE / "readout.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out["verdict"], verdict)
for m in METRICS:
    r = out["readout"][m]
    print(f"{m:11s} " + " ".join(f"{a}={f[a][m]:.4f}" for a in ARMS) + f" | gen045={ref[m]:.4f} | band {r['band']:.4f} b10c128 imp {r['b10c128_improvement']:+.4f}")
for a in ARMS:
    print(a, out["arms"][a]["params"], {k: round(v, 3) for k, v in out["arms"][a]["trajectory_rp_policy"].items()}, "train", {k: round(v, 3) for k, v in out["arms"][a]["train_last_250"].items()})
