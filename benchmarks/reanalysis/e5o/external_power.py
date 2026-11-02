"""E5o: how many external pairs would a promotion signal need? From the paired differences of past
external A/Bs (Pela 2023 + Yixin 2018, 500 ms). Decides nothing; sizes the design.

    python benchmarks/reanalysis/e5o/external_power.py <out.json>
"""

import json
import math
import sys

import numpy as np

ABS = {"e5f": "benchmarks/reanalysis/e5f/ab_e5f_final_vs_T2.json", "e5g": "benchmarks/reanalysis/e5g/ab_e5g_gen045_vs_E4.json",
       "e5l": "benchmarks/reanalysis/e5l/ab_e5l_gen055_vs_gen045.json"}
GAME_SEC = 73 * 60 / 240  # E5l: 240 games (2 arms x 120 pairs) in 73 minutes


def main():
    out = {"source": ABS, "seconds_per_game": GAME_SEC, "rows": {}}
    sds = []
    for k, p in ABS.items():
        d = np.array([r["diff"] for r in json.load(open(p, encoding="utf-8"))["rows"]])
        sds.append(d.std(ddof=1))
        out["rows"][k] = {"pairs": len(d), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1))}
    sd = float(np.mean(sds))
    out["sd_used"] = sd
    out["pairs_needed"] = {}
    for delta in (0.03, 0.05, 0.10):  # two-sided 5%, power 80%, paired (both arms play every pair)
        n = math.ceil((1.96 + 0.8416) ** 2 * sd ** 2 / delta ** 2)
        out["pairs_needed"][f"{delta:.2f}"] = {"pairs": n, "games": 4 * n, "hours": 4 * n * GAME_SEC / 3600}
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
