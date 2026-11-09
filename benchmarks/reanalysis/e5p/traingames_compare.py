"""E5p (descriptive, not pre-registered as a decision): FT1 vs gen_045 on the SAME training-only openings
(renju_train_reanalysis_v1, 120 pairs per engine, 500 ms), paired by opening. gen_045's games were played
2026-11-02/03 (E5o), FT1's 2026-11-05/07 (E5p round 2), so this is not a concurrent comparison.

    python benchmarks/reanalysis/e5p/traingames_compare.py
"""
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("traingames_compare.json")


def by_opening(d: dict) -> dict:
    m = collections.defaultdict(list)
    for g in d["games"]:
        m[g["opening_id"]].append(g["score_a"])
    return {k: sum(v) / len(v) for k, v in m.items()}


def side(d: dict, colour: str) -> float:
    s = [g["score_a"] for g in d["games"] if g["a_color"] == colour]
    return sum(s) / len(s)


res = {}
for eng in ("pela_2023", "yixin_2018", "katagomo_2021r"):
    a = json.loads((REPO / f"runs/evaluation/e5o_traingames_vs_{eng}.json").read_text())
    b = json.loads((REPO / f"runs/evaluation/e5p_r2_traingames_vs_{eng}.json").read_text())
    pa, pb = by_opening(a), by_opening(b)
    ks = sorted(set(pa) & set(pb))
    df = [pb[k] - pa[k] for k in ks]
    n = len(df)
    mu = sum(df) / n
    sd = (sum((x - mu) ** 2 for x in df) / (n - 1)) ** 0.5
    res[eng] = {
        "pairs": n,
        "gen_045": {"score": a["summary"]["score"], "black": side(a, "BLACK"), "white": side(a, "WHITE"),
                    "failures": a["failures"], "file": f"runs/evaluation/e5o_traingames_vs_{eng}.json"},
        "FT1": {"score": b["summary"]["score"], "black": side(b, "BLACK"), "white": side(b, "WHITE"),
                "failures": b["failures"], "file": f"runs/evaluation/e5p_r2_traingames_vs_{eng}.json"},
        "paired_diff": mu, "ci95_normal": [mu - 1.96 * sd / n ** 0.5, mu + 1.96 * sd / n ** 0.5],
        "diff_counts": dict(sorted(collections.Counter(df).items())),
    }
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res.items():
    print(k, round(v["gen_045"]["score"], 3), "->", round(v["FT1"]["score"], 3), round(v["paired_diff"], 3),
          [round(x, 3) for x in v["ci95_normal"]])
