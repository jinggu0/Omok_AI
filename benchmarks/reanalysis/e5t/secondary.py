"""E5t pre-registered secondary analyses of the 400-pair A/B (gen_069 vs gen_045, renju_holdout_e5_v11; same method as e5q/secondary.py):
per opponent, black / white per arm from the strength DB game table, search effort, and the sleep-rule
sensitivity (drop pairs with a game whose elapsed time of either side exceeds 180 s + 1.5 s x plies).

    python benchmarks/reanalysis/e5t/secondary.py
"""
import collections
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
ab = json.loads((HERE / "ab_e5t_gen069_vs_gen045_400.json").read_text(encoding="utf-8"))
ARMS = {"baseline": ab["baseline"]["model_hash"][:12], "candidate": ab["candidate"]["model_hash"][:12]}
con = sqlite3.connect(REPO / "runs/evaluation/results.sqlite")


def ci(x):
    n = len(x)
    mu = sum(x) / n
    sd = (sum((v - mu) ** 2 for v in x) / (n - 1)) ** 0.5
    return {"pairs": n, "mean_diff": mu, "ci95_normal": [mu - 1.96 * sd / n ** 0.5, mu + 1.96 * sd / n ** 0.5]}


out = {"per_opponent": {}, "colour": {}, "search": {}, "sleep": {}}
for opp, v in ab["per_opponent"].items():
    out["per_opponent"][opp] = {"baseline": v["baseline"], "candidate": v["candidate"], "paired": v["paired"],
                                "diff_counts": dict(sorted(collections.Counter(
                                    r["diff"] for r in ab["rows"] if r["opponent"] == opp).items()))}
slept = set()
for arm, h in ARMS.items():
    for opp in ab["per_opponent"]:
        mids = [m for (m,) in con.execute(
            "select match_id from match where opening_set_id='renju_holdout_e5_v11' and engine_a=? and engine_b=?",
            (f"dojo-{h}", opp))]
        assert len(mids) == 1, (arm, opp, mids)
        games = list(con.execute("select opening_id,a_color,score_a,elapsed_a_ms,elapsed_b_ms,plies,search_stats_json "
                                 "from game where match_id=?", (mids[0],)))
        col = {c: [g[2] for g in games if g[1] == c] for c in ("BLACK", "WHITE")}
        out["colour"][f"{arm}/{opp}"] = {c: {"games": len(s), "score": sum(s) / len(s)} for c, s in col.items()}
        st = [json.loads(g[6])["A"] for g in games]
        out["search"][f"{arm}/{opp}"] = {
            "sims_per_move": sum(s["sims"] * s["moves"] for s in st) / sum(s["moves"] for s in st),
            "ms_per_move": sum(g[3] for g in games) / sum(s["moves"] for s in st)}
        for g in games:
            if max(g[3], g[4]) > 180000 + 1500 * g[5]:
                slept.add((opp, g[0]))
                out["sleep"].setdefault("games", []).append({"arm": arm, "opponent": opp, "opening": g[0],
                                                             "elapsed_a_s": g[3] / 1000, "elapsed_b_s": g[4] / 1000})
kept = [r["diff"] for r in ab["rows"] if (r["opponent"], r["opening"]) not in slept]
out["sleep"]["pairs_dropped"] = sorted(map(list, slept))
out["sleep"]["sensitivity"] = ci(kept)
(HERE / "secondary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
