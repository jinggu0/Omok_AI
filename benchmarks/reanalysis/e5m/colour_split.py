"""E5m m3: colour split of the m2 replay (descriptive only).

    python benchmarks/reanalysis/e5m/colour_split.py <replay.json> <out.json>
"""

import json
import sys


def main():
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    g = d["games_detail"]
    out = {"games": len(g)}
    for c in ("BLACK", "WHITE"):
        s = [r["score_a"] for r in g if r["a_color"] == c]
        out[f"a_as_{c.lower()}"] = {"games": len(s), "score": sum(s) / len(s)}
    black = sum((r["score_a"] == 1) == (r["a_color"] == "BLACK") for r in g if r["score_a"] != 0.5)
    draws = sum(r["score_a"] == 0.5 for r in g)
    out |= {"black_wins": black, "white_wins": len(g) - draws - black, "draws": draws, "black_win_share": black / len(g)}
    # pairs where each player won as black: colour, not strength, decides the pair
    by_open: dict = {}
    for r in g:
        by_open.setdefault(r["opening"], []).append(r)
    out["pairs_split_by_colour"] = sum(len(v) == 2 and all(r["score_a"] != 0.5 for r in v) and v[0]["score_a"] != v[1]["score_a"]
                                       and all((r["score_a"] == 1) == (r["a_color"] == "BLACK") for r in v) for v in by_open.values())
    out["pairs"] = len(by_open)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
