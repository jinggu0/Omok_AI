"""Tactical diagnosis of a network on solver-labelled positions (PROGRESS Phase 8 E4, Case B).

    python -m dojozero.tools.reanalysis_eval --ckpt runs/taraguchi_v1/models/gen_001.pt --split test \
        --out benchmarks/reanalysis/eval_baseline_test.json

Labels come from data/reanalysis_v1 (baseline top-k candidates). A model may rank moves that were
never labelled: before measuring, every move in the model's policy top-8 and the move its search
picks is labelled with the same solver budget (cached per position in labels_extra.jsonl.gz), so two
models are compared on the same label rule. Codes: PROVEN_UNSAFE (opponent forced win), PROVEN_WINNING,
UNREFUTED_EXHAUSTED (threat-space search ended without a proof: not a safety proof), UNKNOWN_BUDGET.
"Non-refuted" below = UNREFUTED_EXHAUSTED or PROVEN_WINNING (never UNKNOWN_BUDGET).

Search (fixed visits, PUCT, no root noise, proof/filter off) is classified when its chosen move is
PROVEN_UNSAFE (pre-registered rules):
  ALREADY_LOST  no non-refuted move is known at the position
  POLICY        the best-ranked non-refuted move is outside the policy top-8
  VALUE         a non-refuted move is in the top-8 and got >= 8 visits, and the root Q of the chosen
                (unsafe) move is >= its Q (the evaluations preferred the losing move)
  SEARCH        a non-refuted move is in the top-8 but it got < 8 visits or a higher Q than the chosen
                move (visit allocation / selection picked the losing move)
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..trainer.reanalysis import CODE, PROFILE, SPLITS
from .reanalysis_data import _analyze_move

REPO = Path(__file__).resolve().parents[3]
NAMES = {v: k for k, v in CODE.items()}


def auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pct(x) -> float | None:
    x = list(x)
    return float(np.mean(x)) if x else None


def main(argv: list[str] | None = None) -> int:
    from ..evaluation.tournament import keep_awake
    from ..model.load import load_model, model_sha
    from ..search.driver import NetEvaluator, SearchSettings, search_many
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/reanalysis_v1")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = REPO / args.root
    z = np.load(root / "arrays.npz", allow_pickle=False)
    a = {k: z[k] for k in z.files}
    sel = np.flatnonzero(a["split"] == SPLITS[args.split])
    if args.limit:
        sel = sel[:args.limit]
    pos = {}
    for l in gzip.open(root / "positions.jsonl.gz", "rt", encoding="utf-8"):
        p = json.loads(l)
        pos[p["id"]] = p
    model, spec, _ = load_model(str(REPO / args.ckpt), args.device)
    sha = model_sha(model)
    ev = NetEvaluator(model, args.device)
    states = []
    for i in sel:
        g = _core.GameState(PROFILE)
        for m in pos[str(a["ids"][i])]["moves"]:
            assert g.apply(m)
        states.append(g)
    # Network outputs.
    pol, val, danger, refute = [], [], [], []
    for s in range(0, len(sel), 256):
        feats = np.stack([np.asarray(_core.encode(g, spec.core())) for g in states[s:s + 256]])
        out = ev(feats)
        lg = np.where(a["legal"][sel[s:s + 256]] > 0, out["policy"], -np.inf)
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        pol.append(e / e.sum(axis=1, keepdims=True))
        w = np.exp(out["wdl"] - out["wdl"].max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        val.append(w[:, 0] - w[:, 2])
        if "danger" in out:
            danger.append(1 / (1 + np.exp(-out["danger"])))
            refute.append(1 / (1 + np.exp(-out["refute"])))
    pol, val = np.concatenate(pol), np.concatenate(val)
    danger = np.concatenate(danger) if danger else None
    refute = np.concatenate(refute) if refute else None
    # Search (fixed visits, network only).
    s = SearchSettings(simulations=args.sims, root="PUCT", gumbel_noise=False, max_ms=0.0)
    results = []
    with keep_awake():
        for k in range(0, len(states), 64):
            results += search_many(states[k:k + 64], ev, s, spec.core())
    # Complete labels for the model's top-8 and the searched move (cached).
    extra_path = root / "labels_extra.jsonl.gz"
    extra: dict[tuple[str, str], str] = {}
    if extra_path.exists():
        for l in gzip.open(extra_path, "rt", encoding="utf-8"):
            r = json.loads(l)
            extra[(r["id"], r["m"])] = r["code"]
    code = a["code"][sel].copy()
    by_id: dict[str, list] = {}
    for (pid, m), c in extra.items():
        by_id.setdefault(pid, []).append((m, c))
    for j, i in enumerate(sel):
        for m, c in by_id.get(str(a["ids"][i]), []):
            if c in CODE:
                code[j, _core.str_to_point(m)] = CODE[c]
    todo = []
    for j, i in enumerate(sel):
        want = list(np.argsort(-pol[j][:225])[:8]) + [_core.str_to_point(results[j]["action"])]
        for q in want:
            if pol[j][q] > 0 and code[j, q] < 0 and (str(a["ids"][i]), _core.point_to_str(int(q))) not in extra:
                todo.append((j, int(q)))
    todo = list(dict.fromkeys(todo))
    print(f"labelling {len(todo)} extra moves", flush=True)

    def lab(t):
        j, q = t
        return j, q, _analyze_move(states[j], _core.point_to_str(q))["code"]

    with keep_awake(), ThreadPoolExecutor(max_workers=args.threads) as ex:
        new = list(ex.map(lab, todo))
    with gzip.open(extra_path, "at", encoding="utf-8") as fh:
        for j, q, c in new:
            fh.write(json.dumps({"id": str(a["ids"][sel[j]]), "m": _core.point_to_str(q), "code": c}) + "\n")
            if c in CODE:
                code[j, q] = CODE[c]
    # Metrics.
    side = a["side"][sel]
    res = a["pos_result"][sel]
    rows = []
    for j, i in enumerate(sel):
        p = pol[j][:225]
        order = np.argsort(-p)
        rank = np.empty(225, int)
        rank[order] = np.arange(1, 226)
        c = code[j]
        nonref = np.flatnonzero((c == CODE["UNREFUTED_EXHAUSTED"]) | (c == CODE["PROVEN_WINNING"]))
        unsafe = np.flatnonzero(c == CODE["PROVEN_UNSAFE"])
        r = results[j]
        act = _core.str_to_point(r["action"])
        kids = {_core.str_to_point(k["action"]): k for k in r["children"] if k["action"] != "pass"}
        best_nr = int(nonref[np.argmin(rank[nonref])]) if len(nonref) else None
        row = {"side": int(side[j]), "res": int(res[j]), "value": float(val[j]),
               "top1": NAMES.get(int(c[order[0]]), "UNLABELLED"),
               "unsafe_mass_top8": float(sum(p[q] for q in order[:8] if c[q] == CODE["PROVEN_UNSAFE"])),
               "unsafe_mass": float(p[unsafe].sum()), "has_nonref": bool(len(nonref)),
               "best_nonref_rank": int(rank[best_nr]) if best_nr is not None else None,
               "win_rank": int(rank[a["win_move"][i]]) if a["win_move"][i] >= 0 else None,
               "search": NAMES.get(int(c[act]), "UNLABELLED"), "sims": r["simulations"]}
        for kk in (1, 4, 8, 16):
            row[f"nonref_top{kk}"] = bool(len(nonref)) and bool(any(rank[q] <= kk for q in nonref))
            row[f"unsafe_top{kk}"] = bool(any(rank[q] <= kk for q in unsafe))
        if row["search"] == "PROVEN_UNSAFE":
            if best_nr is None:
                row["failure"] = "ALREADY_LOST"
            elif rank[best_nr] > 8:
                row["failure"] = "POLICY"
            else:
                alts = [q for q in nonref if rank[q] <= 8]
                ch = kids.get(act, {"q": 0.0, "visits": 0})
                strong = [q for q in alts if q in kids and kids[q]["visits"] >= 8]
                row["failure"] = "VALUE" if strong and all(ch["q"] >= kids[q]["q"] for q in strong) else "SEARCH"
        if danger is not None:
            row["danger"] = float(danger[j])
        rows.append(row)

    def summarize(rs: list[dict]) -> dict:
        d = {"n": len(rs)}
        lab1 = [r for r in rs if r["top1"] != "UNLABELLED"]
        d["top1_unsafe_rate"] = pct(r["top1"] == "PROVEN_UNSAFE" for r in lab1)
        d["top1_labelled"] = len(lab1)
        d["unsafe_in_top4_rate"] = pct(r["unsafe_top4"] for r in rs)
        d["unsafe_mass_top8_mean"] = pct(r["unsafe_mass_top8"] for r in rs)
        hn = [r for r in rs if r["has_nonref"]]
        d["positions_with_nonref"] = len(hn)
        for kk in (1, 4, 8, 16):
            d[f"nonref_top{kk}_coverage"] = pct(r[f"nonref_top{kk}"] for r in hn)
        rk = [r["best_nonref_rank"] for r in hn]
        d["best_nonref_rank_mean"] = pct(rk)
        d["best_nonref_rank_median"] = float(np.median(rk)) if rk else None
        wr = [r["win_rank"] for r in rs if r["win_rank"] is not None]
        d["proven_win_move_top1"] = pct(x == 1 for x in wr)
        d["proven_win_move_top4"] = pct(x <= 4 for x in wr)
        d["value_proven_loss_mean"] = pct(r["value"] for r in rs if r["res"] == -1)
        d["value_proven_win_mean"] = pct(r["value"] for r in rs if r["res"] == 1)
        d["value_proven_loss_ge_0_rate"] = pct(r["value"] >= 0 for r in rs if r["res"] == -1)
        pr = [r for r in rs if r["res"] != 0]
        if pr:  # calibration on proven positions: P(win) = (v+1)/2 vs outcome
            pw = np.array([(r["value"] + 1) / 2 for r in pr])
            y = np.array([r["res"] == 1 for r in pr], float)
            bins = np.clip((pw * 5).astype(int), 0, 4)
            d["proven_brier"] = float(np.mean((pw - y) ** 2))
            d["proven_ece5"] = float(sum(abs(pw[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                                         for b in range(5) if (bins == b).any()))
            d["calibration_bins"] = [{"bin": b, "n": int((bins == b).sum()), "pred": float(pw[bins == b].mean()),
                                      "proven_win_rate": float(y[bins == b].mean())} for b in range(5) if (bins == b).any()]
        d["search_codes"] = dict(Counter(r["search"] for r in rs))
        d["search_unsafe_rate"] = pct(r["search"] == "PROVEN_UNSAFE" for r in rs if r["search"] != "UNLABELLED")
        d["failures"] = dict(Counter(r["failure"] for r in rs if "failure" in r))
        d["sims_mean"] = pct(r["sims"] for r in rs)
        if rs and "danger" in rs[0]:
            dl = [r for r in rs if r["res"] != 0]
            if dl:
                sc = np.array([r["danger"] for r in dl])
                y = np.array([r["res"] == -1 for r in dl], int)
                d["danger_auroc_exact"] = auroc(sc, y)
                pred = sc >= 0.5
                d["danger_precision"] = float((pred & (y == 1)).sum() / max(1, pred.sum()))
                d["danger_recall"] = float((pred & (y == 1)).sum() / max(1, (y == 1).sum()))
        return d

    report = {"ckpt": args.ckpt, "model_sha": sha, "split": args.split, "sims": args.sims,
              "extra_labelled": len(todo), "ALL": summarize(rows),
              "BLACK": summarize([r for r in rows if r["side"] == 0]),
              "WHITE": summarize([r for r in rows if r["side"] == 1])}
    if refute is not None:  # per-move refutation head on labelled moves
        ys, ss = [], []
        for j in range(len(sel)):
            for q in np.flatnonzero(code[j] >= 0):
                if code[j][q] in (CODE["PROVEN_UNSAFE"], CODE["PROVEN_WINNING"], CODE["UNREFUTED_EXHAUSTED"]):
                    ys.append(int(code[j][q] == CODE["PROVEN_UNSAFE"]))
                    ss.append(float(refute[j][q]))
        ys, ss = np.array(ys), np.array(ss)
        report["refute_head"] = {"moves": int(len(ys)), "auroc": auroc(ss, ys),
                                 "precision@0.5": float(((ss >= .5) & (ys == 1)).sum() / max(1, (ss >= .5).sum())),
                                 "recall@0.5": float(((ss >= .5) & (ys == 1)).sum() / max(1, (ys == 1).sum()))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report | {"rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
