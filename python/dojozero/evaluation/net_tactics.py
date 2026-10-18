"""Network-only tactical solve rate on the fixed benchmark suites (EVALUATION.md §3, §9).

For every benchmark problem whose expected answer is a win for the side to move, the network
plays its top legal move (no search). The move counts as solving the problem if, judged by the
sound C++ solvers, it wins immediately or leaves the opponent in a proven LOSS. An untrained
network with the same architecture is reported as the baseline.

    python -m dojozero.evaluation.net_tactics --checkpoint runs/<run>/checkpoints/step_N.pt
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .. import _core
from ..features import FeatureSpec
from ..model import ModelConfig, PolicyValueNet
from ..trainer import checkpoint

REPO = Path(__file__).resolve().parents[3]
WIN_RESULTS = {"WIN", "FOUND"}


def win_problems(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.rglob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for p in doc.get("problems", []):
            if p["expected"]["result"] in WIN_RESULTS:
                out.append(p | {"file": path.name})
    return out


@torch.no_grad()
def top_move(model, spec: FeatureSpec, g, device) -> int:
    x = torch.from_numpy(np.asarray(_core.encode(g, spec.core()), dtype=np.float32))[None].to(device)
    logits = model(x)["policy"][0].cpu()
    legal = torch.from_numpy(np.asarray(_core.legal_mask(g))).bool()
    return int(logits.masked_fill(~legal, float("-inf")).argmax())


def solves(g, move: int, budget: int) -> bool:
    if move == 225:
        return False
    g2 = g.copy()
    if not g2.apply(_core.point_to_str(move)):
        return False
    if g2.is_terminal:
        return g2.winner is not None and g2.winner == g.actor
    d = _core.analyze_tactics(g2, budget, budget, budget)
    return d["result"] == "LOSS"


def evaluate_model(model, spec: FeatureSpec, problems: list[dict], device, budget: int) -> dict:
    model.eval()
    by_cat: dict[str, list[int]] = defaultdict(list)
    for p in problems:
        g = _core.GameState.from_spec(json.dumps(p["position"]), p.get("profile", "RENJU_RIF"))
        by_cat[p["category"]].append(int(solves(g, top_move(model, spec, g, device), budget)))
    res = {f"solve_rate/{k}": float(np.mean(v)) for k, v in sorted(by_cat.items())}
    res["solve_rate/all"] = float(np.mean([x for v in by_cat.values() for x in v]))
    res["problems"] = sum(len(v) for v in by_cat.values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--out")
    args = ap.parse_args()
    device = torch.device("cpu")
    ck = checkpoint.load(args.checkpoint)
    fs = ck["feature_spec"]
    spec = FeatureSpec(fs["history_planes"], fs["tactical_planes"], fs["forbidden_plane"])
    mcfg = ModelConfig(**ck["model_config"])
    problems = win_problems(REPO / "benchmarks")
    trained = PolicyValueNet(mcfg)
    trained.load_state_dict(ck["model"])
    torch.manual_seed(ck["config"]["seed"])
    untrained = PolicyValueNet(mcfg)
    result = {"checkpoint": args.checkpoint, "step": ck["step"],
              "trained": evaluate_model(trained, spec, problems, device, args.budget),
              "untrained_baseline": evaluate_model(untrained, spec, problems, device, args.budget)}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
