"""Device benchmark for ADR-04 (training device) and ADR-02 (inference throughput).

    python -m dojozero.tools.bench_device [--blocks 6 --channels 64]

Measures, for CPU and XPU (if available): training steps/s at batch 256 and inference
evaluations/s (eval mode, no grad) at several batch sizes. Synthetic inputs: throughput only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import torch

from ..features import FeatureSpec
from ..model import ModelConfig, PolicyValueNet


def sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()


def bench(device: torch.device, mcfg: ModelConfig, threads: int | None, amp: bool = False) -> dict:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = PolicyValueNet(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    res: dict = {"device": str(device), "threads": torch.get_num_threads(), "amp_bf16": amp}
    ctx = (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    x = torch.rand(256, mcfg.in_planes, 15, 15, device=device)
    for _ in range(3):  # warm-up
        with ctx():
            out = model(x)
        loss = out["policy"].logsumexp(1).mean() + out["wdl"].logsumexp(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync(device)
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 5.0:
        with ctx():
            out = model(x)
        loss = out["policy"].logsumexp(1).mean() + out["wdl"].logsumexp(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    sync(device)
    res["train_steps_per_s_b256"] = steps / (time.perf_counter() - t0)
    model.eval()
    with torch.no_grad():
        for b in (1, 8, 32, 128, 512):
            xb = torch.rand(b, mcfg.in_planes, 15, 15, device=device)
            for _ in range(3):
                with ctx():
                    model(xb)
            sync(device)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < 3.0:
                with ctx():
                    model(xb)
                n += 1
            sync(device)
            dt = time.perf_counter() - t0
            res[f"eval_per_s_b{b}"] = n * b / dt
            res[f"latency_ms_b{b}"] = 1000 * dt / n
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--out")
    args = ap.parse_args()
    mcfg = ModelConfig(in_planes=FeatureSpec().num_planes, blocks=args.blocks, channels=args.channels)
    results = {"model": mcfg.to_dict(), "torch": torch.__version__, "runs": [bench(torch.device("cpu"), mcfg, None)]}
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        results["xpu_name"] = torch.xpu.get_device_name(0)
        results["runs"].append(bench(torch.device("xpu"), mcfg, None))
        results["runs"].append(bench(torch.device("xpu"), mcfg, None, amp=True))
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
