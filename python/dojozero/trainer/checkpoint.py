"""Versioned checkpoints (ARCHITECTURE.md §5.2). Loaded with torch.load(weights_only=True): only
tensors and plain Python containers are stored (no pickled objects).

Batch sampling is a pure function of (seed, step) (trainer/data.py), so model + optimizer +
scheduler + step + torch RNG are sufficient for an exact resume."""

from __future__ import annotations

from pathlib import Path

import torch

FORMAT_VERSION = 1


def save(path: str | Path, *, model, optimizer, scheduler, step: int, config: dict, feature_spec: dict,
         model_config: dict, env: dict, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "feature_spec": feature_spec,
        "model_config": model_config,
        "env": env,
        "metrics": metrics,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic


def load(path: str | Path, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {ckpt.get('format_version')}")
    return ckpt
