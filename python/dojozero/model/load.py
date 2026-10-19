"""Load a trained network from a checkpoint (feature spec and architecture come from the file)."""

from __future__ import annotations

import torch

from ..features import FeatureSpec
from ..trainer import checkpoint
from .net import ModelConfig, PolicyValueNet


def model_sha(model: torch.nn.Module) -> str:
    """SHA-256 of the parameters and buffers (key order), recorded with every self-play game."""
    import hashlib

    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(path: str, device: torch.device | str = "cpu") -> tuple[PolicyValueNet, FeatureSpec, dict]:
    ck = checkpoint.load(path)
    fs = ck["feature_spec"]
    spec = FeatureSpec(fs["history_planes"], fs["tactical_planes"], fs["forbidden_plane"])
    if spec.hash() != fs["hash"]:
        raise ValueError("feature spec hash mismatch: checkpoint was made with a different encoder version")
    model = PolicyValueNet(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, spec, ck
