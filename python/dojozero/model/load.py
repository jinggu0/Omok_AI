"""Load a trained network from a checkpoint (feature spec and architecture come from the file)."""

from __future__ import annotations

import torch

from ..features import FeatureSpec
from ..trainer import checkpoint
from .net import ModelConfig, PolicyValueNet


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
