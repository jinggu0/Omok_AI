"""AlphaZero-style residual policy/value network (TRAINING.md §7, ADR-07 baseline).

Heads:
  policy  [N, 226]   point logits (1x1 conv, translation equivariant) + pass logit (global)
  wdl     [N, 3]     win / draw / loss logits for the actor
  aux     [N, 5, 225] tactical auxiliary logits (immediate_win, forced_block, four, three, forbidden)
  swap    [N, 2]     opening: swap / keep          (trained from Phase 7)
  branch  [N, 2]     opening: direct fifth / ten   (trained from Phase 7)
  propose [N, 225]   opening: ten-proposal policy  (trained from Phase 7)
  select  [N, 225]   opening: select-offered policy(trained from Phase 7)
With ModelConfig.tactical_heads (Phase 8 E4, solver-supervised reanalysis):
  danger  [N]        logit of P(side to move is proven lost: the opponent has a solver-provable
                     forced win against every reply)
  refute  [N, 225]   per point: logit of P(the opponent has a solver-provable forced win after
                     the side to move plays here)
Both are extra modules on the shared trunk, so checkpoints without them load unchanged
(load_partial below) and the policy/value outputs are computed exactly as before.
Legal masks are applied by the losses / search, not inside the network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from ..features import NUM_AUX, NUM_POINTS


@dataclass(frozen=True)
class ModelConfig:
    in_planes: int
    blocks: int = 6
    channels: int = 64
    head_hidden: int = 128
    tactical_heads: bool = False  # danger + refute heads (Phase 8 E4)

    def to_dict(self) -> dict:
        return asdict(self)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.stem = nn.Sequential(nn.Conv2d(cfg.in_planes, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])
        self.point_heads = nn.Conv2d(c, 1 + NUM_AUX + 2, 1)  # policy, aux, propose, select
        g = 2 * c  # global mean + max pooling
        self.global_fc = nn.Sequential(nn.Linear(g, cfg.head_hidden), nn.ReLU())
        self.value_out = nn.Linear(cfg.head_hidden, 3)
        self.pass_out = nn.Linear(cfg.head_hidden, 1)
        self.swap_out = nn.Linear(cfg.head_hidden, 2)
        self.branch_out = nn.Linear(cfg.head_hidden, 2)
        if cfg.tactical_heads:
            self.danger_out = nn.Linear(cfg.head_hidden, 1)
            self.refute_head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(self.stem(x))
        n = h.shape[0]
        pts = self.point_heads(h).reshape(n, -1, NUM_POINTS)
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        z = self.global_fc(pooled)
        policy = torch.cat([pts[:, 0], self.pass_out(z)], dim=1)
        out = {
            "policy": policy,
            "wdl": self.value_out(z),
            "aux": pts[:, 1 : 1 + NUM_AUX],
            "propose": pts[:, 1 + NUM_AUX],
            "select": pts[:, 2 + NUM_AUX],
            "swap": self.swap_out(z),
            "branch": self.branch_out(z),
        }
        if self.cfg.tactical_heads:
            out["danger"] = self.danger_out(z).squeeze(1)
            out["refute"] = self.refute_head(h).reshape(n, NUM_POINTS)
        return out


TACTICAL_HEAD_PREFIXES = ("danger_out.", "refute_head.")


def load_partial(model: PolicyValueNet, state: dict) -> list[str]:
    """Load a state dict that may lack the tactical heads (an older checkpoint): every other
    parameter must match exactly; only tactical-head parameters may be missing (they keep their
    fresh initialisation). Returns the missing keys."""
    res = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith(TACTICAL_HEAD_PREFIXES)]
    if bad_missing or res.unexpected_keys:
        raise ValueError(f"checkpoint mismatch: missing {bad_missing}, unexpected {res.unexpected_keys}")
    return list(res.missing_keys)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
