"""Masked losses (TRAINING.md §4).

* Policy: cross entropy against a target distribution over LEGAL actions only. Illegal logits are
  masked to -inf before the softmax, and a target with mass on an illegal action is a data error.
* Value: W/D/L cross entropy, only where a proven value exists (UNKNOWN is never a loss target).
* Aux: binary cross entropy per point and plane.
Non-finite losses raise immediately (no silent NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class DataError(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    aux: float = 0.5


def masked_policy_loss(logits: torch.Tensor, legal: torch.Tensor, target: torch.Tensor,
                       has_target: torch.Tensor) -> torch.Tensor:
    """logits/legal/target: [N, A]; target is a nonnegative set/distribution; has_target: [N] bool.

    Written with masks/weights only (no boolean indexing): data-dependent shapes crash the PyTorch
    XPU backend (UR_RESULT_ERROR_OUT_OF_RESOURCES, observed with torch 2.14.1+xpu)."""
    has = has_target.bool()
    if not bool(has.any()):
        return logits.new_zeros(())
    legal_b = legal > 0
    tgt = target.float() * has[:, None].float()
    if bool(((tgt > 0) & ~legal_b).any()):
        raise DataError("policy target has mass on an illegal action")
    tgt = tgt / tgt.sum(dim=1, keepdim=True).clamp_min(1e-12)
    # Rows without a target keep raw logits so an empty legal set can never produce NaN.
    masked = torch.where(has[:, None], logits.masked_fill(~legal_b, float("-inf")), logits)
    logp = torch.log_softmax(masked, dim=1)
    per_row = -torch.where(tgt > 0, tgt * logp, torch.zeros_like(logp)).sum(dim=1)
    return per_row.sum() / has.sum().clamp_min(1)


def value_loss(wdl_logits: torch.Tensor, value: torch.Tensor, has_value: torch.Tensor) -> torch.Tensor:
    """value: [N] in {+1 win, 0 draw, -1 loss} for the actor; classes W=0, D=1, L=2."""
    has = has_value.bool()
    if not bool(has.any()):
        return wdl_logits.new_zeros(())
    cls = (1 - value.long()).clamp(0, 2)  # +1 -> 0, 0 -> 1, -1 -> 2
    ce = F.cross_entropy(wdl_logits, cls, reduction="none")
    return (ce * has.float()).sum() / has.sum().clamp_min(1)


def aux_loss(aux_logits: torch.Tensor, aux_target: torch.Tensor) -> torch.Tensor:
    """aux_logits/aux_target: [N, 5, 225]."""
    return F.binary_cross_entropy_with_logits(aux_logits, aux_target.float())


def total_loss(out: dict, batch: dict, w: LossWeights) -> tuple[torch.Tensor, dict]:
    lp = masked_policy_loss(out["policy"], batch["legal"], batch["policy"], batch["has_policy"])
    lv = value_loss(out["wdl"], batch["value"], batch["has_value"])
    la = aux_loss(out["aux"], batch["aux"])
    loss = w.policy * lp + w.value * lv + w.aux * la
    parts = {"loss": loss.detach(), "policy": lp.detach(), "value": lv.detach(), "aux": la.detach()}
    for k, v in parts.items():
        if not torch.isfinite(v):
            raise NonFiniteLoss(f"non-finite {k} loss")
    return loss, parts
