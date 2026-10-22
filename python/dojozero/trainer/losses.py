"""Masked losses (TRAINING.md §4).

* Policy: cross entropy against a target distribution over LEGAL actions only. Illegal logits are
  masked to -inf before the softmax, and a target with mass on an illegal action is a data error.
* Value: W/D/L cross entropy, only where a proven value exists (UNKNOWN is never a loss target).
* Aux: binary cross entropy per point and plane.
* Danger / refute (Phase 8 E4, solver-supervised reanalysis): weighted binary cross entropy only
  where a label exists (weight 0 = no label, e.g. UNKNOWN). Proven labels carry weight 1; heuristic
  negatives (search exhausted without a proof, not a safety proof) carry a smaller weight. An
  UNKNOWN move or position is never a target.
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
    opening: float = 1.0  # swap / branch / propose / select heads (Phase 7)
    danger: float = 0.0   # tactical danger head (Phase 8 E4)
    refute: float = 0.0   # per-move tactical refutation head (Phase 8 E4)


# Head ids of a training position (which head its policy target belongs to).
HEAD_POLICY, HEAD_SWAP, HEAD_BRANCH, HEAD_PROPOSE, HEAD_SELECT = 0, 1, 2, 3, 4
HEAD_NAMES = ["policy", "swap", "branch", "propose", "select"]


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


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean BCE over entries with weight > 0 (weights scale entries; weight 0 = unlabelled)."""
    wsum = weight.float().sum()
    if float(wsum) <= 0:
        return logits.new_zeros(())
    ce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    return (ce * weight.float()).sum() / wsum


def total_loss(out: dict, batch: dict, w: LossWeights) -> tuple[torch.Tensor, dict]:
    """batch["policy"] holds point/pass targets [N, 226] for the policy, propose and select heads
    (selected by batch["head"]); batch["tgt2"] holds swap/branch targets [N, 2]. Batches without
    "head" are plain policy batches (Stage A data)."""
    head = batch.get("head")
    has_policy = batch["has_policy"] if head is None else batch["has_policy"] & (head == HEAD_POLICY)
    lp = masked_policy_loss(out["policy"], batch["legal"], batch["policy"], has_policy)
    lv = value_loss(out["wdl"], batch["value"], batch["has_value"])
    la = aux_loss(out["aux"], batch["aux"])
    lo = la.new_zeros(())
    if head is not None:
        legal_pts = batch["legal"][:, :225]
        ones2 = torch.ones_like(batch["tgt2"], dtype=torch.uint8)
        lo = (masked_policy_loss(out["propose"], legal_pts, batch["policy"][:, :225], head == HEAD_PROPOSE)
              + masked_policy_loss(out["select"], legal_pts, batch["policy"][:, :225], head == HEAD_SELECT)
              + masked_policy_loss(out["swap"], ones2, batch["tgt2"], head == HEAD_SWAP)
              + masked_policy_loss(out["branch"], ones2, batch["tgt2"], head == HEAD_BRANCH))
    ld = lr = la.new_zeros(())
    if "danger_t" in batch and "danger" in out:
        ld = weighted_bce(out["danger"], batch["danger_t"], batch["danger_w"])
    if "refute_t" in batch and "refute" in out:
        lr = weighted_bce(out["refute"], batch["refute_t"], batch["refute_w"])
    loss = w.policy * lp + w.value * lv + w.aux * la + w.opening * lo + w.danger * ld + w.refute * lr
    parts = {"loss": loss.detach(), "policy": lp.detach(), "value": lv.detach(), "aux": la.detach(),
             "opening": lo.detach(), "danger": ld.detach(), "refute": lr.detach()}
    for k, v in parts.items():
        if not torch.isfinite(v):
            raise NonFiniteLoss(f"non-finite {k} loss")
    return loss, parts
