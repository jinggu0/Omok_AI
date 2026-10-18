import pytest
import torch

from dojozero.trainer.losses import DataError, LossWeights, NonFiniteLoss, masked_policy_loss, total_loss, value_loss


def test_policy_loss_ignores_illegal_logits():
    logits = torch.tensor([[10.0, 0.0, 0.0, -100.0]])
    legal = torch.tensor([[0, 1, 1, 0]], dtype=torch.uint8)
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    has = torch.tensor([True])
    # the huge illegal logit must not matter: loss = log(2) with two equal legal logits
    assert torch.isclose(masked_policy_loss(logits, legal, target, has), torch.log(torch.tensor(2.0)))


def test_policy_target_on_illegal_action_is_an_error():
    logits = torch.zeros(1, 3)
    legal = torch.tensor([[1, 0, 1]], dtype=torch.uint8)
    target = torch.tensor([[0.0, 1.0, 0.0]])
    with pytest.raises(DataError):
        masked_policy_loss(logits, legal, target, torch.tensor([True]))


def test_policy_target_set_is_uniform():
    logits = torch.zeros(1, 4)
    legal = torch.ones(1, 4, dtype=torch.uint8)
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert torch.isclose(masked_policy_loss(logits, legal, target, torch.tensor([True])), torch.log(torch.tensor(4.0)))


def test_value_loss_masks_unknown():
    wdl = torch.tensor([[5.0, 0.0, -5.0], [0.0, 0.0, 0.0]])
    value = torch.tensor([1, -1])
    has = torch.tensor([True, False])
    loss = value_loss(wdl, value, has)
    assert loss < 0.01  # only the confident correct win counts


def test_nan_is_never_silent():
    out = {"policy": torch.full((1, 3), float("nan")), "wdl": torch.zeros(1, 3), "aux": torch.zeros(1, 5, 225)}
    batch = {"legal": torch.ones(1, 3, dtype=torch.uint8), "policy": torch.tensor([[1.0, 0, 0]]),
             "has_policy": torch.tensor([True]), "value": torch.tensor([1]), "has_value": torch.tensor([True]),
             "aux": torch.zeros(1, 5, 225)}
    with pytest.raises(NonFiniteLoss):
        total_loss(out, batch, LossWeights())
