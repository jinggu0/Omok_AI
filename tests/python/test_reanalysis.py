"""Phase 8 E4 (Case B, solver-supervised reanalysis): tactical heads, checkpoint compatibility, loss
masking (UNKNOWN is never a target), split leakage, D4 consistency and label codes."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dojozero import _core
from dojozero.features import GATHER, NUM_POINTS, NUM_POLICY, FeatureSpec
from dojozero.model.net import ModelConfig, PolicyValueNet, load_partial
from dojozero.tools.reanalysis_data import _analyze_move, split_for
from dojozero.trainer.losses import LossWeights, total_loss, weighted_bce
from dojozero.trainer.reanalysis import CODE, HEUR_W, ReanalysisData, make_targets

REPO = Path(__file__).resolve().parents[2]


def nets():
    spec = FeatureSpec()
    torch.manual_seed(0)
    old = PolicyValueNet(ModelConfig(in_planes=spec.num_planes, blocks=1, channels=16, head_hidden=32)).eval()
    new = PolicyValueNet(ModelConfig(in_planes=spec.num_planes, blocks=1, channels=16, head_hidden=32,
                                     tactical_heads=True)).eval()
    return spec, old, new


def test_old_checkpoint_loads_into_tactical_model_with_identical_outputs():
    spec, old, new = nets()
    missing = load_partial(new, old.state_dict())
    assert missing and all(k.startswith(("danger_out.", "refute_head.")) for k in missing)
    x = torch.rand(3, spec.num_planes, 15, 15)
    with torch.no_grad():
        a, b = old(x), new(x)
    for k in ("policy", "wdl", "aux", "swap", "branch", "propose", "select"):
        assert torch.equal(a[k], b[k]), k
    assert b["danger"].shape == (3,) and b["refute"].shape == (3, NUM_POINTS)
    assert "danger" not in a and "refute" not in a


def test_partial_load_rejects_other_mismatches():
    _, old, new = nets()
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith("value_out")}
    with pytest.raises(ValueError):
        load_partial(new, sd)


def test_weighted_bce_ignores_unlabelled_entries():
    logits = torch.tensor([[3.0, -2.0, 50.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    weight = torch.tensor([[1.0, 0.3, 0.0]])  # last entry: UNKNOWN (weight 0)
    loss = weighted_bce(logits, target, weight)
    loss.backward()
    assert logits.grad[0, 2] == 0
    assert weighted_bce(logits, target, torch.zeros_like(weight)).item() == 0


def test_total_loss_without_tactical_weights_is_unchanged():
    spec, _, new = nets()
    b = {"x": torch.rand(2, spec.num_planes, 15, 15), "legal": torch.ones(2, NUM_POLICY, dtype=torch.uint8),
         "policy": torch.full((2, NUM_POLICY), 1 / NUM_POLICY), "has_policy": torch.tensor([True, False]),
         "value": torch.tensor([1, -1]), "has_value": torch.tensor([True, True]),
         "aux": torch.zeros(2, 5, NUM_POINTS)}
    out = new(b["x"])
    l0, _ = total_loss(out, b, LossWeights())
    b2 = b | {"danger_t": torch.tensor([1.0, 0.0]), "danger_w": torch.tensor([1.0, 1.0]),
              "refute_t": torch.zeros(2, NUM_POINTS), "refute_w": torch.ones(2, NUM_POINTS)}
    l1, parts = total_loss(out, b2, LossWeights())  # danger/refute weights default to 0
    assert torch.allclose(l0, l1) and float(parts["danger"]) > 0
    l2, _ = total_loss(out, b2, LossWeights(danger=1.0, refute=1.0))
    assert float(l2) > float(l1)


def test_targets_never_use_unknown():
    code = np.full((3, NUM_POINTS), -1, np.int8)
    code[0, [0, 1, 2, 3]] = [CODE["PROVEN_UNSAFE"], CODE["PROVEN_WINNING"], CODE["UNREFUTED_EXHAUSTED"],
                             CODE["UNKNOWN_BUDGET"]]
    code[1, 5] = CODE["UNKNOWN_BUDGET"]  # only an UNKNOWN move, position UNKNOWN
    res = np.array([0, 0, -1], np.int8)
    legal = np.ones((3, NUM_POLICY), np.uint8)
    p0 = np.full((3, NUM_POLICY), 1 / NUM_POLICY, np.float32)
    t = make_targets(code, res, np.array([-1, -1, -1]), legal, p0)
    assert t["refute_t"][0, :4].tolist() == [1, 0, 0, 0]
    assert t["refute_w"][0, :4].tolist() == pytest.approx([1, 1, HEUR_W, 0])
    assert t["refute_w"][0, 4:].sum() == 0 and t["refute_w"][1].sum() == 0
    assert t["danger_w"].tolist() == pytest.approx([HEUR_W, 0, 1]) and t["danger_t"][2] == 1
    assert not t["has_value"][0] and not t["has_value"][1] and t["value"][2] == -1
    # policy: the proven-unsafe move loses its mass, the UNKNOWN and unlabelled moves keep theirs
    assert t["has_policy"][0] and t["policy"][0, 0] == 0 and t["policy"][0, 3] > 0
    assert not t["has_policy"][1] and not t["has_policy"][2]


def test_split_rule_keeps_blunder_openings_in_test():
    assert split_for("renju_balanced_v1", "x", 0) == "test" and split_for("renju_balanced_v1", "x", 11) == "test"
    assert split_for("renju_balanced_v1", "x", 12) in ("train", "val")
    assert split_for("renju_train_reanalysis_v1", "trainre1-0001", 1) in ("train", "val")


def test_collected_positions_do_not_leak_across_splits():
    path = REPO / "data" / "reanalysis_v1" / "positions.jsonl.gz"
    if not path.exists():
        pytest.skip("reanalysis data not generated on this machine")
    split_of_hash, split_of_group = {}, {}
    for line in gzip.open(path, "rt", encoding="utf-8"):
        p = json.loads(line)
        assert split_of_hash.setdefault(p["canonical_hash"], p["split"]) == p["split"]
        assert split_of_group.setdefault(p["group"], p["split"]) == p["split"]
        if p["group"].startswith("renju_balanced_v1/"):
            assert not (24 <= p["opening_index"] < 104), "external A/B openings must not be used"


def test_d4_batch_moves_labels_with_the_board():
    n = 2
    a = {"feats": np.zeros((n, 3, NUM_POINTS), np.uint8), "aux": np.zeros((n, 5, NUM_POINTS), np.uint8),
         "legal": np.ones((n, NUM_POLICY), np.uint8), "policy": np.zeros((n, NUM_POLICY), np.float32),
         "has_policy": np.ones(n, bool), "value": np.zeros(n, np.int64), "has_value": np.zeros(n, bool),
         "danger_t": np.zeros(n, np.float32), "danger_w": np.zeros(n, np.float32),
         "refute_t": np.zeros((n, NUM_POINTS), np.float32), "refute_w": np.zeros((n, NUM_POINTS), np.float32),
         "split": np.zeros(n, np.int8)}
    q = _core.str_to_point("C3")
    a["feats"][:, 0, q] = 1
    a["refute_t"][:, q] = 1
    a["refute_w"][:, q] = 1
    a["policy"][:, q] = 1
    d = ReanalysisData.__new__(ReanalysisData)
    d.a, d.n = a, n
    for t in range(8):
        b = d.batch(np.array([0, 1]), np.array([t, t]), "cpu")
        stone = b["x"][0, 0].reshape(-1).argmax().item()
        assert b["refute_t"][0].argmax().item() == stone and b["refute_w"][0].argmax().item() == stone
        assert b["policy"][0, :NUM_POINTS].argmax().item() == stone
        assert stone == int(np.flatnonzero(GATHER[t] == q)[0])


def test_move_label_codes():
    g = _core.GameState("RENJU_GOMOCUP")
    for m in ["H8", "A1", "I8", "A3", "J8"]:  # black open three, white to move
        assert g.apply(m)
    assert _analyze_move(g, "A5")["code"] == "PROVEN_UNSAFE"   # ignores the three: black wins by force
    assert _analyze_move(g, "K8")["code"] in ("UNREFUTED_EXHAUSTED", "UNKNOWN_BUDGET")  # never claimed safe
    h = _core.GameState("RENJU_GOMOCUP")
    for m in ["H8", "A1", "I8", "A3", "J8", "A5", "K8", "A7"]:
        assert h.apply(m)
    assert _analyze_move(h, "L8")["code"] == "PROVEN_WINNING"  # black five
    rec = json.loads(json.dumps(_analyze_move(g, "A5")))
    assert rec["code"] in CODE and rec["depth"] >= 1
