"""Capacity probe (PROGRESS Phase 8 E5r): a random-init checkpoint of another trunk size loads with that size."""
from pathlib import Path

import pytest

from dojozero.model.load import load_model
from dojozero.tools import capacity_probe as cp

REF = cp.REPO / cp.REF_CKPT


@pytest.mark.skipif(not REF.exists(), reason="reference checkpoint not present")
def test_init_checkpoint_loads_with_requested_trunk(tmp_path: Path):
    out = tmp_path / "init_b2c32.pt"
    assert cp.main(["init", "--blocks", "2", "--channels", "32", "--out", str(out)]) == 0
    model, spec, _ = load_model(str(out), "cpu")
    assert (model.cfg.blocks, model.cfg.channels, model.cfg.tactical_heads) == (2, 32, True)
    ref, ref_spec, _ = load_model(str(REF), "cpu")
    assert spec.hash() == ref_spec.hash()
    assert cp.n_params(model) < cp.n_params(ref)
