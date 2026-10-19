"""A/B decision rule (EVALUATION §19): INCONCLUSIVE is never reported as success."""

from dojozero.evaluation.ab import decide, paired_stats


def test_paired_stats_and_decisions():
    up = paired_stats([0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5] * 5)
    assert up["mean_diff"] > 0 and up["ci95"][0] > 0
    assert decide(up, 0) == "ACCEPT"
    assert decide(up, 1) == "REJECT"  # any candidate failure blocks adoption
    down = paired_stats([-0.5, -0.5, 0.0, -0.5] * 10)
    assert decide(down, 0) == "REJECT"
    flat = paired_stats([0.5, -0.5, 0.0, 0.5, -0.5, 0.0] * 5)
    assert flat["ci95"][0] < 0 < flat["ci95"][1] and decide(flat, 0) == "INCONCLUSIVE"
