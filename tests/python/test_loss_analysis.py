"""Loss classifier (EVALUATION §21.1) on constructed games: categories follow the solver proofs."""

from dojozero.evaluation.loss_analysis import NetProbe, classify_game

BUDGET = {"vcf": 100000, "vct": 50000, "loss": 20000, "ms": 2000}
NO_NET = NetProbe(None)


def test_ignoring_an_open_three_is_a_tactical_blunder():
    # Black H8 I8 J8 is an open three; White (DojoZero) ignores it with A5, Black makes an open four.
    moves = ["H8", "A1", "I8", "A3", "J8", "A5", "K8", "A7", "L8"]
    r = classify_game(moves, 5, "WHITE", NO_NET, BUDGET)
    assert r["category"] == "tactical_blunder", r
    assert r["primary"]["ply"] == 5 and r["primary"]["move"] == "A5" and r["primary"]["after"] == "WIN"


def test_skipping_a_winning_open_four_is_a_missed_win():
    # Black (DojoZero) to move with an open three could make an open four (K8 or G8) but plays O15;
    # White blocks K8 and the win is gone at Black's next turn.
    moves = ["H8", "A1", "I8", "A3", "J8", "A5", "O15", "K8", "B10", "G8"]
    r = classify_game(moves, 6, "BLACK", NO_NET, BUDGET)
    assert r["events"][0]["event"] == "missed_win", r["events"]
    assert r["events"][0]["solver_move"] in ("K8", "G8")


def test_playing_the_win_is_not_an_event():
    moves = ["H8", "A1", "I8", "A3", "J8", "A5", "K8", "A7", "G8"]  # Black converts: open four, five next
    r = classify_game(moves, 6, "BLACK", NO_NET, BUDGET)
    assert r["events"] == [] and r["category"] == "unclassified"


def test_summary_is_json_serialisable():
    import json
    from dojozero.evaluation.loss_analysis import summarize
    moves = ["H8", "A1", "I8", "A3", "J8", "A5", "K8", "A7", "L8"]
    r = classify_game(moves, 5, "WHITE", NO_NET, BUDGET) | {"dojo_color": "WHITE"}
    json.dumps(summarize([r]))  # regression: tuple keys made the first full run fail at the end
