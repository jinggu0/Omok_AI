"""GUI-only options of the Python pbrain host (pbrain-DojoZero.exe): Yixin Board rule dialect and the
opening guard. Both are off by default, so evaluation paths are unchanged."""

import io

from dojozero import _core
from dojozero.protocol.pbrain import GUARD_STONES, make_think, opening_guard, serve


def run(lines, dialect="gomocup"):
    out = io.StringIO()
    serve(make_think("heuristic", "cpu", 1), inp=io.StringIO("".join(l + "\n" for l in lines)), out=out,
          dialect=dialect)
    return out.getvalue().splitlines()


def is_move(reply):
    x, _, y = reply.partition(",")
    return x.isdigit() and y.isdigit()


def test_gomocup_dialect_keeps_rule_2_as_unsupported_continuous_game():
    replies = run(["START 15", "INFO rule 2", "BEGIN", "END"])
    assert replies[0] == "OK"
    assert replies[1].startswith("ERROR") and "Renju" in replies[1]


def test_both_dialects_play_renju_with_rule_4():
    for dialect in ("gomocup", "yixin"):
        replies = run(["START 15", "INFO rule 4", "BEGIN", "END"], dialect)
        assert replies[0] == "OK" and is_move(replies[1])


def test_yixin_dialect_maps_rule_2_to_renju_and_nothing_else():
    # The command sequence Yixin Board sends (recorded 2026-10-21), lower-case start/board/done included.
    replies = run(["info show_detail 3", "yxshowinfo", "START 15", "INFO timeout_turn 2000", "INFO rule 2",
                   "yxquerydatabaseallt", "done", "start 15", "board", "done", "END"], "yixin")
    moves = [r for r in replies if is_move(r)]
    assert len(moves) == 1 and not any(r.startswith("ERROR") for r in replies)
    # Freestyle / standard gomoku stay rejected in the Yixin dialect.
    for rule in ("0", "1"):
        replies = run(["START 15", f"INFO rule {rule}", "BEGIN", "END"], "yixin")
        assert replies[1].startswith("ERROR")


def test_yixin_rule_2_position_is_still_judged_by_the_canonical_rules():
    # Opponent move on an occupied point is rejected by the session/GameState as before.
    replies = run(["START 15", "INFO rule 2", "TURN 7,7", "TURN 7,7", "END"], "yixin")
    assert is_move(replies[1]) and replies[2].startswith("ERROR")


def state_with(*moves):
    g = _core.GameState("RENJU_GOMOCUP")
    for m in moves:
        assert g.apply(m)
    return g


def kid(action, visits, prior=0.0):
    return {"action": action, "visits": visits, "prior": prior}


def test_guard_plays_center_on_an_empty_board():
    assert opening_guard(state_with(), {"action": "B2", "children": [kid("B2", 50)]}) == "H8"


def test_guard_keeps_a_search_move_near_the_stones():
    assert opening_guard(state_with("H8"), {"action": "I9", "children": []}) == "I9"


def test_guard_replaces_an_edge_move_with_the_most_visited_move_near_the_stones():
    r = {"action": "C1", "children": [kid("C1", 40), kid("J10", 5), kid("G7", 9), kid("H12", 30)]}
    assert opening_guard(state_with("H8"), r) == "G7"  # H12 is 4 rows away: outside the radius


def test_guard_uses_prior_when_nothing_near_was_visited():
    r = {"action": "A1", "children": [kid("A1", 10), kid("I8", 0, 0.01), kid("H9", 0, 0.2)]}
    assert opening_guard(state_with("H8", "A15"), r) == "H9"


def test_guard_falls_back_to_the_point_nearest_the_center():
    assert opening_guard(state_with("H8"), {"action": "O15", "children": []}) in {"G7", "H7", "I7", "G8", "I8",
                                                                                     "G9", "H9", "I9"}


def test_guard_is_off_from_six_stones():
    g = state_with("H8", "I9", "J8", "H9", "I8", "G8")
    assert sum(1 for v in g.board() if v) == GUARD_STONES
    assert opening_guard(g, {"action": "A1", "children": [kid("A1", 1)]}) == "A1"
