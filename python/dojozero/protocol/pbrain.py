"""Piskvork / Gomocup engine hosted in Python (PRD FR-21b, ARCHITECTURE §3.12).

    python -m dojozero.protocol.pbrain --think heuristic
    python -m dojozero.protocol.pbrain --think engine:runs/.../gen_001.pt --device xpu

The protocol codec and engine-side state machine are the C++ `EngineSession` (no second codec in
Python); this module only supplies the move function and the stdin/stdout loop. It is the interim
DojoZero protocol engine until `pbrain-DojoZero.exe` is decided (ADR-08), and with `--fault` it is
the misbehaving mock engine for TournamentRunner tests.

Time: per-move budget = timeout_turn (or time_left / 25 when a match clock is given, whichever is
smaller, and at most --max-move-ms) minus a safety margin; the search uses a PUCT root with a
wall-clock limit.

GUI-only options (pbrain-DojoZero.exe passes them from its ini; evaluation paths never set them):
  --rule-dialect yixin  Yixin Board sends `INFO rule 2` for Renju. In this dialect the host hands the
                        session `INFO rule 4` instead; nothing else changes (the session and the
                        canonical GameState still judge every move). Default `gomocup`: rule 2 is a
                        continuous game and stays rejected.
  --opening-guard       usability guard, not strength: taraguchi_v1 never saw a free opening and plays
                        edge moves there (PROGRESS Phase 8 GUI launcher). See `opening_guard`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .. import _core

ABOUT = 'name="DojoZero", version="0.8-dev", author="DojoZero project", country="-"'


def move_budget_ms(limits: dict, default_ms: int = 1000, max_ms: float = 0) -> float:
    turn = limits.get("timeout_turn_ms") or 0
    left = limits.get("time_left_ms", -1)
    budget = float(turn) if turn > 0 else float(default_ms)
    if left is not None and left >= 0:
        budget = min(budget, left / 25.0)
    if max_ms > 0:
        budget = min(budget, max_ms)
    return max(10.0, budget * 0.85 - 30.0)  # margin for process I/O and network warm-up


# Opening guard. In training the network reaches NORMAL_RENJU only after the six Taraguchi-10 opening
# moves, whose first four stones are confined to the central 1/3/5/7 squares (RULES §8). A free
# Gomocup game with fewer than six stones is therefore outside what it has seen, and it answers with
# edge moves (after H8 it answered 2,0). Below that count the guard plays H8 on an empty board and
# otherwise keeps the search's preference only among points within GUARD_RADIUS (Chebyshev) of a
# stone, which is where every opening stone of the training distribution lies.
GUARD_STONES = 6
GUARD_RADIUS = 2
CENTER = "H8"


def opening_guard(state, result: dict) -> str:
    """Move to play under the opening guard, given the search result for `state` (`action` and
    `children` with visits/prior). From GUARD_STONES stones on, the search's action is returned."""
    board = state.board()
    stones = [p for p in range(len(board)) if board[p] != 0]
    if len(stones) >= GUARD_STONES:
        return result["action"]
    legal = set(state.legal_actions())
    if not stones and CENTER in legal:
        return CENTER
    xy = [_core.point_to_gomocup(p) for p in stones]

    def near(a: str) -> bool:
        if a not in legal or a == "pass":
            return False
        x, y = _core.point_to_gomocup(_core.str_to_point(a))
        return any(max(abs(x - sx), abs(y - sy)) <= GUARD_RADIUS for sx, sy in xy)

    if near(result["action"]):
        return result["action"]
    kids = [c for c in result.get("children", []) if near(c["action"])]
    if kids:
        return max(kids, key=lambda c: (c["visits"], c["prior"]))["action"]
    # No ranked candidate near the stones (cannot happen with a full root expansion): nearest to H8.
    cx, cy = _core.point_to_gomocup(_core.str_to_point(CENTER))
    area = [a for a in legal if near(a)]
    if not area:
        return result["action"]

    def dist(a: str):
        x, y = _core.point_to_gomocup(_core.str_to_point(a))
        return (max(abs(x - cx), abs(y - cy)), a)
    return min(area, key=dist)


def make_think(spec: str, device: str, seed: int, guard: bool = False, max_move_ms: float = 0):
    if spec == "heuristic":
        counter = [0]

        def think(state, limits):
            counter[0] += 1
            return _core.str_to_point(_core.heuristic_action(state, seed * 1000 + counter[0]))
        return think
    if spec.startswith("engine:"):
        from ..evaluation.match import make_player
        from ..search.driver import search_many
        player = make_player(spec, device, {})
        player.settings.root = "PUCT"
        player.settings.gumbel_noise = False
        player.settings.simulations = max(player.settings.simulations, 100000)  # cap; time decides
        # Warm-up before serving: the first network calls in a fresh process (XPU kernel set-up)
        # take far longer than a move budget (PROGRESS Phase 8: timeouts without this).
        from ..evaluation.tournament import warm_up
        warm_up(player)

        def think(state, limits):
            player.settings.max_ms = move_budget_ms(limits, max_ms=max_move_ms)
            r = search_many([state], player.evaluator, player.settings, player.feature_config)[0]
            return _core.str_to_point(opening_guard(state, r) if guard else r["action"])
        return think
    raise ValueError(f"unknown think spec {spec}")


def dialect_line(line: str, dialect: str) -> str:
    """Rewrite a GUI dialect command into its Gomocup form (only Yixin Board's Renju rule number)."""
    if dialect == "yixin":
        parts = line.split()
        if len(parts) == 3 and parts[0].upper() == "INFO" and parts[1].lower() == "rule" and parts[2] == "2":
            return "INFO rule 4"
    return line


def serve(think, fault: str | None = None, fault_after: int = 0, inp=None, out=None,
          dialect: str = "gomocup") -> int:
    """Run the protocol loop until END or EOF. Fault injection (tests only): after `fault_after`
    own moves, the next move request triggers the fault instead of an answer."""
    inp = inp or sys.stdin
    out = out or sys.stdout
    session = _core.EngineSession(think, ABOUT)
    moves = 0
    if fault == "start-crash":
        os._exit(3)

    def emit(lines):
        for line in lines:
            out.write(line + "\n")
        out.flush()

    for raw in inp:
        line = dialect_line(raw.strip(), dialect)
        if not line:
            continue
        head = line.split(" ")[0].upper()
        wants_move = head in ("BEGIN", "TURN") or (head == "DONE")
        if fault == "slow-first" and wants_move and moves == 0:
            time.sleep(1.5)  # lazy initialisation on the first search (e.g. AlphaGomoku 2026)
        elif fault and fault != "slow-first" and wants_move and moves >= fault_after:
            if fault == "crash":
                os._exit(3)
            if fault == "timeout":
                time.sleep(3600)
            if fault == "malformed":
                emit(["I think 7,7 is nice"])
                continue
            if fault == "error":
                emit(["ERROR something went wrong"])
                continue
            if fault == "illegal":
                session.feed(line)  # keep the state in sync, then answer an occupied point
                st = session.state
                occupied = next(p for p in range(225) if st.board()[p] != 0)
                x, y = _core.point_to_gomocup(occupied)
                emit([f"{x},{y}"])
                continue
        replies = session.feed(line)
        if wants_move and replies and "," in replies[0] and not replies[0].startswith("ERROR"):
            moves += 1
        emit(replies)
        if session.quit:
            return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", default="heuristic")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fault", choices=("crash", "timeout", "malformed", "error", "illegal", "start-crash", "slow-first"))
    ap.add_argument("--fault-after", type=int, default=0)
    ap.add_argument("--rule-dialect", choices=("gomocup", "yixin"), default="gomocup")
    ap.add_argument("--opening-guard", action="store_true", help="GUI usability guard (see module doc)")
    ap.add_argument("--max-move-ms", type=float, default=0, help="cap on the per-move budget (0 = none)")
    args = ap.parse_args(argv)
    if args.think.startswith("engine:"):
        # Under pbrain-DojoZero.exe the venv python.exe is a redirector: the launcher's EcoQoS opt-out
        # does not reach this interpreter, so opt out here as well.
        from ..evaluation.tournament import disable_power_throttling
        disable_power_throttling()
    think = make_think(args.think, args.device, args.seed, args.opening_guard, args.max_move_ms)
    return serve(think, args.fault, args.fault_after, dialect=args.rule_dialect)


if __name__ == "__main__":
    raise SystemExit(main())
