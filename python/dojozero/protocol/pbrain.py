"""Piskvork / Gomocup engine hosted in Python (PRD FR-21b, ARCHITECTURE §3.12).

    python -m dojozero.protocol.pbrain --think heuristic
    python -m dojozero.protocol.pbrain --think engine:runs/.../gen_001.pt --device xpu

The protocol codec and engine-side state machine are the C++ `EngineSession` (no second codec in
Python); this module only supplies the move function and the stdin/stdout loop. It is the interim
DojoZero protocol engine until `pbrain-DojoZero.exe` is decided (ADR-08), and with `--fault` it is
the misbehaving mock engine for TournamentRunner tests.

Time: per-move budget = timeout_turn (or time_left / 25 when a match clock is given, whichever is
smaller) minus a safety margin; the search uses a PUCT root with a wall-clock limit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .. import _core

ABOUT = 'name="DojoZero", version="0.8-dev", author="DojoZero project", country="-"'


def move_budget_ms(limits: dict, default_ms: int = 1000) -> float:
    turn = limits.get("timeout_turn_ms") or 0
    left = limits.get("time_left_ms", -1)
    budget = float(turn) if turn > 0 else float(default_ms)
    if left is not None and left >= 0:
        budget = min(budget, left / 25.0)
    return max(10.0, budget * 0.85 - 30.0)  # margin for process I/O and network warm-up


def make_think(spec: str, device: str, seed: int):
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
            player.settings.max_ms = move_budget_ms(limits)
            r = search_many([state], player.evaluator, player.settings, player.feature_config)[0]
            return _core.str_to_point(r["action"])
        return think
    raise ValueError(f"unknown think spec {spec}")


def serve(think, fault: str | None = None, fault_after: int = 0, inp=None, out=None) -> int:
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
        line = raw.strip()
        if not line:
            continue
        head = line.split(" ")[0].upper()
        wants_move = head in ("BEGIN", "TURN") or (head == "DONE")
        if fault and wants_move and moves >= fault_after:
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
    ap.add_argument("--fault", choices=("crash", "timeout", "malformed", "error", "illegal", "start-crash"))
    ap.add_argument("--fault-after", type=int, default=0)
    args = ap.parse_args(argv)
    return serve(make_think(args.think, args.device, args.seed), args.fault, args.fault_after)


if __name__ == "__main__":
    raise SystemExit(main())
