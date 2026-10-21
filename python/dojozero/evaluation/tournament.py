"""TournamentRunner (PRD FR-22, EVALUATION §13-§17, ARCHITECTURE §3.12).

    python -m dojozero.evaluation.tournament --engine-a dojo:runs/taraguchi_v1/models/gen_001.pt \
        --engine-b ext:rapfi_2025 --openings renju_development_v1 --pairs 100 --time-ms 1000 --tier B

Player specs:
  dojo:<ckpt>[:sims=N]       DojoZero in-process (PUCT root, no noise). With --time-ms the search is
                             time-limited (sims = cap); without it, fixed visits.
  heuristic | random         internal bots (in-process)
  ext:<engine_id>            registered external engine (external_engines/, sha256-pinned)
  pbrain-py:<think>          the Python-hosted DojoZero protocol engine as a separate process

Standard Renju only (profile RENJU_GOMOCUP: RIF rules, no pass): Gomocup engines do not support
Taraguchi-10 actions (EVALUATION §15.1). Each opening is played twice; player A takes the side to
move after the opening in game 1 and the other side in game 2 (pair = statistics unit, §4.1).

Referee: every move goes through the canonical GameState. A rejected move ends the game as
FORBIDDEN_MOVE (black forbidden point) or ILLEGAL_MOVE; process failures end it as CRASH /
TIMEOUT / MALFORMED_RESPONSE / PROTOCOL_ERROR / START_FAILED. The failing side loses; failures are
counted separately from normal losses. A DojoZero-side failure raises (it is a bug and blocks
promotion, EVALUATION §1-5).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import _core
from . import corpus as corpus_mod
from .external import engine_config, load_meta, local_path, pbrain_py_config
from .match import make_player, summarize
from .strength_db import StrengthDB

PROFILE = "RENJU_GOMOCUP"
REPO = Path(__file__).resolve().parents[3]


def disable_power_throttling() -> bool:
    """Opt this process out of Windows power throttling (EcoQoS). A windowless background process may
    otherwise run on slow cores: a solver benchmark ran ~8x slower that way (PROGRESS Phase 8 E3b).
    Process-local; no system setting is changed. Engines started by the C++ launcher are opted out
    there. Returns True when the call succeeded."""
    if sys.platform != "win32":
        return False
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    st = _State(1, 0x1, 0)  # PROCESS_POWER_THROTTLING_CURRENT_VERSION, EXECUTION_SPEED, off
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    return bool(k32.SetProcessInformation(ctypes.c_void_p(k32.GetCurrentProcess()), 4,  # ProcessPowerThrottling
                                          ctypes.byref(st), ctypes.sizeof(st)))


class keep_awake:
    """Hold a per-process 'system required' request on Windows while a long, time-controlled
    evaluation runs (an idle sleep once stretched a round robin by 3 hours, PROGRESS Phase 8), and
    opt the process out of power throttling (disable_power_throttling).
    Released on exit; no power setting is changed; a closed lid still sleeps."""

    def __enter__(self):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
            self.unthrottled = disable_power_throttling()
        return self

    def __exit__(self, *exc):
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return False


class DojoZeroError(RuntimeError):
    """DojoZero itself produced an illegal move or crashed: a bug, never scored as a game."""


# ---------------------------------------------------------------------------------------------
def warm_up(player) -> None:
    """First network calls in a process are slow (XPU kernel set-up); run them before timing moves."""
    from ..search.driver import search_many
    s = player.settings
    saved = (s.simulations, s.max_ms)
    s.max_ms = 0
    for sims in (16, 64):
        s.simulations = sims
        search_many([_core.GameState(PROFILE)], player.evaluator, s, player.feature_config)
    s.simulations, s.max_ms = saved


class InProcessPlayer:
    """DojoZero search or an internal bot, called directly."""

    def __init__(self, spec: str, device: str, cache: dict, time_ms: float | None, seed: int):
        self.spec = spec
        self.time_ms = time_ms
        self.seed = seed
        self.counter = 0
        if spec in ("heuristic", "random"):
            self.kind, self.player = spec, None
            self.engine_id, self.engine_kind = spec, "internal_bot"
            return
        if not spec.startswith("dojo:"):
            raise ValueError(f"not an in-process spec: {spec}")
        self.kind = "dojo"
        self.player = make_player("engine:" + spec[len("dojo:"):], device, cache)
        s = self.player.settings
        s.root, s.gumbel_noise = "PUCT", False
        if time_ms:
            s.simulations = max(s.simulations, 100000)  # cap only
            s.max_ms = float(time_ms)
        self.engine_kind = "dojo"
        warm_up(self.player)
        from ..model.load import model_sha
        self.model_hash = model_sha(self.player.evaluator.model)
        # Root defense filter: "-def{k}" = filter with the ee1d011 solver (E3a ids); the optimized solver
        # (Phase 8 E3b) adds "-fast" and, with the fixed ee1d011 time split, "-nocarry".
        self.engine_id = f"dojo-{self.model_hash[:12]}" + (
            f"-def{s.defense_k}" + ("" if s.defense_impl == "ref" else "-fast" + ("" if s.defense_carry else "-nocarry"))
            if s.defense_k else "")

    def start_game(self):
        return None

    def play(self, g, moves: list[int]) -> dict:
        self.counter += 1
        t0 = time.perf_counter()
        if self.kind == "heuristic":
            a = _core.heuristic_action(g, self.seed * 100_003 + self.counter)
            stats = {}
        elif self.kind == "random":
            a = _core.random_action(g, self.seed * 100_003 + self.counter)
            stats = {}
        else:
            from ..search.driver import search_many
            self.player.settings.seed = self.seed * 100_003 + self.counter
            evals0 = self.player.evaluator.evaluations
            r = search_many([g], self.player.evaluator, self.player.settings, self.player.feature_config)[0]
            a = r["action"]
            stats = {"sims": r["simulations"], "nodes": r["nodes"],
                     "nn_evals": self.player.evaluator.evaluations - evals0}
            d = r.get("defense")
            if d is not None:  # per-move filter statistics (means over moves in the game record)
                checks = [c for c in d.get("checks", []) if c.get("result") not in ("ILLEGAL", "TERMINAL")]
                stats |= {"def_ms": d.get("ms", 0.0), "def_checked": d.get("checked", 0),
                          "def_refuted": len(d.get("refuted", [])), "def_changed": float(d.get("changed", False)),
                          "def_solver_calls": len(checks),
                          "def_win": sum(c["result"] == "WIN" for c in checks),
                          "def_unknown": sum(c["result"] == "UNKNOWN" for c in checks),
                          "def_nodes": sum(c.get("nodes", 0) for c in checks),
                          "def_solver_ms": sum(c.get("ms", 0.0) for c in checks)}
        ms = (time.perf_counter() - t0) * 1000
        if a == "pass":
            raise DojoZeroError("in-process player chose PASS in a no-pass profile")
        return {"move": _core.str_to_point(a), "failure": "NONE", "elapsed_ms": ms, "detail": "", "stats": stats}

    def end_game(self, final: bool = False):
        pass

    def describe(self) -> dict:
        if self.kind != "dojo":
            return {"kind": self.kind}
        s = self.player.settings
        return {"kind": "dojo", "spec": self.spec, "root": s.root, "simulations": s.simulations, "max_ms": s.max_ms,
                "root_tactics_nodes": s.root_tactics_nodes, "defense_k": s.defense_k,
                "defense_share": s.defense_share, "defense_threads": s.defense_threads,
                "defense_impl": s.defense_impl, "defense_carry": s.defense_carry, "model_hash": self.model_hash}


class ProcessPlayer:
    """A Gomocup engine in its own process (fresh process per game)."""

    def __init__(self, engine_id: str, config, meta: dict | None = None, kind: str = "external",
                 reuse_process: bool = False):
        self.engine_id = engine_id
        self.config = config
        self.meta = meta or {}
        self.engine_kind = kind
        self.reuse_process = reuse_process  # RESTART between games instead of a new process
        self.engine = None
        self.synced = 0  # number of moves the engine has seen

    def start_game(self):
        self.synced = 0
        if self.reuse_process and self.engine is not None and self.engine.alive():
            failure, detail = self.engine.restart()
            if failure == "NONE":
                return None
            self.engine.end()  # fall back to a fresh process
        self.engine = _core.ExternalEngine(self.config)
        failure, detail = self.engine.start()
        return None if failure == "NONE" else (failure, detail)

    def play(self, g, moves: list[int]) -> dict:
        if self.synced == 0 and not moves:
            r = self.engine.begin()
        elif self.synced == 0 or self.synced != len(moves) - 1:
            r = self.engine.board(moves)
        else:
            r = self.engine.turn(moves[-1])
        if r["failure"] == "NONE":
            self.synced = len(moves) + 1
        return {"move": r["move"], "failure": r["failure"], "elapsed_ms": r["elapsed_ms"],
                "detail": r["detail"], "stats": {"messages": len(r["messages"])}, "xy": (r["x"], r["y"])}

    def end_game(self, final: bool = False):
        if self.engine is not None and (final or not self.reuse_process or not self.engine.alive()):
            self.engine.end()
            self.engine = None

    def describe(self) -> dict:
        return {"kind": self.engine_kind, "engine_id": self.engine_id, "path": self.config.path,
                "timeout_turn_ms": self.config.timeout_turn_ms, "rule": self.config.rule,
                "reuse_process": self.reuse_process}


def make_tournament_player(spec: str, device: str, cache: dict, time_ms: float | None, seed: int,
                           engine_dir: Path | None = None):
    if spec.startswith("ext:"):
        eid = spec[4:]
        kw = {"engine_dir": engine_dir} if engine_dir else {}
        meta = load_meta(eid, **kw)
        cfg = engine_config(meta, local_path(eid, **kw), timeout_turn_ms=int(time_ms) if time_ms else None)
        return ProcessPlayer(eid, cfg, meta)
    if spec.startswith("pbrain-py:"):
        think = spec[len("pbrain-py:"):]
        cfg = pbrain_py_config(think, int(time_ms or 1000), device=device)
        # Python + torch start-up and network warm-up take seconds: keep one process per match.
        return ProcessPlayer(f"pbrain-py-{think.split(':')[0]}", cfg, reuse_process=True,
                             kind="dojo" if think.startswith("engine:") else "internal_bot")
    return InProcessPlayer(spec, device, cache, time_ms, seed)


# ---------------------------------------------------------------------------------------------
def classify_rejected(g, p: int | None) -> str:
    if p is None:
        return "MALFORMED_RESPONSE"
    if np.asarray(g.board())[p] != 0:
        return "ILLEGAL_MOVE"
    if g.side_to_move == "BLACK" and g.black_verdict(p).startswith("FORBIDDEN"):
        return "FORBIDDEN_MOVE"
    return "ILLEGAL_MOVE"


def play_game(a, b, opening: dict, opening_set, a_first: bool, pair_id: int) -> dict:
    """One game. a_first: A plays the side to move after the opening."""
    g = opening_set.start_state(opening)
    moves = [_core.str_to_point(m) for m in opening["moves"]]
    first = g.side_to_move
    a_color = first if a_first else ("WHITE" if first == "BLACK" else "BLACK")
    players = {a_color: a, ("WHITE" if a_color == "BLACK" else "BLACK"): b}
    elapsed = {"A": 0.0, "B": 0.0}
    stats = {"A": [], "B": []}
    rec = {"pair_id": pair_id, "opening_id": opening["opening_id"], "a_first": a_first, "a_color": a_color,
           "termination_reason": None, "failed_side": None, "detail": ""}
    started = []
    try:
        for side, pl in (("A", a), ("B", b)):
            f = pl.start_game()
            started.append(pl)
            if f:
                rec.update(termination_reason=f[0] if f[0] != "NONE" else "START_FAILED", failed_side=side,
                           detail=f[1])
                break
        while rec["termination_reason"] is None and not g.is_terminal:
            color = g.side_to_move
            pl = players[color]
            side = "A" if pl is a else "B"
            r = pl.play(g, moves)
            elapsed[side] += r["elapsed_ms"]
            if r.get("stats"):
                stats[side].append(r["stats"])
            if r["failure"] != "NONE":
                if pl.engine_kind == "dojo" and isinstance(pl, InProcessPlayer):
                    raise DojoZeroError(r["detail"])
                rec.update(termination_reason=r["failure"], failed_side=side, detail=r["detail"])
                break
            p = r["move"]
            if p is None or not g.apply(_core.point_to_str(p)):
                reason = classify_rejected(g, p)
                if isinstance(pl, InProcessPlayer):
                    raise DojoZeroError(f"{pl.spec} produced {reason} at {p}")
                rec.update(termination_reason=reason, failed_side=side,
                           detail=f"move {r.get('xy')} -> {_core.point_to_str(p) if p is not None else None}")
                break
            moves.append(p)
    finally:
        failed = rec["termination_reason"] is not None
        for pl in started:
            pl.end_game(final=failed)  # never reuse a process after a failure in the game
    if rec["termination_reason"] is None:
        rec["termination_reason"] = "DRAW" if g.result == "DRAW" else "NORMAL_WIN"
        winner = {"BLACK_WIN": "BLACK", "WHITE_WIN": "WHITE"}.get(g.result)
        rec["result"] = g.result
    else:  # the failing side loses
        failed_color = a_color if rec["failed_side"] == "A" else ("WHITE" if a_color == "BLACK" else "BLACK")
        winner = "WHITE" if failed_color == "BLACK" else "BLACK"
        rec["result"] = f"{winner}_WIN"
    rec["winner"] = winner
    rec["score_a"] = 0.5 if winner is None else float(winner == a_color)
    rec["moves"] = [_core.point_to_str(p) for p in moves]
    rec["plies"] = len(moves)
    rec["elapsed_a_ms"], rec["elapsed_b_ms"] = elapsed["A"], elapsed["B"]
    rec["search_stats"] = {k: _mean_stats(v) for k, v in stats.items()}
    return rec


def _mean_stats(rows: list[dict]) -> dict:
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r.get(k, 0) for r in rows])) for k in sorted(keys)} | {"moves": len(rows)}


def hardware_id() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        import torch
        gpu = torch.xpu.get_device_name(0) if hasattr(torch, "xpu") and torch.xpu.is_available() else "none"
    except Exception:  # noqa: BLE001 - metadata only
        gpu = "unknown"
    return f"{platform.system()} {platform.release()} | {cpu} | gpu={gpu}"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def run_match(a, b, opening_set, pairs: int, db: StrengthDB | None, *, tier: str, time_ms: float | None,
              command: str = "", verbose: bool = True) -> dict:
    if pairs > len(opening_set.openings):
        raise ValueError(f"{opening_set.set_id} has only {len(opening_set.openings)} openings (asked {pairs} pairs)")
    compute_mode = "fixed_time" if time_ms else "fixed_visits"
    budget = {"time_ms_per_move": time_ms, "a": a.describe(), "b": b.describe()}
    match_id = None
    if db is not None:
        for pl in (a, b):
            db.register_engine(pl.engine_id, pl.engine_kind, name=pl.engine_id,
                               model_hash=getattr(pl, "model_hash", ""), config=pl.describe(),
                               executable_sha256=(pl.meta.get("executable_sha256", "") if isinstance(pl, ProcessPlayer) else ""),
                               version=(pl.meta.get("version", "") if isinstance(pl, ProcessPlayer) else ""),
                               git_commit=git_commit())
        match_id = db.start_match(a.engine_id, b.engine_id, tier=tier, rule_profile=PROFILE,
                                  profile_hash=str(_core.profile_hash(PROFILE)), opening_set_id=opening_set.set_id,
                                  opening_set_hash=opening_set.set_hash, compute_mode=compute_mode, budget=budget,
                                  hardware_id=hardware_id(), concurrent_games=1, git_commit=git_commit(),
                                  command=command)
    t0 = time.time()
    games = []
    for k, op in enumerate(opening_set.openings[:pairs]):
        for a_first in (True, False):
            rec = play_game(a, b, op, opening_set, a_first, k)
            games.append(rec)
            if db is not None:
                db.add_game(match_id, rec)
        if verbose:
            done = len(games) // 2
            print(f"  pair {done}/{pairs}: score {np.mean([x['score_a'] for x in games]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    summary = summarize([{"opening": r["pair_id"], "score_a": r["score_a"], "a_color": r["a_color"],
                          "plies": r["plies"]} for r in games])
    failures = {}
    for r in games:
        if r["termination_reason"] not in ("NORMAL_WIN", "DRAW"):
            key = f"{r['failed_side']}:{r['termination_reason']}"
            failures[key] = failures.get(key, 0) + 1
    compute = {"wall_s": time.time() - t0, "mean_ms_per_move_a": _per_move(games, "a"),
               "mean_ms_per_move_b": _per_move(games, "b"), "search_a": _mean_side(games, "A"),
               "search_b": _mean_side(games, "B"), "concurrent_games": 1}
    for pl in (a, b):
        pl.end_game(final=True)
    if db is not None:
        db.finish_match(match_id, compute)
    return {"match_id": match_id, "summary": summary, "failures": failures, "compute": compute,
            "opening_set": {"id": opening_set.set_id, "hash": opening_set.set_hash}, "compute_mode": compute_mode,
            "games": games}


def _per_move(games: list[dict], side: str) -> float:
    ms = sum(g[f"elapsed_{side}_ms"] for g in games)
    n = sum(g["search_stats"][side.upper()].get("moves", 0) for g in games) or 1
    return ms / n


def _mean_side(games: list[dict], side: str) -> dict:
    rows = [g["search_stats"][side] for g in games if g["search_stats"][side].get("moves")]
    keys = {k for r in rows for k in r if k != "moves"}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--openings", required=True, help="opening set id (benchmarks/openings/<id>.json) or path")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--time-ms", type=float, help="per-move time for both sides (fixed_time); omit = fixed visits")
    ap.add_argument("--tier", choices=("A", "B", "C"), default="A")
    ap.add_argument("--device", default="xpu" if _xpu() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--db", default=str(REPO / "runs" / "evaluation" / "results.sqlite"))
    ap.add_argument("--out", help="JSON summary path")
    args = ap.parse_args(argv)
    s = corpus_mod.load(args.openings)
    if s.status.startswith("PROVISIONAL") and args.tier != "A":
        raise SystemExit(f"{s.set_id} is provisional ({s.status}); use it for Tier A only")
    cache: dict = {}
    a = make_tournament_player(args.engine_a, args.device, cache, args.time_ms, args.seed)
    b = make_tournament_player(args.engine_b, args.device, cache, args.time_ms, args.seed + 1)
    db = StrengthDB(args.db)
    with keep_awake():
        res = run_match(a, b, s, args.pairs, db, tier=args.tier, time_ms=args.time_ms, command=" ".join(sys.argv))
    out = {k: v for k, v in res.items() if k != "games"} | {"a": args.engine_a, "b": args.engine_b}
    print(json.dumps(out, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


def _xpu() -> bool:
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
