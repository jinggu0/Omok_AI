"""Strength Database (PRD FR-24, EVALUATION §17): every evaluation match accumulates here.

    runs/evaluation/results.sqlite   (WAL, schema_version table)

Rating names follow EVALUATION §16: `dojo_external_elo`, `dojo_internal_elo`,
`dojo_taraguchi_elo`; published ratings (`gomocup_reference_elo`) live only in engine metadata.
Re-analysis results go to `analysis` and never into `game` (EVALUATION §21).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = 1
RATING_SYSTEMS = ("dojo_external_elo", "dojo_internal_elo", "dojo_taraguchi_elo")
TERMINATIONS = ("NORMAL_WIN", "DRAW", "FORBIDDEN_MOVE", "ILLEGAL_MOVE", "CRASH", "TIMEOUT", "MALFORMED_RESPONSE",
                "PROTOCOL_ERROR", "START_FAILED", "ADJUDICATED", "MAX_PLIES")
DECISIONS = ("ACCEPT", "REJECT", "INCONCLUSIVE")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS engine (
  engine_id TEXT PRIMARY KEY, name TEXT, version TEXT, kind TEXT NOT NULL,
  model_hash TEXT, executable_sha256 TEXT, git_commit TEXT, config_json TEXT, config_hash TEXT);
CREATE TABLE IF NOT EXISTS match (
  match_id TEXT PRIMARY KEY, engine_a TEXT NOT NULL REFERENCES engine(engine_id),
  engine_b TEXT NOT NULL REFERENCES engine(engine_id), tier TEXT, rule_profile TEXT, profile_hash TEXT,
  opening_set_id TEXT, opening_set_hash TEXT, compute_mode TEXT, budget_json TEXT, hardware_id TEXT,
  concurrent_games INTEGER, start_time TEXT, end_time TEXT, git_commit TEXT, command TEXT, compute_json TEXT);
CREATE TABLE IF NOT EXISTS game (
  game_id TEXT PRIMARY KEY, match_id TEXT NOT NULL REFERENCES match(match_id), pair_id INTEGER,
  opening_id TEXT, a_first INTEGER, a_color TEXT, result TEXT, winner TEXT, score_a REAL, moves TEXT,
  termination_reason TEXT, failed_side TEXT, plies INTEGER, elapsed_a_ms REAL, elapsed_b_ms REAL,
  search_stats_json TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS rating_snapshot (
  snapshot_id TEXT PRIMARY KEY, date TEXT, rating_system TEXT NOT NULL, anchor_engine TEXT,
  engine_id TEXT NOT NULL, rating REAL, lower_ci REAL, upper_ci REAL, n_games INTEGER, match_ids TEXT);
CREATE TABLE IF NOT EXISTS experiment (
  experiment_id TEXT PRIMARY KEY, baseline TEXT, candidate TEXT, changed_feature TEXT, hypothesis TEXT,
  primary_metric TEXT, tier TEXT, match_ids TEXT, result_json TEXT, decision TEXT, notes TEXT, date TEXT);
CREATE TABLE IF NOT EXISTS analysis (
  analysis_id TEXT PRIMARY KEY, game_id TEXT REFERENCES game(game_id), ply INTEGER, kind TEXT,
  budget_json TEXT, result_json TEXT, date TEXT);
CREATE INDEX IF NOT EXISTS game_match ON game(match_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class StrengthDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA foreign_keys=ON")
        self.con.executescript(_SCHEMA)
        row = self.con.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self.con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        elif row[0] != SCHEMA_VERSION:
            raise RuntimeError(f"{path}: schema version {row[0]}, expected {SCHEMA_VERSION}")
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    # ---- writes ----
    def register_engine(self, engine_id: str, kind: str, name: str = "", version: str = "", model_hash: str = "",
                        executable_sha256: str = "", git_commit: str = "", config: dict | None = None) -> None:
        if kind not in ("dojo", "external", "internal_bot"):
            raise ValueError(f"unknown engine kind {kind}")
        cfg = json.dumps(config or {}, sort_keys=True)
        import hashlib
        self.con.execute(
            "INSERT INTO engine VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(engine_id) DO UPDATE SET "
            "name=excluded.name, version=excluded.version, kind=excluded.kind, model_hash=excluded.model_hash, "
            "executable_sha256=excluded.executable_sha256, git_commit=excluded.git_commit, "
            "config_json=excluded.config_json, config_hash=excluded.config_hash",
            (engine_id, name, version, kind, model_hash, executable_sha256, git_commit, cfg,
             hashlib.sha256(cfg.encode()).hexdigest()[:16]))
        self.con.commit()

    def start_match(self, engine_a: str, engine_b: str, *, tier: str, rule_profile: str, profile_hash: str,
                    opening_set_id: str, opening_set_hash: str, compute_mode: str, budget: dict, hardware_id: str,
                    concurrent_games: int, git_commit: str, command: str) -> str:
        if compute_mode not in ("fixed_time", "fixed_visits"):
            raise ValueError("compute_mode must be fixed_time or fixed_visits (never mixed, EVALUATION §18)")
        match_id = uuid.uuid4().hex[:16]
        self.con.execute(
            "INSERT INTO match VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (match_id, engine_a, engine_b, tier, rule_profile, profile_hash, opening_set_id, opening_set_hash,
             compute_mode, json.dumps(budget, sort_keys=True), hardware_id, concurrent_games, _now(), None,
             git_commit, command, None))
        self.con.commit()
        return match_id

    def add_game(self, match_id: str, g: dict) -> str:
        if g["termination_reason"] not in TERMINATIONS:
            raise ValueError(f"unknown termination_reason {g['termination_reason']}")
        game_id = f"{match_id}-{g['pair_id']}-{int(not g['a_first'])}"
        self.con.execute(
            "INSERT INTO game VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (game_id, match_id, g["pair_id"], g["opening_id"], int(g["a_first"]), g["a_color"], g["result"],
             g["winner"], g["score_a"], json.dumps(g["moves"]), g["termination_reason"], g.get("failed_side"),
             g["plies"], g["elapsed_a_ms"], g["elapsed_b_ms"], json.dumps(g.get("search_stats", {})),
             g.get("detail", "")))
        self.con.commit()
        return game_id

    def finish_match(self, match_id: str, compute: dict) -> None:
        self.con.execute("UPDATE match SET end_time=?, compute_json=? WHERE match_id=?",
                         (_now(), json.dumps(compute, sort_keys=True), match_id))
        self.con.commit()

    def add_rating(self, rating_system: str, engine_id: str, rating: float, lower: float, upper: float, n_games: int,
                   anchor_engine: str, match_ids: list[str]) -> None:
        if rating_system not in RATING_SYSTEMS:
            raise ValueError(f"rating_system must be one of {RATING_SYSTEMS} (EVALUATION §16)")
        self.con.execute("INSERT INTO rating_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex[:16], _now(), rating_system, anchor_engine, engine_id, rating, lower,
                          upper, n_games, json.dumps(match_ids)))
        self.con.commit()

    def add_experiment(self, baseline: str, candidate: str, changed_feature: str, hypothesis: str,
                       primary_metric: str, tier: str, match_ids: list[str], result: dict, decision: str,
                       notes: str = "") -> str:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS} (EVALUATION §19)")
        eid = uuid.uuid4().hex[:16]
        self.con.execute("INSERT INTO experiment VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (eid, baseline, candidate, changed_feature, hypothesis, primary_metric, tier,
                          json.dumps(match_ids), json.dumps(result, sort_keys=True), decision, notes, _now()))
        self.con.commit()
        return eid

    def add_analysis(self, game_id: str, ply: int, kind: str, budget: dict, result: dict) -> None:
        self.con.execute("INSERT INTO analysis VALUES (?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex[:16], game_id, ply, kind, json.dumps(budget), json.dumps(result), _now()))
        self.con.commit()

    # ---- reads ----
    def match(self, match_id: str) -> dict:
        r = self.con.execute("SELECT * FROM match WHERE match_id=?", (match_id,)).fetchone()
        if r is None:
            raise KeyError(match_id)
        return dict(r)

    def games(self, match_id: str) -> list[dict]:
        rows = self.con.execute("SELECT * FROM game WHERE match_id=? ORDER BY pair_id, a_first DESC", (match_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["moves"] = json.loads(d["moves"])
            d["search_stats"] = json.loads(d.pop("search_stats_json") or "{}")
            out.append(d)
        return out

    def matches(self) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM match ORDER BY start_time")]

    def engine(self, engine_id: str) -> dict | None:
        r = self.con.execute("SELECT * FROM engine WHERE engine_id=?", (engine_id,)).fetchone()
        return dict(r) if r else None
