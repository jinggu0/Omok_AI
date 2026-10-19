"""External evaluation infrastructure (PRD FR-20..FR-24): Gomocup process client against the
Python-hosted mock engine, TournamentRunner referee and failure reasons, engine registry hash
guard, opening corpus validation, Strength DB constraints, time-limited search."""

import json
import time

import numpy as np
import pytest

from dojozero import _core
from dojozero.evaluation import corpus
from dojozero.evaluation.external import EngineError, engine_config, load_meta, pbrain_py_config, sha256_file
from dojozero.evaluation.strength_db import StrengthDB
from dojozero.evaluation.tournament import InProcessPlayer, ProcessPlayer, classify_rejected, run_match

SMOKE = corpus.load("renju_smoke_v0")


def mock(timeout_ms=1000, fault=None, fault_after=0):
    cfg = pbrain_py_config("heuristic", timeout_ms, fault=fault, fault_after=fault_after)
    cfg.slack_ms = 300
    return ProcessPlayer(f"mock-{fault or 'ok'}", cfg, kind="internal_bot")


def test_external_engine_plays_legal_moves_over_the_protocol():
    e = _core.ExternalEngine(pbrain_py_config("heuristic", 1000))
    assert e.start() == ("NONE", "")
    assert 'name="DojoZero"' in e.about()
    r = e.begin()
    assert r["failure"] == "NONE" and r["move"] == 112  # H8, centre
    g = _core.GameState("RENJU_GOMOCUP")
    assert g.apply("H8")
    assert g.apply("I9")
    r = e.turn(_core.str_to_point("I9"))
    assert r["failure"] == "NONE" and g.apply(_core.point_to_str(r["move"]))
    # BOARD resynchronisation from a corpus opening (engine = side to move)
    st = SMOKE.start_state(SMOKE.openings[0])
    r = e.board([_core.str_to_point(m) for m in SMOKE.openings[0]["moves"]])
    assert r["failure"] == "NONE" and st.apply(_core.point_to_str(r["move"]))
    e.end()
    assert not e.alive()


def test_match_records_paired_games_in_the_db(tmp_path):
    db = StrengthDB(tmp_path / "r.sqlite")
    a = InProcessPlayer("heuristic", "cpu", {}, None, seed=3)
    res = run_match(a, mock(), SMOKE, 2, db, tier="A", time_ms=None, verbose=False)
    assert res["summary"]["games"] == 4 and res["summary"]["pairs"] == 2
    assert res["failures"] == {}
    games = db.games(res["match_id"])
    assert len(games) == 4
    assert {g["termination_reason"] for g in games} <= {"NORMAL_WIN", "DRAW"}
    m = db.match(res["match_id"])
    assert m["opening_set_hash"] == SMOKE.set_hash and m["compute_mode"] == "fixed_visits"
    for g in games:  # every recorded game replays legally from its opening
        st = _core.GameState("RENJU_GOMOCUP")
        for mv in g["moves"]:
            assert st.apply(mv)
        assert st.is_terminal
    # each pair swaps the side A plays
    by_pair = {}
    for g in games:
        by_pair.setdefault(g["pair_id"], set()).add(g["a_color"])
    assert all(v == {"BLACK", "WHITE"} for v in by_pair.values())


@pytest.mark.parametrize("fault,reason", [("crash", "CRASH"), ("timeout", "TIMEOUT"), ("malformed", "MALFORMED_RESPONSE"),
                                          ("error", "PROTOCOL_ERROR"), ("illegal", "ILLEGAL_MOVE"),
                                          ("start-crash", "CRASH")])
def test_external_failures_are_recorded_with_their_reason(fault, reason, tmp_path):
    a = InProcessPlayer("heuristic", "cpu", {}, None, seed=1)
    # Short turn limit only where a TIMEOUT is expected; elsewhere a loaded machine must not turn
    # the expected failure into a TIMEOUT (one unreproduced flake of the illegal case, Phase 8).
    b = mock(timeout_ms=300 if fault == "timeout" else 2000, fault=fault, fault_after=1)
    t0 = time.time()
    res = run_match(a, b, SMOKE, 1, StrengthDB(tmp_path / "f.sqlite"), tier="A", time_ms=None, verbose=False)
    assert time.time() - t0 < 30
    for g in res["games"]:
        assert g["termination_reason"] == reason, g
        assert g["failed_side"] == "B" and g["score_a"] == 1.0
    assert res["failures"] == {f"B:{reason}": 2}


def test_missing_binary_is_start_failed(tmp_path):
    cfg = _core.EngineConfig()
    cfg.path = str(tmp_path / "no_such_engine.exe")
    b = ProcessPlayer("ghost", cfg)
    res = run_match(InProcessPlayer("heuristic", "cpu", {}, None, 1), b, SMOKE, 1, None, tier="A", time_ms=None,
                    verbose=False)
    assert {g["termination_reason"] for g in res["games"]} == {"START_FAILED"}


def test_rejected_moves_are_classified_by_the_canonical_rules():
    spec = json.dumps({"black": ["F8", "G8", "H6", "H7"], "white": ["A1", "A3", "A5", "A7"], "side": "BLACK"})
    g = _core.GameState.from_spec(spec, "RENJU_GOMOCUP")
    assert g.side_to_move == "BLACK"
    assert classify_rejected(g, _core.str_to_point("H8")) == "FORBIDDEN_MOVE"   # black double three
    assert classify_rejected(g, _core.str_to_point("F8")) == "ILLEGAL_MOVE"     # occupied
    assert classify_rejected(g, None) == "MALFORMED_RESPONSE"


def test_registry_refuses_unpinned_or_drifted_binaries(tmp_path):
    exe = tmp_path / "pbrain-fake.exe"
    exe.write_bytes(b"not really an engine")
    meta = {"schema_version": 1, "engine_id": "fake_2026", "engine_name": "Fake", "version": "1",
            "executable_sha256": "", "source": "test", "license": "UNVERIFIED", "supported_rules": ["RENJU"],
            "protocol": "gomocup", "config": {"timeout_turn_ms": 500}}
    (tmp_path / "fake_2026.json").write_text(json.dumps(meta), encoding="utf-8")
    m = load_meta("fake_2026", engine_dir=tmp_path)
    with pytest.raises(EngineError, match="not pinned"):
        engine_config(m, exe)
    m["executable_sha256"] = "0" * 64
    with pytest.raises(EngineError, match="version drift"):
        engine_config(m, exe)
    m["executable_sha256"] = sha256_file(exe)
    cfg = engine_config(m, exe)
    assert cfg.timeout_turn_ms == 500 and cfg.rule == 4
    meta["supported_rules"] = ["FREESTYLE"]
    (tmp_path / "fake_2026.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(EngineError, match="RENJU"):
        load_meta("fake_2026", engine_dir=tmp_path)


def test_corpus_validation_hash_and_leakage(tmp_path):
    assert len(SMOKE.openings) == 24 and SMOKE.status.startswith("PROVISIONAL")
    hashes = SMOKE.canonical_hashes()
    assert len(hashes) == 24
    assert corpus.leaked(SMOKE, {next(iter(hashes))}) != []
    assert corpus.leaked(SMOKE, set()) == []
    d = json.loads(SMOKE.path.read_text(encoding="utf-8"))
    dup = dict(d, openings=d["openings"] + [dict(d["openings"][0], opening_id="dup")])
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(dup), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        corpus.load(p)
    bad = dict(d, openings=[dict(d["openings"][0], moves=["H8", "H8"])])
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="legality"):
        corpus.load(p)
    changed = dict(d, openings=d["openings"][1:])
    assert corpus.set_hash(changed["openings"]) != SMOKE.set_hash


def test_strength_db_enforces_names_and_decisions(tmp_path):
    db = StrengthDB(tmp_path / "s.sqlite")
    db.register_engine("x", "external")
    with pytest.raises(ValueError):
        db.add_rating("gomocup_reference_elo", "x", 3000, 2900, 3100, 10, "x", [])  # metadata only (§16)
    db.add_rating("dojo_external_elo", "x", 0, -10, 10, 10, "x", [])
    with pytest.raises(ValueError):
        db.add_experiment("a", "b", "f", "h", "m", "B", [], {}, "PROBABLY_GOOD")
    assert db.add_experiment("a", "b", "f", "h", "m", "B", [], {}, "INCONCLUSIVE")
    with pytest.raises(ValueError):
        db.start_match("x", "x", tier="A", rule_profile="R", profile_hash="0", opening_set_id="s", opening_set_hash="h",
                       compute_mode="mixed", budget={}, hardware_id="h", concurrent_games=1, git_commit="g", command="")
    db.close()
    StrengthDB(tmp_path / "s.sqlite").close()  # reopen: schema version check passes


def test_time_limited_batched_search():
    from dojozero.features import FeatureSpec
    from dojozero.search import UniformNumpyEvaluator
    from dojozero.search.driver import SearchSettings, search_many

    class Slow(UniformNumpyEvaluator):
        def __call__(self, feats):
            time.sleep(0.003)
            return super().__call__(feats)

    g = SMOKE.start_state(SMOKE.openings[1])
    s = SearchSettings(simulations=1_000_000, root="PUCT", max_ms=150, root_tactics_nodes=0, max_inflight=4)
    t0 = time.perf_counter()
    r = search_many([g], Slow(), s, FeatureSpec().core())[0]
    ms = (time.perf_counter() - t0) * 1000
    assert 1 <= r["simulations"] < 1_000_000
    assert 150 <= ms < 150 + 400
    assert g.apply(r["action"])


def test_report_renders_matches_and_marks_unmeasured_sections(tmp_path):
    from dojozero.evaluation.report import render
    db = StrengthDB(tmp_path / "r.sqlite")
    run_match(InProcessPlayer("heuristic", "cpu", {}, None, 2), mock(fault="malformed", fault_after=3), SMOKE, 2, db,
              tier="A", time_ms=None, verbose=False)
    text = render(db)
    assert "heuristic vs mock-malformed" in text and SMOKE.set_hash[:16] in text
    assert "B:MALFORMED_RESPONSE" in text
    assert "## Diagnosis\nUNKNOWN" in text


def test_process_reuse_restarts_between_games_and_is_dropped_after_failures(tmp_path):
    b = mock()
    b.reuse_process = True
    pids = []
    orig = b.start_game

    def tracking():
        f = orig()
        pids.append(b.engine.pid)
        return f
    b.start_game = tracking
    res = run_match(InProcessPlayer("heuristic", "cpu", {}, None, 4), b, SMOKE, 2, None, tier="A", time_ms=None,
                    verbose=False)
    assert res["failures"] == {} and len(set(pids)) == 1 and len(pids) == 4
    assert b.engine is None  # ended after the match


def test_first_move_gets_an_initialisation_allowance(tmp_path):
    """Engines that initialise lazily on their first search must not be scored as TIMEOUT for it
    (AlphaGomoku 2026 at 200 ms/move, PROGRESS Phase 8); later moves keep the normal limit."""
    b = mock(timeout_ms=300, fault="slow-first")
    res = run_match(InProcessPlayer("heuristic", "cpu", {}, None, 5), b, SMOKE, 1, None, tier="A", time_ms=None,
                    verbose=False)
    assert res["failures"] == {}
    b.config.first_move_slack_ms = 0
    res = run_match(InProcessPlayer("heuristic", "cpu", {}, None, 5), b, SMOKE, 1, None, tier="A", time_ms=None,
                    verbose=False)
    assert res["failures"] == {"B:TIMEOUT": 2}
