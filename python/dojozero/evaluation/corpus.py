"""Versioned opening corpus (PRD FR-23, EVALUATION §15.2).

    benchmarks/openings/<set_id>.json

Every opening is replayed by the C++ GameState (expected legality checked), duplicates by
canonical hash are rejected, and the set hash (sha256 of the canonical JSON of the openings list)
goes into every match record so results on different sets are never compared.

    python -m dojozero.evaluation.corpus build-smoke --n 24 --seed 1   # provisional smoke set
    python -m dojozero.evaluation.corpus check benchmarks/openings/renju_smoke_v0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .. import _core

REPO = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO / "benchmarks" / "openings"


@dataclass
class OpeningSet:
    set_id: str
    profile: str
    openings: list[dict]
    set_hash: str
    status: str
    path: Path | None = None

    def start_state(self, opening: dict):
        g = _core.GameState(opening.get("rule_profile", self.profile))
        for m in opening["moves"]:
            if not g.apply(m):
                raise ValueError(f"{self.set_id}/{opening['opening_id']}: illegal move {m}")
        return g

    def canonical_hashes(self) -> set[int]:
        return {self.start_state(o).canonical_hash() for o in self.openings}


def set_hash(openings: list[dict]) -> str:
    blob = json.dumps(openings, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load(path_or_id: str | Path) -> OpeningSet:
    p = Path(path_or_id)
    if not p.suffix:
        p = CORPUS_DIR / f"{path_or_id}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("schema_version") != 1:
        raise ValueError(f"{p}: unsupported schema_version {d.get('schema_version')}")
    s = OpeningSet(d["set_id"], d["profile"], d["openings"], set_hash(d["openings"]), d.get("status", ""), p)
    validate(s)
    return s


def validate(s: OpeningSet) -> None:
    seen: dict[int, str] = {}
    ids = set()
    for o in s.openings:
        for key in ("opening_id", "moves", "rule_profile", "source", "source_version", "expected_legal",
                    "approx_balance", "creation_method"):
            if key not in o:
                raise ValueError(f"{s.set_id}: opening without {key}: {o.get('opening_id')}")
        if o["opening_id"] in ids:
            raise ValueError(f"{s.set_id}: duplicate opening_id {o['opening_id']}")
        ids.add(o["opening_id"])
        try:
            g = s.start_state(o)
            legal = not g.is_terminal
        except ValueError:
            legal = False
        if legal != o["expected_legal"]:
            raise ValueError(f"{s.set_id}/{o['opening_id']}: legality {legal} != expected {o['expected_legal']}")
        if legal:
            h = g.canonical_hash()
            if h in seen:
                raise ValueError(f"{s.set_id}: {o['opening_id']} duplicates {seen[h]} (same canonical position)")
            seen[h] = o["opening_id"]


def leaked(s: OpeningSet, start_hashes: set[int]) -> list[str]:
    """Opening ids whose start position appears among training start positions (PRD §15)."""
    return [o["opening_id"] for o in s.openings if s.start_state(o).canonical_hash() in start_hashes]


def stones_of_taraguchi(actions: list[str], profile: str = "OMOKDOJANG_DAN") -> list[str]:
    """Stone placements (in order) of a Taraguchi-10 action sequence; proposals place no stone,
    the selected fifth does. Colours alternate from black because swaps exchange players, not
    colours."""
    g = _core.GameState(profile)
    out = []
    for a in actions:
        before = int((g.board() != 0).sum())
        assert g.apply(a)
        if int((g.board() != 0).sum()) == before + 1:
            out.append(a.split(":")[-1])
    return out


def build_smoke(n: int, seed: int) -> dict:
    """PROVISIONAL Standard Renju smoke set: stone sequences of random Taraguchi-10 openings
    (6 stones, black to move). Balance is NOT assessed -> for harness tests and Tier A only; not
    for Tier B/C until the corpus source is decided (PRD OQ-14)."""
    from .match import make_openings
    openings = []
    for k, acts in enumerate(make_openings(n, seed)):
        moves = stones_of_taraguchi(acts)
        openings.append({"opening_id": f"smoke0-{k:04d}", "moves": moves, "rule_profile": "RENJU_GOMOCUP",
                         "source": "generated: random Taraguchi-10 opening (dojozero.evaluation.match.make_openings)",
                         "source_version": f"seed={seed}", "expected_legal": True,
                         "approx_balance": {"value_black": None, "method": "not assessed", "uncertainty": None},
                         "creation_method": "stones_of_taraguchi(make_openings(n, seed))"})
    return {"schema_version": 1, "set_id": "renju_smoke_v0", "profile": "RENJU_GOMOCUP",
            "status": "PROVISIONAL - harness tests / Tier A only; balance not assessed (PRD OQ-14)",
            "openings": openings}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-smoke")
    b.add_argument("--n", type=int, default=24)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", default=str(CORPUS_DIR / "renju_smoke_v0.json"))
    c = sub.add_parser("check")
    c.add_argument("path")
    args = ap.parse_args(argv)
    if args.cmd == "build-smoke":
        d = build_smoke(args.n, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        s = load(args.out)
        print(f"{s.set_id}: {len(s.openings)} openings, hash {s.set_hash[:16]}")
    else:
        s = load(args.path)
        print(f"{s.set_id}: {len(s.openings)} openings OK, hash {s.set_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
