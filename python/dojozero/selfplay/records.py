"""Self-play game records (TRAINING.md §3, ARCHITECTURE.md §5.1).

A record stores the ACTION SEQUENCE from a fresh GameState, not boards: positions are rebuilt by
the C++ engine at training time, so the single rule implementation stays the source of truth and
auxiliary targets are always recomputed with the current rules.

Chunk file: gzip JSON Lines, one game per line, written atomically (tmp -> rename).
Game line:
  {"format_version": 1, "meta": {...}, "actions": ["H8", "keep", ...],
   "positions": [{"i": <index into actions>, "actor": "P0", "policy": {"H9": 0.71, ...},
                  "value": <root value for the actor>, "proven": 2}]}
Only positions with a full search are listed (playout cap randomization, TRAINING.md §2.4).
The value target z is derived from meta["winner"] and the position's actor at load time.

Binary .dzg chunks (ARCHITECTURE.md §5.1) are not implemented yet; JSONL is the format in use.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

FORMAT_VERSION = 1


def write_chunk(path: str | Path, games: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for g in games:
            if g.get("format_version") != FORMAT_VERSION:
                raise ValueError("game record without the current format_version")
            fh.write(json.dumps(g, separators=(",", ":")) + "\n")
    os.replace(tmp, path)  # atomic on the same volume: readers never see a partial chunk


def read_chunk(path: str | Path) -> list[dict]:
    games = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            g = json.loads(line)
            if g.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported record format {g.get('format_version')}")
            games.append(g)
    return games


def z_for(actor: str, winner: str | None) -> int:
    """Game outcome for the actor at a position: +1 win, 0 draw, -1 loss (TRAINING.md §1)."""
    if winner is None:
        return 0
    return 1 if winner == actor else -1
