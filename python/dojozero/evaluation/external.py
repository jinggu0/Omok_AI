"""External engine registry (PRD FR-20, EVALUATION §12.1).

    external_engines/<engine_id>.json   metadata, tracked in git (no binaries)
    external_engines/local_paths.json   {"<engine_id>": "C:/engines/pbrain-rapfi.exe"}, NOT tracked

Engine names and published ratings are metadata only; nothing in the code ranks engines by them.
A binary runs only when its sha256 equals the pinned `executable_sha256` (version drift guard).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .. import _core

REPO = Path(__file__).resolve().parents[3]
ENGINE_DIR = REPO / "external_engines"
REQUIRED = ("schema_version", "engine_id", "engine_name", "version", "executable_sha256", "source", "license",
            "supported_rules", "protocol", "config")


class EngineError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta(engine_id: str, engine_dir: Path = ENGINE_DIR) -> dict:
    p = engine_dir / f"{engine_id}.json"
    if not p.exists():
        raise EngineError(f"no metadata {p}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise EngineError(f"{p}: missing {missing}")
    if meta["engine_id"] != engine_id:
        raise EngineError(f"{p}: engine_id {meta['engine_id']} != file name")
    if meta["protocol"] != "gomocup":
        raise EngineError(f"{engine_id}: unsupported protocol {meta['protocol']}")
    if "RENJU" not in meta["supported_rules"]:
        raise EngineError(f"{engine_id}: metadata does not list RENJU support")
    return meta


def local_path(engine_id: str, engine_dir: Path = ENGINE_DIR) -> Path:
    p = engine_dir / "local_paths.json"
    if not p.exists():
        raise EngineError(f"{p} not found: binaries are user-provided (PRD OQ-04); map engine ids to paths there")
    paths = json.loads(p.read_text(encoding="utf-8"))
    if engine_id not in paths:
        raise EngineError(f"{engine_id} has no local path in {p}")
    return Path(paths[engine_id])


def engine_config(meta: dict, path: str | Path, timeout_turn_ms: int | None = None) -> "_core.EngineConfig":
    """EngineConfig for a pinned binary; refuses to run a binary whose hash differs."""
    path = Path(path)
    if not path.exists():
        raise EngineError(f"{meta['engine_id']}: binary {path} not found")
    want = meta["executable_sha256"]
    if not want:
        raise EngineError(f"{meta['engine_id']}: executable_sha256 is not pinned (PRD OQ-11)")
    got = sha256_file(path)
    if got != want:
        raise EngineError(f"{meta['engine_id']}: sha256 {got} != pinned {want} (version drift) - register a new engine_id")
    c = meta["config"]
    cfg = _core.EngineConfig()
    cfg.path = str(path)
    cfg.args = list(c.get("args", []))
    cfg.working_dir = str(path.parent)
    cfg.rule = int(c.get("rule", 4))
    cfg.timeout_turn_ms = int(timeout_turn_ms if timeout_turn_ms is not None else c.get("timeout_turn_ms", 1000))
    cfg.timeout_match_ms = int(c.get("timeout_match_ms", 0))
    cfg.max_memory_bytes = int(c.get("max_memory_mb", 0)) * 1024 * 1024
    cfg.extra_info = [(str(k), str(v)) for k, v in c.get("extra_info", {}).items()]
    return cfg


def pbrain_py_config(think: str, timeout_turn_ms: int, fault: str | None = None, fault_after: int = 0,
                     device: str = "cpu") -> "_core.EngineConfig":
    """The Python-hosted DojoZero protocol engine as an external process (interim pbrain, ADR-08;
    with `fault` the misbehaving mock engine used by tests)."""
    boot = f"import sys; sys.path.insert(0, r'{REPO / 'python'}'); from dojozero.protocol.pbrain import main; sys.exit(main())"
    args = ["-c", boot, "--think", think, "--device", device]
    if fault:
        args += ["--fault", fault, "--fault-after", str(fault_after)]
    cfg = _core.EngineConfig()
    cfg.path = sys.executable
    cfg.args = args
    cfg.working_dir = str(REPO)
    cfg.timeout_turn_ms = timeout_turn_ms
    cfg.start_timeout_ms = 60000  # Python + torch import
    return cfg
