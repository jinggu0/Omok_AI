"""Self-play RL generation loop (TRAINING.md §2.4, §5, §8; EVALUATION.md §5).

    python -m dojozero.rl.loop --config configs/training/stage_d_selfplay.yaml [--resume]

Each generation:
  1. self-play with the current BEST model -> runs/<run>/selfplay/gen_NNN.jsonl.gz (atomic)
  2. ingest into the replay buffer (C++ replay + record verification)
  3. continue training the learner on replay (+ optional Stage A tactical mix) -> models/gen_NNN.pt
  4. paired SPRT match: candidate vs best; promote on H1 (registry.json is the single source of truth)
  5. anchor match vs the heuristic bot for the best model (external reference, echo-chamber watch)
Phase 7 (TRAINING.md §2.5-§2.6): self-play may start from the empty board / opening prefixes and
record every Taraguchi-10 decision; a share of games is played against a league pool (fixed
anchors + past best models) chosen by PFSP; matches can start from opening prefixes so the
engines make the opening decisions; opening statistics, an opening value DB and the ten-proposal
regret diagnostic are logged per generation.
Components talk through files only (chunks, checkpoints, registry), so self-play / training /
evaluation can later run as separate workers (filesystem queue, ARCHITECTURE.md §3.11).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..evaluation.match import make_openings, make_player, make_prefix_openings, play
from ..evaluation.opening import OpeningDB, opening_stats, proposal_regret
from ..evaluation.sprt import sprt
from ..features import FeatureSpec
from ..model.load import load_model, model_sha
from ..search.driver import NetEvaluator
from ..selfplay.generate import Opponent, SelfPlaySettings, play_games
from ..selfplay.records import write_chunk
from ..trainer import checkpoint
from ..trainer.data import TacticalDataset
from ..trainer.losses import LossWeights, total_loss
from ..trainer.train import REPO, environment, git, pick_device

KEYS = ("x", "legal", "policy", "has_policy", "value", "has_value", "aux", "head", "tgt2")


def rel(p: Path) -> str:
    """Repository-relative path when inside the repo (portable registry), else absolute."""
    try:
        return str(Path(p).resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(Path(p).resolve()).replace("\\", "/")


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    import os
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class RunLock:
    """runs/<run>/LOCK holds the owner PID. A second loop on the same run directory refuses to start
    (two concurrent loops once overwrote each other's chunks, checkpoints and registry; CLAUDE.md §6)."""

    def __init__(self, run_dir: Path):
        import os
        self.path = run_dir / "LOCK"
        if self.path.exists():
            try:
                owner = int(self.path.read_text(encoding="utf-8").split()[0])
            except (ValueError, IndexError):
                owner = -1
            if owner > 0 and owner != os.getpid() and pid_alive(owner):
                raise RuntimeError(f"{run_dir} is locked by running process {owner}")
            self.path.unlink()  # stale lock from a dead process
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(sys.argv)}".encode())
        os.close(fd)
        import atexit
        atexit.register(self.release)

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class Registry:
    """runs/<run>/registry.json: best model, generation history (atomic writes)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"best": None, "generations": []}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def save_model(path: Path, model, opt, spec: FeatureSpec, mcfg: dict, cfg: dict, step: int, env: dict, metrics: dict):
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)  # constant-LR placeholder state
    checkpoint.save(path, model=model, optimizer=opt, scheduler=sched, step=step, config=cfg,
                    feature_spec=spec.to_dict(), model_config=mcfg, env=env, metrics=metrics)


def eval_openings(ecfg: dict, n: int, seed: int) -> list[list[str]]:
    return (make_prefix_openings if ecfg.get("openings", "full") == "prefix" else make_openings)(n, seed)


def pfsp_weights(stats: dict, names: list[str], eps: float = 0.05) -> list[float]:
    """Prioritized fictitious self-play, "hard" variant: weight (1 - p)^2 + eps where p is the
    learner's score against the opponent (prior 0.5 from one pseudo-game per opponent)."""
    out = []
    for n in names:
        e = stats.get(n, {"score": 0.0, "games": 0})
        p = (e["score"] + 0.5) / (e["games"] + 1)
        out.append((1.0 - p) ** 2 + eps)
    return out


def league_results(games: list[dict]) -> dict:
    """Learner score per league opponent from this generation's league games."""
    out: dict[str, dict] = {}
    for g in games:
        m = g["meta"]
        if m.get("opponent", "self") == "self":
            continue
        e = out.setdefault(m["opponent"], {"score": 0.0, "games": 0})
        e["games"] += 1
        e["score"] += 0.5 if m["winner"] is None else float(m["winner"] == m["main_player"])
    return out


def league_members(cfg: dict, best_history: list[dict]) -> list[dict]:
    """Fixed league members from the config plus the most recent past best models (the current
    best is the learner side, so it is excluded). Names are unique."""
    lc = cfg.get("league") or {}
    members = [dict(m) for m in lc.get("fixed", [])]
    k = int(lc.get("past_best", 0))
    if k > 0:
        members += [{"name": f"best_v{h['version']}", "spec": f"engine:{h['checkpoint']}"} for h in best_history[:-1][-k:]]
    seen, out = set(), []
    for m in members:
        if m["name"] not in seen and m["spec"] not in seen:  # same model under two names: keep the first
            seen.update((m["name"], m["spec"]))
            out.append(m)
    return out


def build_league(members: list[dict], device, cache: dict) -> list[Opponent]:
    out = []
    for m in members:
        kind, _, ckpt = m["spec"].partition(":")
        if kind in ("heuristic", "random"):
            out.append(Opponent(m["name"], kind))
            continue
        if ckpt not in cache:  # each model is loaded once per run
            model, fspec, _ = load_model(str(REPO / ckpt), device)
            cache[ckpt] = (NetEvaluator(model, device), fspec.core())
        out.append(Opponent(m["name"], "engine", *cache[ckpt]))
    return out


def evaluate_pair(cand_ckpt: str, best_ckpt: str, ecfg: dict, device: str, seed: int, log) -> dict:
    cache: dict = {}
    sims = ecfg["sims"]
    a = make_player(f"engine:{cand_ckpt}:sims={sims}", device, cache)
    b = make_player(f"engine:{best_ckpt}:sims={sims}", device, cache)
    scores: list[float] = []
    rnd = 0
    res = sprt(scores, ecfg["elo0"], ecfg["elo1"], ecfg["alpha"], ecfg["beta"])
    while len(scores) < ecfg["max_pairs"]:
        openings = eval_openings(ecfg, ecfg["pairs_per_round"], seed * 1000 + rnd)
        results = play(a, b, openings, seed * 1000 + rnd, verbose=False)
        by_open: dict[int, list[float]] = {}
        for r in results:
            by_open.setdefault(r["opening"], []).append(r["score_a"])
        scores += [float(np.mean(v)) for v in by_open.values()]
        rnd += 1
        res = sprt(scores, ecfg["elo0"], ecfg["elo1"], ecfg["alpha"], ecfg["beta"])
        log(f"    SPRT after {res.pairs} pairs: score {res.score:.3f} LLR {res.llr:.2f} [{res.lower:.2f}, {res.upper:.2f}]")
        if res.decision != "continue" and len(scores) >= ecfg["min_pairs"]:
            break
    return {"pairs": res.pairs, "score": res.score, "llr": res.llr, "decision": res.decision}


def anchor_match(ckpt: str, ecfg: dict, device: str, seed: int) -> dict:
    from ..evaluation.match import summarize
    cache: dict = {}
    a = make_player(f"engine:{ckpt}:sims={ecfg['sims']}", device, cache)
    b = make_player("heuristic", device, cache)
    s = summarize(play(a, b, eval_openings(ecfg, ecfg["anchor_pairs"], 900_000 + seed), seed, verbose=False))
    return {k: s[k] for k in ("score", "score_ci95", "elo", "elo_ci95", "a_score_as_black", "a_score_as_white")}


def run(cfg: dict, resume: bool = False, quiet: bool = False) -> Registry:
    device = pick_device(cfg.get("device", "auto"))
    run_dir = REPO / cfg["run_dir"] / cfg["run_name"]
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    lock = RunLock(run_dir)  # noqa: F841  (released at exit)
    log_path = run_dir / "loop.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if not quiet:
            print(line, flush=True)

    reg = Registry(run_dir / "registry.json")
    if reg.data["best"] is not None and not resume:
        raise RuntimeError(f"{run_dir} already has a registry; pass --resume")
    if not resume:
        (run_dir / "config.merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        env = environment(device)
        (run_dir / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
        if env["git_dirty"]:
            (run_dir / "git_diff.patch").write_text(git("diff"), encoding="utf-8")
    (run_dir / "command.txt").open("a", encoding="utf-8").write(" ".join(sys.argv) + "\n")
    env_small = {"git_commit": git("rev-parse", "HEAD"), "torch": str(torch.__version__), "device": str(device)}

    # Learner: initialised from the init checkpoint, or the latest generation on resume.
    learner_ckpt = cfg["init_checkpoint"] if not reg.data["generations"] else reg.data["generations"][-1]["candidate"]
    learner, spec, ck = load_model(str(REPO / learner_ckpt), device)
    mcfg = ck["model_config"]
    learner.train()
    o = cfg["train"]
    opt = torch.optim.AdamW(learner.parameters(), lr=o["lr"], weight_decay=o["weight_decay"], betas=tuple(o["betas"]),
                            foreach=False if device.type == "xpu" else None)  # PROGRESS Phase 4: XPU foreach bug
    if reg.data["generations"]:
        opt.load_state_dict(ck["optimizer"])
    step = ck["step"] if reg.data["generations"] else 0
    if reg.data["best"] is None:
        reg.data["best"] = {"version": 0, "checkpoint": cfg["init_checkpoint"], "sha": model_sha(learner)}
        reg.data["best_history"] = [dict(reg.data["best"])]
        reg.data["league"] = {}
        reg.save()
    reg.data.setdefault("best_history", [dict(reg.data["best"])])
    reg.data.setdefault("league", {})
    league_cache: dict = {}
    opening_db = OpeningDB(run_dir / "opening_db.json")

    from ..replay import ReplayBuffer
    replay = ReplayBuffer(spec, window=cfg["replay"]["window"], capacity=cfg["replay"]["capacity"])
    for rec in reg.data["generations"][-cfg["replay"]["window"]:]:
        replay.ingest(REPO / rec["chunk"], rec["generation"])
    tactical = None
    mix = float(o.get("tactical_mix", 0.0))
    if mix > 0:
        tactical = TacticalDataset(REPO / o["tactical_data"], spec, verbose=False)
        tactical_pool = tactical.split(5.0).train
    weights = LossWeights(**o["loss"])
    amp = bool(cfg.get("amp_bf16", True)) and device.type == "xpu"
    sp = SelfPlaySettings(**cfg["selfplay"])
    first = len(reg.data["generations"]) + 1
    for gen in range(first, first + cfg["generations"]):
        t_gen = time.time()
        best = reg.data["best"]
        log(f"generation {gen}: best = v{best['version']} ({best['sha'][:12]})")
        # 1. self-play with the best model
        best_model, _, _ = load_model(str(REPO / best["checkpoint"]), device)
        t0 = time.time()
        best_eval = NetEvaluator(best_model, device)
        league = build_league(league_members(cfg, reg.data["best_history"]), device, league_cache) \
            if sp.league_fraction > 0 else []
        names = [o.name for o in league]
        weights_l = pfsp_weights(reg.data["league"], names, float((cfg.get("league") or {}).get("pfsp_eps", 0.05)))
        if league:
            log("  league: " + ", ".join(f"{n} w={w:.2f}" for n, w in zip(names, weights_l)))
        games = play_games(best_eval, spec.core(), sp, seed=cfg["seed"] * 1000 + gen,
                           meta={"generation": gen, "model_version": best["version"], "model_sha": best["sha"]},
                           verbose=not quiet, opponents=league, opponent_weights=weights_l)
        chunk = run_dir / "selfplay" / f"gen_{gen:03d}.jsonl.gz"
        write_chunk(chunk, games)
        winners = [g["meta"]["winner"] for g in games]
        black_wins = sum(1 for g in games if g["meta"]["winner"] is not None and g["meta"]["winner"] == g["meta"]["black_holder"])
        t_sp = time.time() - t0
        n_pos = replay.ingest(chunk, gen)
        log(f"  self-play: {len(games)} games, {n_pos} target positions in {t_sp:.0f}s; black wins "
            f"{black_wins}/{len(games)}, draws {winners.count(None)}; mean plies "
            f"{np.mean([g['meta']['plies'] for g in games]):.1f}; replay {len(replay)} positions {replay.value_balance()} "
            f"heads {replay.head_counts()}")
        lres = league_results(games)
        for n, e in lres.items():
            tot = reg.data["league"].setdefault(n, {"score": 0.0, "games": 0})
            tot["score"] += e["score"]
            tot["games"] += e["games"]
        if lres:
            log("  league results (learner score): " + ", ".join(f"{n} {e['score']:.1f}/{e['games']}" for n, e in lres.items()))
        ostats = opening_stats([g for g in games if g["meta"].get("opponent", "self") == "self"])
        opening_db.add_games(games, best["version"])
        opening_db.save()
        regret = proposal_regret(best_eval, spec.core(), games, max_positions=int(cfg.get("regret_positions", 20)))
        log(f"  opening: {ostats}; db {opening_db.summary()}; proposal regret {regret}")
        # 2. train the learner
        t0 = time.time()
        learner.train()
        bs = o["batch_size"]
        n_tac = int(round(bs * mix)) if tactical is not None else 0
        agg: dict[str, list[float]] = {}
        for k in range(o["steps_per_generation"]):
            b = replay.sample(bs - n_tac, cfg["seed"], step, device)
            if n_tac:
                rng = np.random.default_rng([cfg["seed"], step, 11])
                tb = tactical.batch(tactical_pool[rng.integers(0, len(tactical_pool), n_tac)],
                                    rng.integers(0, 8, n_tac), device)
                tb["head"] = torch.zeros(n_tac, dtype=torch.int64, device=device)  # HEAD_POLICY
                tb["tgt2"] = torch.zeros(n_tac, 2, dtype=torch.float32, device=device)
                b = {key: torch.cat([b[key], tb[key]]) for key in KEYS}
            ctx = torch.autocast(device_type="xpu", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)
            with ctx:
                out = learner(b["x"])
            out = {kk: v.float() for kk, v in out.items()}
            loss, parts = total_loss(out, b, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(learner.parameters(), o["grad_clip"],
                                                foreach=False if device.type == "xpu" else None)
            if not torch.isfinite(gn):
                raise RuntimeError(f"non-finite gradient at step {step}")
            opt.step()
            step += 1
            for kk, v in parts.items():
                agg.setdefault(kk, []).append(float(v))
        train_metrics = {kk: float(np.mean(v)) for kk, v in agg.items()}
        cand_rel = Path(cfg["run_dir"]) / cfg["run_name"] / "models" / f"gen_{gen:03d}.pt"
        save_model(REPO / cand_rel, learner, opt, spec, mcfg, cfg, step, env_small, train_metrics)
        cand_sha = model_sha(learner)
        log(f"  train: {o['steps_per_generation']} steps in {time.time() - t0:.0f}s, {train_metrics}")
        # 3. SPRT promotion
        t0 = time.time()
        ev = evaluate_pair(str(REPO / cand_rel), str(REPO / best["checkpoint"]), cfg["eval"], str(device),
                           cfg["seed"] * 7 + gen, log)
        promoted = ev["decision"] == "H1"
        log(f"  promotion: {ev} -> {'PROMOTED' if promoted else 'kept best'} ({time.time() - t0:.0f}s)")
        if promoted:
            reg.data["best"] = {"version": best["version"] + 1, "checkpoint": str(cand_rel).replace("\\", "/"),
                                "sha": cand_sha, "generation": gen}
            reg.data["best_history"].append(dict(reg.data["best"]))
        # 4. anchor vs heuristic (external reference)
        anchor = None
        if cfg["eval"].get("anchor_pairs", 0) > 0:
            anchor = anchor_match(str(REPO / reg.data["best"]["checkpoint"]), cfg["eval"], str(device), gen)
            log(f"  anchor (best vs heuristic): {anchor}")
        reg.data["generations"].append({
            "generation": gen, "chunk": rel(chunk),
            "candidate": str(cand_rel).replace("\\", "/"), "candidate_sha": cand_sha, "selfplay_model": best["sha"],
            "games": len(games), "positions": n_pos, "black_wins": black_wins, "draws": winners.count(None),
            "train": train_metrics, "step": step, "eval": ev, "promoted": promoted, "anchor": anchor,
            "league": lres, "opening": ostats, "proposal_regret": regret,
            "seconds": time.time() - t_gen})
        reg.save()
        del best_model, best_eval
    return reg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--generations", type=int)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.generations:
        cfg["generations"] = args.generations
    run(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
