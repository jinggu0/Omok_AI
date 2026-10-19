# DojoZero

Standalone Renju (RIF) + Taraguchi-10 engine aiming at Omokdojang dan-level rule compatibility.
Design docs: [docs/PRD.md](docs/PRD.md) · [RULES](docs/RULES.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) ·
[TRAINING](docs/TRAINING.md) · [EVALUATION](docs/EVALUATION.md) · [ASSUMPTIONS](docs/ASSUMPTIONS.md) ·
status: [PROGRESS](docs/PROGRESS.md).

Current state: **Phase 6 complete** — rule engine, tactical solver, supervised tactical network, MCTS/Gumbel search, self-play RL loop with replay buffer and SPRT promotion (3 generations run). Opening (Taraguchi-10) policy learning not yet.

## Build (Windows, MSVC 2022, CMake ≥ 3.24)
```bash
cmake --preset msvc
cmake --build build/msvc --config Release
```

## Test
```bash
./build/msvc/tests/Release/dojozero_tests.exe
```
```bash
./build/msvc/cpp/Release/dojozero_fuzz.exe --comparisons 10000000 --threads 8 --seed 1
```

## Use
```bash
./build/msvc/cpp/Release/dojozero_engine.exe forbidden --profile RENJU_RIF --position '{"black":["F8","G8","H6","H7"],"white":["A15","C15","E15","G15"]}' --point H8
```
```bash
./build/msvc/cpp/Release/dojozero_engine.exe legal --profile OMOKDOJANG_DAN --position '{"moves":["H8","keep","I9"]}'
```

```bash
./build/msvc/cpp/Release/dojozero_engine.exe tactics --position '{"black":["F8","G8","H8","I6","I7"],"white":["E8","B15","E15","H15","K15"],"side":"BLACK"}' --verify
```
```bash
./build/msvc/cpp/Release/dojozero_engine.exe benchmark --suite benchmarks
```

## Python (training)
```bash
python -m venv .venv
```
```bash
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/xpu
```
```bash
./.venv/Scripts/python.exe -m pip install -e .[dev]
```
```bash
cmake --preset msvc -DPython_EXECUTABLE=.venv/Scripts/python.exe
```
```bash
./build/msvc/cpp/Release/dojozero_datagen.exe --out data/stage_a_v1 --samples 120000 --seed 1 --threads 8 --vcf-nodes 8000 --vct-nodes 3000 --loss-nodes 1500
```
```bash
./.venv/Scripts/python.exe -m dojozero.trainer.train --config configs/training/stage_b_tactical.yaml
```

```bash
PYTHONPATH=python ./.venv/Scripts/python.exe -m dojozero.evaluation.match --a engine:runs/stage_b_b6c64_v1/checkpoints/step_6000.pt:sims=64 --b heuristic --pairs 50
```

```bash
PYTHONPATH=python ./.venv/Scripts/python.exe -m dojozero.rl.loop --config configs/training/stage_d_selfplay.yaml
```

Coordinates: columns A–O (left to right), ranks 1–15 (bottom to top), `H8` = centre.
