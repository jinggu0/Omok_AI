# DojoZero

Standalone Renju (RIF) + Taraguchi-10 engine aiming at Omokdojang dan-level rule compatibility.
Design docs: [docs/PRD.md](docs/PRD.md) · [RULES](docs/RULES.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) ·
[TRAINING](docs/TRAINING.md) · [EVALUATION](docs/EVALUATION.md) · [ASSUMPTIONS](docs/ASSUMPTIONS.md) ·
status: [PROGRESS](docs/PROGRESS.md).

Current state: **Phase 2 (rule engine) complete.** No AI yet.

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

Coordinates: columns A–O (left to right), ranks 1–15 (bottom to top), `H8` = centre.
