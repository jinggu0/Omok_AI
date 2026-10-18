# Fixed benchmark suites

Run all: `dojozero_engine benchmark --suite benchmarks` (also part of `dojozero_tests`).
Every WIN/LOSS answer is re-checked by the independent full-width verifier (`cpp/tactics/verify.*`,
GameState rules only). A claim the verifier refutes is a **false claim** and must stay at 0.

| Directory | Contents | Source |
|---|---|---|
| `tactical/handmade.json` | win-in-1, forced block, no-defense, VCF, forbidden induction, black 4-4 not allowed, white 3-3 VCT, proven loss | hand-made, reasoning in each `note` |
| `vcf/generated.json` | 40 VCF wins (PV 5–23 plies), RENJU_RIF + OMOKDOJANG_DAN | `dojozero_tactics generate --kind vcf --seed 1`, verified |
| `vct/generated.json` | 25 wins that need threes (VCF exhaustively absent) | `--kind vct --seed 2`, verified |
| `vct/generated_loss.json` | 20 positions where the side to move is proven lost | `--kind loss --seed 3`, verified |
| `rules/`, `forbidden/` | rule cases live in `tests/forbidden/cases/` (EVALUATION §9) | — |

Problem format (`schema_version` 1):
```json
{"id": "...", "category": "VCF", "source": {"kind": "handmade|generated+verified", "note": "..."},
 "profile": "RENJU_RIF|TARAGUCHI10|OMOKDOJANG_DAN",
 "position": {"black": [...], "white": [...], "side": "BLACK"},
 "query": "decide|vcf|vct|loss", "budget_nodes": 200000,
 "expected": {"result": "WIN|LOSS|FOUND|NONE|NOT_WIN|UNKNOWN", "reason": "...", "move": "J8", "first_moves": ["I8"]}}
```
- `NONE` (vcf) passes only if the search was exhaustive (no depth cut).
- Generated problems carry no `first_moves`: another winning first move would be equally correct.
- Generated expectations are **solver results confirmed by the verifier**, not independent ground
  truth for *non*-wins; there are no generated "no win" problems for that reason.
