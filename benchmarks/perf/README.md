# Tactical-solver performance corpora and results (PROGRESS Phase 8 E3b)

## Corpora (`python -m dojozero.tools.tactics_corpus --ckpt runs/taraguchi_v1/models/gen_001.pt`)
| file | positions | sha256 (canonical positions list) | content |
|---|---|---|---|
| `bench_problems.json` | 102 | `7b5d31c1…a93589e` | every fixed benchmark problem (`benchmarks/{tactical,vcf,vct}`) |
| `after_blunder.json` | 247 | `57123c09…122b33cb` | position right after each classified DojoZero tactical blunder (E2c), opponent to move |
| `filter_workload.json` | 1976 | `54303e54…44e410581` | position after each of the top-8 policy moves at each blunder position: what the root defense filter checks |

## Tools (same source `cpp/tools/tactics_profile.cpp`, two builds)
- `dojozero_tbench` — production build. `--impl ref` = frozen ee1d011 solver (`cpp/tactics/reference`), `--impl live` = optimized, `--impl pair` = ref then live on each position back to back (same thread, same moment; use this for speed ratios). Flags: `--mode vcf|vct|analyze --nodes N --nodes2 N(analyze VCT) --ms T --threads K --gen-cache 0|1 --lazy 0|1 --fast-refute 0|1 --carry 0|1`.
- `dojozero_tprof` — `DZ_TPROF` instrumented build: per-section inclusive time and calls (`cpp/tactics/tprof.h`). Timer overhead ≈ 80–95 ns per scope inflates fine-grained sections; call counts are exact.
- Both opt their own process out of Windows power throttling (EcoQoS).

## Measurement caveat: Windows power throttling
Runs started from a windowless background shell were throttled by Windows (EcoQoS): the same ref solver ran at
3.2k nodes/s throttled vs 24.8k nodes/s opted out (single thread, after_blunder VCT). Everything in
`results/throttled/` was measured that way: ratios inside one file are usable, absolute speeds are not, and
files run at different times are not comparable (the clock state drifted by up to 5x). Valid speed numbers
are the `pair_*` and `proofms_pair_*` files.

## Results
| file | what |
|---|---|
| `results/baseline_ee1d011_after_blunder_{vcf,vct}_tprof.json` | pre-optimization section profile (instrumented live code = ee1d011 at that point; 8 threads; throttled timings, exact call counts) |
| `results/optimized_after_blunder_vct_tprof_t1.json` | post-optimization section profile (1 thread, not throttled) |
| `results/pair_after_blunder_vct_t{1,2,4,8}.json` | ref vs live, VCT 50k nodes, node budget only (deterministic trees), thread scaling |
| `results/proofms_pair_filter_workload_carry{0,1}_{125,250,500,1000}ms.json` | proofs found within a wall-clock budget, analyze_tactics as the filter calls it (VCF 100k / VCT 50k nodes, 8 threads), ref vs live, VCT time carry off/on |
| `results/nodes_only_filter_workload_live.json` | analyze on the filter workload without a clock (ceiling: 1836 WIN / 1976) |
| `results/pair_{after_blunder,bench_problems}_{vcf,analyze}_t1.json` | ref vs live, VCF and analyze_tactics (filter call shape), node budgets only |
| `results/background_load_rerun.csv` | CPU used by other processes during the rerun (every ~30 s) |
| `results/differential/*` | ref vs live and switch on/off differentials (result and node count identical everywhere) |
| `results/replay_247_k8_t8.json` | 247 blunder positions replayed through search + filter: none / A (ee1d011 solver) / B (optimized) at defense shares 10–50% (`python -m dojozero.tools.filter_replay`) |
| `results/coverage_247.json` | Case C (Phase 8 E3c): every legal move at the 247 pre-blunder positions (and 154 earliest provably-lost decisions) judged by the solver (VCF 200k / VCT 100k nodes, no clock), with policy and 500 ms visit ranks (`python -m dojozero.tools.coverage_replay --earlier`) |
| `results/ab/ab_e3b_fastsolver_k8_s30_rerun.json` | external paired A/B (Pela 2023, Yixin 2018, 40 pairs each): the run used for the verdict |
| `results/ab/ab_e3b_fastsolver_k8_s30_INVALID_rapfi_load.json` | first A/B, invalid: a background Rapfi analysis engine starved DojoZero in the Yixin half |
| `results/replay_e3d_{decision,blunder}.json` | Case A (E3d): proof-aware MCTS arms vs none / B30 filter, on the first losing decision positions and on the blunder positions |
| `results/ab/ab_e3d_proof6.json` | external paired A/B, proof-aware MCTS (6 solver threads) vs off |
| `results/background_load_e3d.csv` | CPU used by other processes during the E3d runs |
| `results/replay_e3e_{decision,blunder}.json`, `results/ab/ab_e3e_proof3_d1.json`, `results/background_load_e3e.csv` | E3e: light proof-aware MCTS (2-3 threads, depth 1) replays and the A/B of the selected 3-thread setting |
| `results/throttled/*` | early runs under power throttling (see caveat) |
