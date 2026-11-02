# Solver-supervised tactical reanalysis results (PROGRESS Phase 8 E4, Case B)

Data lives in `data/reanalysis_v1/` (git-ignored); `manifest_reanalysis_v1.json` records its hashes, counts,
solver budgets and split statistics.

    python -m dojozero.tools.reanalysis_data openings --n 120   # training-only openings (seed 101)
    python -m dojozero.tools.reanalysis_data collect            # positions + opening-group split
    python -m dojozero.tools.reanalysis_data label              # solver labels (VCF 50k / VCT 30k / loss 30k)
    python -m dojozero.trainer.reanalysis build                 # training arrays + targets
    python -m dojozero.trainer.reanalysis_train --config configs/training/e4_reanalysis.yaml
    python -m dojozero.tools.reanalysis_eval --ckpt <ckpt> --split test --out <json>

| file | what |
|---|---|
| `manifest_reanalysis_v1.json` | dataset manifest (positions / labels / arrays sha256, split counts, label counts) |
| `eval_{baseline,candidate}_test.json` | held-out tactical diagnosis (policy coverage, value calibration, search failure classes), per position rows |
| `replay_{blunder,decision}_{baseline,candidate}.json` | 247 blunder positions and the 247 first losing decisions replayed at 500 ms (proof/filter off) |
| `visits_{baseline,candidate}.json` | fixed 64/256/1024 sims vs Pela 2023 / Yixin 2018 (opponent 500 ms), 12 pairs each |
| `ab_e4_reanalysis_holdout.json` | external paired A/B on the fresh holdout `renju_holdout_e4_v1` (60 pairs per opponent) |
| `e5/` | E5: RL arms T/C registries and logs, held-out diagnosis of both bests, final A/Bs (T vs C, T vs E4, C vs E4), load logs, `value_drift.json` (why proven-win values fell: 14 checkpoints on proven positions, `python -m dojozero.tools.value_drift`) |
| `e5b/` | E5b balanced value gate: gate set manifest (`gate_balanced_v1.json`, E4 val 468 per colour x result cell) and the E5 counterfactual (`gate_counterfactual_e5.json`, `python -m dojozero.tools.balanced_gate`) |
| `e5c/` | E5c fixed-data screen of colour x result balanced value weights (`screen_*.json`, `python -m dojozero.tools.fixed_replay_screen`; pre-registered judge: STOP) |
| `e5d/` | E5d/E5e fixed-data screens (EMA, lr/2, EMA + balanced weights; seeds 7-9), pre-registrations, group judges |
| `e5e/` | E5e T'' run (registry, log), held-out diagnosis, value balance on test / gate, external A/B on `renju_holdout_e5_v3` (INCONCLUSIVE), load logs |
| `e5f/` | E5f T'' continued to generation 47: pre-registration + addendum, registry, loop / driver logs, held-out diagnosis and value balance on test / gate of the final best gen_045, external A/B gen_045 vs gen_002 on `renju_holdout_e5_v4` (ACCEPT, CI lower bound +0.003), load logs |
| `e5g/` | E5g pre-registration, run log (attempt 1 stopped before any A/B game), external A/B T'' gen_045 vs E4 candidate on `renju_holdout_e5_v5` (ACCEPT, +0.142 [+0.083, +0.200]), load log |
| `e5h/` | E5h external ladder v2 of T'' gen_045 (`python -m dojozero.evaluation.ladder calibrate --config configs/evaluation/ladder_v2.yaml`): pre-registration, calibration output (all games and decided-only refit; -318 [-410, -226], anchor Yixin 2018), load log |
| `e5i/` | E5i value drift diagnosis (E4 candidate + T'' generations 1-52 on the balanced gate set vs the E4 test proven positions), pre-registration, T'' generations 48-52 (registry snapshot, loop / driver logs; no promotion, best stays gen_045), load log |
| `e5j/` | E5j rebuilt balanced gate: pre-registration, gate v2a (E4 val, DojoZero-to-move) and v2 (fresh `renju_gate_e5_v1` games, 730 per cell) manifests, measurements of E4 candidate + T'' generations 1-52 (`measure_gate.py`), bias by arm (`measure_by_arm.py`), pool games and manifest, load log. Gate arrays are in `data/` (git-ignored) |
| `e5k/` | E5k T'' generations 53-57 with the promotion gate on gate v2: pre-registration, registry snapshot, loop / driver logs, gate v1 and v2 values (`measure_generations.py`; v1 is a diagnostic), load log. New internal best v5 gen_055 |
| `background_load_e4_*.csv` | CPU used by other processes during each phase (sleep gaps visible as time jumps) |

Label codes: PROVEN_UNSAFE / PROVEN_WINNING are exact; UNREFUTED_EXHAUSTED is a heuristic negative (the threat-space
search ended within budget without a proof: not a safety proof); UNKNOWN_BUDGET carries no label.
