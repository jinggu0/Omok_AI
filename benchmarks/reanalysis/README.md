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
| `background_load_e4_*.csv` | CPU used by other processes during each phase (sleep gaps visible as time jumps) |

Label codes: PROVEN_UNSAFE / PROVEN_WINNING are exact; UNREFUTED_EXHAUSTED is a heuristic negative (the threat-space
search ended within budget without a proof: not a safety proof); UNKNOWN_BUDGET carries no label.
