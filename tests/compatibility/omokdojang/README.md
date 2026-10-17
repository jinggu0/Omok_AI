# Omokdojang compatibility cases

Evidence collected **by hand** from the 오목도장 app (no automation, see docs/PRD.md Non-goals).
Procedure and schema: [docs/ASSUMPTIONS.md §2.1](../../../docs/ASSUMPTIONS.md).

- One file per observation: `<C-ID>_<slug>.json` (e.g. `C-04_false_33_depth2.json`).
- Start from `TEMPLATE.json.example`.
- Supported checks: `expected_legal`, `expected_forbidden` (`NONE` / `OVERLINE` / `DOUBLE_FOUR` / `DOUBLE_THREE`),
  `expected_legal_moves`, `expected_phase`, `apply` + `expected_result`, `expected_load_error`.
- `dojozero_tests --gtest_filter=OmokdojangCompatibility.*` runs them. A failure means the engine
  default differs from the app: update the C-xx row in ASSUMPTIONS.md and the profile in
  `configs/rules/omokdojang_dan.json`; keep the case as the regression test.

Currently empty: every C-xx item is `UNVERIFIED_APP_BEHAVIOR`.
