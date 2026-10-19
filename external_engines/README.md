# External engines (PRD FR-20, EVALUATION §12)

Metadata only. **Binaries are never committed** and are used only when the user provides or
approves them (PRD OQ-04, ASSUMPTIONS D-05).

1. Copy `_template.json` to `<engine_id>.json` (e.g. `rapfi_2025.json`); fill in name, exact
   version, source, license (keep `UNVERIFIED` until checked) and `executable_sha256`
   (`python -c "from dojozero.evaluation.external import sha256_file; print(sha256_file('path'))"`).
   An empty hash means "not pinned": the runner refuses to start the engine (PRD OQ-11).
2. Map the id to the local binary in `local_paths.json` (git-ignored):
   `{"rapfi_2025": "C:/engines/rapfi/pbrain-rapfi.exe"}`
3. Before using an engine in Tier B/C, run the rule conformance check (EVALUATION §12.2) and record
   the result under `conformance`.

A new version is a new `engine_id`; the runner rejects a binary whose sha256 differs from the
pinned value. Published ratings go into `public_reference_rating` as metadata only and are never
used as `dojo_external_elo`.
