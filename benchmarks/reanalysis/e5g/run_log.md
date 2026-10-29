# E5g run log

- 2026-10-29 12:36 attempt 1 started (AC confirmed): holdout v5 generation.
- 2026-10-29 ~12:44 attempt 1 stopped by hand at 100/400 holdout games, before any A/B game:
  the user had to close the laptop lid (sleep would corrupt the 500 ms timed games).
  No holdout file was written and no A/B result exists; nothing of it is used.
  The pre-registered run restarts from scratch (same seed 205, same commands) when the lid can stay open.
- 2026-10-29 12:58:55 the laptop resumed (Kernel-Power 507; standby since 12:44:12, event 506), on AC.
- 2026-10-29 13:00:37 attempt 2 started from scratch (same commands). Holdout v5 13:00-13:16 (60 balanced of 400, 0 failures,
  4b2778eb...), A/B 13:16-14:28. No Kernel-Power sleep event after 13:00; load-log max gap 81 s.
  Result: +0.142 [+0.083, +0.200] -> ACCEPT (pre-registered rule).
