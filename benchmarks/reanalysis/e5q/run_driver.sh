#!/bin/bash
# E5q driver (benchmarks/reanalysis/e5q/preregistration.json): holdout v10 (200 openings) -> 400-pair A/B gen_063 vs gen_045.
cd /c/temp_git/Omok_AI
export PYTHONPATH=python
PY=.venv/Scripts/python.exe
S='C:/Users/jingg/AppData/Local/Temp/claude/4bf386c95b33b8d4/6d8070cf-753e-4993-af2c-209a92d9c3e9/scratchpad'
E=benchmarks/reanalysis/e5q
H=benchmarks/openings/renju_holdout_e5_v10.json
EXCL="renju_balanced_v1 renju_provisional_v0 renju_smoke_v0 renju_train_reanalysis_v1 renju_holdout_e4_v1 renju_holdout_e5_v2 renju_holdout_e5_v3 renju_holdout_e5_v4 renju_holdout_e5_v5 renju_gate_e5_v1 renju_holdout_e5_v6 renju_holdout_e5_v7 renju_holdout_e5_v8 renju_holdout_e5_v9"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
need_ac() { local n=0; while [ "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d '\r')" = "1" ]; do [ $((n % 15)) -eq 0 ] && say "on battery - waiting for AC"; n=$((n+1)); sleep 120; done; }
die() { say "STOP: $*"; exit 1; }
nohup powershell -NoProfile -ExecutionPolicy Bypass -File "$S/loadlog2.ps1" -out $E/background_load_e5q.csv >/dev/null 2>&1 &
say "E5q driver start"
if [ ! -f $H ]; then
  need_ac; say "holdout v10 start"
  $PY -m dojozero.evaluation.engine_games balance --candidates 720 --target 200 --seed 211 --set-id renju_holdout_e5_v10 --exclude-sets $EXCL 2>&1 | grep -v Warning | tail -3
  [ -f $H ] || die "no holdout"
fi
K=$($PY -c "import json;print(len(json.load(open('$H'))['openings']))")
P=$(( K < 200 ? K : 200 ))
say "holdout v10 openings $K, pairs per opponent $P"
OUT=$E/ab_e5q_gen063_vs_gen045_400.json
if [ ! -f $OUT ]; then
  need_ac; say "ab start"
  $PY -m dojozero.evaluation.ab --baseline dojo:runs/e5e_rl_T/models/gen_045.pt --candidate dojo:runs/e5e_rl_T/models/gen_063.pt --opponents pela_2023 yixin_2018 --openings renju_holdout_e5_v10 --pairs $P --time-ms 500 --feature "T'' gen_063 vs gen_045, E5q 400-pair external power check" --hypothesis "gen_063 scores differently from gen_045 against external engines" --out $OUT 2>&1 | grep -v Warning | grep -E "decision|mean_diff" | head -3
  [ -f $OUT ] || die "ab failed"
fi
say "ab done: $($PY -c "import json;d=json.load(open('$OUT',encoding='utf-8'));print(d['decision'], d['pooled'], 'cand failures', d['candidate_failures'])")"
say "ALL DONE"
