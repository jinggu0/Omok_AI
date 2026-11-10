#!/bin/bash
# E5r driver (benchmarks/reanalysis/e5r/preregistration.json)
cd /c/temp_git/Omok_AI
export PYTHONPATH=python
PY=.venv/Scripts/python.exe
S='C:/Users/jingg/AppData/Local/Temp/claude/4bf386c95b33b8d4/6d8070cf-753e-4993-af2c-209a92d9c3e9/scratchpad'
E=benchmarks/reanalysis/e5r
R=runs/capacity_probe
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
need_ac() { local n=0; while [ "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d '\r')" = "1" ]; do [ $((n % 15)) -eq 0 ] && say "on battery - waiting for AC"; n=$((n+1)); sleep 120; done; }
die() { say "STOP: $*"; exit 1; }
nohup powershell -NoProfile -ExecutionPolicy Bypass -File "$S/loadlog2.ps1" -out $E/background_load_e5r.csv >/dev/null 2>&1 &
say "E5r driver start"
for a in "4 48" "6 64" "10 96" "10 128"; do set -- $a
  [ -f $R/init_b$1c$2.pt ] || $PY -m dojozero.tools.capacity_probe init --blocks $1 --channels $2 --out $R/init_b$1c$2.pt 2>&1 | grep -v Warning
done
if [ ! -f $E/throughput.json ]; then
  need_ac; say "throughput start"
  $PY -m dojozero.tools.capacity_probe throughput --ckpts runs/e5e_rl_T/models/gen_045.pt $R/init_b4c48.pt $R/init_b6c64.pt $R/init_b10c96.pt $R/init_b10c128.pt --positions 40 --out $E/throughput.json 2>&1 | grep -v Warning || die "throughput failed"
fi
[ -f $E/val_gen045.json ] || $PY -m dojozero.tools.capacity_probe val --ckpt runs/e5e_rl_T/models/gen_045.pt --out $E/val_gen045.json 2>&1 | grep -v Warning | cut -c1-300
for a in "6 64 1" "6 64 2" "4 48 1" "10 96 1" "10 128 1"; do set -- $a
  d=$R/b$1c$2_s$3
  [ -f $d/step_6000.pt ] && { say "fit b$1c$2 s$3 exists - skip"; continue; }
  need_ac; say "fit b$1c$2 s$3 start"
  $PY -m dojozero.tools.capacity_probe fit --blocks $1 --channels $2 --seed $3 --out-dir $d > "$S/e5r_fit_b$1c$2_s$3.out" 2>&1 || { tail -5 "$S/e5r_fit_b$1c$2_s$3.out"; say "fit b$1c$2 s$3 FAILED (arm stopped)"; continue; }
  cp $d/metrics.jsonl $E/metrics_b$1c$2_s$3.jsonl
  say "fit b$1c$2 s$3 done: $(grep -v train_ $d/metrics.jsonl | grep '"step": 6000' | cut -c1-260)"
done
say "ALL DONE"
