#!/bin/bash
# E5p driver (benchmarks/reanalysis/e5p/preregistration.json). Each stage is skipped if its output exists.
cd /c/temp_git/Omok_AI
export PYTHONPATH=python
PY=.venv/Scripts/python.exe
S='C:/Users/jingg/AppData/Local/Temp/claude/4bf386c95b33b8d4/6d8070cf-753e-4993-af2c-209a92d9c3e9/scratchpad'
E=benchmarks/reanalysis/e5p
G045=runs/e5e_rl_T/models/gen_045.pt
FT1=runs/e5p_ft_r1/checkpoints/step_1500.pt
FT2=runs/e5p_ft_r2/checkpoints/step_1500.pt
D2=data/reanalysis_e5p_r2
EXCL="renju_balanced_v1 renju_provisional_v0 renju_smoke_v0 renju_train_reanalysis_v1 renju_holdout_e4_v1 renju_holdout_e5_v2 renju_holdout_e5_v3 renju_holdout_e5_v4 renju_holdout_e5_v5 renju_gate_e5_v1 renju_holdout_e5_v6 renju_holdout_e5_v7"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
need_ac() { local n=0; while [ "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d '\r')" = "1" ]; do [ $((n % 15)) -eq 0 ] && say "on battery - waiting for AC"; n=$((n+1)); sleep 120; done; }
die() { say "STOP: $*"; exit 1; }
nohup powershell -NoProfile -ExecutionPolicy Bypass -File "$S/loadlog2.ps1" -out $E/background_load_e5p.csv >/dev/null 2>&1 &
say "E5p driver start"

finetune() { # $1 round
  local ck=runs/e5p_ft_r$1/checkpoints/step_1500.pt
  [ -f $ck ] && { say "FT$1 exists - skip"; return; }
  say "fine-tune round $1 start"
  $PY -m dojozero.trainer.reanalysis_train --config configs/training/e5p_ft_r$1.yaml > "$S/e5p_ft_r$1.out" 2>&1 || { tail -5 "$S/e5p_ft_r$1.out"; die "fine-tune $1 failed"; }
  say "fine-tune round $1 done: $(tail -1 runs/e5p_ft_r$1/metrics.jsonl)"
}
holdout() { # $1 version $2 seed
  [ -f benchmarks/openings/renju_holdout_e5_v$1.json ] && { say "holdout v$1 exists - skip"; EXCL="$EXCL renju_holdout_e5_v$1"; return; }
  need_ac; say "holdout v$1 start"
  $PY -m dojozero.evaluation.engine_games balance --candidates 200 --target 60 --seed $2 --set-id renju_holdout_e5_v$1 --exclude-sets $EXCL 2>&1 | grep -v Warning | tail -3
  [ -f benchmarks/openings/renju_holdout_e5_v$1.json ] || die "no holdout v$1"
  EXCL="$EXCL renju_holdout_e5_v$1"
}
ab() { # $1 candidate $2 holdout version $3 out $4 feature
  [ -f $3 ] && { say "$3 exists - skip"; return; }
  need_ac; say "ab start: $1 on v$2"
  $PY -m dojozero.evaluation.ab --baseline dojo:$G045 --candidate dojo:$1 --opponents pela_2023 yixin_2018 --openings renju_holdout_e5_v$2 --pairs 60 --time-ms 500 --feature "$4" --hypothesis "the fine-tuned model scores higher than gen_045 against external engines" --out $3 2>&1 | grep -v Warning | grep -E "decision|mean_diff" | head -3
  [ -f $3 ] || die "ab $3 failed"
  say "ab done: $($PY -c "import json;d=json.load(open('$3'));print(d['decision'], d['pooled'], 'cand failures', d['candidate_failures'])")"
}
diag() { # $1 ckpt $2 out
  [ -f $2 ] && return; say "diag start $2"
  $PY -m dojozero.tools.reanalysis_eval --ckpt $1 --split test --out $2 > "$S/e5p_diag_$(basename $2).out" 2>&1 || { tail -5 "$S/e5p_diag_$(basename $2).out"; say "diag $2 failed (descriptive only, continuing)"; return; }
  say "diag done $2"
}

# ---- round 1
finetune 1
holdout 8 209
ab $FT1 8 $E/ab_e5p_ft1_vs_gen045.json "E5p FT1: E4 reanalysis fine-tune of gen_045 on the E5o buffer (1,500 steps)"
dec=$($PY -c "import json;print(json.load(open('$E/ab_e5p_ft1_vs_gen045.json'))['decision'])")
diag $FT1 $E/eval_ft1_test.json
[ "$dec" = "REJECT" ] && die "round 1 REJECT - pre-registered stop after round 1"

# ---- round 2
FT1ID=dojo-$($PY -c "import json;print([json.loads(l) for l in open('runs/e5p_ft_r1/metrics.jsonl')][-1]['model_sha'][:12])")
say "FT1 engine id $FT1ID"
for eng in pela_2023 yixin_2018 katagomo_2021r; do
  out=runs/evaluation/e5p_r2_traingames_vs_$eng.json
  [ -f $out ] && { say "games vs $eng exist - skip"; continue; }
  need_ac; say "games vs $eng start"
  $PY -m dojozero.evaluation.tournament --engine-a dojo:$FT1 --engine-b ext:$eng --openings benchmarks/openings/renju_train_reanalysis_v1.json --pairs 120 --time-ms 500 --tier A --out $out > "$S/e5p_games_$eng.out" 2>&1 || { tail -5 "$S/e5p_games_$eng.out"; die "games vs $eng failed"; }
  af=$($PY -c "import json;d=json.load(open('$out'));print(sum(v for k,v in d['failures'].items() if k.startswith('A:')))")
  say "games vs $eng done: $($PY -c "import json;d=json.load(open('$out'));print(d['summary']['score'], d['summary']['a_score_as_black'], d['summary']['a_score_as_white'], 'failures', d['failures'])")"
  [ "$af" = "0" ] || die "DojoZero failures in training games vs $eng"
done
if [ ! -f $D2/positions.jsonl.gz ]; then
  say "collect"; $PY -m dojozero.tools.reanalysis_data collect --out $D2 --dojo-engine $FT1ID > "$S/e5p_collect.out" 2>&1 || die "collect failed"
fi
if [ ! -f $D2/labels.jsonl.gz ]; then
  need_ac; say "label start"; $PY -m dojozero.tools.reanalysis_data label --out $D2 --ckpt $FT1 > "$S/e5p_label.out" 2>&1 || { tail -5 "$S/e5p_label.out"; die "label failed"; }
  say "label done"
fi
if [ ! -f $D2/arrays.npz ]; then
  say "arrays"; $PY -m dojozero.trainer.reanalysis build --root $D2 --ckpt $FT1 > "$S/e5p_build.out" 2>&1 || { tail -5 "$S/e5p_build.out"; die "build failed"; }
fi
cp $D2/manifest.json $E/manifest_reanalysis_e5p_r2.json
finetune 2
holdout 9 210
ab $FT2 9 $E/ab_e5p_ft2_vs_gen045.json "E5p FT2: E4 reanalysis fine-tune of FT1 on FT1's own external games (1,500 steps)"
diag $FT2 $E/eval_ft2_test.json
say "ALL DONE"
