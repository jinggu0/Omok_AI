# DojoZero — PROGRESS.md

Phase-by-phase log: what was done, what was **verified** (with the command), what deviated from plan,
and what is still open. Newest phase first.

---

## Current Strength Snapshot (갱신: 2026-10-19, PRD v0.2 / EVALUATION §16 이름 규칙)

측정하지 않은 항목은 `UNKNOWN`. 추정값을 실측처럼 쓰지 않는다. 외부 엔진 평가 인프라(E1)는 아직 없다.

| 항목 | 값 |
|---|---|
| version | Phase 7 best = `taraguchi_v1` registry v1 (Phase 6 best `selfplay_v2` v2에서 이어 학습) |
| best checkpoint | `runs/taraguchi_v1/models/gen_001.pt` |
| external calibration date | UNKNOWN (외부 엔진 대국 0회) |
| strongest opponent with score ≥ 50% | 내부: Phase 6 best `selfplay_v2` v2 (prefix 오프닝 0.67, 전체 무작위 오프닝 0.51) — 외부: UNKNOWN |
| weakest opponent with score < 50% | UNKNOWN (점수 50% 미만 상대 미측정) |
| Rapfi / KataGomo / AlphaGomoku / JAX / Embryo score | UNKNOWN / UNKNOWN / UNKNOWN / UNKNOWN / UNKNOWN |
| `dojo_external_elo` (CI) | UNKNOWN |
| `dojo_internal_elo` | taraguchi v1 vs selfplay v2: +7 [−88, +102] (무작위 오프닝을 NORMAL까지 진행한 시작점, 50쌍, 64 sims); selfplay v2 vs v0 +210 [+97, +322]; heuristic 대비 0.95–0.975 (20쌍 앵커) |
| `dojo_taraguchi_elo` | taraguchi v1 vs selfplay v2: **+123 [+22, +224]** (prefix 오프닝 50쌍, 64 sims, 엔진이 opening 결정) — 절대 척도 아님(기준점 = selfplay v2) |
| tactical score | 네트워크 단독 83.6%, 탐색 포함 91.8% (102문제, Phase 4–5; 이후 모델은 미측정) |
| primary known weakness | 측정 근거 있는 것: 탐색 속도(EVALUATION §8.2 미달 항목), opening 결정 붕괴(2·3수 후 swap 100%, offer10 98%), 색 편향(흑으로 0.74–0.83 / 백으로 0.28–0.35). 외부 상대 기준 약점은 UNKNOWN |

---

## Phase 7 — Taraguchi-10 opening RL + league (2026-10-19)

### Delivered
| 영역 | 파일 |
|---|---|
| opening 헤드 손실(swap/branch/propose/select, `L_opening`) | `python/dojozero/trainer/losses.py` |
| replay: 모든 head의 target 기록·검증(phase, 합법 action, legal mask, 헤드 혼합 거부), `final_hash` 무결성 검사 | `python/dojozero/replay/buffer.py` |
| self-play: 시작 모드(empty / random_opening / mixed prefix), opening 결정 기록, league 상대(엔진·heuristic·random) 그룹 탐색, learner 수만 기록 | `python/dojozero/selfplay/generate.py` |
| PFSP-hard league(고정 멤버 + 직전 best, 같은 checkpoint 중복 제거), prefix 오프닝 평가, 세대별 opening 통계·Opening DB·제안 regret | `python/dojozero/rl/loop.py`, `python/dojozero/evaluation/opening.py` |
| prefix 오프닝 생성기(서로 다른 (canonical hash, phase)), `--openings prefix` | `python/dojozero/evaluation/match.py` |
| 설정 | `configs/training/stage_f_taraguchi.yaml` |
| 문서 v0.2(지속적 Strength Evaluation System 명세 — 코드 없음) | PRD / EVALUATION Part II / ARCHITECTURE §3.12 / ASSUMPTIONS / TRAINING / 이 파일 Snapshot |

### Verified
| 검증 | 결과 |
|---|---|
| `pytest tests/python` | **41/41** (신규 8: opening target 기록·헤드 도달·손실, 잘못된 헤드 target 거부, league 기록 범위, PFSP·league 구성, prefix 오프닝, opening 통계/DB/regret, Phase 7 1세대 루프) |
| C++ | Phase 7에서 C++ 변경 없음 → 재실행 안 함(Phase 6 기준 246 + 1 skip) |
| `taraguchi_v1` (3세대, 73분, XPU, 단일 프로세스 확인·LOCK) | 세대1 **승격**(prefix 오프닝 SPRT 40쌍 0.688); 세대2·3 **INCONCLUSIVE**(각 80쌍 0.588, LLR 2.14 < 2.94 → v1 유지). 세대2·3의 최종 수치가 소수점까지 같아 확인: 라운드별 경로가 다름(0.550/0.600/0.600 vs 0.650/0.613/0.608) — 독립 평가가 같은 pentanomial 개수로 끝난 우연 |
| v1 vs selfplay v2, **prefix 오프닝** 50쌍 64 sims (`runs/phase7_eval/t1_vs_v2_prefix.json`) | **0.67, +123 [+22, +224]** — opening 결정까지 포함한 강도 개선(CI 하한 > 0) |
| v1 vs selfplay v2, **NORMAL까지 진행한 무작위 오프닝** 50쌍 (`..._full.json`) | 0.51, +7 [−88, +102] — **INCONCLUSIVE**: 중반 강도 변화는 감지되지 않음(희생 증거도, 개선 증거도 없음; RQ-08 예비 결과) |
| 앵커(best vs heuristic, prefix 20쌍) | 0.975 / 0.95 / 0.95 |

세대별 opening 지표(self-play, prefix로 정해진 결정 제외):
| 세대 | 흑 승 | P0 점수 / 흑 holder 점수 | swap@1 / @2 / @3 | offer10 | 제안 regret(평균, 20국면) | NORMAL 시작 국면 서로 다름 |
|---|---|---|---|---|---|---|
| 1 (v0) | 144/256 | 0.544 / 0.578 | 0.19 / 1.00 / 0.74 | 0.97 | 0.54 | 204/204 |
| 2 (v1) | 127/256 | 0.514 / 0.519 | 0.93 / 1.00 / 1.00 | 0.98 | 0.26 | 185/185 |
| 3 (v1) | 147/256 | 0.426 / 0.568 | 0.90 / 1.00 / 1.00 | 0.99 | 0.26 | 188/188 |
Opening DB 5,324 항목(`runs/taraguchi_v1/opening_db.json`), 세대당 학습 타깃 ~6.5k(opening head ~2k).

### 사고·발견 (중요)
1. **opening 결정 붕괴**: 2·3수 후 swap이 100%, Choice는 offer10 97–99%. Gumbel 노이즈만으로는 다양성이 유지되지 않음(PRD §30 "self-play 다양성 붕괴"). swap 확률이 정말 최적인지, 학습 신호 부족으로 굳은 것인지는 **미판정** — 분석 탐색 기준(EVALUATION §7)이 아직 없다.
2. **제안 regret 0.54 → 0.26**은 같은 네트워크의 value로 같은 네트워크의 제안을 잰 값 — 내부 일관성 지표이지 절대 품질이 아니다.
3. **League 점수는 강도 지표가 아니다**: learner 쪽은 64/16 sims, 상대는 32 sims라 같은 모델(selfplay_v2_best = 초기 best)에게도 0.76(세대1). PFSP 가중치가 이 compute 차이로 왜곡될 수 있음.
4. **self-play 시간 증가 미해명**: 같은 256판에 688 → 858 → 1,097초(Phase 6: 134–212초). Phase 6 대비 증가는 빈 보드 시작(opening 전 과정 탐색)·league 상대 탐색 그룹 추가로 설명되지만 세대 간 증가 원인(발열·메모리·모델 캐시)은 측정하지 않았다.
5. 기록 손상 테스트(action) 실패 → 원인은 코드가 아니라 테스트: OFFER 단계에서 `"A1"`은 `propose:A1`로 해석되어 **다른 합법 게임**이 된다. 테스트를 불법 착수(점유된 점)로 바꾸고, 기록 무결성을 위해 meta `final_hash` 검증을 추가.
6. replay `_Block` 생성 시 위치 인자 순서 오류(root value가 head 자리로)를 실행 전 발견·수정(키워드 인자).

### Milestone 5 판정
- opening 전체 학습 가능, swap/branch/제안/선택 action 지원 — **충족**.
- "opening regret 감소" — 네트워크 기준 regret은 감소(0.54 → 0.26), 분석 탐색 기준 regret은 **미측정**.
- Strength gate(PRD v0.2): Taraguchi-10 full-game 보고를 Standard 보고와 분리 — prefix(+123) vs 무작위 오프닝(+7)으로 분리 측정은 **충족**, 별도 rating DB·외부 비교는 E1 이후.

### Open
- opening 결정 탐험(swap/branch에 별도 노이즈·온도), 분석 탐색 기준 regret, Approach B(top-k) 비교(ADR-06).
- 세대2·3 연속 INCONCLUSIVE → 세대당 데이터·학습량 또는 SPRT 상한(80쌍) 재검토(§19: INCONCLUSIVE는 성공 아님).
- PFSP를 compute 대칭 조건으로 측정, 비이행성 지표 미구현.
- 외부 Strength Evaluation 인프라(E1)는 Phase 8 착수 시.

---

## Phase 6 — Self-play RL, replay buffer, promotion (2026-10-19)

### Delivered
| 영역 | 파일 |
|---|---|
| Self-play 생성(게임 병렬, Gumbel+노이즈, playout cap randomization) | `python/dojozero/selfplay/generate.py` |
| 기록 형식(gzip JSONL, action 수순 + 위치별 개선 정책/actor/루트값 + 메타, 원자적 쓰기) | `python/dojozero/selfplay/records.py` |
| Replay(C++ 재생·기록 검증·feature 재계산, 세대 창, D4 증강) | `python/dojozero/replay/buffer.py` |
| SPRT(pentanomial 정규근사) | `python/dojozero/evaluation/sprt.py` |
| 세대 루프(self-play → replay → 학습 → SPRT 승격 → 휴리스틱 앵커), 레지스트리, **실행 잠금** | `python/dojozero/rl/loop.py`, `configs/training/stage_d_selfplay.yaml` |
| Gumbel 루트 정체 수정, 탐색 진단 덤프 | `cpp/search/search.*`, `cpp/bindings/module.cpp` |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release / Debug(탐색) | 247개 중 246 통과 + 1 skip / 12/12 |
| `pytest tests/python` | 33/33: 기록 왕복, 손상 기록 4종(action·정책·결과·actor) 거부, z 부호, 창 축출, 증강 후 정책 합법성, SPRT, 오프닝 중복 없음, 실행 잠금, 1세대 루프 + 재개 |
| `selfplay_v2` (3세대, 23분, XPU) | 세대1 승격(SPRT 60쌍 0.633), 세대2 미승격(80쌍 0.513, 결론 없음 → 유지), 세대3 승격(40쌍 0.638) |
| 공정 비교(새 오프닝 세트, 50쌍씩) | v0 vs 휴리스틱 0.88 (+346 [+202, +497]); **v2 vs 휴리스틱 0.97 (+604 [+346, …])**; **v2 vs v0 0.77 (+210 [+97, +322])** |
| 재현성 | 같은 시드의 1세대 self-play가 별도 실행에서 동일한 통계(256판, 타깃 3,460, 흑 148승) |

Self-play 통계: 세대당 256판, 학습 타깃 3.5–4.1k, 평균 36수, **흑 승률 58–66%**, 무승부 0. 세대당 self-play 134–212초, 학습 150 step 14–54초, 승격 판정 93–264초.

### 사고·발견 (중요)
1. **중복 실행 사고**: 루프가 처음 "exit 127로 종료"된 것처럼 보였으나 실제로는 bash 래퍼만 종료되고 Python 프로세스가 살아 있었다. 다시 시작한 루프와 **두 프로세스가 같은 실행 디렉터리의 청크·체크포인트·레지스트리를 덮어썼다**(로그 모든 줄 중복, 노트북 절전까지 겹쳐 self-play 40,845초). 해당 실행은 `runs/selfplay_v1_INVALID_duplicate_processes`로 격리하고 결과를 쓰지 않았다. 대응: `runs/<run>/LOCK`(살아 있는 PID면 시작 거부, 죽은 PID면 인수) + 테스트, 재시작 전 프로세스 확인.
2. **오프닝 생성기 무한 루프**: 중앙 탐욕형 오프닝은 서로 다른 오프닝이 적어 256개 요청 시 끝나지 않았다(50·96개는 됐음). 중앙 7×7 균등 무작위로 바꾸고 시도 상한 초과 시 예외. → Phase 5 매치 수치(이전 오프닝)와 Phase 6 수치(새 오프닝)는 직접 비교 불가: **같은 v0가 휴리스틱 상대로 0.69(이전) vs 0.88(새)** — 평가 결과가 오프닝 분포에 크게 좌우된다.
3. **Gumbel 루트 정체 수정은 원인 확정이 아니다**: 후보 집합이 전부 증명된 패배일 때 다른 수가 남아도 선택 불가한 경로를 고쳤고, 수정 후 같은 시드의 self-play가 완주했지만, 새 스트레스 테스트는 수정 전 코드에서도 정체를 재현하지 못했다. 재발 시 원인 파악을 위해 정체 트리 덤프를 예외 메시지에 포함.
4. 메타데이터에 `TorchVersion` 객체를 넣어 안전 로딩(`weights_only=True`)이 거부 → 문자열로 저장(안전 로딩 정책이 의도대로 작동).
5. 데이터량 대비 학습량: 처음 설정(96판·p_full 0.35)은 판당 타깃 4.4개라 세대당 600 step이면 과적합 → 256판·0.75·150 step으로 조정.

### Milestone 4 판정 (self-play 안정 / 승격 동작 / 세대별 Elo 상승)
- 승격: 동작(승격 2회, 결론 없음 1회 유지). Elo 상승: v2가 v0 대비 +210 [+97, +322], 외부 앵커(휴리스틱)도 상승.
- **"24시간 연속 실행 실패 0"은 미검증** — 3세대(23분)만 실행. 오프닝 정책(swap/branch/proposal) 학습 없음(Phase 7).

### Open
- 흑 승률 58–66%: 무작위 Taraguchi 오프닝이 흑에 유리할 가능성 — Phase 7 오프닝 학습/평가에서 측정.
- 장시간 실행(수십 세대), 재분석(reanalysis), 우선순위 샘플링 미구현.

---

## Phase 5 — MCTS / Gumbel search and search verification (2026-10-18)

### Delivered
| 영역 | 파일 |
|---|---|
| 탐색 트리: PUCT·Gumbel 루트(SH), MCTS-solver, actor-aware 백업(P0 관점 값), 리프/루트 전술, virtual loss, 동기·배치 API, 불변식 검사 | `cpp/search/search.*` |
| 기준 봇: random-legal, 1-ply 휴리스틱 | `cpp/search/bots.*` |
| 바인딩: `SearchConfig`, `SearchBatch`(다중 트리 collect/apply, 병렬 `add_many`), 봇, `black_holder` | `cpp/bindings/module.cpp` |
| Python 배치 탐색 드라이버(PyTorch XPU bf16, 버킷 패딩) | `python/dojozero/search/driver.py`, `model/load.py` |
| 쌍대국 매치 러너 + 통계(pentanomial, 정규·Wilson CI, Elo) | `python/dojozero/evaluation/match.py` |
| 벤치마크: 탐색 포함 전술 해결률(`net_tactics --sims`), 트리 비용(`dojozero_bench_search`) | |

### Verified — 탐색 정확성
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release / Debug(탐색·오프닝·규칙) | 246개 중 245 통과 + 1 skip / 28/28 |
| `pytest tests/python` | 22/22 (배치 크기 1 vs 256 결과 동일, 모든 phase에서 합법 action, 망 탐색이 즉시승 증명) |
| swap 결정(플레이어≠색) | 첫 결정으로 승자가 고정되는 일관된 평가기에서 PUCT·Gumbel 모두 올바른 swap/keep 선택, 자식 Q = ±0.9 정확 |
| 같은 플레이어 연속 결정 (CHOICE_AFTER_4 → SWAP_BEFORE_DIRECT_5 / OFFER) | 두 모드 모두 올바른 분기, 자식 Q가 정확값(0.9/0.6)과 1e-3 이내 — negamax식 부호 반전이면 실패하도록 설계 |
| MCTS-solver | 루트 전술 없이 균등 prior로 열린 삼 → straight four 승리를 트리에서 증명 |
| 배치 경로 = 동기 경로 | in-flight 1에서 방문 수까지 동일 (PUCT·Gumbel) |
| virtual loss 배치(in-flight 8) | 트리 불변식(visits = 1 + Σ자식) 유지, in-flight 0으로 종료 |

### Verified — 강도 (체크포인트 `runs/stage_b_b6c64_v1/checkpoints/step_6000.pt`, OMOKDOJANG_DAN, 무작위 Taraguchi 오프닝 50쌍 = 100판, 역할 교대)
| A vs B | 점수(A) | Elo [95% CI] | A 흑/백 점수 |
|---|---|---|---|
| 엔진(PUCT 64 sims) vs random | 1.00 (100-0) | ≥ +446 | 1.00 / 1.00 |
| 엔진(PUCT 64) vs 휴리스틱 | 0.69 (68-2-30) | **+139 [+36, +242]** | **0.48** / 0.90 |
| 정책만(탐색 없음) vs 휴리스틱 | 0.31 | −139 [−242, −36] | 0.16 / 0.46 |
| 엔진(PUCT 64) vs 정책만 | 0.92 | +424 [+254, +612] | 0.88 / 0.96 |
고정 벤치마크 승리 문제 73개: 정책만 83.6% → **탐색 64 sims(루트 전술 끔) 91.8%** (VCF 100%, VCT 76.9%); 미학습+탐색 16.4%.

### 성능
- 트리 비용(균등 평가기, 단일 스레드): PUCT 34–39k sims/s, Gumbel 11–17k sims/s — 목표 500k **크게 미달**. 확장마다 `legal_actions`(빈 점별 금수 판정 ~40µs)가 지배. Gumbel은 σ 계산이 루트 엣지를 매번 순회.
- 망 평가(탐색 중, B6C64 XPU bf16): 실행마다 0.4k–7.5k eval/s로 편차 큼(XPU 워밍업·내장 GPU 클럭). 고정 버킷 패딩 적용.
- 매치 50쌍(엔진 vs 휴리스틱) 525초. 루트 전술(5,000 노드) 직렬 실행이 라운드당 42초를 만들던 것을 `add_many` 병렬화로 18.7초 → 이후 ~5초.

### 계획 대비 차이 / 발견
1. **Milestone 3 "random/heuristic 봇 결정적 격파"는 부분 달성**: random은 결정적, 휴리스틱은 통계적으로 유의(+139)하지만 **엔진이 흑일 때 0.48** — 흑 플레이(금수 제약)가 약점. 학습 데이터가 무작위 국면 전술 위주이고 self-play가 아직 없음.
2. **ADR-03(PUCT vs Gumbel)은 미결**: 매치는 PUCT로만 했다. Gumbel은 정확성 테스트만 통과.
3. 첫 테스트 실패 2건은 엔진 버그가 아니라 테스트 설계 오류였다(균등 prior + FPU 감점으로 E8 미탐색; 이후 swap으로 흑을 다시 가져갈 수 있어 "흑이 이긴다" 평가기가 트리와 모순). 트리와 일관된 평가기로 교체 — 이것이 actor-aware 백업을 정확히 검증하는 형태.
4. 트리 재사용, 루트 Dirichlet 노이즈, 시간 기반 한도 미구현(Phase 6).
5. 오프닝 결정 헤드(swap/branch/propose/select)는 아직 학습되지 않아 사실상 무작위 prior.

### Open
- 흑 약점 원인 분석(금수 관련 수 선택? 가치 head 보정?) — Phase 6 self-play 데이터로 재평가 필요.
- 증분 금수/위협 캐시(ADR-01 C안): 트리 비용·전술 solver·legal_actions 공통 병목.

---

## Phase 4 — Neural network, dataset pipeline, supervised pretraining (2026-10-18)

### Environment
`.venv` (Python 3.11.9): torch **2.14.1+xpu** (공식 XPU 인덱스), numpy 2.4.6, PyYAML 6.0.3, pytest 9.1.1 (`runs/venv_freeze.txt`). Intel Arc 140V 드라이버 32.0.101.8626. `torch.xpu.is_available() == True`.

### Delivered
| 영역 | 파일 |
|---|---|
| C++ feature encoder (26 plane 기본, 옵션 history/tactical/forbidden), legal mask, aux targets | `cpp/features/features.*` |
| pybind11 바인딩 `dojozero._core` (GameState, encode, encode_boards[멀티스레드], aux, transform table, analyze_tactics) | `cpp/bindings/module.cpp`, CMake `_core` |
| Stage A 데이터 생성기 (solver 라벨, NPY + manifest) | `cpp/tools/datagen.cpp` → `dojozero_datagen` |
| Python 패키지: feature spec·D4 증강, ResNet policy/value/aux/opening heads, 마스킹 손실, 데이터셋(캐시·결정적 split·벤치마크 누출 제외), 체크포인트, 트레이너, 평가 | `python/dojozero/`, `pyproject.toml` |
| 장치 벤치마크, 네트워크 전술 해결률 평가 | `python/dojozero/tools/bench_device.py`, `evaluation/net_tactics.py` |
| 설정 | `configs/training/stage_b_tactical.yaml` |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release | 235개 중 234 통과, 1 skip (변화 없음) |
| `pytest tests/python` | 17/17 통과: D4 증강 = C++ 재인코딩(8변환, 2 profile), legal mask = legal_actions, encode_boards = 단일 encode, 손실 마스킹/불법 target 거부/NaN 감지, **CPU 결정적 모드에서 체크포인트 재개가 손실 단위로 정확히 일치**, XPU foreach 회피 |
| XPU 수치 | 같은 배치·가중치에서 CPU 대비: fp32 손실 일치(소수 5자리), 기울기 상대오차 1.2e-4; bf16 기울기 상대오차 8.7%(bf16 정밀도 수준) |
| Stage A 데이터 | 120,000 샘플, 중복 1건 제거, 라벨 분포는 TRAINING §2.1 |
| Stage B 학습 (`runs/stage_b_b6c64_v1`, B6C64 422k 파라미터, 6,000 step × b256, XPU bf16) | holdout(5%, 벤치마크 국면 제외): policy top-1 즉시승 100% / VCF 70.9% / VCT 54.5% / 강제차단 100%(n=38) — 무작위 합법수 기준선 0.5–1.2%; 증명 가치 정확도 98.0% |
| 네트워크 단독(탐색 없음) 고정 벤치마크 승리 문제 73개 | **83.6% 해결**(VCF 90.9%, VCT 69.2%, 금수유도 100%, 1수승 100%) vs 미학습 13.7%. 해결 판정은 C++ solver(네트워크 수 이후 상대가 증명된 패배) |
| XPU 크래시 후 재개 | step 2000 체크포인트에서 재개, 같은 step 손실이 원 실행과 0.1% 이내 |

### 계획 대비 차이 / 발견 (중요)
1. **PyTorch XPU 백엔드 문제 3건 발견·우회**:
   (a) 불리언 마스크 인덱싱(동적 shape)이 `UR_RESULT_ERROR_OUT_OF_RESOURCES`/네이티브 assert → 손실을 마스크 가중합으로 재작성(수학적으로 동일, 테스트로 확인);
   (b) **foreach AdamW가 간헐적으로 잘못된 업데이트**(한 스텝에 |w| 1→31, 이후 NaN) → `foreach=False`;
   (c) `clip_grad_norm_`의 foreach 커널에서 네이티브 크래시(학습 2,800 step에서 종료) → `foreach=False`. 회귀 테스트 추가.
   NaN 감지가 (b)를 즉시 잡아냈다(조용한 실패 없음). PRD 리스크 R-04가 실제로 발생한 사례.
2. **ONNX export 미구현** — onnx/onnxruntime 미설치. 추론 런타임 결정(ADR-02)은 Phase 5로.
3. 학습 데이터·벤치마크가 **같은 생성 방식(무작위 대국)** 이라 벤치마크 해결률은 분포 내 진단이다. 실전 기력 근거 아님.
4. 데이터 생성 중 노트북 절전으로 추정되는 벽시계 공백(~6시간)이 있었다(datagen `seconds` 값과 pytest 소요시간 표기에 영향, 결과에는 무영향).
5. 보조 head 중 **forbidden recall 14.8%**, three_threat recall 68%: 양성 라벨이 드물어 BCE가 0 쪽으로 치우침(forbidden은 입력 plane에도 있어 사실상 복사 문제). pos_weight 조정은 다음 단계 후보.
6. policy 라벨이 "증명된 첫수 1개"라 다른 승리수를 둔 경우 오답 처리됨 → holdout top-1은 하한 추정.

### Open
- OQ-03(목표 기력 정의·외부 GPU), OQ-04/05(외부 엔진) 미결.
- Phase 5 전제: 추론 런타임(ADR-02) 측정, MCTS nodes/s, 전술 solver 속도(Phase 3 미달 항목).

---

## Phase 3 — Tactical solver + tactical benchmark (2026-10-18)

### Delivered
| 영역 | 파일 |
|---|---|
| ThreatDetector (five point, four, 이중위협, three; 흑 금수 반영) | `cpp/tactics/threats.*` |
| VCF solver (DFS, 실패 TT, 노드·깊이·시간 예산, 금수 유도) | `cpp/tactics/vcf.*` |
| df-pn proof-number solver = VCT/threat-space (PROVE_WIN / PROVE_LOSS) | `cpp/tactics/dfpn.*` |
| 독립 증명 검증기 (GameState만, 방어자 full-width) | `cpp/tactics/verify.*` |
| 전술 의사결정 `analyze_tactics` (즉시승 → 방어불가/강제차단 → VCF → VCT → 증명된 패배) | `cpp/tactics/engine.*` |
| 벤치마크 러너 + CLI `tactics`, `benchmark`; 도구 `dojozero_tactics generate|fuzz|perf` | `cpp/tactics/benchmark.*`, `cpp/cli`, `cpp/tools/tactics_tool.cpp` |
| 고정 벤치마크 102문제 (수작업 17 + 생성·검증 85) | `benchmarks/` |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release | 235개 중 234 통과, 1 skip (호환성 증거 없음) |
| `dojozero_tests` Debug (assert 활성, 생성 스위트 제외·수작업 스위트 포함) | 150개 중 149 통과, 1 skip |
| `dojozero_engine benchmark --suite benchmarks` | 102/102 통과, **false claim 0**, 증명 기대 문제 97/97 해결 |
| `dojozero_tactics fuzz --positions 4000 --threads 7 --seed 11` | WIN/LOSS 주장 3,648건(승 3,018 / 패 630) 중 3,627 VERIFIED, 21 TOO_LARGE, **REFUTED 0** |
| 검증기 음성 테스트 | 가짜 WIN·가짜 LOSS·금수 증명수 모두 REFUTED |
| VCF PV 재생 | GameState 재생 시 공격자 승리로 종국 |
| 생성 문제 | 3세트 모두 생성 시점에 full-width 검증 통과한 것만 수록 (검증 불가 5건은 제외), RENJU_RIF·OMOKDOJANG_DAN 혼합 |

### 성능 (Release, 단일 스레드, 무부하, 300 국면) — EVALUATION §8.2
VCF 70–78k nodes/s (목표 1M **미달**), df-pn 13.5–14k nodes/s, `analyze_tactics` 지연 p99 4.5s(노드 예산만) / 311ms 이하(`max_ms=500`).
최적화 1건 적용 후 수치(적용 전 VCF 37k, df-pn 5.2k, p99 13.5s). 결과(증명 수)는 최적화 전후 동일.

### 계획 대비 차이 / 발견
1. **별도 ThreatSpaceSearch 모듈 없음** — df-pn의 threat-restricted OR 노드가 그 역할 (ARCHITECTURE §3.7 표).
2. **VCF 성능 목표 미달** — 근본 해결은 증분 위협 추적(line code). ADR-01 C안과 함께 Phase 5 전 재검토.
3. 검증기는 속도를 위해 위협 검출기의 **힌트**(후보 수 제안)를 쓴다. 승패 판정은 GameState만 하므로 힌트 오류는 "검증 실패"로만 나타나고 거짓 VERIFIED는 불가.
4. 생성기 버그 발견·수정: 무작위 대국 생성기가 opening 결정 phase에서 멈춰 OMOKDOJANG_DAN 국면이 전혀 생성되지 않았음 → 결정도 무작위로 두도록 수정 후 세트 재생성.
5. 시간 제한(`max_ms`)은 기본 0(끔): 테스트·벤치마크 재현성을 위해 노드 예산만 사용.

### Open
- "VCF 없음"(NONE) 주장은 독립 검증 수단이 없다 — 수작업 세트에서 자명한 경우(합법 four 자체가 없음)만 기대값으로 사용.
- 전술 문제의 독립 출처(문헌 문제집·타 엔진) 0건 — PRD R-07 유지.
- df-pn 증명은 메모리 상한 없는 TT를 쓴다(노드 예산으로만 제한).

### Milestone 2 판정
"VCF solver functional / fixed tactical benchmark pass / WIN 오답 0" — 충족. 성능 목표는 미달(위 기록).

---

## Phase 2 — Rule engine + tests (2026-10-18)

### Delivered
| 영역 | 파일 |
|---|---|
| 좌표·D4 변환·Zobrist·보드(A: padded array, 8중 증분 hash) | `cpp/board/` |
| 흑 금수 판정 (fast, 완전 재귀) + 설명(`ForbiddenInfo`, pre-order 재귀 경로) | `cpp/rules/forbidden.{h,cpp}` |
| 참조 oracle (다른 정식화: 5칸 window, 보드 복사) | `cpp/rules/reference_rules.{h,cpp}` |
| RuleProfile + JSON (`configs/rules/*.json`, 미지 키 거부, hash) | `cpp/rules/rule_profile.*` |
| Taraguchi-10 상태 머신, PlayerId/Color/Role 분리, swap, 10제안 | `cpp/opening/`, `cpp/game/game_state.*` |
| 10제안 대칭 판정(E1/E3) + 검증 | `cpp/opening/symmetry.*` |
| 국면 입력(JSON: moves/black/white/lines/diagram) + malformed 거부 | `cpp/game/position_spec.*` |
| CLI `dojozero_engine rules-test | forbidden | legal` | `cpp/cli/main.cpp` |
| 퍼징 `dojozero_fuzz`, 벤치마크 `dojozero_bench` | `cpp/tools/` |
| 테스트 124개 (GoogleTest) + 호환성 러너 + 템플릿 | `tests/` |

### Verified (실행한 명령과 결과)
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release | 123 통과, 1 skip (오목도장 증거 0건이라 호환성 러너 skip — 의도된 동작) |
| `dojozero_tests` Debug (assert 활성) | 동일: 123 통과, 1 skip |
| JSON 금수 사례 76개 (`tests/forbidden/cases/`) | 각 사례마다 손으로 쓴 기대값 = fast = explain = oracle = D4 8변환 전부 일치 |
| 1D 전수 열거 (±5칸, 모든 열, 354,051 패턴) | fast = oracle, A-R3·A-R4 반례 0건 |
| `dojozero_fuzz --comparisons 100000000 --threads 8 --seed 1` | 100,000,503 비교 / 980,775 국면, **불일치 0**, 847초. 겉보기 3-3 1,424,446곳 중 거짓금수 387,365곳; 재귀 최대 깊이 분포 1:1,310,769 / 2:101,116 / 3:11,130 / 4:1,313 / 5:113 / 6:5 |
| E1 = E2 (10제안 대칭) | 도달 가능한 4수 국면 8,464개 × 모든 빈 점 일치 |
| Opening 결정 경로 | 40개 경로 전부 actor/전이/undo 확인 |
| 무작위 대국 apply/undo/replay/D4 property | RENJU_RIF 40판, TARAGUCHI10 25판, OMOKDOJANG_DAN 25판 위반 0 |
| 깊은 거짓금수 3건 (깊이 2·3·4) | 퍼저가 찾은 국면을 RULES 정의로 단계별 수작업 검토 → 엔진과 일치, 회귀 사례로 고정 |

### 성능 (Release, 단일 스레드, 무부하) — EVALUATION §8.2
apply+undo 3.4M/s (목표 10M **미달**), Board place+remove 74M/s, 흑 금수 판정 2.1–2.9M/s (목표 1M 충족),
legal_actions 25k/s (목표 200k **미달**). 최적화 2건 적용 후 수치(적용 전 1.95M/s / 8.5M/s / 23k/s).

### 계획 대비 차이 (deviations)
1. **Board C(line code + 생성 테이블) 미구현** — ADR-01을 A로 부분 결정. 테이블 생성기가 없으므로 "테이블 = oracle" 열거 검증도 해당 없음(대신 fast = oracle 열거).
2. **pybind11 바인딩 미구현** — ADR-05를 Phase 4로 연기.
3. "금수 엔진 수백 개 unit test" 목표: 독립 손검토 사례는 76개(각 9중 검증)이며, 그 외는 열거·퍼징으로 대체. 개별 사례 수 기준으로는 목표 미달.
4. A-R5(흑이 둘 곳이 모두 금수 → 무승부) 경로는 구현만 되어 있고 **테스트 없음** (해당 국면 구성 미완).

### 아직 검증되지 않은 것 (open)
- **독립 출처 부족 (PRD R-07)**: oracle도 같은 사람이 같은 RULES.md 해석으로 작성했으므로, fast = oracle 일치는 "구현 간 일관성"이지 "RIF 해석의 정확성"이 아니다. 문헌 문제·타 엔진 차등 비교(OQ-04)·오목도장 앱 증거(OQ-01)가 아직 0건.
- 오목도장 호환성 C-01..C-20 전부 `UNVERIFIED_APP_BEHAVIOR`.
- 금수 판정 지연 max 0.4–1.1ms 이상치 원인 미조사.
- Linux/Clang 빌드 미확인 (MSVC만).

### 다음 (Phase 3 진입 조건)
규칙 테스트 100% 통과 — 충족. Phase 3(전술 solver)은 위 open 항목과 독립적으로 진행 가능.
