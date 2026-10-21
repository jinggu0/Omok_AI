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
| external calibration date | 2026-10-19 (ladder v1b: 균형 corpus `renju_balanced_v1`, 500 ms/수, 엔진 간 라운드로빈 포함) |
| strongest opponent with score ≥ 50% | 내부: Phase 6 best `selfplay_v2` v2 — 외부: **없음** (10개 엔진 모두 50% 미만) |
| weakest opponent with score < 50% | Pela 2023 (사다리 최하위 엔진, −411; DojoZero 프로브 0.125) |
| Rapfi / KataGomo / AlphaGomoku / JAX / Embryo score | 균형 corpus 4쌍 프로브: Rapfi 2025 0.0 / KataGomo 2026 0.0 / AlphaGomoku 2026 0.0 / JAX 사용 불가 / Embryo 2026 0.0 |
| `dojo_external_elo` (CI) | **−491 [−634, −348]** (anchor Yixin 2018 = 0; Rapfi 2025 +357, Pela 2023 −411) — 500 ms/수 기준 |
| `dojo_internal_elo` | taraguchi v1 vs selfplay v2: +7 [−88, +102] (무작위 오프닝을 NORMAL까지 진행한 시작점, 50쌍, 64 sims); selfplay v2 vs v0 +210 [+97, +322]; heuristic 대비 0.95–0.975 (20쌍 앵커) |
| `dojo_taraguchi_elo` | taraguchi v1 vs selfplay v2: **+123 [+22, +224]** (prefix 오프닝 50쌍, 64 sims, 엔진이 opening 결정) — 절대 척도 아님(기준점 = selfplay v2) |
| tactical score | 네트워크 단독 83.6%, 탐색 포함 91.8% (102문제, Phase 4–5; 이후 모델은 미측정) |
| primary known weakness | **방어 전술**: 패배의 87%가 상대에게 솔버가 증명하는 강제승(VCF/VCT)을 허용한 한 수, value망은 그 직전 평균 +0.52로 과신. 탐색량 10배로도 개선 없음(Phase 8 E2c). 흑에서 더 약함(Pela 상대 흑 ~0.12 / 백 ~0.41). E3b: 솔버 5× 가속 후에도 블런더 대부분 남음. E3c: 블런더 직전 국면의 74.5%는 전 합법수가 강제패(2–4 ply 전에 이미 짐), 대안이 top-32 밖에만 있는 국면 1.6% → 병목은 후보 범위가 아니라 수 사이에 걸친 깊은(5k–100k 노드) 패배 증명 |

---

## Phase 8 (E3e) — Light proof-aware MCTS: 2–3 solver threads, depth 1 (2026-10-22)

E3d(6스레드·깊이 3)에서 증명 스레드가 MCTS sims를 35–45% 빼앗은 것을 받아, 스레드를 줄이고 루트 자식만 증명(깊이 1)하는 설정을 측정했다. 코드 변경 없음(E3d 코드, 설정만 다름).

**사전 확정 규칙**: 재생(처음 진 결정 국면)에서 진 수가 적은 쪽을 외부 A/B arm으로 고르되, 차이가 3 이하면 스레드가 적은 P2. A/B 판정 규칙은 E3a–E3d와 동일(쌍대 차이 95% 구간 하한 > 0, candidate 실패 0 → ACCEPT).
환경: 전원 연결, GUI·pbrain 종료(사용자 승인 후 종료), 측정 중 다른 프로세스 평균 ~2코어(OneDrive 등, `benchmarks/perf/results/background_load_e3e.csv`). 첫 실행은 사용자 요청으로 중지 후 처음부터 재실행.

### 재생 (노드 심판, UNKNOWN은 진 수로 세지 않음)
| 국면 | arm | 진 수 | 흑 / 백 | 심판 UNKNOWN | sims/수 |
|---|---|---|---|---|---|
| 처음 진 결정 247 | 필터 없음 | 221 | 119/142 · 102/105 | 26 | 192 |
| | P2 깊이 1 | 188 | 94 · 94 | 59 | 185 |
| | **P3 깊이 1** | **170** | 80 · 90 | 77 | 175 |
| | (E3d P6 깊이 3) | 170 | 78 · 92 | 77 | 122 |
| 블런더 247 | 필터 없음 | 232 | 128 · 104 | 15 | 193 |
| | P2 깊이 1 | 207 | 113 · 94 | 40 | 216* |
| | P3 깊이 1 | 201 | 107 · 94 | 46 | 234* |
국면 단위(처음 진 결정): 필터 없음→P3 제거 52 / 신규 1, P2→P3 제거 18 / 신규 0. 규칙에 따라 **P3 선택**(P2보다 18 적음).
\* sims에는 증명된 노드 재방문(신경망 평가 없음)도 포함되어 증명이 많은 arm에서 부풀려진다. 실제 신경망 평가 수는 아래 A/B의 `nn_evals`.

### 외부 paired A/B (A = 증명 끔, B = 증명 3스레드·깊이 1; 500 ms/수, opening 24–63, Pela·Yixin 각 40쌍, 색 교대, arm 교대)
| 상대 | A | B | 쌍대 차이 [95%] | ΔElo (로컬, 점추정) |
|---|---|---|---|---|
| Pela 2023 | 0.312 (흑 0.200 / 백 0.425) | 0.425 (0.375 / 0.475) | +0.1125 [−0.001, +0.226] | +84 |
| Yixin 2018 | 0.025 (0.025 / 0.025) | 0.087 (0.100 / 0.075) | +0.0625 [+0.011, +0.114] | +229 (점수가 0에 가까워 매우 불확실) |
| **합산 80쌍** | 0.169 | 0.256 | **+0.0875 [+0.025, +0.150] → ACCEPT (사전 등록 규칙)** | +92 |
쌍 차이: 0 ×58, +0.5 ×15, −0.5 ×5, +1.0 ×2. 실패 0, DojoZero 금수패 0(Pela 금수패 A 3 / B 1). 수당 시간 A 404–411 ms / B 380–420 ms. 실제 신경망 평가/수: A 162·165 → B 136·153 (−16% / −7%; E3d P6은 −42% / −34%). B의 수당 증명 시도 8.8–9.3, 성공 0.6–1.0, 솔버 노드 1.3–1.6만. 상대 엔진 수당 시간 126 → 141 ms, 151 → 164 ms(B에서 증가).

### 판정과 주의
**ACCEPT (사전 등록 규칙) — 단, 기력 증거는 아직 누적 중**:
- 같은 opening 세트(24–63)로 한 **세 번째** A/B다(E3b +0.025, E3d +0.0125, E3e +0.0875). 세 실행 모두 점추정은 +이지만 다중 비교가 있다 → 기본값 변경 전에 **보류 opening(64–103) 확인 A/B** 권장.
- **같은 계산량 비교가 아니다**: DojoZero는 GPU + CPU 3스레드를 더 쓰고, 상대 엔진의 생각 시간도 B에서 ~10% 늘었다(CPU 경합).
- 측정 중 다른 프로세스가 ~2코어 사용 — 두 arm은 opening마다 교대라 같은 영향을 받았지만 절대 점수는 조용한 머신과 다를 수 있다.
- 흑에서 개선이 큼(재생 처음 진 결정: 흑 119 → 80; Pela 흑 0.200 → 0.375).
- 기본값(`proof_threads = 0`)은 유지. 대국·평가에 쓰려면 스펙 `:proof=3:proofdepth=1`. 셀프플레이는 트리 수백 개가 같은 풀을 쓰므로 별도 측정 전에는 켜지 않는다.

결과: `benchmarks/perf/results/replay_e3e_{decision,blunder}.json`, `benchmarks/perf/results/ab/ab_e3e_proof3_d1.json` + DB `experiment`.

---

## Phase 8 (E3d) — Case A: proof-aware MCTS (2026-10-21)

질문: E3c 권고대로 솔버 증명을 탐색 트리 안(사후 필터가 아니라)에서 MCTS와 동시에 쓰면, 실제로 진 결정 국면에서 지는 수를 피하고 외부 성능이 오르는가?

### Delivered
| 영역 | 파일 |
|---|---|
| 프로세스 전역 증명 풀(솔버 스레드, 트리별 mailbox — 작업자는 트리를 건드리지 않음, 트리 파괴 후 결과는 버림) + 증명 캐시(정확한 WIN/LOSS와 승리 수만 결과로 저장, UNKNOWN은 "시도한 예산"만 따로 기록 — 결과로 쓰지 않음) | `cpp/search/proof_pool.{h,cpp}` |
| 트리 통합: 일정 시뮬레이션마다 깊이 ≤ `proof_max_depth`의 방문 많은 미해결 노드를 골라 "둘 차례인 쪽의 VCF/VCT 강제승" 증명 요청, UNKNOWN이면 다음 시도에서 VCT 예산 4배(4k → 256k), 결과는 기존 MCTS-solver 경로로 표시·전파(증명된 패배 자식은 선택·최종 선택에서 제외), 루트 승리 증명은 그 수를 둠 | `cpp/search/search.{h,cpp}` (`SearchConfig.proof_*`, `SearchResult.proof_*`) |
| Python 설정(`SearchSettings.proof_*`, 스펙 옵션 `:proof=N:proofdepth=D`), 대국 기록에 수마다 증명 통계, 엔진 ID `-proofN` | `search/driver.py`, `evaluation/{match,tournament}.py`, `bindings/module.cpp` |
| 재생 도구: `--positions decision`(E3c backtrack으로 복원한 "처음 진 결정 국면": 블런더에서 거슬러 올라간 연속 REFUTED 수 중 가장 이른 수 직전, 없으면 블런더 국면), 증명 arm, `--judge-ms 0` | `python/dojozero/tools/filter_replay.py` |
| 테스트 | `tests/search/test_proof_mcts.cpp`(5), `tests/python/test_search.py`(1) |

### 정확성
| 검증 | 결과 |
|---|---|
| 증명 인지 탐색이 증명 표시한 루트 자식 → 독립 df-pn 재증명 + GameState 검증기 | 218개 중 216 검증, 2 예산 내 미결, **REFUTED·모순 0** |
| 솔버만으로 찾은 루트 승리(루트 전술 끔) | 그 수를 두고, 둔 뒤 상대가 증명된 패배 |
| 트리 16개 × 4스레드 동시, 증명 6스레드, 대기·실행 중 작업이 남은 채 트리 파괴 20회 | 트리 불변식 위반 0, 크래시 0 |
| 증명 끔 = 기존 탐색(같은 시드) | 동일 |
| 캐시가 UNKNOWN을 결과로 바꾸지 않음 | 통과 |
| 기존 탐색 테스트 13개, Python 탐색·필터·토너먼트 | 통과 |

### 측정 (사전 확정 설정: 솔버 6스레드, 깊이 3, VCT 4k→256k, 방문 ≥ 4; 측정 국면으로 튜닝하지 않음)
환경: 전원 연결, Rapfi·GUI 종료, OneDrive가 측정 내내 ~1–1.2코어 사용(`benchmarks/perf/results/background_load_e3d.csv`). 심판 = 노드 예산만(VCF 200k / VCT 100k, 결정적), UNKNOWN은 진 수로 세지 않음(안전 증명도 아님).

**처음 진 결정 국면 247**
| arm | 진 수 | 흑 / 백 | 심판 UNKNOWN | sims/수 | 증명 시도·성공 / 노드 (수당) | 증명된 패배 루트 자식 |
|---|---|---|---|---|---|---|
| 필터 없음 | 221 | 122/142 · 99/105 | 26 | 214 | — | 0.48 |
| **P6 (깊이 3)** | **170** | 78 · 92 | 77 | 122 | 21.7 · 4.8 / 4.4만 | 3.89 |
| P6 깊이 1 | 167 | 74 · 93 | 80 | 123 | 9.9 · 3.5 / 2.0만 | 5.39 |
| E3b 필터 B30 | 180 | 89 · 91 | 67 | 147 | (필터 증명 57%) | 0.34 |
국면 단위: 필터 없음→P6 제거 56 / 신규 5, B30→P6 제거 16 / 신규 6, 깊이 1→3 제거 5 / 신규 8(차이 없음). "제거"의 상당 부분은 심판 UNKNOWN(26 → 77)이라 안전이 증명된 것은 아님.

**블런더 국면 247**: 필터 없음 234(흑 129 · 백 105) / P6 202(108 · 94), sims 206 → 184 / B30 204(109 · 95), sims 139. 필터 없음→P6 제거 32 / 신규 0.

**외부 paired A/B** (A = 증명 끔, B = P6; 500 ms/수, opening 24–63, Pela·Yixin 각 40쌍, 색 교대, arm 교대)
| 상대 | A | B | 쌍대 차이 [95%] | ΔElo (로컬, 점추정) |
|---|---|---|---|---|
| Pela 2023 | 0.250 (흑 0.150 / 백 0.350) | 0.300 (0.200 / 0.400) | +0.050 [−0.071, +0.175] | +44 |
| Yixin 2018 | 0.062 (0.100 / 0.025) | 0.037 (0.075 / 0.000) | −0.025 [−0.087, +0.037] | −93 |
| **합산 80쌍** | 0.156 | 0.169 | **+0.0125 [−0.056, +0.081] → INCONCLUSIVE** | +16 |
쌍 차이: 0 ×56, −0.5 ×12, +0.5 ×10, +1.0 ×2. 실패 0, DojoZero 금수패 0(Pela 금수패 A 3 / B 0). 수당 시간 A 400–407 ms / B 392–429 ms. **sims/수 A 171·184 → B 96·121 (−44% / −34%)**, B의 수당 증명 시도 19–22, 성공 1.2–2.0, 솔버 노드 2.5–3.0만. **상대 엔진의 수당 시간도 B에서 늘었다**(Pela 123 → 136 ms, Yixin 143 → 170 ms): 솔버 스레드가 외부 엔진과도 CPU를 나눔 — 외부 엔진은 단일 스레드, DojoZero는 GPU + CPU 6스레드 추가이므로 같은 계산량 비교가 아니다.

### 결론
**INCONCLUSIVE**.
- 처음 진 결정 국면에서 증명 인지 탐색은 진 수를 221 → 170(B30 필터 180)으로 줄였고, 흑에서 크게(122 → 78) 줄었다. 그러나 감소분의 상당 부분은 심판 UNKNOWN 증가이고, 외부 A/B는 0을 포함한다(+0.0125).
- **가설 일부 기각**: "솔버를 MCTS와 동시에 돌리면 MCTS 시간을 빼앗지 않는다"는 틀렸다. 이 노트북(4P + 4LPE)에서 솔버 6스레드는 MCTS sims를 35–45% 줄이고 상대 엔진도 느리게 한다.
- 깊이 1과 3의 차이 없음 → 깊은 증명은 이 예산에서 거의 기여하지 않음.
- 기본값은 **끔**(`proof_threads = 0`), 코드는 실험 옵션으로 둔다.
- 다음 후보: (1) 스레드 수를 줄인 증명(2–3스레드, 깊이 1)으로 sims 손실과 증명 효과의 균형 측정, (2) Case B — 가치망·정책망이 위협을 보게 하는 전술 재분석 학습(이번 단계 금지 사항이던 retraining이 필요하므로 사용자 결정 필요), (3) 측정 시 상대 엔진과 CPU 경합을 없애는 프로세스 친화도(affinity) 고정.

### 재현성
코드 = 이 커밋(`phase-8-eval-infra`), checkpoint `gen_001.pt`(model_sha `9ccd7fab…`), `renju_balanced_v1` `1043ff68…`, Pela·Yixin exe 해시는 E3b와 동일, coverage `benchmarks/perf/results/coverage_247.json`(E3c). 결과: `benchmarks/perf/results/replay_e3d_{decision,blunder}.json`, `benchmarks/perf/results/ab/ab_e3d_proof6.json` + DB `experiment`.

---

## Phase 8 (E3c) — Case C 진단: policy 후보 범위(candidate coverage) (2026-10-21)

질문: "남은 블런더 국면에서 안전한 수가 policy/MCTS 상위 후보 밖에 있는가, 아니면 블런더 전에 이미 진 국면인가?"
재학습·새 구조 없음. 측정만 했다. 도구 `python -m dojozero.tools.coverage_replay --earlier --out benchmarks/perf/results/coverage_247.json`
(코드 `python/dojozero/tools/coverage_replay.py`, 실행 시점 HEAD `90ba5ef` + 이 도구).

### 방법
- 국면: E2c tactical_blunder 247개의 **블런더 직전** 국면(DojoZero 차례; `filter_replay.load_positions`와 같은 DB 재구성, `after_blunder.json`과 착수·차례 일치 assert).
- 판정: **모든 합법수**(국면당 중앙값 ~200)를 둔 뒤 상대 차례에서 `analyze_tactics(h, 200000, 100000, 0, 0.0)` — 노드 예산만, 시계 없음, 결정적. 총 48,918회.
  - REFUTED = 상대의 강제승 증명(WIN). WINNING = DojoZero 승리 증명(증명된 안전). UNKNOWN은 패배도 안전도 아님: `UNK_DONE`(노드 < VCT 예산 — VCF·위협공간 df-pn이 증명 없이 끝남; 위협공간 밖 공격·깊이 40 제한이 있어 **안전 증명 아님**) / `UNK_BUDGET`(예산 소진 가능).
  - "반박되지 않은 수" = WINNING + UNKNOWN. 이 API에는 "강제승 없음"을 증명하는 경로가 없어(`engine.cpp`: VCT 실패는 UNKNOWN) 증명된 안전은 사실상 즉승 수뿐이다.
- 순위(1부터, 합법 착점 기준): policy = gen_001.pt raw logit, visit = 500 ms PUCT(노이즈 없음, 필터 끔, filter_replay와 같은 설정) 루트 방문 수(동률·미방문은 prior 순).
- 역추적: DojoZero가 실제로 둔 이전 수(ply−2, −4, … 최대 10수)를 같은 판정으로 검사 → "이미 증명 가능하게 진 연속 구간". `--earlier`: 전 합법수가 반박된 국면마다 그 연속 구간의 **가장 이른** DojoZero 결정 국면에서 위 전수 분석을 반복(154국면).

### 1. 블런더 직전 국면 (247)
| | 전체 | 흑 | 백 |
|---|---|---|---|
| 국면 | 247 | 142 | 105 |
| **전 합법수 반박(이미 진 국면)** | **184 (74.5%)** | 96 (67.6%) | 88 (83.8%) |
| 반박되지 않은 수 존재 | 63 (25.5%) | 46 | 17 |
| └ 증명된 안전(WINNING) 존재 | 0 | 0 | 0 |
| └ UNK_DONE 수 존재 | 34 | 28 | 6 |
| 반박되지 않은 수 / 국면 (중앙값, 존재 국면) | 3 | 4 | 2 |
| 블런더 수 자체 REFUTED | 247/247 | 142 | 105 |
| 현재 탐색(500 ms) 선택이 반박되지 않은 수 | 14 | 13 | 1 |

수 단위 합계: REFUTED 48,608 / UNK_BUDGET 220 / UNK_DONE 90 / WINNING 0.

**반박되지 않은 최선 수의 순위** (63국면; 흑 46 / 백 17):
| | 중앙값 | p75 | p90 | 최대 | 1위 | ≤8 | ≤16 | ≤32 |
|---|---|---|---|---|---|---|---|---|
| policy 순위 | 2 | 5 | 9 | 53 | 17 | 54 | 59 | 59 |
| visit 순위 | 2 | 5 | 10 | 53 | 14 | 56 | 59 | 59 |
| policy 순위, UNK_DONE만(34) | 3 | 6 | 21 | 123 | 6 | 26 | 30 | 32 |

**반박되지 않은 수가 top-k 밖에만 있는 국면** (분모 247 / 분모 63):
| k | policy: 전체 | 흑 | 백 | visit: 전체 | 흑 | 백 |
|---|---|---|---|---|---|---|
| 8 | **9 (3.6% / 14.3%)** | 7 | 2 | 7 (2.8%) | 5 | 2 |
| 16 | 4 (1.6% / 6.3%) | 2 | 2 | 4 | 2 | 2 |
| 32 | **4 (1.6% / 6.3%)** | 2 | 2 | 4 | 2 | 2 |

E3b B30 재생 분류와 교차: "상위 8 전부 반박" 123국면 중 **118은 전 합법수 반박**(후보 범위와 무관하게 이미 진 국면), 5만 top-8 밖에 UNKNOWN 수가 있음. "예산 부족" 80 중 66 전 합법수 반박. "심판 UNKNOWN" 44는 전부 반박되지 않은 수 있음(일관).
반박되지 않은 수가 있는 63국면에서 policy 1위 수가 반박된 경우 46, 블런더 수의 policy 순위 1위 23.

### 2. 이미 진 국면(184)은 언제 졌나 (역추적)
DojoZero가 실제 둔 직전 수부터 연속으로 REFUTED인 수의 개수:
| 연속 반박 수 | 0 | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---|---|---|---|---|---|---|
| 국면 (전체) | 30 | 60 | 62 | 18 | 11 | 2 | 1 |
| 흑 / 백 | 13 / 17 | 35 / 25 | 28 / 34 | 13 / 5 | 5 / 6 | 1 / 1 | 1 / 0 |

- 154/184는 블런더 2–14 ply 전(대부분 2–4 ply) DojoZero의 수가 이미 강제패로 증명됨. 30은 직전 수가 UNKNOWN(UNK_BUDGET 18 / UNK_DONE 12) — 솔버 지평선(VCT 100k 노드) 밖에서 진 국면, 시점 **UNKNOWN**.

**가장 이른 증명 가능 패배 결정 국면(154)의 전수 분석**:
| | 전체 | 흑 | 백 |
|---|---|---|---|
| 국면 | 154 | 83 | 71 |
| 그 국면도 전 합법수 반박 | 44 | 19 | 25 |
| 반박되지 않은 수 존재 | **110** | 64 | 46 |
| 그중 best policy 순위 ≤8 | **96 (87%)** | 56 | 40 |
| policy 순위 >32 | 8 | 3 | 5 |
| 현재 탐색이 반박되지 않은 수 선택 | 22 | 17 | 5 |

best policy 순위 중앙값 2(p90 11, 최대 99), visit 중앙값 2. 반박되지 않은 대안은 대부분 UNK_BUDGET(88/110; UNK_DONE 22) — 이 대안들도 더 깊이 지는지는 **UNKNOWN**(상한 추정치).
실제 둔 수의 policy 순위 1위 48, visit 순위 1위 72/110. **그 수를 반박하는 데 필요한 노드: 중앙값 32k, p75 59k, p90 85k, 110개 모두 >5k**(>20k 75, >50k 34). E3b 필터(8후보·8스레드·수당 30%, 후보당 ~150 ms ≈ 스레드당 30k nodes/s × 0.15 s ≈ 4–5k 노드)로는 하나도 증명할 수 없는 크기다. 루트 value 중앙값 +0.045(증명된 패배 국면).

### 3. 판정 (사전 질문에 대한 답)
- **후보 생성(candidate generation)은 병목이 아니다**: 블런더 시점에 반박되지 않은 수가 top-32 policy 밖에만 있는 국면은 247 중 4(1.6%), top-8 밖 9(3.6%). 더 이른 결정 시점에서도 110 중 96이 top-8 안.
- **주 원인은 "이미 진 국면"(74.5%, 흑 67.6% / 백 83.8%)**: 대부분 2–4 ply 전 DojoZero의 수가 강제패였고, 그 시점에는 policy 상위(중앙값 2위)에 반박되지 않은 대안이 있었다. 그러나 그 수를 반박하는 증명은 20k–100k 노드 규모라 수당 500 ms의 루트 필터로는 찾을 수 없다.
- 흑/백: 백이 이미 진 비율이 더 높고(83.8%), 흑은 블런더 시점에 UNKNOWN 대안이 남은 비율이 높다(46/142). 흑의 top-8 밖 대안 국면 7 / 백 2.

### 권고: **Case A (proof-aware MCTS)** 우선, Case B는 그다음
- 후보 생성 변경(top-k 확대, policy 외 후보 추가): 최대 개선 여지 9/247(top-8 → 전수), top-32 확대로는 5/247 수준 → **권하지 않음**.
- Case A 근거: 놓친 결정의 대안은 이미 상위 후보에 있고(순위 문제 아님), 부족한 것은 상대의 깊은(5k–100k 노드) 강제승을 **여러 수에 걸쳐** 증명·전파하는 능력. 증명이 같은 게임에서 2–4 ply 지속되므로, 트리 내부 노드(상대 응수)에서의 솔버 증명, 수 사이 TT·증명 재사용, 증명된 패배 자식의 방문 차단이 직접 대응한다. 재학습이 필요 없다.
- Case B(value/policy 전술 재분석) 근거: 증명된 패배 국면의 루트 value 중앙값 ≈ +0.05 — 망이 위협을 못 본다. 다만 policy는 대안을 이미 높게 두고 있어(중앙값 2위) policy 쪽 이득은 작고, 효과는 재학습 후에만 측정 가능(**UNKNOWN**). Case A가 만든 증명을 학습 라벨로 쓰는 순서가 자연스럽다.
- 주의: 반박되지 않은 대안 대부분이 UNK_BUDGET이라 "그 대안을 두면 살았다"는 증명되지 않았다. Case A의 기대 효과 상한은 110/154 결정점, 실제 효과는 A/B로만 확인 가능.

### 정확성·재현성
- 결정성: 판정은 노드 예산만 사용. 10국면(25번째마다) 1,998수를 단일 스레드로 재판정 → 결과·노드 수 불일치 **0** (본 실행은 8스레드).
- 측정된 것 / 안 된 것: 블런더 직전·가장 이른 패배 결정 국면의 전 합법수 판정과 policy/visit 순위는 측정. 대안 수를 실제로 두었을 때의 결과, 실전 A/B 효과, 30국면의 패배 시점, 국면당 시간(아래)은 **UNKNOWN / 참고용**.
- MCTS 순위는 500 ms 시간 제한이라 결정적이지 않다(sims 중앙값 203). 측정 중 OneDrive ~1.3–2.4코어, 같은 폴더의 다른 세션(GUI 런처 작업)이 기동 측정을 동시에 수행 → sims와 시간은 부하 영향을 받음; 솔버 판정은 영향 없음.
- 실행 시간: 본 분석(247국면, 8스레드) 397 s, 이른 결정점(154) 809 s, 역추적(단일 스레드, 2,470회 중 다수 UNK_BUDGET) 포함 총 18,629 s.
- 강제 수 루트(상대 4, sims=1): 블런더 시점 3국면 — visit 순위가 의미 없음.
- corpus `after_blunder.json` sha256 `57123c09…`, checkpoint `runs/taraguchi_v1/models/gen_001.pt`, XPU, Intel Core Ultra 7 258V 8스레드, Windows 11 10.0.26200, `keep_awake`(EcoQoS opt-out). 시작 전 Rapfi/engine.exe 등 엔진 프로세스 없음 확인.

---

## Phase 8 — GUI 엔진 런처 `pbrain-DojoZero.exe` (2026-10-21)

목표: Gomocup GUI(Piskvork, Rapfi-YixinBoard)에서 DojoZero와 둘 수 있게 한다. ADR-08(C++ 추론 vs 런처)은
결정하지 않고, 기존 Python 호스트를 감싸는 **임시 런처**만 추가했다. 사용법은 [GUI.md](GUI.md).

### Delivered
| 영역 | 파일 |
|---|---|
| 런처: `<repo>/.venv/Scripts/python.exe -m dojozero.protocol.pbrain --think engine:<model> --device xpu`, PYTHONPATH=`<repo>/python`, stdin/stdout 핸들 상속(복사·재인코딩 없음), 종료 코드 전달, EcoQoS opt-out, kill-on-close Job(런처가 죽으면 Python도 종료), stderr는 exe 옆 로그 파일(기본) | `cpp/cli/pbrain_launcher.cpp`, `cpp/CMakeLists.txt`(target `pbrain_dojozero`, 정적 CRT) |
| 저장소 위치: exe 옆 `pbrain-DojoZero.ini`(repo/python/model/device/args/stderr) 또는 빌드 시 기본값 | 같은 파일 |
| Python 호스트가 자기 인터프리터도 EcoQoS opt-out(engine 모드) | `python/dojozero/protocol/pbrain.py` |
| 한국어 사용법(Piskvork / Yixin Board, 시간 설정, 한계) | `docs/GUI.md`, README, ARCHITECTURE §3.12 |

### Verified
| 검증 | 결과 |
|---|---|
| 빌드 `cmake --build build/msvc --config Release --target pbrain_dojozero` | 성공, /W4 경고 0 |
| 파이프 직접 구동(START/INFO/BEGIN/TURN/ABOUT/END) | 정상 응답, 종료 코드 0 |
| 좌표 확인: BOARD로 자기 사목(가로 y=3, 세로 x=2) 제시 | 둘 다 오목 완성점(`3,3`, `2,13`) 응답 |
| TournamentRunner(프로젝트 Gomocup 클라이언트) vs **Pela 2023**, `renju_balanced_v1` 4쌍, 1000 ms/수, RESTART 재사용 | 3회 실행 24판: 첫 빌드 8판 실패 0 · 최종 빌드 8판 **A:TIMEOUT 1건**(CPU 100% 시점, 첫 쌍 132초) · 최종 빌드 재실행 8판 실패 0. 나머지 전부 NORMAL_WIN, DojoZero 평균 628–697 ms/수 |
| TournamentRunner vs heuristic(in-process), 2쌍, **매 판 새 프로세스**(START/END/종료 경로), 첫 빌드·최종 빌드 각 1회 | 8판 실패 0, 8판 모두 DojoZero 승 |
| Yixin Board 복사본(scratch, 원본 폴더 미변경) + 기록용 엔진으로 실제 실행 | 명령 도달 확인(아래 발견 1·2), Yixin 종료 시 엔진도 종료(잔존 프로세스 0) |
| 기동·첫 수 지연(콜드 스타트 6회) | START→OK: 6.9–10.5초(가벼운 부하), 47–70초(CPU 100%: 다른 세션의 `coverage_replay`가 ~5.5코어 점유, `import torch`만 26초). 첫 수: 2초 제한 1.68초, 1초 제한 0.83–0.85초. 부하 중 1초 제한에서 한 번 1.86초 |

### 사고·발견
0. **TIMEOUT 1건(원인 미확정)**: 최종 빌드 Pela 매치 첫 판에서 DojoZero 쪽 수가 1000 ms + slack 1000 ms를 넘었다. 다른 세션 작업이 CPU 100%를 쓰던 시점이고(같은 시점 1초 제한에서 1.86초 사례), 부하가 줄어든 재실행 8판은 실패 0. 런처는 프로세스 기동 후 관여하지 않으므로 부하 원인으로 **추정**하지만 확인하지 않았다 → GUI 권장 수당 시간 2초 이상.
1. **런처 버그(수정됨)**: Yixin Board는 GLib gspawn으로 엔진을 띄우며 파이프를 CRT 디스크립터 표(lpReserved2)로 넘기고 프로세스 표준 핸들은 숨은 콘솔로 남긴다. 처음 구현(`GetStdHandle`)은 Python에 콘솔 핸들을 넘겨 명령을 하나도 받지 못했고, GUI를 닫아도 EOF가 없어 고아 프로세스가 남았다. `_get_osfhandle(0/1/2)`로 바꾼 뒤 정상. TournamentRunner는 `STARTF_USESTDHANDLES`라 이 차이를 드러내지 못한다.
2. **Yixin Board 규칙 번호 불일치(→ 아래 후속에서 GUI 방언 옵션으로 해결)**: 렌주 설정이 `INFO rule 2`(Yixin 방언; Gomocup 표준에서 2 = 연속 게임)를 보내고, `EngineSession`은 렌주 = 비트 4만 받으므로 모든 착수 요청에 `ERROR DojoZero supports only Renju (INFO rule 4)`. 같은 명령열에서 `rule 4`로 바꾸면 착수한다. 코덱이 Yixin의 `rule 2`를 렌주로 받을지는 결정 필요(이번 작업 범위 밖, 코덱 미변경). Yixin은 자기 폴더의 `engine.exe`만 실행하고, `yx...` 확장 명령에는 `UNKNOWN`으로 답한다(무해).
3. **빈 판·초반 수가 비정상(모델 결함)**: 빈 판 흑 첫 수 `14,1`, 사람 중앙(`7,7`) 뒤 백 `2,0`, `7,7`·`8,8` 뒤 `2,14`. in-process 탐색(CPU, 64/400 sims)에서도 빈 판에 `B2`/`O14` → 런처가 아니라 taraguchi_v1이 Taraguchi-10 오프닝으로만 학습되어 빈 판 자유 오프닝을 본 적이 없기 때문으로 **추정**. 평가 매치는 5–6돌 오프닝 국면에서 시작해 이 결함이 수치에 드러나지 않았다. GUI 대국 품질에 직접 영향.
4. venv의 `python.exe`는 실제 인터프리터를 자식으로 띄우는 리다이렉터다(프로세스 트리로 확인). 그래서 `GomocupProcess`/런처의 자식 EcoQoS opt-out은 실제 인터프리터에 닿지 않는다 → Python 호스트가 스스로 opt-out하도록 추가(효과 자체는 미측정).

### 후속: 사용자가 GUI로 직접 테스트할 수 있게 (2026-10-21)

| 영역 | 내용 | 파일 |
|---|---|---|
| Yixin 방언 | `--rule-dialect yixin`(ini `rule_dialect=yixin`)일 때만 호스트가 `INFO rule 2` → `INFO rule 4`로 바꿔 세션에 전달. 기본(gomocup)은 rule 2 = 연속 게임으로 계속 거부. 규칙 판정은 그대로 `EngineSession` + 정식 `GameState`(새 규칙 로직 없음) | `python/dojozero/protocol/pbrain.py` |
| 오프닝 가드 | `--opening-guard`(런처 기본 on, 평가·토너먼트 경로는 off): 빈 판 H8, 돌 6개 미만에서는 기존 돌로부터 Chebyshev 거리 2 이내로 제한하고 그 안에서 방문 수(없으면 prior) 최대인 수. 임계값 6 = 학습에서 NORMAL_RENJU 국면은 Taraguchi-10 6수 이후에만 나오고 그 전 돌은 모두 중앙 1/3/5/7 영역에 있음(RULES §8). **사용 편의 장치이며 기력 아님** | 같은 파일 |
| 수당 상한 | `--max-move-ms`(ini `max_move_ms`): Yixin "unlimited"가 보내는 2,000,000 ms 대비 | 같은 파일, 런처 |
| 테스트 | 두 방언(rule 2 거부/수용, rule 0·1 거부, Yixin 실제 명령열, 정식 규칙 판정 유지), 가드 6종 | `tests/python/test_pbrain_gui.py` |
| 사용자용 설치 | `C:\dojozero_gui\YixinBoard-DojoZero`(Rapfi-YixinBoard 복사본, `engine.exe` = 런처, ini `rule_dialect=yixin`, 사람 흑·DojoZero 백·수당 5초), `C:\dojozero_gui\engine`(커밋 스냅샷 + 그 코드로 빌드한 `_core`·런처). 원본 Yixin 폴더는 변경 없음 | 저장소 밖 |
| 문서 | 한국어 단계별 사용법 | `docs/GUI.md` |

Verified:
| 검증 | 결과 |
|---|---|
| `pytest tests/python/test_pbrain_gui.py tests/python/test_tournament.py` (스냅샷 `_core`) | 27/27 |
| 런처 빌드(`build/msvc-gui`, 스냅샷 `C:\dojozero_gui\engineuild`) | 성공, 런처 경고 0 (`_core`는 기존 C4390 경고 2건) |
| **실제 Yixin Board 창**에서 DojoZero 대 DojoZero(양쪽 컴퓨터, 수당 3초) | 23수 끝까지 진행, Yixin↔엔진 ERROR 0, 첫 수 중앙 `7,7`(가드), 기보를 정식 `GameState`로 재생 → BLACK_WIN(FIVE). Yixin 종료 시 엔진 종료(잔존 0) |

발견:
1. **작업 트리 의존 위험(실제 발생)**: 처음엔 엔진이 작업 트리(`C:	emp_git\Omok_AI`)를 그대로 썼는데, 다른 세션이 Case A로 `driver.py`를 고친 상태에서 `_core`가 재빌드되기 전이라 엔진이 시작 직후 `AttributeError: proof_threads`로 죽었다. → GUI 엔진은 커밋 스냅샷(`git archive`) + 별도 빌드한 `_core`로 분리. 저장소의 `_core`·`build/msvc`는 건드리지 않았다.
2. Yixin은 `use database moves` = 0이면 START 후 착수 요청을 보내지 않았다(1에서는 정상, 원인 미확인). 수가 바뀔 때마다 `yxblockreset`·좌표 줄·`start 15`·`board ... done`을 보낸다. 앞의 것들에는 `UNKNOWN`으로 답하며 무해.
3. Yixin settings의 시간 단위는 초(turn 3 → `INFO timeout_turn 3000`).

미검증(UNKNOWN): 사람이 직접 클릭해서 두는 경로(엔진 응답 경로는 같은 `board ... done`이라 동작할 것으로 추정), Yixin swap 규칙(3/5/6), Piskvork 전체.

### Open
- Piskvork 실제 확인(다운로드는 사용자 승인 필요).
- 빈 판·초반 자유 오프닝의 근본 대응(모델 학습 범위). 현재는 GUI 가드로만 가림.
- 스냅샷 갱신 절차: 새 모델·코드로 GUI 엔진을 바꾸려면 `git archive` 스냅샷을 새로 만들고 `_core`를 다시 빌드.
- 기동 지연 10–70초: GUI 시작 대기 제한이 짧으면 실패. ADR-08 결정 시 고려.

---

## Phase 8 (E3b) — Tactical solver profiling and acceleration, replay, paired A/B rerun (2026-10-21)

질문: "모델은 그대로 두고 전술 솔버를 충분히 빠르게 만들면 실제 tactical blunder와 외부 성능이 개선되는가?"
기준선 = `ee1d011`. 같은 바이너리에 ee1d011 솔버를 동결 복사(`dz::tactics::ref`)해 모든 비교를 같은 빌드에서 수행.

### Delivered
| 영역 | 파일 |
|---|---|
| 동결 참조 솔버(ee1d011 threats/vcf/dfpn/engine 복사, 차분 oracle·같은 바이너리 기준선) | `cpp/tactics/reference/*` |
| 구간 프로파일러(`DZ_TPROF` 빌드 전용 매크로, 일반 빌드에서는 빈 매크로) | `cpp/tactics/tprof.{h,cpp}`, 솔버 계측 |
| 벤치·프로파일 도구(`dojozero_tbench`/`dojozero_tprof`, `--impl ref/live/pair`, 할당 수, 스레드, 스로틀링 해제) | `cpp/tools/tactics_profile.cpp`, CMake |
| 성능 corpus 3종(벤치 문제 102 / 블런더 직후 247 / 필터 워크로드 1,976) | `python/dojozero/tools/tactics_corpus.py`, `benchmarks/perf/*.json` |
| df-pn 정확 가속 3종(아래), 스위치로 on/off | `cpp/tactics/dfpn.{h,cpp}` |
| `five_points_near` 셀 복사 제거(기본 흑 옵션·백) | `cpp/tactics/threats.cpp` |
| analyze_tactics VCT 시간 이월(opt-in, 필터만 사용) | `cpp/tactics/engine.{h,cpp}` (`carry_unused_time`) |
| 필터: 솔버 선택(`defimpl=ref/live`)·이월(`defcarry`)·후보별 결과 기록, 대국 기록에 수마다 필터 통계, 엔진 ID 구분(`-def8` = ee1d011 솔버, `-def8-fast`) | `search/driver.py`, `evaluation/{match,tournament}.py`, `bindings/module.cpp` |
| 247 블런더 재생 도구(arm별, 결정적 노드 심판 + E3a 1초 심판) | `python/dojozero/tools/filter_replay.py` |
| Windows 전원 스로틀링 해제(프로세스 단위: 벤치 도구, 평가 진입점, 외부 엔진 자식 프로세스) | `tournament.disable_power_throttling`, `protocol/gomocup_process.cpp` |
| 정확성 테스트 9종 + Python 3종 | `tests/tactics/test_accel.cpp`, `tests/python/test_defense.py`, `tests/python/test_bindings.py` |

### 1. Profiling (최적화 전, ee1d011 코드, 블런더 직후 247국면 VCT 5만 노드)
VCF는 병목이 아님(247국면 합 5,035노드, 국면 중앙값 0.04 ms). VCT 구간 비중(포함 시간, 8스레드; 호출 수는 결정적):

| 구간 | 호출 | 비중 |
|---|---|---|
| AND 노드 수 생성 | 32.8만 | **77.8%** |
| └ 방어수마다 `refuted(d)` (착수 → 방어 5목 점 → 공격 쌍위협 재검사) | 3,814만 (AND 노드당 116) | 45.8% |
| └ 합법성(흑 방어 시 금수 판정), 모든 점 | 4,529만 (노드당 138) | 18.3% |
| OR 노드 위협수 생성(`threat_moves`) | 17.1만 | 15.6% |
| 전체 보드 5목 점 스캔 | 109만 | 5.4% |
| TT 조회·저장 | 402만 | 1.6% |

노드당 할당 **145회**. 원인: (1) TT가 pn/dn만 저장해 df-pn이 노드를 재방문할 때마다 수 생성(AND 노드 전 빈 점 반박 검사)을 반복, (2) 모든 빈 점에 금수 판정 후 반박 검사, (3) 반박 검사가 매번 착수·해시 갱신·셀 625바이트 복사. 원본: `benchmarks/perf/results/baseline_ee1d011_after_blunder_{vcf,vct}_tprof.json`(시간은 스로틀링 아래 측정, 호출 수는 정확).

### 2. 최적화 (1–4는 정확: 같은 탐색 트리, 같은 노드 수, 같은 결과)
1. **생성 캐시**: 노드 키별 수 생성 결과 재사용. OR 노드의 깊이 컷은 원래 순서(즉승 검사 → 깊이 컷)대로 재적용, 깊이 컷 결과는 캐시하지 않음.
2. **지연 합법성**: AND 노드에서 반박되지 않은 빈 점에만 금수 판정(같은 집합·같은 순서).
3. **fast_refute** (ADR-01 C안의 최소 범위: 노드 단위 위협 정보 + local invalidation): AND 노드마다 공격 쌍위협 수 e와 그 5목 점 집합을 한 번 계산. 방어 돌 d는 공격 연속 길이를 바꾸지 않으므로 e의 5목 점 수 = 기존 집합에서 d를 뺀 크기; 방어 5목 점은 9칸 창 사전 필터를 통과할 때만 원래 검사; **공격 흑의 합법성은 canonical `black_verdict_cells`로 정확히 재판정**(재귀 거짓금수는 국소성이 보장되지 않으므로). 비기본 흑 옵션은 원래 경로.
4. `five_points_near` 셀 복사 제거, 반박 버퍼 재사용 → 노드당 할당 145 → 5.2.
5. (정확 아님, 별도 측정) **VCT 시간 이월**: ee1d011의 40/40/20 분할에서 빠른 VCF·꺼진 loss 검색이 남긴 시간이 버려져, 필터의 VCT는 후보 예산의 40%만 사용했다. opt-in으로 VCT가 남은 시간을 사용. 결과는 UNKNOWN → WIN으로만 바뀌고 WIN/LOSS가 뒤바뀌지 않음. MCTS 루트 전술은 ee1d011 분할 유지 — 처음에 기본값을 켬으로 두었다가 MCTS 시간을 빼앗아 `Search.TimeLimitStopsSearch`가 실패한 것을 발견, opt-in으로 수정.

TT 검토: UNKNOWN은 증명으로 저장하지 않음(예산 중단 시 pn/dn ≥ 1로만 저장), 깊이 컷 DISPROVEN 비저장 — 예산 의존 결과가 증명으로 캐시되는 경로 없음(변경 없음). 수 정렬은 변경하지 않음(트리 보존이 차분 검증의 전제); 정책 prior 미사용.

### 3. 정확성
| 검증 | 결과 |
|---|---|
| 차분: 벤치 102(VCF/VCT/analyze), 블런더 247(VCF/VCT/analyze), 필터 워크로드 1,976(analyze, 노드 예산) ref vs live | 결과·노드 수 불일치 **0** |
| 스위치 8조합(캐시·지연·fast_refute on/off) vs ref, 무작위 국면 RIF/Omokdojang, WIN/LOSS 모드 | 비트 동일 (`SolverAccel.DfpnMatchesReferenceAndSwitchesAreNeutral`) |
| 비기본 흑 옵션 / 흑 밀집 보드(장목·4-4·3-3·재귀 거짓금수), 양 색 공격 | 비트 동일 |
| 노드 예산 30/300/3000 × 이월 on/off의 모든 WIN/LOSS 주장 → 독립 검증기 | REFUTED 0 |
| D4 8이미지 | WIN/LOSS 모순 0 |
| 입력 불변, play/undo 400회 스트레스 후 셀·8개 해시·차례 복원 | 통과 |
| 8스레드 병렬 호출 = 순차 결과 | 통과 |
| 예산 소진 → UNKNOWN (LOSS 아님) | 통과 |
| 시간 제한 측정에서 live가 ref의 WIN을 잃은 국면 | 0 (8조건 × 1,976) |
| `dojozero_tactics fuzz --positions 2000 --seed 2026` (analyze_tactics 주장 → 독립 검증기) | 주장 1,837(WIN 1,522 / LOSS 315), 검증 1,828, 너무 큼 9, **REFUTED 0** |
| 전체 테스트 | C++ 265개 중 264 통과·1 skip(기존 앱 증거 없음), Python 76 통과 |

### 4. 속도 (같은 바이너리, 국면마다 ref → live 교대 `--impl pair`, 스로틀링 해제)
측정 환경 주의: 1차 측정 중에는 사용자 GUI의 Rapfi 분석 엔진이 백그라운드에서 평균 ~1.7코어 사용(발견 후 종료), 재측정 중에는 OneDrive가 ~1.4코어 사용(`benchmarks/perf/results/background_load_rerun.csv`). 절대 nodes/s는 부하에 따라 30–50% 달라진다; 비율은 교대 측정이라 견고하다.

| VCT, 블런더 247, 5만 노드 | 1차 ref → live | 재측정 ref → live | 배율(재측정) | 국면별 중앙값 |
|---|---|---|---|---|
| 1스레드 | 24.8k → 81.5k | 16.7k → 55.4k | **3.3×** | 4.1× |
| 2스레드 (스레드당) | 23.0k → 84.1k | 15.4k → 54.4k | 3.5× | 4.4× |
| 4스레드 (스레드당) | 18.7k → 80.2k | 9.5k → 39.3k | 4.2× | 5.0× |
| 8스레드 (스레드당, 필터 설정) | 10.0k → 53.5k | 5.8k → 30.9k | **5.3×** | 6.3× |

ref는 스레드가 늘수록 스레드당 속도가 크게 떨어짐(1→8: −65%), live는 −44%. analyze(필터 호출 형태, 1스레드): 블런더 3.4×, 벤치 2.7×. VCF: 변경 없음(블런더 109k → 125k nodes/s; 벤치 문제는 합 598노드라 무의미). nodes-to-proof(중앙값 1,635 / p95 5,783)는 트리가 같으므로 불변.

**proof@시간** (필터 워크로드 1,976, VCF 100k / VCT 50k 노드, 8스레드, 재측정; 노드 예산만의 상한 1,836 WIN):

| 시간 | ref | live | live + 이월 |
|---|---|---|---|
| 125 ms | 1,137–1,155 | 1,421 | **1,614** |
| 250 ms | 1,219–1,224 | 1,569 | **1,744** |
| 500 ms | 1,318–1,327 | 1,707 | 1,796 |
| 1000 ms | 1,460–1,473 | 1,783 | 1,816 |

개선판 125 ms(1,614)가 ee1d011 1000 ms(약 1,465)보다 많이 증명한다.

### 5. 247 블런더 재생 (k=8, 8스레드, 500 ms/수, 노드 심판: VCF 200k / VCT 100k, 시계 없음)
E3a 수치(187 / 149)는 1초 시간 제한 심판·스로틀링 조건이라 직접 비교할 수 없다 — 같은 실행에서 전 arm을 재측정했다. 3회 실행(Rapfi 부하 중 2회 + 제거 후 1회) 모두 같은 경향; 아래는 마지막 실행(OneDrive 부하 기록됨).

| arm | 블런더 | 흑 / 백 | 필터 증명 | UNKNOWN | 후보 검사 / 반박 | 수 교체 | 남은 sims | 방어 ms 평균/p95 |
|---|---|---|---|---|---|---|---|---|
| 필터 없음 | 234 | 131/142 · 103/105 | — | — | — | — | 269 | — |
| A: ee1d011 솔버 50% | 214 | 117 · 97 | 67.5% | 32.5% | 7.91 / 5.34 | 25.9% | 109 | 119 / 223 |
| B 10% | 213 | 115 · 98 | 75.8% | 24.2% | 7.91 / 6.00 | 30.0% | 203 | 59 / 109 |
| B 20% | 206 | 111 · 95 | 82.7% | 17.3% | 7.91 / 6.55 | 32.4% | 171 | 105 / 227 |
| **B 30%** | **203** | 109 · 94 | 86.4% | 13.6% | 7.91 / 6.84 | 28.3% | **152** | 114 / 182 |
| B 50% | 203 | 109 · 94 | 89.8% | 10.2% | 7.91 / 7.11 | 21.9% | 106 | 149 / 281 |
| B 50% 이월 없음 | 208 | 112 · 96 | 82.0% | 18.0% | 7.91 / 6.49 | 31.2% | 103 | 92 / 154 |

국면 단위: A → B30 블런더 제거 11 / 신규 0, 필터 없음 → B30 제거 32 / 신규 1 ("제거"는 심판이 더 이상 강제승을 증명하지 못함; 심판 UNKNOWN 포함). **남은 실패 원인이 바뀜**: A는 예산 부족 179 / 상위 8 전부 반박 35, B30은 80 / **123**, B50은 44 / 159 — 솔버가 빨라지자 "상위 k 후보 중 안전한 수가 없음"(또는 이미 진 국면)이 주 병목. 흑 블런더 비율 77%(109/142), 백 90%(94/105).
방어 비율 30%를 A/B arm으로 **사전 확정**(재생 국면 기준, A/B opening과 별개): 50%와 블런더 동일, sims +43%.

### 6. 외부 paired A/B (사전 등록, Tier B)
A = 필터 끔, B = 개선 솔버 + 필터(k=8, 수당 시간 30%, 8스레드, 이월). 500 ms/수, 같은 체크포인트, `renju_balanced_v1` opening 24–63(E3a와 동일), Pela 2023·Yixin 2018 각 40쌍, 색 교대, arm 교대, 외부 엔진 단일 프로세스(스레드 수 지정 안 함 — E3a와 동일).

| 상대 | A | B | 쌍대 차이 [95%] | ΔElo (로컬, 점추정) |
|---|---|---|---|---|
| Pela 2023 | 0.263 (흑 0.150 / 백 0.375) | 0.300 (흑 0.175 / 백 0.425) | +0.0375 [−0.050, +0.119] | +32 |
| Yixin 2018 | 0.025 (0.025 / 0.025) | 0.037 (0.025 / 0.050) | +0.0125 [−0.043, +0.068] | +73 (점수가 0에 가까워 매우 불확실) |
| **합산 80쌍** | 0.144 | 0.169 | **+0.025 [−0.025, +0.075] → INCONCLUSIVE** | +33 |

쌍 차이: 0 ×64, +0.5 ×10, −0.5 ×6. 실패 0, DojoZero 금수패 0(금수패 2+2는 Pela). 수당 시간 A 409–422 ms / B 403–409 ms, sims/수 A 172·162 → B 117·118(−32%). B의 필터: 수당 솔버 호출 6.3, 증명(WIN) 2.2–3.5, 솔버 노드 1.3–1.7만, 솔버 CPU 323–385 ms(8스레드 합), 방어 벽시계 112–115 ms, 수 교체 3.7–6.3%. 실전에서 UNKNOWN 45–65%는 대부분 "강제승 없음"(안전한 후보)이라 실패율이 아니다. 대국 내 tactical blunder 비율: **UNKNOWN**(대국 분류 미실행).
**무효 실행 1건(기록 유지)**: 첫 A/B(`runs/evaluation/ab_e3b_fastsolver_k8_s30.json`)는 사용자 GUI의 Rapfi 분석 엔진이 CPU를 점유해 Yixin 구간에서 DojoZero가 수당 1.3 sims·1.25 s로 굶었다(같은 국면 재현 시 정상) → 판정에 사용하지 않음. 그 실행의 Pela 구간: +0.025 [−0.063, +0.113].
E3a(−0.019 [−0.075, +0.038])와 비교: 점추정 부호가 +로 바뀌었으나 두 실행 모두 0을 포함. 조건(스로틀링 해제로 외부 엔진도 전속력, 배경 부하)이 달라 "같은 조건"이 아니다 — 비교는 각 실행 내부 arm 간에만 유효.

### 결론
**INCONCLUSIVE** (명세 §13: "솔버 속도는 크게 개선됐지만 replay·Elo 개선이 약함").
- 솔버 가속 자체는 정확성 회귀 없이 채택(결과 비트 동일, 기본 경로). 필터 기본값은 계속 **끔**.
- 충족: 필터 설정(8스레드) 5.3×, proof@125–250 ms +40%, 필터의 sims 손실 감소, 정확성 회귀 0. 미충족: 1스레드 3.3×·100k nodes/s, 재생 블런더 감소 소폭(214 → 203), 외부 A/B CI가 0 포함.
- **tactical solver 속도는 병목 중 하나였을 뿐 전체 병목은 아니다**: 솔버가 빨라지자 남은 블런더의 다수(123/203)는 상위 8 후보가 전부 강제패로 증명되는 국면 — 후보 범위(policy coverage) 또는 그 전 수에서 이미 진 국면 문제.
- 다음 단계 권고: **Case C (policy candidate coverage)** — 블런더 국면에서 안전한 수가 존재하는지(전 합법수 솔버 검사), 그 수의 policy/visit 순위, top-k 밖 안전 방어 비율 측정 → 결과로 Case B(value/policy tactical reanalysis) 또는 Case A(proof-aware MCTS) 결정. 흑 측 약점(Case D)은 계속 기록.

### 재현성
baseline `ee1d011`; corpus sha256: bench_problems `7b5d31c1…`, after_blunder `57123c09…`, filter_workload `54303e54…`; checkpoint `runs/taraguchi_v1/models/gen_001.pt` (파일 sha256 `3067e05e…`, model_sha `9ccd7fab…`); `renju_balanced_v1` `1043ff68…`; Pela 2023 exe `b97df7a3…`, Yixin 2018 exe `f6d1a01b…`; Windows 11 Home 10.0.26200, Intel Core Ultra 7 258V (4P+4LPE, 8스레드), 32 GB, 전원 연결(재측정·A/B) / 배터리(1차 측정); MSVC 19.43.34808 Release, CMake 4.3.2; 결과 파일 `benchmarks/perf/results/`(README 참조), A/B `runs/evaluation/ab_e3b_fastsolver_k8_s30_rerun.json` + DB `experiment`. 솔버 설정: VCF 100k / VCT 50k 노드, depth 40, 필터 k=8·8스레드·후보당 `500 × share` ms.

---

## Phase 8 (E3a) — Root defense filter: implementation and paired A/B (2026-10-20)

### Delivered
| 영역 | 파일 |
|---|---|
| 루트 방어 필터(탐색 후 방문 수 상위 k 후보를 순서대로 솔버로 확인, 상대 강제승이 증명된 수는 반박되지 않은 대안이 있을 때만 건너뜀; 모두 반박·증명된 승리·opening 단계면 원래 선택 유지). 시간 제한 시 수당 시간의 `defense_share`를 필터에 할당, 후보 검사는 병렬 스레드 | `python/dojozero/search/driver.py` (`defend`, `SearchSettings.defense_*`), 스펙 옵션 `:defense=k:defshare=x:defthreads=n` |
| paired A/B 러너(같은 opening·상대·시간, opening마다 두 arm 교대, 사전 등록 판정, DB `experiment`) | `python/dojozero/evaluation/ab.py` |

### Verified
| 검증 | 결과 |
|---|---|
| 단위 테스트 | 필터 4 + A/B 판정 1 (반박된 선택 교체, 전부 반박 시 유지, 증명된 승리·opening 불변, 시간 제한 안 동작; ACCEPT/REJECT/INCONCLUSIVE) |
| 전술 benchmark 회귀(§19 조건 2; 승리 문제 73개, 64 sims) | 필터 없이 0.945 / 필터 k=8 0.945 — **회귀 없음** |
| 메커니즘 확인(기록된 tactical_blunder 국면 247개 재생, 500 ms/수, 1 s 솔버로 판정) | 필터 없음: 같은 실수 187 / k=8·30%·4스레드: 163 / k=8·50%·8스레드: 149(안전 98) / k=16·50%·8스레드: 157. 놓친 원인 대부분 **예산 부족**(필터의 짧은 솔버 시간으로는 증명 못 하는 VCT; 116–143건), 상위 k 전부 반박 16–33건 |

**A/B (사전 등록, Tier B)**: baseline = taraguchi v1, candidate = 같은 모델 + 필터(k=8, 수당 시간의 50%, 8 스레드). 500 ms/수, `renju_balanced_v1` opening 24–63(진단·calibration에 쓰지 않은 것), 상대 Pela 2023·Yixin 2018 각 40쌍, arm 교대. 판정 규칙: 쌍대 차이 95% 구간 하한 > 0 이고 candidate 실패 0이면 ACCEPT.
| 상대 | baseline | candidate | 쌍대 차이 [95%] |
|---|---|---|---|
| Pela 2023 | 0.250 | 0.225 | −0.025 [−0.125, +0.075] |
| Yixin 2018 | 0.050 | 0.037 | −0.013 [−0.068, +0.043] |
| **합산 80쌍** | | | **−0.019 [−0.075, +0.038] → INCONCLUSIVE** |
쌍 차이 분포: −0.5 ×12, 0 ×59, +0.5 ×9. candidate 실패 0. DojoZero 평균 sims/수: baseline 135.5 → candidate 64.0.

### 결론
- **채택하지 않음**(§19: INCONCLUSIVE는 성공이 아님). 기본값은 필터 끔(`defense_k = 0`); 코드는 실험 옵션으로 남김.
- 해석(가설): 필터가 일부 실수를 막지만(메커니즘 확인 +38/247) 탐색 시간을 절반으로 줄인 비용과 상쇄. **진짜 병목은 솔버 속도** — 500 ms 안에서 상대 VCT를 증명하지 못한다(EVALUATION §8.2: VCT 13.5–14k nodes/s, VCF 70–78k, 목표 대비 크게 미달). 다음 후보: 증분 위협 추적 등 솔버 가속(ADR-01 C안), 필터를 탐색 트리 안(MCTS-solver의 루트 자식 증명)으로 옮겨 시간 재분배, 가치망이 위협을 보도록 학습.

---

## Phase 8 (E2c) — Weakness diagnosis: scaling, fixed visits, loss classification (2026-10-20)

질문: DojoZero(taraguchi v1)가 외부 사다리 최하위인 원인이 탐색량(performance)인가, 평가/탐색 품질인가 (EVALUATION §20).

### Delivered
| 영역 | 파일 |
|---|---|
| 시간 scaling / fixed-visits 실험 러너 | `python/dojozero/evaluation/diagnose.py` |
| 패배 분류기(솔버 증명 + 네트워크 value/policy, EVALUATION §21.1 1차판) | `python/dojozero/evaluation/loss_analysis.py` |

### 실험 (균형 corpus 첫 12개 opening, 조건마다 12쌍 = 24판, 모든 조건 같은 opening — 조건 간 paired)
시간 scaling (양쪽 같은 수당 시간; 외부 엔진 단일 스레드 CPU):
| 상대 | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|
| Pela 2023 | 0.250 (134 sims) | 0.292 (286) | 0.292 (604) | 0.333 (1,418) |
| Yixin 2018 | 0.042 (158) | 0.000 (248) | 0.125 (637) | 0.042 (1,570) |

Fixed visits (DojoZero 고정 sims, 상대는 0.5 s 고정):
| 상대 | 64 | 256 | 1024 |
|---|---|---|---|
| Pela 2023 | 0.125 | 0.250 | 0.250 |
| Yixin 2018 | 0.000 | 0.083 | 0.083 |
모든 CI가 넓고 서로 겹친다(12쌍). 색별(Pela, 7조건 합산 84판씩): **흑 ~0.12, 백 ~0.41**.

패배 분류(비승리 284판, 솔버 예산 VCF 200k / VCT 100k / LOSS 50k 노드, 1 s):
| 범주 | 판 수 | 비고 |
|---|---|---|
| tactical_blunder | **247 (87%)** | 착수 전 솔버가 패를 증명하지 못한 국면 → 착수 후 상대 강제승 증명. 흑 142/151, 백 105/133 |
| strategic_loss | 31 (11%) | 이미 증명된 패(솔버 시야 밖에서 짐). 백 24, 흑 7. opening 직후 패 0 |
| missed_win | 6 (2%) | 증명된 승리를 놓침 |
- 결정적 수 직전 **네트워크 value(DojoZero 관점) 평균 +0.52**, 247건 중 196건이 +0.2 이상 — 상대의 강제승을 전혀 예상하지 못함.
- tactical_blunder 247건 중 **123건은 policy 상위 10수 안에 솔버가 반박하지 못한 대안**이 있었고(43건은 policy 1순위 수가 대안), 둔 수가 policy 1순위인 경우 99건.
- 분포는 compute에 거의 무관(0.5 s: 전술 33·전략 6·놓친 승 2 / 5 s: 34·5·0 / 64 sims: 41·4 / 1024 sims: 35·5).

### 결론 (측정 근거와 한계를 함께)
1. **탐색량 부족은 주원인이 아니다**: 수당 sims를 ~10배(134→1,418, 52→797) 늘려도 점수 변화가 CI 안. EVALUATION §20의 "performance problem"은 기각(이 상대·이 표본 범위에서).
2. **주원인은 방어 전술**: 패배의 87%가 "상대에게 솔버가 보는 강제승을 허용한 한 수". 우리 C++ 솔버는 그 국면을 ~1 s 안에 증명하지만 DojoZero 탐색은 루트 후보마다 "이 수 뒤 상대에게 VCF/VCT가 있는가"를 확인하지 않는다(루트 전술은 자기 승리/증명된 패만, 리프 전술은 즉시 5·단일 방어만). value망도 그 위협을 보지 못한다(+0.52).
3. **흑에서 더 약함**: 흑 패배의 94%가 전술 실수(백 79%). 흑은 금수 때문에 막을 수 없는 점이 있어 방어가 더 어렵다는 것과 맞지만, 원인 확정은 아님.
4. 한계: 분류는 솔버 예산에 의존(UNKNOWN은 "안전"이 아님), 첫 결정적 사건만 1차 범주로 셈, 표본 12쌍/조건. 분류 정확도의 사람 검수는 아직 안 함. 이번 실행의 요약 저장 단계가 JSON 버그(튜플 키)로 실패해 요약은 DB `analysis` 행에서 다시 계산(분류 자체는 정상 저장; 버그 수정 + 회귀 테스트).

### 다음 후보 (가설 — §19 A/B로 검증해야 함)
- **루트 방어 필터**: 루트 상위 k 후보마다 "착수 후 상대 강제승?"을 시간 예산 안에서 솔버로 확인, 대안이 있으면 그런 수를 배제(기존 최선 유지 + 확실할 때만 이탈하는 하한 원칙과 같은 구조). 근거: 실수 123건에 반박되지 않은 대안 존재.
- 학습: 상대 위협(VCF/VCT) 인지를 value/aux target에 반영, 위협 국면 비중 확대.

---

## Phase 8 (E2b) — Balanced corpus, engine round robin, calibration v1 (2026-10-19)

### Delivered
| 영역 | 파일 |
|---|---|
| 균형 opening corpus `renju_balanced_v1`(120개): 무작위 Taraguchi 오프닝의 첫 5돌(백 차례), Rapfi 2025 vs AlphaGomoku 2026 색 교대 2판(200 ms)에서 **한 색이 두 판을 다 이기지 않은** 국면만 채택 (OQ-14 방법: 사용자 결정 2026-10-19) | `python/dojozero/evaluation/engine_games.py`, `benchmarks/openings/renju_balanced_v1.json` |
| 외부 엔진 간 라운드로빈(병렬 스레드, 엔진별 프로세스) | `engine_games.py round-robin` |
| ladder `refit`(저장된 대국으로 재적합, 실패 종료 대국 제외 옵션), `anchor: auto` | `python/dojozero/evaluation/ladder.py`, `configs/evaluation/ladder_v1.yaml` |
| 첫 수 초기화 여유(`first_move_slack_ms`, 기본 5 s, time_left에는 포함) | `cpp/protocol/external_engine.*` |
| 장시간 평가 중 유휴 절전 방지(`keep_awake`: 프로세스 단위 SetThreadExecutionState, 설정 변경 없음) | `tournament.py` 외 4개 진입점 |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` / `pytest` | 255 + 1 skip / **64/64** (신규: 5돌 후보 고유성, 균형 필터 판정, 첫 수 여유 회귀) |
| 균형 필터 | 후보 500개 중 처리분: 흑 2승 89, 백 2승 53, 무승부 포함 6, 실패 4, **균형 120 채택** |
| 라운드로빈 v1b (10엔진 × 45쌍 × 4쌍 = 360판, 500 ms/수, 동시 3판) | 실패 3(Rapfi 2021 CRASH 2, Yixin MALFORMED 1) + 금수 패 18(규칙상 패) |
| calibration v1 (DojoZero 500 ms/수, 균형 corpus) | 프로브 4쌍: Yixin 0.25, Rapfi 2021·KataGomo 2021r·Pela 0.125, 나머지 6개 엔진 **0.0**. 주 매치 Yixin 24쌍: **0.062, −470 [−1600, −209]** |

**Ladder v1b** (`dojo_external_elo`, anchor Yixin 2018 = 0; 500 ms/수, 외부 엔진 단일 스레드 CPU, `renju_balanced_v1` hash `1043ff68…`; DB rating_snapshot):
| 엔진 | Elo [95% CI] |
|---|---|
| Rapfi 2025 | +357 [+223, +492] |
| Embryo 2026 | +242 [+119, +364] |
| Embryo 2021r | +208 [+88, +328] |
| AlphaGomoku 2026 | +186 [+67, +305] |
| KataGomo 2026 | +97 [−19, +213] |
| Rapfi 2021 | +29 [−87, +145] |
| AlphaGomoku 2023 | +18 [−98, +134] |
| Yixin 2018 | 0 (anchor) |
| KataGomo 2021r | −14 [−131, +103] |
| Pela 2023 | −411 [−575, −247] |
| **DojoZero (taraguchi v1)** | **−491 [−634, −348]** |
전체 대국 적합과 실패 제외 적합의 차이는 35 Elo 이내. 외부 기준 확인: Gomocup 2025 Renju 1위 Rapfi가 1위(단 이 짧은 시간·단일 스레드에서는 AlphaGomoku 2026이 Embryo보다 아래 — 시간 규정이 다르면 순서가 바뀔 수 있음).

### 사고·발견 (중요)
1. **DojoZero는 사다리 최하위**: Pela 2023보다도 약 80 Elo 아래(CI 겹침). 내부 지표(heuristic 0.95+, 세대 간 승격)는 외부 강도를 전혀 대변하지 못했다(CLAUDE.md §1).
2. **첫 라운드로빈(v1)은 폐기**: TIMEOUT 154건(KataGomo 2021r 65 등). 같은 설정을 keep-awake로 재실행(v1b)하니 TIMEOUT **0**. 노트북이 세션 유휴 중 약 3시간 절전에 들어간 구간과 겹쳤다 — 앱의 keep-awake는 세션이 유휴(백그라운드 대기)면 풀린다. "동시 실행 경합" 가설은 직접 비교(동시 3 vs 1, 각 8판, 모두 실패 0)로 **기각**. v1 데이터는 DB에 남기되 ladder에는 v1b만 사용.
3. 균형 필터 첫 실행: 500개 중 278개가 실패 — AlphaGomoku 2026이 **첫 수**(신경망 지연 로딩)에서 200 ms+1 s를 넘김. 첫 수 초기화 여유 추가 후 실패 4.
4. 엔진들의 금수 착수(규칙상 패)는 실제로 일어남(라운드로빈 18건, 주로 Rapfi 2021) — 적합성 함정 5종으로는 걸러지지 않는 수준. 패로 집계하되 사유를 분리 기록.
5. 균형 필터는 근사(국면당 2판, 한 시간 규정, 두 엔진)이고 필터 엔진(Rapfi 2025, AlphaGomoku 2026)이 사다리에도 있어 그 두 엔진에 약간 유리/불리할 수 있음 — 미측정.

### Open
- DojoZero 열세 원인 분리: fixed-time scaling curve(0.5–5 s), fixed-visits 비교, 전술/가치 패배 분류(E3). 지금 수치로는 "performance"와 "value/search" 중 어느 쪽인지 판단 불가.
- 시간 규정(OQ-13) 확정 시 사다리 재측정. 엔진 간 대국 쌍 수 확대(현재 쌍당 4쌍).

---

## Phase 8 (E2) — External ladder: engines, conformance, rating, first calibration (2026-10-19)

### Delivered
| 영역 | 파일 |
|---|---|
| 외부 엔진 11종 메타데이터(사용자 승인 후 공식 Gomocup 배포 https://download.gomocup.com/ai/ 에서 받음, exe sha256 고정; 바이너리는 `C:/dojozero_engines/`, 저장소 밖) | `external_engines/*.json` |
| 규칙 적합성 검사(함정 5종 + 짧은 대국) | `python/dojozero/evaluation/conformance.py` |
| ladder rating(최대우도 Elo, anchor 0, 약한 prior) + adaptive 선택 + calibrate | `python/dojozero/evaluation/ladder.py`, `configs/evaluation/ladder.yaml` |
| 매치 시계 항상 전송(`timeout_match`, 수마다 `time_left`) | `cpp/protocol/external_engine.*` |
| 임시 opening 세트 100개 `renju_provisional_v0`(균형 미평가) | `benchmarks/openings/renju_provisional_v0.json` |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` / `pytest` | 255 + 1 skip / **61/61** (신규: 최대우도 복원·무한대 방지·anchor, 선택 규칙 4경우, 함정 정의 검증) |
| 적합성(`conformance --engines all`) | **10/11 eligible**(금수 위반 0, 실패 0, 즉승 놓침 0): Rapfi 2021/2025, AlphaGomoku 2023/2026, Embryo 2021r/2026, KataGomo 2021r/2026, Yixin 2018, Pela 2023. **JAX 2024 불가**: 신경망 평가 순간 0xC0000005로 종료(BEGIN 정석수만 동작), 원인 미확정 |
| calibration v0 (`runs/evaluation/calibration_v0.json`): DojoZero = taraguchi v1, **500 ms/수**(외부 엔진 단일 스레드 CPU), `renju_provisional_v0`, Tier A | 프로브 4쌍 × 10엔진 → 주 매치 24쌍 × {Yixin 2018, KataGomo 2021r, Pela 2023} |

주 매치 결과(DojoZero 관점, 24쌍):
| 상대 | score | ΔElo (95% CI) | 판정 |
|---|---|---|---|
| Yixin 2018 | 0.146 | **−307 [−501, −119]** | DojoZero 열세(유의) |
| Pela 2023 | 0.188 | **−255 [−426, −84]** | DojoZero 열세(유의); Pela가 금수 착수 1회(FORBIDDEN_MOVE로 기록, 함정 검사는 통과했었음) |
| KataGomo 2021r | 0.333 | −120 [−264, +23] | 결론 없음 |

### 사고·발견 (중요)
1. **외부 강도는 내부 지표가 보여준 것보다 훨씬 낮다**: 내부(heuristic 0.95–0.97, 세대 간 +123 등)와 달리, 500 ms/수에서 DojoZero는 2018년 엔진(Yixin)과 2023년 약한 엔진(Pela)에도 유의하게 진다(CLAUDE.md §1 echo chamber 경고의 실측 사례). 원인 후보(가설): fixed time에서 탐색량 부족(수당 수십 sims, Phase 8 E1), 가치망 품질. EVALUATION §20 기준 "performance" 또는 "value/search" — 다음 실험(scaling curve, fixed-visits 비교)으로 구분 필요.
2. **임시 opening 세트는 정보가 거의 없다**: 프로브 80판 중 흑 승 66판, DojoZero는 백으로 40판 전패, 대부분의 쌍이 1:1(각자 흑으로 승). 6돌·흑 차례 시작 국면이 흑에 크게 유리 → 균형 corpus(OQ-14)가 실제로 필요함을 확인. 4쌍 프로브는 상대 강도를 거의 구분하지 못했다(Yixin 프로브 0.5 → 주 매치 0.146).
3. **사다리 순서는 아직 측정되지 않았다**: 외부 엔진끼리 대국이 없어(별 모양 그래프) 엔진 간 rating은 사실상 prior이고(예: Rapfi 2025 = Rapfi 2021 = +9), anchor(Embryo 2026)도 8판뿐이라 DojoZero의 `dojo_external_elo` −182 [−404, +40]은 신뢰할 수 없다. **이 수치를 기력으로 인용하지 않는다** — 위 주 매치의 쌍대 ΔElo만 유효.
4. Embryo 2021/2026은 매치 시계 INFO가 없으면 turn 제한을 무시해 TIMEOUT → 클라이언트가 항상 시계를 보내도록 수정(ASSUMPTIONS A-P4 REJECTED). 수정 후 eligible.
5. Windows 오류 보고가 크래시한 프로세스를 잠시 살려 두어 CRASH가 TIMEOUT으로 보일 수 있음(JAX) — 실패 사유 분류의 한계로 기록.

### Open
- 균형 opening corpus(OQ-14) — 후보: 5돌(백 차례) 국면 + 강한 엔진 자체 대국 결과로 균형 필터.
- 외부 엔진 간 라운드로빈으로 사다리 연결, anchor는 대국 수가 많은 엔진으로.
- fixed-time scaling curve(예: 0.5/1/2/5 s)와 fixed-visits 비교로 열세 원인 분리.
- 재귀 거짓금수 함정, JAX 2025 등 대체 엔진(다운로드는 사용자 승인 필요).

---

## Phase 8 (E1) — External evaluation infrastructure (2026-10-19)

Phase 8의 첫 단계로 PRD §33 E1을 구현했다. **외부 엔진 바이너리가 없어(OQ-04) 실제 external calibration(M3 소급 gate)은 아직 수행하지 않았다** — 인프라는 mock 엔진과 DojoZero 자신(별도 프로세스)으로만 검증.

### Delivered
| 영역 | 파일 |
|---|---|
| 시간 제한 탐색(`SearchConfig.max_ms`, 루트 전술은 그 25%까지, ≥1 시뮬레이션 보장) | `cpp/search/search.*`, `python/dojozero/search/driver.py` |
| Gomocup 코덱 + engine-side 세션(좌표 변환 1곳) | `cpp/protocol/piskvork_adapter.*` |
| 외부 프로세스(Windows CreateProcess + Job object, 리더 스레드, timeout; POSIX 경로 미검증) | `cpp/protocol/gomocup_process.*` |
| ExternalEngine 클라이언트(START/BEGIN/TURN/BOARD/RESTART/ABOUT/END, 실패 분류) | `cpp/protocol/external_engine.*`, 바인딩 |
| 규칙 profile `RENJU_GOMOCUP`(RIF, 패스 없음) | `cpp/rules/rule_profile.*`, RULES §7.3 |
| Python 호스트 pbrain(임시 `pbrain-DojoZero`, 장애 주입 mock 겸용, 네트워크 warm-up) | `python/dojozero/protocol/pbrain.py` |
| TournamentRunner(paired, canonical GameState 심판, FORBIDDEN/ILLEGAL/CRASH/TIMEOUT/MALFORMED/PROTOCOL_ERROR/START_FAILED, RESTART 재사용) | `python/dojozero/evaluation/tournament.py` |
| 엔진 레지스트리(sha256 고정 필수, drift 거부), 템플릿·README | `python/dojozero/evaluation/external.py`, `external_engines/` |
| Strength DB(EVALUATION §17 스키마 v1, rating 이름·decision 값 강제) | `python/dojozero/evaluation/strength_db.py` |
| opening corpus(검증·set hash·누출 검사), **임시** `renju_smoke_v0`(24개, 균형 미평가, Tier A 전용) | `python/dojozero/evaluation/corpus.py`, `benchmarks/openings/renju_smoke_v0.json` |
| Strength Report v0(측정 안 한 절은 UNKNOWN) | `python/dojozero/evaluation/report.py` |

### Verified
| 검증 | 결과 |
|---|---|
| `dojozero_tests` Release | 255 통과 + 1 skip (신규: 시간 제한 탐색 2, 프로토콜 7) |
| `pytest tests/python` | **57/57** (신규 16: 프로토콜 왕복, paired 기록·DB, 실패 6종 사유 기록, 바이너리 없음 → START_FAILED, 금수/불법 분류, 레지스트리 hash 가드, corpus 검증·누출, DB 제약, 시간 제한 배치 탐색, report, 프로세스 재사용) |
| 스모크: taraguchi v1(in-process, 200 ms/수) vs heuristic(**별도 프로세스, 프로토콜 경유**), smoke 12쌍 | 0.958, 실패 0, DojoZero 평균 120 ms/수·**~30 sims/수** |
| 스모크: taraguchi v1 in-process vs 같은 모델 pbrain-py(별도 프로세스), 6쌍 200 ms | 첫 실행: **B 쪽 TIMEOUT 3건, 1,300초** → 원인: 새 프로세스마다 torch 기동 + 첫 XPU 평가가 수 초. 수정(warm-up + RESTART 재사용) 후 재실행: **실패 0, 63초**, 0.667 [−147, +388](차이 감지 안 됨, 같은 모델과 일관) |

### 사고·발견
1. **fixed time에서 탐색량이 매우 적다**: 200 ms/수에서 ~30 sims(배치 1–8 XPU 지연 지배). fixed-visits 평가(64 sims)보다 적어, 외부 엔진과의 fixed-time 비교에서 불리할 것으로 예상 — EVALUATION §20 "performance problem" 후보. 아직 외부 상대로 측정하지 않았으므로 가설.
2. 위 timeout은 "외부 엔진 실패"가 실제로 기록·분리되는 것을 보여준 사례(설계대로 B:TIMEOUT으로 집계)이자, 같은 문제가 DojoZero를 외부 매니저에 올릴 때도 생긴다는 경고(ADR-08 결정 시 warm-up 필수).
3. 장애 주입 테스트 중 `illegal` 케이스 1회 실패(전체 스위트 실행 중), 재현 안 됨(단독 3회, 전체 2회 통과). 300 ms 제한이 부하에서 TIMEOUT으로 바뀐 것으로 **추정**(원인 미확정) → TIMEOUT 케이스 외에는 2,000 ms로 완화.
4. self-match에서도 양쪽이 같은 XPU를 공유하고 외부 쪽 예산은 85%−30 ms라 대칭이 아니다. Tier B/C에서는 동시 실행 작업·장치 공유를 기록·통제해야 함(E-10).

### Open (E1 잔여 → 사용자 결정 필요)
- **실제 외부 엔진 calibration**: 바이너리 확보(OQ-04), 버전 고정(OQ-11), 시간 규정(OQ-13), Standard Renju corpus 출처(OQ-14) 결정 후.
- 규칙 적합성 사전 검사 자동화(EVALUATION §12.2), ladder rating 최대우도 적합, adaptive selection(E2), 병렬 대국, POSIX 프로세스 경로 실행 검증.

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
