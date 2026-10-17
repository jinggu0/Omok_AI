# DojoZero — EVALUATION.md

Status: **Draft v0.1** · 2026-10-18

---

## 1. 원칙
1. **Training loss로 강함을 판단하지 않는다.** 강함 = 통제된 대국 결과 + 고정 벤치마크.
2. **자기 대전 지표는 대리 지표다.** 내부 Elo 상승이 실제 기력 상승인지 확인하기 위해 **외부 기준 엔진(anchor)** 과의 대국을 Milestone 3부터 정기적으로 수행한다(§6). 내부 Elo만 오르고 외부 지표가 정체하면 echo chamber로 간주하고 학습 방향을 재검토한다.
3. 변경 하나당 측정 하나. 여러 변경을 묶어 한 번에 검증하지 않는다.
4. 하한 보장: 새 모델/새 탐색은 기존 best를 기본값으로 두고, 통계적으로 유의하게 나을 때만 교체(§5).
5. 안정성 위반(불법 action, 크래시, 타임아웃, NaN)은 기력과 무관하게 **0건**이어야 승격 가능.

---

## 2. 규칙 정확성 평가 (Milestone 1 게이트)
| 지표 | 기준 |
|---|---|
| 규칙 단위 테스트 통과율 (RULES §12 카테고리 전부) | 100% |
| oracle vs fast 차등 퍼징 | ≥10⁷ (국면, 착수점) 판정 비교, 불일치 0 — Phase 2 실측: 1.0×10⁸ 비교 / 980,775 국면, 불일치 0 |
| 1차원 window 전수 열거 (테이블 생성기 = oracle) | 불일치 0 |
| D4 가환성 property (금수·합법수·결과) | 위반 0 |
| apply/undo 왕복 property (hash·상태 완전 복원) | 위반 0 |
| opening 상태 머신 전 경로 | 위반 0 |
| `tests/compatibility/omokdojang/*.json` | 기본 profile과 불일치 시 **실패 보고** (ASSUMPTIONS C-xx 갱신 절차) |
| self-play·대국 중 불법 action 발생 | 0 (런타임 assert + 기보 재생 검증) |

forbidden-rule accuracy 지표(NN aux head의 금수 예측 정확도)는 별개다 — 규칙 엔진은 100%가 요구사항이고, NN 금수 head는 진단용.

---

## 3. 전술 평가 (Milestone 2 게이트)
- 세트: `benchmarks/{tactical,vcf,vct,endgame,adversarial}` (학습 데이터에 포함 금지 — 해시로 누출 검사).
- 문제 출처: 손으로 구성·검토한 문제, 공개 문제집(출처·라이선스 기록, 좌표만), 차등 비교로 발견한 사례. **자기 solver가 만든 문제만으로 구성하지 않는다**(순환 검증).
- 지표: solve rate(시간/노드 budget별 곡선), 평균 노드·시간, **오답률(WIN을 잘못 주장)** — 0이어야 함. 모든 WIN 주장은 PV 재생으로 검증.
- `UNKNOWN`과 `LOSS` 혼동 검사: LOSS로 보고된 모든 문제는 독립 검증(완전 탐색 가능한 소형 사례 또는 교차 엔진).
- 금수 관련 전술 하위 세트: 금수 유도로만 이기는 백 VCF, 거짓금수를 이용한 흑 승리, 흑 차단점이 금수인 방어 실패.

---

## 4. 대국 프로토콜

### 4.1 Paired games (MUST)
- 모든 평가 대국은 쌍으로 진행: 같은 시작 조건에서 (A=흑, B=백), (B=흑, A=백).
- 결과 단위 = **쌍(pair)**: 5가지 결과(0, 0.5, 1, 1.5, 2점) — pentanomial 통계(§5).

### 4.2 시작 조건 세트
| 세트 | 용도 | 구성 |
|---|---|---|
| `balanced_renju` | 중반 기력(일반 Renju) | Taraguchi-10 규칙으로 도달 가능한 5–6수 국면 중 강한 분석으로 \|v\|<0.15 인 것 N개(초기 N=100), 고정·버전 관리 |
| `rif26` | 외부 비교 표준 | RIF 26 opening + 대표 5수 |
| `taraguchi_full` | opening 포함 전체 게임 | 빈 보드, Taraguchi-10 전 과정 |

### 4.3 Taraguchi-10 paired protocol
- swap 때문에 색 배정이 결정에 따라 바뀐다. 따라서 쌍의 대칭축은 색이 아니라 **역할(role)**: 게임1 = A가 P0(임시 흑, 1수 착수자), 게임2 = B가 P0. 그 외 조건 동일.
- 보조 프로토콜: 4수 국면(고정 세트)부터 시작하여 "4수 후 흑 holder" 역할을 교대 → Choice/제안/선택 능력만 분리 측정.
- 기록: 최종 색 배정, 각 swap/branch 결정, 제안 집합, 선택 — opening 통계(§7).

### 4.4 조건 통제
- compute 통제: 고정 visits 또는 고정 시간(같은 하드웨어, 동시 실행 대국 수 고정). 보고서에 반드시 명시.
- 탐색 노이즈: 평가 대국은 노이즈 없음, 다양성은 시작 세트로 확보. 같은 시작 국면 중복 대국 금지(쌍 내부 제외).

---

## 5. 통계와 승격(promotion)
- Elo 추정: pentanomial 모델로 점수·분산 → Elo와 95% CI. 다수 모델 ladder는 BayesElo/Ordo식 최대우도 일괄 추정.
- **승격 = SPRT**(pentanomial, Fishtest 방식): H0: ΔElo ≤ 0, H1: ΔElo ≥ elo1, α=β=0.05. 초기 elo1=+15 (측정 노이즈 보고 조정), 최소 100쌍, 최대 2000쌍(최대 도달 시 H0 유지 = 승격 안 함).
- 고정 표본 보고 시 CI 하한 > 0 이어야 "우위"로 표기. "5승 3패" 같은 소표본 판단 금지.
- 승격 추가 게이트: §2 안정성 0건, §3 전술 세트 회귀 없음(solve rate 하락이 CI 밖이면 차단), 외부 anchor 대비 하락 없음(정기 점검 시).

---

## 6. 상대·기준점(anchors)
| 상대 | 목적 |
|---|---|
| Random-legal | 최저 기준, 합법성 검사 |
| Heuristic bot (1-ply 위협 평가) | Milestone 3 "결정적 승리" 대상 |
| Solver bot (VCF/VCT + heuristic) | 전술만으로의 강도, Ablation C 기준 |
| Pure neural baseline (solver OFF) | Milestone 6 비교 대상 |
| Historical checkpoints | 세대 간 Elo 사다리 |
| **외부 엔진: Rapfi** (Gomocup 상위, Renju 규칙 지원, GPL-3.0, 별도 프로세스) | 실제 기력의 외부 기준. 일반 Renju(`balanced_renju`, `rif26`)에서 paired, 여러 시간 budget으로 강도 곡선 |

- 외부 엔진은 Gomocup(piskvork) 프로토콜 어댑터로 대국. Taraguchi swap은 프로토콜이 지원하지 않으므로 외부 비교는 일반 Renju 시작 국면에서만.
- 외부 엔진 바이너리 다운로드·사용은 사용자 승인 후(PRD OQ-04).
- 금수 판정 차등 비교: 외부 엔진의 금수 판정과 불일치하는 국면은 자동 수집 → 사람이 RULES.md 기준으로 판정 → 회귀 테스트로 고정(누가 틀렸든 기록).

---

## 7. Opening 평가
- 지표: swap 결정 일치율(강한 분석 탐색 = 기준 결정) 및 **regret**(선택 결정의 기준 가치 손실), Choice 1/2 선택 분포, 제안 집합 품질 = min-value regret (`max_{orbit-valid Q*} min V − min_{q∈Q} V`, 기준 V는 ANALYSIS budget), 선택 정확도(백이 최소 가치 제안을 골랐는가).
- opening self-play의 최종 색별 승률(흑/백 advantage), opening key 분포 다양성(엔트로피).
- 기준값은 Opening Value DB의 고-budget 항목. 기준 자체의 불확실성(분산)을 함께 보고.

---

## 8. 성능 지표

### 8.1 측정 환경 (기준 머신, ASSUMPTIONS E-01..E-08)
Intel Core Ultra 7 258V (4P+4LPE), 32 GB, Intel Arc 140V iGPU, Windows 11, MSVC 14.43 Release(/O2), 전원 연결·고성능 모드, 다른 부하 없음. 결과에는 커밋·빌드 플래그·전원 모드 기록.

### 8.2 지표와 잠정 목표 (Phase 2–5 실측 후 갱신)
| 지표 | 잠정 목표 | Phase 2 실측 (Board A, 2026-10-18) | 비고 |
|---|---|---|---|
| apply+undo /s (단일 스레드) | ≥ 10M | GameState 3.4M (합법성 검사 포함) / Board place+remove 74M | **미달**. GameState 비용 대부분은 흑 착수마다의 금수 판정(~0.4µs). ADR-01 참조 |
| 흑 금수 판정 /s (중반 국면, 무작위 빈 점) | ≥ 1M | 2.1M (중반 평균 43돌), 2.9M (흑 밀집 퍼징 국면); p50 300–400ns, p99 700–900ns, max 0.4–1.1ms | 충족. max 이상치(깊은 재귀 또는 OS 지터)는 미조사 |
| legal_actions 생성 /s | ≥ 200k | 25k (NORMAL_RENJU, 흑 차례 포함) | **미달**. 빈 점마다 금수 판정 → 증분 금수 캐시 필요 (Phase 5 전 ADR-01 재검토) |
| VCF nodes/s | ≥ 1M | (Phase 3) | |
| MCTS nodes/s (uniform evaluator) | ≥ 500k | (Phase 5) | NN 제외 순수 트리 비용 |
| batched NN eval/s | 실측 후 설정 | (Phase 4) | 모델 크기·배치별 표 |
| 평균 탐색 지연 (FAST/STANDARD) | 실측 후 설정 | (Phase 5) | |
| self-play games/hour | 실측 후 설정 | (Phase 6) | Stage별 compute 계획의 입력 |
| 메모리 (트리·TT·replay 인덱스) | 실측 | — | |
| GPU 사용률 | 실측 | — | |

측정 명령: `dojozero_bench --positions 300 --seed 7` (단일 스레드, Release, 다른 부하 없음 — 퍼징 종료 후 측정). 원본 출력: `runs/bench_rules_phase2.txt` (git 미추적).

"잠정 목표"는 요구사항이 아니라 첫 측정의 비교 기준이다. 미달 시 원인 분석 후 목표 또는 설계를 갱신하고 이 표에 이력을 남긴다.

---

## 9. 고정 벤치마크 세트 (`benchmarks/`)
| 디렉터리 | 내용 | 실행 시점 |
|---|---|---|
| `rules/` | 기본 규칙 국면(5목, 장목, 백/흑 비대칭, 종국) | 매 커밋 |
| `forbidden/` | 금수·거짓금수(재귀 단계별) | 매 커밋 |
| `tactical/` | win-in-1, block, 4-3, 방어 희생, 금수 유도 | 매 모델 |
| `vcf/`, `vct/` | 증명 문제 | 매 모델·solver 변경 |
| `opening/` | Taraguchi 상태/제안 검증 + 결정 품질 세트 | 매 모델 |
| `endgame/` | 판이 거의 찬 국면, 패스·무승부 경계 | 매 모델 |
| `adversarial/` | 이전 버전이 틀렸던 국면, 차등 퍼징 최소화 사례, 외부 엔진 불일치 사례 | 매 커밋/모델 |

각 릴리스마다 동일 세트 실행 → 결과를 `runs/benchmarks/<version>.json`에 저장, 이전 버전과 diff 리포트.

---

## 10. Ablation 계획
각 항목은 동일 compute에서 paired match + 전술 세트로 비교. 보고: Elo per compute, wall-clock, 추론 지연, tactical score.

| ID | 비교 | 고정 조건 |
|---|---|---|
| A | ResNet vs Hybrid attention | 파라미터·FLOP 대응 크기 2쌍 |
| B | PUCT vs Gumbel root | 같은 모델, visits 16/64/256 |
| C | Tactical solver OFF vs ON (+ 호출 임계값 sweep) | 같은 모델 |
| D | Aux tactical heads OFF vs ON | 같은 데이터 양 |
| E | Tactical feature planes OFF vs ON | |
| F | Random-start RL vs tactical pretraining | 같은 self-play compute |
| G | Self-play only vs league | |
| H | Replay vs replay + reanalysis | 같은 총 compute |

Research questions(PRD §Research) 각각은 위 ablation 중 하나 이상에 매핑된다.

---

## 11. 리포트
- `dojozero evaluate`는 Markdown + JSON 리포트 생성: 조건(커밋, 모델 hash, compute, 시작 세트 버전), 결과(WDL, 쌍 통계, Elo±CI, SPRT LLR), 실패 0건 확인, 결정 분포.
- 리포트는 "실행됨"과 "검증됨"을 구분해 기술하고, 측정하지 못한 항목을 명시한다.
