# DojoZero — EVALUATION.md

Status: **Draft v0.2** · 2026-10-19 (v0.1 2026-10-18; v0.2 = 지속적 Strength Evaluation System §12–§23)

---

## 1. 원칙
1. **Training loss로 강함을 판단하지 않는다.** 강함 = 통제된 대국 결과 + 고정 벤치마크.
2. **자기 대전 지표는 대리 지표다.** 내부 Elo 상승이 실제 기력 상승인지 확인하기 위해 **외부 기준 엔진(anchor)** 과의 대국을 Milestone 3부터 정기적으로 수행한다(§6). 내부 Elo만 오르고 외부 지표가 정체하면 echo chamber로 간주하고 학습 방향을 재검토한다.
3. 변경 하나당 측정 하나. 여러 변경을 묶어 한 번에 검증하지 않는다.
4. 하한 보장: 새 모델/새 탐색은 기존 best를 기본값으로 두고, 통계적으로 유의하게 나을 때만 교체(§5).
5. 안정성 위반(불법 action, 크래시, 타임아웃, NaN)은 기력과 무관하게 **0건**이어야 승격 가능.
6. **강도 기반 개발(Strength-driven development)**: 기능을 구현했다는 사실만으로 완료로 보지 않는다. 채택은 §19 결정 규칙(`ACCEPT / REJECT / INCONCLUSIVE`)으로.
7. **Taraguchi-10 평가 ≠ 일반 Renju 평가**: 두 강도 보고를 하나의 Elo로 섞지 않는다(§15).

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
- 지표: solve rate(시간/노드 budget별 곡선), 평균 노드·시간, **오답률(WIN을 잘못 주장)** — 0이어야 함. 모든 WIN/LOSS 주장은 독립 검증기(`cpp/tactics/verify.*`: GameState 규칙만 사용, 방어자 노드 full-width)로 재검증하며, 검증기가 REFUTED를 내면 false claim.
- `UNKNOWN`과 `LOSS` 혼동 검사: LOSS로 보고된 모든 문제는 독립 검증(완전 탐색 가능한 소형 사례 또는 교차 엔진).
- 금수 관련 전술 하위 세트: 금수 유도로만 이기는 백 VCF, 거짓금수를 이용한 흑 승리, 흑 차단점이 금수인 방어 실패.

---

## 4. 대국 프로토콜

### 4.1 Paired games (MUST)
- 모든 평가 대국은 쌍으로 진행: 같은 시작 조건에서 (A=흑, B=백), (B=흑, A=백).
- 결과 단위 = **쌍(pair)**: 5가지 결과(0, 0.5, 1, 1.5, 2점) — pentanomial 통계(§5).

### 4.2 시작 조건 세트
버전 관리 파일·메타 필드·해시 규칙은 §15 (Opening corpus). 아래 이름은 corpus 파일과 1:1로 대응한다: `balanced_renju` → `renju_development_v*`, `rif26`(+ balanced 부분집합) → `renju_external_v*`, `taraguchi_full` → `taraguchi_full_v*`.

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
- **위 파라미터(최소 100쌍, elo1=+15)는 Tier B 채택 판정(§13) 기준이다.** RL 루프 안의 세대 승격(TRAINING §2.4: 20쌍 단위, 20–80쌍, elo1=+30)은 compute 제약으로 축소한 **Tier A 성격의 내부 게이트**이며, 그 결과는 `dojo_internal_elo`로만 보고하고 기능 채택 근거로 쓰지 않는다. 축소 파라미터는 TRAINING에 명시된 deviation.

---

## 6. 상대·기준점(anchors)
| 상대 | 목적 |
|---|---|
| Random-legal | 최저 기준, 합법성 검사 |
| Heuristic bot (1-ply 위협 평가) | Milestone 3 "결정적 승리" 대상 |
| Solver bot (VCF/VCT + heuristic) | 전술만으로의 강도, Ablation C 기준 |
| Pure neural baseline (solver OFF) | Milestone 6 비교 대상 |
| Historical checkpoints | 세대 간 Elo 사다리 |
| **외부 엔진 사다리** (Embryo · JAX · AlphaGomoku · KataGomo · Rapfi · Gomocup 구버전 등) | 실제 기력의 외부 기준 — **§12 External Strength Ladder**. Rapfi는 사다리의 최상단 reference 중 하나이며 단일 anchor가 아니다 |

- 위 표의 Random/Heuristic/Solver bot/Pure neural/Historical은 **내부 anchor**이며 결과는 `dojo_internal_elo`(§16). 외부 엔진 결과만 `dojo_external_elo`.
- 외부 엔진은 Gomocup(piskvork) 프로토콜 어댑터로 대국(ARCHITECTURE §3.12). Taraguchi swap은 프로토콜이 지원하지 않으므로 외부 비교는 Standard Renju 시작 국면에서만(§15).
- 외부 엔진 바이너리 다운로드·사용은 사용자 승인 후(PRD OQ-04).
- 금수 판정 차등 비교: 외부 엔진의 금수 판정과 불일치하는 국면은 자동 수집 → 사람이 RULES.md 기준으로 판정 → 회귀 테스트로 고정(누가 틀렸든 기록).

---

## 7. Opening 평가
- 지표: swap 결정 일치율(강한 분석 탐색 = 기준 결정) 및 **regret**(선택 결정의 기준 가치 손실), Choice 1/2 선택 분포, 제안 집합 품질 = min-value regret (`max_{orbit-valid Q*} min V − min_{q∈Q} V`, 기준 V는 ANALYSIS budget), 선택 정확도(백이 최소 가치 제안을 골랐는가).
- opening self-play의 최종 색별 승률(흑/백 advantage), opening key 분포 다양성(엔트로피).
- 기준값은 Opening Value DB의 고-budget 항목. 기준 자체의 불확실성(분산)을 함께 보고.
- **Phase 7 구현(간이판)**: `python/dojozero/evaluation/opening.py`. 제안 regret의 기준 V는 ANALYSIS 탐색이 아니라 **네트워크 가치**(후보 q를 제안·선택한 WHITE_MOVE_6 국면의 V_black) — 탐색 기준은 미구현. 기준 집합 = 엔진의 합법 PROPOSE action(대칭 제약 포함)으로 V_black 탐욕 선택 10개. 강도 비교 매치는 `--openings prefix`(무작위 오프닝을 1..len-1 수에서 자른 서로 다른 (canonical hash, phase) 시작점, 나머지 opening 결정은 엔진이 직접) — §4.3 역할 교대 그대로.

---

## 8. 성능 지표

### 8.1 측정 환경 (기준 머신, ASSUMPTIONS E-01..E-08)
Intel Core Ultra 7 258V (4P+4LPE), 32 GB, Intel Arc 140V iGPU, Windows 11, MSVC 14.43 Release(/O2), 전원 연결·고성능 모드, 다른 부하 없음. 결과에는 커밋·빌드 플래그·전원 모드 기록. **주의(ASSUMPTIONS E-11)**: 창 없는 백그라운드 프로세스는 Windows 전원 스로틀링(EcoQoS)으로 최대 ~8배 느려질 수 있다 — 벤치마크·평가 프로세스는 스로틀링을 프로세스 단위로 끈다(Phase 8 E3b부터).

### 8.2 지표와 잠정 목표 (Phase 2–5 실측 후 갱신)
| 지표 | 잠정 목표 | Phase 2 실측 (Board A, 2026-10-18) | 비고 |
|---|---|---|---|
| apply+undo /s (단일 스레드) | ≥ 10M | GameState 3.4M (합법성 검사 포함) / Board place+remove 74M | **미달**. GameState 비용 대부분은 흑 착수마다의 금수 판정(~0.4µs). ADR-01 참조 |
| 흑 금수 판정 /s (중반 국면, 무작위 빈 점) | ≥ 1M | 2.1M (중반 평균 43돌), 2.9M (흑 밀집 퍼징 국면); p50 300–400ns, p99 700–900ns, max 0.4–1.1ms | 충족. max 이상치(깊은 재귀 또는 OS 지터)는 미조사 |
| legal_actions 생성 /s | ≥ 200k | 25k (NORMAL_RENJU, 흑 차례 포함) | **미달**. 빈 점마다 금수 판정 → 증분 금수 캐시 필요 (Phase 5 전 ADR-01 재검토) |
| VCF nodes/s | ≥ 1M | 70–78k (Phase 3, `dojozero_tactics perf`) | **미달 (~13배)**. 노드마다 전체 보드 스캔으로 위협 재계산. 증분 위협 추적(ADR-01 C안 line code) 필요 |
| df-pn (VCT) nodes/s | — | 13.5–14k | AND 노드가 방어자 전체 합법수를 실제 보드에서 반박 검사(건전성 비용) |
| **Phase 8 E3b 솔버 가속** (2026-10-21, 같은 바이너리에서 ee1d011 참조 구현과 국면마다 교대 측정 `--impl pair`, 전원 스로틀링 해제) | VCT ≥ 100k nodes/s 또는 ≥ 5× | VCT(블런더 직후 247국면, 5만 노드) 배율: 1스레드 **3.3×**, 2: 3.5×, 4: 4.2×, 8스레드(필터 설정) **5.3×**; 국면별 중앙값 4.1–6.3×. 절대값은 배경 부하에 따라 1스레드 live 55k–82k nodes/s(ref 17k–25k). VCF(변경 없음) 109k → 125k. 노드 예산 하 결과·노드 수 **ref와 비트 동일** | 1스레드 목표 **미달**(3.3×, < 100k), 8스레드 5×는 충족. 남은 비용: AND 노드 반박 루프(빈 점마다), OR 노드 위협수 생성, 전체 보드 5목 점 스캔 — 보드 전역 증분 line 상태(ADR-01 C 본체)가 다음 후보 |
| **proof@시간** (필터 워크로드 1,976국면, analyze VCF 100k/VCT 50k, 8스레드; 상한 = 노드 예산만 1,836 WIN) | — | ref → live + VCT 시간 이월: 125 ms 1,137–1,155 → **1,614** / 250: ~1,220 → 1,744 / 500: ~1,320 → 1,796 / 1000: ~1,465 → 1,816. live가 ref의 WIN을 잃은 국면 0 | 125 ms의 개선판이 1000 ms의 ee1d011보다 많이 증명. 이월이 125 ms에서 +193 (PROGRESS Phase 8 E3b) |
| analyze_tactics 지연 | — | 노드 예산만: p50 0ms, p90 ~20ms, p99 4.5s, max 6.1s / `max_ms=500`: p99 308ms, max 311ms | MCTS 통합 시 시간 상한 필수 |
| MCTS nodes/s (uniform evaluator) | ≥ 500k | PUCT 34–39k / Gumbel 11–17k sims/s (Phase 5, `dojozero_bench_search`) | **미달**. 확장 시 legal_actions(금수 판정) 지배 |
| batched NN eval/s | 실측 후 설정 | B6C64, PyTorch: CPU fp32 ≤ ~316/s(b512), XPU bf16 ≤ ~5.6k/s(b512), b1은 10–31/s (측정 편차 큼) | `runs/bench_device_b6c64.json`. 배치 1이 매우 느려 MCTS는 배치 추론 필수 |
| 학습 step/s (B6C64, b256) | — | CPU 0.18, XPU fp32 0.74–1.7, XPU bf16 3.4(벤치) / 실제 학습 1.2–2.0(foreach 끔, 데이터 준비 포함) | ADR-04 |
| 평균 탐색 지연 (FAST/STANDARD) | 실측 후 설정 | 50트리 동시 64 sims ≈ 2.5–7s(망 워밍업 상태에 따라) | 시간 기반 한도 **구현(Phase 8 E1)**: `SearchConfig.max_ms`(루트 전술은 그 25%까지), 시뮬레이션 수는 상한 |
| self-play games/hour | 실측 후 설정 | ~4,300–6,900 games/h (256판 134–212초, Gumbel full 64/fast 16 sims, p_full 0.75, 평균 36수, B6C64 XPU) | 짧은 대국(무작위 오프닝 이후 ~25수) 기준 |
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

Research questions(PRD §Research) 각각은 위 ablation 중 하나 이상에 매핑된다. RQ-01..RQ-08(외부 강도 관련)의 실험 설계는 §23.

---

## 11. 리포트
- `dojozero evaluate`는 Markdown + JSON 리포트 생성: 조건(커밋, 모델 hash, compute, 시작 세트 버전), 결과(WDL, 쌍 통계, Elo±CI, SPRT LLR), 실패 0건 확인, 결정 분포.
- 리포트는 "실행됨"과 "검증됨"을 구분해 기술하고, 측정하지 못한 항목을 명시한다.
- 개발 주기 전체를 덮는 자동 Strength Report 형식은 §22.

---

# Part II — 지속적 Strength Evaluation System (v0.2)

§1–§11은 그대로 유효하다. 아래는 그 위에 쌓는 확장이며 paired games(§4.1), pentanomial·SPRT·Elo CI(§5), ablation(§10)을 재정의하지 않고 사용한다. 구현 시점은 PRD §33 E0–E4.

## 12. External Strength Ladder

```
DojoZero ─▶ internal checkpoints (historical best, 내부 봇)
        ─▶ weak / medium 외부 엔진 (Gomocup 구버전 등)
        ─▶ Embryo ─▶ JAX ─▶ AlphaGomoku(MK) ─▶ KataGomo ─▶ Rapfi
```
- 사다리 순서는 **우리 환경에서 직접 대국해 측정한 상대 강도로 정한다**. 위 순서는 초기 가설이며 공개 rating으로 고정하지 않는다. Gomocup 구버전을 넣어 간격을 촘촘하게 할 수 있다.
- 목적: "최상위 AI에게 몇 판 이겼나"가 아니라 (1) 현재 대략적 기력, (2) 승률이 급락하는 구간, (3) 격차의 원인(전술/전략/search, §20), (4) 새 기능이 실제 Elo를 올렸는가를 지속 측정.

### 12.1 엔진 메타데이터 `external_engines/<engine_id>.json`
```json
{ "schema_version": 1, "engine_id": "rapfi_2025", "engine_name": "Rapfi", "version": "...",
  "executable_sha256": "...", "source": "URL 또는 '사용자 제공'", "license": "GPL-3.0 | UNVERIFIED",
  "supported_rules": ["RENJU"], "protocol": "gomocup",
  "config": {"threads": 1, "timeout_turn_ms": 1000, "timeout_match_ms": 0, "max_memory_mb": 1024, "extra_info": {}},
  "public_reference_rating": {"system": "gomocup", "value": null, "date": null, "note": "참고 메타데이터 전용"},
  "conformance": {"checked": false, "forbidden_violations": null, "date": null} }
```
- **코드에 엔진 이름이나 공개 Elo를 절대 기준으로 하드코딩하지 않는다.** ladder 구성은 이 파일들과 Strength DB(§17)에서 읽는다.
- 바이너리는 저장소에 넣지 않는다. 경로는 사용자 로컬 설정(`external_engines/local_paths.json`, git 제외)에만, 실행 전 sha256 대조.
- 같은 엔진의 다른 버전은 **다른 engine_id**(예: `rapfi_2024`, `rapfi_2025`). 설정(threads, 시간)이 다른 경우도 결과는 별도 player로 기록(engine_id + config hash).

### 12.2 규칙 적합성 사전 검사 (ladder 등록 조건)
1. `INFO rule`로 Renju를 전달했을 때 금수 국면 세트(`benchmarks/forbidden/`에서 선택)에서 흑으로 금수점을 두지 않는가 — 위반 수를 기록.
2. 흑 장목·백 장목 처리, 정확한 5 우선 국면에서 종국 판정이 canonical `GameState`와 일치하는가(심판은 항상 DojoZero `GameState`).
3. 짧은 시험 대국 20판에서 crash/timeout/malformed 0건.
- 위반이 있는 엔진은 사용할 수 있으나 해당 사유로 끝난 대국은 `termination_reason`으로 구분하고(§17), 위반율을 report에 표기. 금수 착수가 잦은 엔진은 사다리에서 제외(임계값 config).
- **구현(Phase 8 E2)**: `python -m dojozero.evaluation.conformance --engines all --update-meta`. 함정 5종(즉승점, 흑 장목·4-4·3-3 회피, 백 장목 승리) — 함정 국면과 정답은 C++ 규칙이 자체 검증 — + heuristic 상대 짧은 대국 2쌍. 위반 0·실패 0일 때만 `eligible`(임계값 0). 결과는 `external_engines/<id>.json`의 `conformance`와 `runs/evaluation/conformance/`. 재귀 거짓금수 함정은 아직 없음.

## 13. Evaluation Tiers

| Tier | 목적 | 상대 | 조건 | 실행 시점 | 판정 용도 |
|---|---|---|---|---|---|
| **A — Regression** | 변경이 기존 기력을 망가뜨렸는지 빠르게 탐지 | previous best, heuristic bot, solver bot (+ §3 전술 세트, §2 규칙 세트) | 낮은 compute, 작은 opening set(예: 20–50쌍), 수 분 | search/model/solver의 의미 있는 변경 후, RL 세대 승격 | 회귀 차단만. **절대 Elo 추정에 쓰지 않음** |
| **B — Development** | 개발 방향 결정 | historical best + ladder 중 점수 25–75% 상대 2–4개(§14) | 충분한 쌍(§5: 최소 100쌍 또는 SPRT), `renju_development_v*` + Taraguchi 기능이면 `taraguchi_full_v*`, Elo ± 95% CI | 기능 후보마다, 정기(예: 승격 N회마다) | **기능 채택의 주 기준(§19)** |
| **C — Reference** | 최상위 AI와의 객관적 격차 | Rapfi, KataGomo, AlphaGomoku, JAX, 기타 현행 강자 | 하드웨어·compute·opening set(`renju_external_v*`) 엄격 고정, 대국 1개씩 순차, 여러 budget | milestone, release candidate | 장기 benchmark history 보존. 개발 판단의 단독 근거로 쓰지 않음 |

- 모든 Tier에서 §4.1 paired, §1-5 안정성 0건 원칙 유지. 외부 엔진 쪽 실패는 §17 `termination_reason`으로 분리 기록하고, **DojoZero 쪽 crash/timeout/illegal은 Tier와 무관하게 승격 실패**.

## 14. Adaptive Opponent Selection
정보가 없는 매치(거의 다 이기거나 다 지는)를 줄이고, DojoZero 실제 강도 근처에 대국을 집중한다.
```
score(opponent) > hi   (기본 0.75) → 더 강한 ladder 상대를 후보에 추가
score(opponent) < lo   (기본 0.25) → 더 약한 ladder 상대를 후보에 추가
lo ≤ score ≤ hi                   → primary calibration opponent
```
- 임계값·후보 수·탐색 쌍 수(예: 첫 10쌍으로 구간 판정 후 본 매치)는 config(`configs/evaluation/ladder.yaml`).
- 선택 로그(어떤 점수로 어떤 상대를 골랐는가)를 DB에 저장.
- 최상위 엔진과의 대국은 Tier C로 유지하되, Rapfi 상대 0–2% 같은 결과만으로 개발 성능을 판단하지 않는다.
- **ladder v1b(2026-10-19)**: 균형 corpus + 엔진 간 라운드로빈(쌍당 4쌍) + DojoZero 프로브·주 매치로 적합 — 결과는 PROGRESS Phase 8 (E2b). 엔진 간 대국은 병렬(동시 3판) 실행, 동시 실행 수 기록.
- **구현(Phase 8 E2)**: `python -m dojozero.evaluation.ladder calibrate`(`configs/evaluation/ladder.yaml`). calibration v0의 교훈: (1) 프로브 4쌍은 흑 유리 opening에서 강도를 거의 구분하지 못함 → 균형 corpus 전에는 프로브를 늘리거나 생략, (2) DojoZero하고만 대국한 엔진들의 rating은 prior가 지배 → 사다리 순서를 내려면 **엔진 간 대국**이 필요, (3) anchor는 대국 수가 많은 엔진으로 정한다.
- `dojo_external_elo` 추정: 사다리 전체 결과를 BayesElo/Ordo식 최대우도로 일괄 적합(§5), 외부 엔진 하나를 기준점(0)으로 고정하고 그 선택을 report에 명시.

## 15. Benchmark Opening Policy & Opening Corpus

### 15.1 두 개의 강도 보고 (혼합 금지)
| 보고 | 시작 국면 | 상대 | rating 이름 |
|---|---|---|---|
| **Standard Renju strength** | 고정 balanced Renju 국면(Taraguchi로 도달 가능한 5–6수 국면 포함), 같은 국면에서 색 교환 paired | 외부 Gomocup 엔진 + 내부 | `dojo_external_elo`, `dojo_internal_elo` |
| **Taraguchi-10 full-game strength** | 빈 보드 또는 opening prefix(§7 Phase 7 구현), swap·direct fifth·offer-ten·select 포함, 역할(P0/P1) 교대 paired(§4.3) | DojoZero 계열·내부 봇(Taraguchi action 지원 상대만) | `dojo_taraguchi_elo` |
- Taraguchi action을 지원하지 않는 외부 엔진은 Taraguchi rating에 넣지 않는다.
- RQ-08: Taraguchi 개선이 Standard Renju 강도를 희생하는지 두 보고를 함께 추적.

### 15.2 Corpus 파일 `benchmarks/openings/`
`renju_external_v1.json`, `renju_development_v1.json`, `taraguchi_full_v1.json`. 항목:
```json
{ "schema_version": 1, "set_id": "renju_development_v1", "profile": "RENJU_RIF",
  "openings": [ { "opening_id": "rd1-0001", "moves": ["H8", "..."], "rule_profile": "RENJU_RIF",
                  "source": "generated|literature|...", "source_version": "...", "expected_legal": true,
                  "approx_balance": {"value_black": 0.05, "method": "multi-engine consensus", "uncertainty": 0.1},
                  "creation_method": "..." } ] }
```
- **set hash** = 정규화 JSON(openings 배열, 키 정렬)의 sha256. 모든 match report·DB `Match.opening_set_hash`에 기록. 해시가 다른 실험끼리는 비교하지 않는다.
- 로드 시 모든 수순을 C++ `GameState`로 재생해 `expected_legal` 검증, 중복 canonical hash 거부.
- balance는 순환 판정을 피하려고 복수 엔진(가능하면 외부 엔진 포함)과 고-budget 분석의 합의로 기록하고 불확실성을 함께 저장(출처는 OQ-14).
- **누출 검사**: corpus 시작 국면의 canonical hash가 학습 데이터·self-play 시작점에 있으면 실패(PRD §15). self-play opening 생성기는 corpus 해시 집합을 제외 목록으로 받는다.

## 16. Rating 이름 규칙 (혼용 금지)
| 이름 | 뜻 | 출처 |
|---|---|---|
| `gomocup_reference_elo` | Gomocup 등 공개 rating | 외부 공개 수치. **메타데이터 전용**, 계산에 쓰지 않음 |
| `dojo_external_elo` | 우리 환경에서 외부 사다리와 직접 대국해 추정한 Elo (Standard Renju) | §14 최대우도 |
| `dojo_internal_elo` | self-play 세대, historical checkpoint, 내부 봇(heuristic/solver/random) 대비 Elo | §5 |
| `dojo_taraguchi_elo` | Taraguchi-10 full-game Elo | §15.1 |
- "Gomocup Elo 3000 엔진에 50%이므로 DojoZero = 3000" 같은 동일시 금지. 보고서는 항상 필드 이름과 기준점(어느 엔진·버전 = 0)을 함께 쓴다.

## 17. Strength Database (`runs/evaluation/results.sqlite`)
stdout으로 끝내지 않고 모든 평가를 누적한다(SQLite, WAL, 스키마 버전 테이블). 최소 스키마:
| 테이블 | 열 |
|---|---|
| `engine` | engine_id, name, version, kind(dojo/external/internal_bot), model_hash, executable_sha256, git_commit, config_json, config_hash |
| `match` | match_id, engine_a, engine_b, tier, rule_profile, profile_hash, opening_set_id, opening_set_hash, compute_mode(fixed_visits/fixed_time), budget_json, hardware_id, concurrent_games, start_time, end_time, git_commit, command |
| `game` | game_id, match_id, pair_id, opening_id, a_player(P0/P1), a_color, result, winner, moves(action 수순), termination_reason, plies, elapsed_a, elapsed_b, search_stats_json(nodes, sims, nn_evals, depth, …) |
| `rating_snapshot` | snapshot_id, date, rating_system(`dojo_external_elo`/`dojo_internal_elo`/`dojo_taraguchi_elo`), anchor_engine, engine_id, rating, lower_ci, upper_ci, n_games, match_ids |
| `experiment` | experiment_id, baseline, candidate, changed_feature, hypothesis, primary_metric, tier, match_ids, result(score, Δelo, CI, LLR), decision(ACCEPT/REJECT/INCONCLUSIVE), notes |
| `analysis` | analysis_id, game_id, ply, kind(loss_category / reanalysis), budget, result_json — 실전 결과와 분리(§21) |
- `termination_reason`: `NORMAL_WIN`, `DRAW`, `FORBIDDEN_MOVE`(금수 착수 — 규칙상 패), `CRASH`, `TIMEOUT`, `MALFORMED_RESPONSE`, `ILLEGAL_MOVE`, `PROTOCOL_ERROR`, `ADJUDICATED`, `MAX_PLIES`. 외부 엔진 실패 사유는 점수 계산과 별개로 집계해 보고한다(점수 처리 방식은 config에 명시: 기본 = 실패한 쪽 패, 단 report에 실패 수 표기).
- 기존 JSON 산출물(`runs/**/match*.json`, registry)은 가져오기(import) 스크립트로 DB에 넣을 수 있게 한다(E1).

## 18. Compute-normalized Metrics & Strength Scaling
- 모든 match에 기록(PRD NFR-08): wall time, CPU threads, GPU device·사용률, 메모리, nodes, MCTS sims, NN evals, NN batch, NN eval/s, nodes/s, 동시 대국 수, 전원 모드.
- 비교 지표: raw strength / Elo @ fixed time / Elo @ fixed simulations / time 대비 scaling.
- **fixed visits와 fixed time 결과를 같은 표에 섞지 않는다.** 외부 엔진은 시간으로만 통제 가능하므로 외부 비교는 fixed time이 기본(DojoZero 시간 기반 SearchLimits는 E1 구현 항목 — 현재 미구현, §8.2).
- **Scaling curve**: 주요 상대와 여러 budget(예: 50 ms, 100 ms, 500 ms, 1 s, 5 s 또는 FAST/STANDARD/STRONG)에서 paired 매치 → `score(t)`, `Δelo(t)` ± CI 곡선. 해석: 작은 budget에서만 경쟁력 → search scaling 문제, 모든 budget에서 일정한 격차 → NN/평가 품질, budget 증가 시 전술 패배 비중 감소 여부 → solver 병목(§20).
- 공정성(PRD R-09): Tier C는 대국 1개씩, 양쪽 동일 wall time, CPU 스레드 수 명시, DojoZero의 GPU 사용 사실을 report에 표기. "같은 시간"과 "같은 하드웨어 자원"은 다를 수 있음을 결론에 적는다.
- 모델 선택은 파라미터 수나 validation loss가 아니라 Elo/compute(§10 Ablation A와 연결).

## 19. Development Decision Policy
새 search/model/training 기능은 가능한 경우 baseline vs candidate paired A/B(Tier B)로 판정한다.

채택 조건(모두 필요):
1. 안정성 regression 없음(§1-5, §2).
2. tactical/rule benchmark regression 없음(§3; CI 밖 하락 차단).
3. primary strength metric 개선(사전에 `experiment.primary_metric`으로 지정 — 사후 변경 금지).
4. compute 비용 대비 이득(fixed time 결과 또는 Elo/compute로 확인).
5. 통계적으로 불명확하면 기본값으로 승격하지 않는다.

결과 상태:
| 상태 | 조건 |
|---|---|
| `ACCEPT` | 1–4 충족, SPRT H1 또는 고정 표본 CI 하한 > 0 |
| `REJECT` | 1·2 위반, 또는 SPRT H0 / CI 상한 < 0 |
| `INCONCLUSIVE` | 그 외. **성공으로 해석하지 않음**: 기존 기본값 유지, 필요하면 표본을 늘려 재실험 |
- 모든 판정은 `experiment` 테이블과 PROGRESS에 기록. Phase 6 세대 2(80쌍 0.513 → 유지)는 이 규칙의 INCONCLUSIVE 사례.

## 20. Diagnosis Matrix
평가 결과 → 다음 개발 후보. 판단은 가설이며 §19 실험으로 확인한다.
| 현상 | 판단 | 다음 후보 |
|---|---|---|
| VCF/VCT benchmark 실패 | tactical weakness | ThreatDetector / VCF / VCT / df-pn 개선 |
| 전술 benchmark는 강하나 외부 장기전에서 약함 | strategic / value / search weakness | value network, MCTS, 학습 데이터 |
| fixed visits에선 강하나 fixed time에서 약함 | performance problem | NN 추론, 배칭, legal_actions, search 최적화(§8.2 미달 항목) |
| 짧은 시간은 경쟁력, 긴 시간에서 격차 증가 | search scaling weakness | tree policy, transposition, solver 통합, value 정확도 |
| previous best엔 강하나 외부 anchor는 정체 | self-play echo chamber | opponent 다양성, league, 데이터 분포, search 재검토 |
| 특정 opening subset에서만 큰 패배 | opening weakness | opening curriculum, Opening Value DB |
| 특정 외부 엔진에만 점수 상승, held-out 엔진은 정체 | opponent overfitting | 상대 순환, Tier C 교차 확인 |
| 외부 대국 패배에서 timeout 비중이 큼 | time management | 시간 배분, 조기 종료, 배치 지연 |
| 새 아키텍처(예: ResNet → Transformer) validation 개선, 실제 Elo 하락 | inference cost / search 상호작용 | 기본 모델로 승격하지 않음, Elo/compute 재측정 |

## 21. 패배 분석과 Match Reanalysis

### 21.1 패배 원인 분류 (post-game pipeline)
패배 기보마다(가능하면 무승부도) 다음을 기록. 초기 분류기는 단순 규칙이어도 되며 정확도를 표본 검수로 보고한다.
| 범주 | 검출 방법(초기) |
|---|---|
| first major evaluation swing | 실전 root value 시계열에서 첫 급변(임계값 config) 지점 |
| missed immediate tactic | 그 국면에 즉시 5 / 필수 방어가 있었는데 다른 수 |
| missed VCF / VCT | 고-budget `analyze_tactics`가 WIN을 증명한 국면에서 다른 수(증명은 독립 검증기로 재확인, §3) |
| illegal / forbidden issue | 금수 관련 종국·금수 유도 패배 |
| opening disadvantage | 시작 국면 직후 가치가 이미 불리(분석 기준) |
| search timeout | 시간 초과·시간 부족 상태의 착수 |
| value misprediction | 실전 value와 재분석 value 차이 큼 |
| policy candidate miss | 재분석 최선수가 실전 policy 상위 k 밖 |
- 결과 요약 예: "20패 중 VCF 탐지 실패 7, opening 5, long-horizon value 6, timeout 2" → §20으로 다음 과제 선택.
- **구현(Phase 8 E2c)**: `python -m dojozero.evaluation.loss_analysis` — 범주 `tactical_blunder / strategic_loss(lost_from_start) / missed_win / unclassified`, 솔버(건전한 증명)로 착수 전후 판정, policy 상위 10수 중 반박되지 않은 대안 탐색, 네트워크 value 과신 표시. 결과는 DB `analysis`(kind `loss_category`). 사람 표본 검수는 아직 없음.

### 21.2 Reanalysis queue
- 우선순위: 예상 밖 패배 > 고신뢰 blunder > 외부 엔진과의 불일치 > solver 불일치 > 후보 모델 regression.
- analysis budget(ANALYSIS 프로필, 실전보다 훨씬 큰 compute)으로 재탐색 + solver. 결과는 DB `analysis` 테이블에만 저장하고 **실전 결과·rating에 섞지 않는다**.
- 재분석에서 확인된 전술 국면은 사람 검토 후 `benchmarks/adversarial/`로 승격 가능(학습 데이터 누출 검사 대상).

## 22. 자동 Strength Report (`dojozero report`)
Markdown + JSON, Strength DB에서 생성. 측정하지 않은 항목은 `UNKNOWN`, 추정은 "추정"으로 표시.
```
# DojoZero Strength Report
Commit / Model(hash) / Rule profile / Opening set(id + hash) / Hardware / Search budget(compute_mode)

## Internal comparison      Candidate vs Previous Best: W/D/L, score, ΔElo, 95% CI, SPRT LLR/decision
## External ladder          상대별(Embryo, JAX, AlphaGomoku, KataGomo, Rapfi, …): W/D/L, score, pair score,
                             local Elo difference(dojo_external_elo 기준점 명시), CI, 실패 사유 수
## Taraguchi-10              dojo_taraguchi_elo, 역할별 점수, P0/흑 holder 점수
## Tactical                  VCF, VCT, TSS, forbidden adversarial (solve rate, 오답 0 확인)
## Performance               nodes/s, NN eval/s, latency, memory
## Opening                   swap 결정 분포, proposal regret, selection regret
## Regression                rule failures, illegal moves, timeouts, crashes (양쪽 분리)
## Diagnosis                 가장 큰 약점 3개 (§20, §21 근거 수치와 함께)
## Recommendation            후보별 expected impact / implementation cost / evidence.
                             측정값이 뒷받침하지 않는 추천은 "가설"로 표기하고 단정하지 않는다.
```

## 23. Research Questions — 실험 설계
| RQ | 독립변수 | 종속변수 | 고정 조건 | 성공 기준 / 판정 |
|---|---|---|---|---|
| RQ-01 fixed visits vs fixed time 순위 | 통제 방식(visits / time), 모델 2개 이상 | 모델 간 Δelo의 부호·크기 | 같은 opening set, 하드웨어 | 두 방식에서 순위가 바뀌는지 CI로 판정 |
| RQ-02 NN 규모 vs Elo/compute | 모델 크기(B6C64 → B10C128 …) | Elo @ fixed time, Elo @ fixed visits | 같은 데이터·학습 compute | fixed time Elo가 CI 하한 > 0으로 개선되면 규모 증가 채택 |
| RQ-03 solver가 줄이는 패배 유형 | solver ON/OFF (Ablation C) | §21 범주별 패배 수, 외부 상대 점수 | 같은 모델, 같은 상대·budget | 범주별 차이를 비율 검정으로 보고 |
| RQ-04 solver 임계값 vs 상대 강도 | 호출 임계값 × ladder 상대 | score | 같은 모델 | 상대별 최적 임계값이 CI상 다른지 |
| RQ-05 budget scaling | search budget (§18) | `Δelo(t)` vs 주요 상대 | 같은 모델·opening | 곡선 기울기와 CI 보고 |
| RQ-06 internal vs external 상관 | 세대(checkpoint) | `dojo_internal_elo`, `dojo_external_elo` | 같은 ladder·opening set | 순위 상관(Spearman)과 CI; 낮으면 echo chamber 경고 |
| RQ-07 특정 외부 엔진 과적합 | 세대 | 학습·튜닝에 쓴 상대 vs held-out 상대 점수 | ladder 고정 | 두 추이의 괴리 |
| RQ-08 Taraguchi 개선의 midgame 비용 | Taraguchi 학습 유무/세대 | `dojo_taraguchi_elo`, Standard Renju score | 같은 compute | Standard Renju 하락이 CI 밖이면 "희생 있음" |
