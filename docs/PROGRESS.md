# DojoZero — PROGRESS.md

Phase-by-phase log: what was done, what was **verified** (with the command), what deviated from plan,
and what is still open. Newest phase first.

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
