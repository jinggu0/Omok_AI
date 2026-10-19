# DojoZero — Product Requirements Document

Status: **Draft v0.2 — 리뷰 대기** · 2026-10-19 (v0.1 2026-10-18) · Owner: Jinggu0_
관련 문서: [RULES](RULES.md) (규칙 규범) · [ARCHITECTURE](ARCHITECTURE.md) · [TRAINING](TRAINING.md) · [EVALUATION](EVALUATION.md) · [ASSUMPTIONS](ASSUMPTIONS.md)

규범 우선순위: RULES.md > PRD.md > 나머지. 충돌 시 상위 문서를 따르고 하위 문서를 고친다.

---

## 1. Problem Statement
오목도장 앱의 유단자 대국 규칙(가정: RIF Renju + Taraguchi-10)으로 대국·분석할 수 있는 **독립 실행형** 고기력 AI가 없다. 기존 공개 엔진들은 (a) Taraguchi-10의 swap·10제안 opening을 지원하지 않거나, (b) 흑 금수의 재귀적 거짓금수 판정이 완전한지 검증 근거가 공개되어 있지 않거나, (c) 앱 규칙과의 차이가 문서화되어 있지 않다.
DojoZero는 **정확성이 검증된 규칙 엔진** 위에 전술 solver와 신경망 탐색을 결합해, 국면별로 "합법수·금수와 그 이유·강제승 여부·승률·최선수·opening 결정"에 답하는 엔진을 만든다.

## 2. Background
- Renju: 15×15, 흑에게만 3-3·4-4·장목 금수, 흑은 정확한 5로만 승리, 백은 5 이상으로 승리(RULES §2).
- 3-3 판정은 extension point의 합법성까지 따지는 **재귀** 정의(RIF Art. 9.3) — 고정 패턴 매칭으로는 정확하지 않다.
- Taraguchi-10(RIF 승인 opening rule): 1–4수를 점점 넓은 중앙 영역에 두며 매 수 swap 가능, 4수 후 "9×9 안 직접 5수 + swap" 또는 "비대칭 5수 후보 10개 제안 → 백 선택" (RULES §8).
- AlphaZero/KataGo/Gumbel AlphaZero 계열 self-play RL은 오목류에서 강력하지만 compute 집약적이다. 렌주 강자 엔진(Rapfi 등)은 강한 전술 탐색(VCF)과 경량 평가를 결합한다.
- 이 저장소는 2026-10-18 기준 빈 git 저장소다(ASSUMPTIONS E-09). 개발 머신은 CUDA GPU가 없는 노트북이다(E-04).

## 3. Users / Stakeholders
| 사용자 | 필요 |
|---|---|
| 프로젝트 오너(연구·개발자) | 재현 가능한 실험, 정확한 규칙, 강한 모델, 정직한 지표 |
| 렌주 학습자/분석가 | 국면 분석(최선수·PV·금수 이유·강제승), 텍스트 CLI 대국 |
| 향후 기여자 | 명확한 모듈 경계, 문서화된 가정, 테스트 |

## 4. Goals
| ID | 목표 | 측정 |
|---|---|---|
| G1 | RIF Renju + Taraguchi-10 규칙을 정확히 구현 | 규칙 테스트 100%, oracle 차등 퍼징 불일치 0 (EVALUATION §2) |
| G2 | 금수·거짓금수의 판정 근거를 설명 | `ForbiddenInfo` 재귀 경로 출력 |
| G3 | 정확한 전술 solver (VCF/VCT/df-pn) | 고정 전술 벤치마크 solve rate, 오답 0 |
| G4 | 신경망+탐색 엔진이 기준 봇들을 결정적으로 이김 | paired SPRT 통과 (heuristic, solver bot) |
| G5 | Taraguchi-10 opening 결정을 학습 | opening regret 지표 감소, Elo |
| G6 | 하이브리드(전술+신경망) > 순수 신경망 | 통계적으로 유의한 paired 결과 |
| G7 | **버전 고정된 External Strength Ladder**에서 기력 위치를 지속적으로 측정 (Rapfi 단일 anchor → 내부 checkpoint · 약/중 외부 엔진 · Embryo · JAX · AlphaGomoku · KataGomo · Rapfi 사다리) | `dojo_external_elo` ± CI, ladder 위치(50% 경계 상대), 상대·budget별 score 곡선 (EVALUATION §12–§18). 절대 목표치는 OQ-03 결정 후 설정 |
| G8 | 완전 재현 가능한 릴리스 | 체크포인트로 결과 재현, resume 일치 테스트 |
| G9 | **Strength-driven development**: 기능 채택은 구현 완료가 아니라 측정된 강도로 결정 | 변경마다 paired A/B 결과 `ACCEPT / REJECT / INCONCLUSIVE` 기록 (EVALUATION §19) |

## 5. Non-goals
- 오목도장 앱(또는 어떤 온라인 서비스) 자동 조작, 화면 인식 봇, 온라인 대국 부착 — **구현하지 않는다**.
- 앱 데이터 대량 수집·스크래핑.
- 15×15 이외 보드, Freestyle/Standard Gomoku, Swap2·Soosõrv-8 등 다른 opening (v1 범위 밖; profile 구조는 확장 가능하게).
- 그래픽 GUI·모바일 앱 (텍스트 CLI만; GUI는 Future Work).
- 시간 관리·초읽기 등 대회 운영 규정(엔진은 SearchLimits만 받음).

## 6. Scope
- v1 포함: C++ 규칙 엔진, opening 상태 머신, 대칭 모듈, 전술 solver, 신경망 학습 파이프라인(Python), MCTS/Gumbel 탐색, self-play, replay, league, 평가·벤치마크, CLI, 문서.
- 작업 순서: Phase 0–8 (§33 Milestones). 각 Phase는 테스트 통과 후에만 다음으로.

## 7. Rule Definition
[RULES.md](RULES.md)가 규범이다. 요약:
- 좌표: `row`(아래 0)·`col`(왼쪽 0), `Point = row*15+col`, 표기 `A–O` + `1–15`. 다른 좌표계 금지.
- 패턴: FIVE(정확히 5), OVERLINE(6+), FOUR(completion 기반, 흑은 정확한 5로만), STRAIGHT FOUR, THREE(진짜 = 합법적 extension으로 straight four 가능, **재귀**), DOUBLE FOUR(한 직선 포함), DOUBLE THREE(서로 다른 방향 진짜 three 2+), FALSE FORBIDDEN, LEGAL FOUR-THREE.
- Exact-five precedence: 흑의 정확한 5는 어떤 금수 모양과 동시여도 승리(RIF 9.2).
- Taraguchi-10 상태 머신 15 phase, PlayerId/StoneColor/SideAssignment 분리, swap = 색 배정 교환.
- 10제안 대칭 = 5돌 국면 D4 동치(= stabilizer orbit, 증명 포함).

## 8. App Compatibility Assumptions
- `OMOKDOJANG_DAN` = RIF Renju + Taraguchi-10이라는 **가정**(C-11).
- 공식 문구에 없는 결정(Choice 결정자, swap 결정자, 대칭 정의 등)과 앱별 동작(금수점 처리, 패스, 재귀 깊이 등)은 [ASSUMPTIONS.md](ASSUMPTIONS.md) Compatibility Matrix(C-01..C-20)에 `UNVERIFIED_APP_BEHAVIOR`로 기록. 모든 항목은 profile 파라미터로 노출되어 앱 증거에 따라 코드 수정 없이 바꿀 수 있다.
- 앱 증거는 사용자가 수동 수집 → `tests/compatibility/omokdojang/*.json` 회귀 테스트.

## 9. Functional Requirements
| ID | 요구사항 |
|---|---|
| FR-01 | `GameState`: apply/undo/legal_actions/is_legal/result/forbidden(point)/phase/actor/hash/canonical_hash/transform (ARCHITECTURE §3.6) |
| FR-02 | 흑 금수 판정은 RULES §4 알고리즘(완전 재귀). 판정 결과와 `ForbiddenInfo` 설명 제공 |
| FR-03 | `RuleProfile`: board_size, black/white win condition, black/white forbidden rules, opening_rule, pass_allowed, illegal_move_behavior, exact_five_precedence, Taraguchi 파라미터 |
| FR-04 | Taraguchi-10 상태 머신(RULES §8.3) 전 phase, undo 가능, malformed state 거부 |
| FR-05 | 10제안 검증·대칭 판정 모듈(E1/E3 모드), canonical orbit id |
| FR-06 | D4 변환을 보드·action·마스크·제안 집합·opening 메타데이터에 일관 적용 |
| FR-07 | 전술 solver: ThreatDetector, VCF, VCT, TSS, df-pn; `ProofResult{WIN,LOSS,UNKNOWN}`, TT, 노드/시간 budget, PV |
| FR-08 | AI 의사결정 우선순위: 즉시 5 → 강제 방어 → 증명된 전술 → neural MCTS |
| FR-09 | 신경망: 입력 planes(ARCHITECTURE §3.8), policy/WDL/aux/opening heads, phase별 legal mask |
| FR-10 | 탐색: PUCT, Gumbel root + Sequential Halving, MCTS-solver, actor-aware backup, 배치 추론, 트리 재사용, budget profile 4종 |
| FR-11 | Opening 엔진: legal_opening_actions, evaluate_opening(swap/branch), generate_ten_fifths |
| FR-12 | Self-play: game-parallel 워커, 기보 기록(TRAINING §3), 모델 hash 기록, 원자적 파일 쓰기 |
| FR-13 | Replay: 디스크 기반, window/감쇠, prioritized 옵션, 균형 메타데이터, canonical dedupe, reanalysis |
| FR-14 | 학습: Stage A–G, 손실(TRAINING §4), AdamW/SGD, warmup+cosine, 체크포인트 resume |
| FR-15 | 평가(내부): paired match, Taraguchi role-paired protocol, SPRT/Elo CI, 리포트. 외부 엔진 평가는 FR-20..FR-27로 분리 |
| FR-16 | 고정 벤치마크 A–H 러너, 버전 간 diff |
| FR-17 | CLI: play, analyze, selfplay, train, evaluate, benchmark, rules-test (ARCHITECTURE §6) |
| FR-18 | 호환성 테스트 JSON 자동 수집·실행 |
| FR-19 | Python은 규칙을 재구현하지 않고 pybind11 바인딩만 사용 |
| FR-20 | **ExternalEngine 레지스트리**: `external_engines/<engine_id>.json` 메타(engine_id, name, version, executable sha256, source, license, supported_rules, protocol, search/time 설정, public reference rating = 참고 메타만). 바이너리는 저장소에 포함하지 않고 사용자가 제공·승인한 경로만 사용, 실행 전 sha256 대조 (EVALUATION §12) |
| FR-21 | **Piskvork/Gomocup 프로토콜** (v1 요구사항): (a) 외부 엔진 클라이언트 — 프로세스 기동, START/RECTSTART(미지원 명시)/BEGIN/TURN/BOARD/INFO/END/ABOUT; (b) DojoZero 엔진 측 — tournament manager가 쓸 수 있는 `pbrain-DojoZero.exe`. Renju 규칙 전달(`INFO rule`), 좌표 변환은 어댑터 경계 1곳에서만. 프로토콜 파서는 AI/search와 분리되고 규칙 판정은 canonical `GameState`만 사용 (ARCHITECTURE §3.12) |
| FR-22 | **TournamentRunner** (`dojozero tournament`): paired opening 배정, 프로세스 수명·timeout·crash·malformed 응답·illegal/금수 착수 감지, 기보 기록, 결과 집계. 외부 실패는 일반 패배와 별도 `termination_reason`으로 기록 |
| FR-23 | **버전 관리된 opening corpus** `benchmarks/openings/*.json`(opening_id, 수순, profile, source/version, expected legality, approximate balance, 생성 방법) + 세트 해시를 모든 match report에 기록, 학습 데이터 누출 검사(§15) 연동 |
| FR-24 | **Strength Database** `runs/evaluation/results.sqlite`: Engine / Match / Game / RatingSnapshot / Experiment 테이블. 모든 평가 결과를 누적 (EVALUATION §17) |
| FR-25 | **3-tier 평가 + adaptive opponent selection**: Tier A regression / Tier B development / Tier C reference, 점수 구간에 따라 ladder 상대 자동 선택(임계값 config) (EVALUATION §13–§14) |
| FR-26 | **자동 Strength Report** (Markdown + JSON): internal / external ladder / tactical / performance / opening / regression / diagnosis / recommendation 섹션. 측정하지 않은 항목은 `UNKNOWN` (EVALUATION §22) |
| FR-27 | **패배 분석 + match reanalysis 큐**: 패배 기보의 원인 분류(첫 큰 평가 급변, 놓친 즉시 전술·VCF/VCT, 금수 문제, opening 불리, timeout, value 오판, policy 후보 누락)와 고-budget 재분석. 실전 결과와 재분석 결과는 별도 저장 (EVALUATION §21) |

## 10. Non-functional Requirements
| ID | 요구사항 |
|---|---|
| NFR-01 정확성 | 규칙 테스트 100%, 퍼징 불일치 0이 학습 착수 조건 |
| NFR-02 성능 | EVALUATION §8.2 잠정 목표 대비 측정·보고; 핫패스에서 힙 할당·보드 복사 없음 |
| NFR-03 재현성 | 결정적 테스트 모드, 모든 run 메타데이터 기록 (§21) |
| NFR-04 이식성 | Windows/MSVC 1차, Linux 빌드 가능 유지 |
| NFR-05 안정성 | 불법 action·크래시·NaN·타임아웃 0 (스트레스 테스트로 확인) |
| NFR-06 관측성 | 구조화 로그, TensorBoard, 평가 리포트 |
| NFR-07 유지보수 | 모듈 의존 단방향, 규칙 단일 구현, 문서-코드 ID 상호참조 |
| NFR-08 compute 기록 | 모든 대국 평가에 wall time, CPU threads, GPU device·사용률, 메모리, nodes, MCTS sims, NN evals·batch·eval/s, nodes/s, 동시 대국 수를 기록. 모델 선택은 파라미터 수·validation loss가 아니라 Elo/compute로 (EVALUATION §18) |

## 11. AI Requirements
- 어떤 profile·phase에서도 illegal action을 선택하지 않는다(legal mask, 런타임 assert).
- 출력: action, W/D/L, visits, PV, tactical proof, nodes, elapsed (SearchResult).
- 결정적 모드에서 같은 입력·seed·budget이면 같은 출력.
- 증명된 전술 결과는 신경망 값보다 우선.
- swap/branch/제안/선택을 일반 착수와 다른 action 타입으로 다룬다(flat 225 강제 금지).

## 12. Tactical Search Requirements
- 지원: immediate five, mandatory defense, open four, four, three, 4-3, legal/illegal 3-3 & 4-4, VCF, VCT, TSS, df-pn.
- 금수 반영: 흑 공격수는 합법수만, 흑 방어 차단점이 금수면 차단 불가(금수 유도).
- 건전성: WIN은 PV 재생 검증, LOSS는 모든 합법(또는 증명된 건전 축소) 방어 실패 증명 시에만, 그 외 UNKNOWN. "VCF 없음" ≠ LOSS.
- TT + Zobrist, move ordering, proof/disproof numbers, 노드·시간 budget.

## 13. RL Requirements
- 보상: +1/0/−1 (actor PlayerId 관점) 외 shaping 금지.
- self-play target: MCTS visit 분포 또는 Gumbel improved policy, z, aux target(엔진 재계산).
- 커리큘럼 Stage A–G (TRAINING §2), 진단 순서(TRAINING §1.6).
- 학습 중 illegal action 확률 0 (마스킹 후 softmax), target에 illegal 질량이 있으면 데이터 오류로 중단.

## 14. Opening Requirements
- 학습 대상 결정: swap(1·2·3수 후), 2/3/4수, Choice 1/2, swap(직접 5수 전·후), 직접 5수, 10제안, 제안 선택.
- 10제안 방식 A(autoregressive)/B(top-k + Sequential Halving, 기본)/C(beam) 비교 (ADR-06). 설계 근거: 정확한 가치 하에서 최적 제안 = 비대칭 제약 하 상위 10 orbit.
- Opening Value DB(키: canonical opening state) 구조 제공.
- 흑/백 advantage 별도 측정.

## 15. Data Requirements
- 학습 데이터: solver 생성(Stage A), 자체 self-play, (선택) 라이선스 확인된 외부 기보.
- 모든 샘플: source, license, generator/model version, seed, profile hash.
- 벤치마크 세트는 학습 데이터와 해시로 분리·누출 검사.
- 평가용 opening corpus(`benchmarks/openings/*`, FR-23)도 같은 누출 검사 대상: 학습 데이터에 corpus 시작 국면(canonical hash)이 시작점으로 들어가면 실패. 외부 엔진 대국 기보는 OQ-05 결정 전까지 학습에 쓰지 않는다.
- 기보는 action 수순으로 저장, 국면은 엔진 재생으로 복원(ARCHITECTURE §5.1).

## 16. Model Architecture
- Baseline: AlphaZero ResNet (B6C64 → B10C128). Strong 후보: conv stem + residual + global/axial attention. ADR-07로 Elo/compute 비교 후 선택.
- 헤드: policy 226, WDL 3, aux 5×225(BCE), opening(swap 2, branch 2, propose 225, select 225).
- 세부: [TRAINING.md §7](TRAINING.md).

## 17. Search Architecture
PUCT 기본 + Gumbel root(Sequential Halving) 옵션, MCTS-solver, actor-aware backup(opening 연속 행동), 전술 solver 선택 호출, game-parallel 배치, NN 캐시, 트리 재사용, FAST/STANDARD/STRONG/ANALYSIS. 세부: [ARCHITECTURE.md §3.10](ARCHITECTURE.md).

## 18. Training Architecture
1 trainer + N self-play + M evaluation 워커, 파일시스템 큐, 버전·해시된 ONNX 모델 배포, 디스크 replay + SQLite 인덱스. 세부: [TRAINING.md](TRAINING.md).

## 19. Evaluation Methodology
paired games(pentanomial), Taraguchi role-paired protocol, SPRT 승격, Elo±95% CI, 고정 벤치마크 A–H, ablation A–H. 세부: [EVALUATION.md](EVALUATION.md). 추가 원칙(v0.2):
- **3-tier 평가**: Tier A regression(빠름, 변경마다) / Tier B development(채택 결정의 주 기준, DojoZero와 비슷한 강도의 ladder 상대 2–4개) / Tier C reference(milestone·release candidate, 최상위 엔진, 엄격히 고정된 조건).
- **External Strength Ladder**: 외부 anchor는 Rapfi 하나가 아니라 버전 고정된 사다리. 정보량이 큰 대국(점수 25–75%)에 compute를 집중하고, 0–2% 같은 최상위 상대 결과만으로 개발 판단을 하지 않는다.
- **두 개의 강도 보고를 섞지 않는다**: Standard Renju(외부 엔진 비교, balanced Renju 시작 국면) vs Taraguchi-10 full-game(DojoZero opening 능력). 외부 엔진이 Taraguchi action을 지원하지 않으면 Taraguchi rating에 포함하지 않는다.
- **세 종류의 rating 이름을 혼용하지 않는다**: `gomocup_reference_elo`(공개 수치, 메타데이터) / `dojo_external_elo`(동일 환경 직접 대국으로 추정한 외부 사다리 Elo) / `dojo_internal_elo`(self-play·checkpoint·내부 봇 Elo). 공개 Elo를 우리 척도로 동일시하지 않는다. Taraguchi-10 full-game 강도는 별도 `dojo_taraguchi_elo`(EVALUATION §16).
- **Strength-driven decision**: 기능은 baseline vs candidate paired A/B로 `ACCEPT / REJECT / INCONCLUSIVE` 판정. INCONCLUSIVE는 성공이 아니다.

## 20. Performance Requirements
측정 환경과 잠정 목표는 [EVALUATION.md §8](EVALUATION.md). 목표는 Phase 2–5 실측 후 갱신하며 갱신 이력을 남긴다. NN 추론은 반드시 배치 지원.

## 21. Reproducibility
모든 run: git commit(+dirty diff), 병합 config, seed, model hash, 데이터 생성 범위, optimizer state, PyTorch/드라이버 버전, 하드웨어, 메트릭. 체크포인트로 완전 resume, resume 일치 테스트. 결정적 테스트 모드.

## 22. Interfaces
C++ API(ARCHITECTURE §3.6, §3.7, §3.9, §3.10): `GameState{apply, undo, legal_actions, result, forbidden}`, `AIEngine::search(state, limits) → SearchResult`, `OpeningEngine{legal_opening_actions, evaluate_opening, generate_ten_fifths}`, `Evaluator`. Python 바인딩 `dojozero_core`.

## 23. CLI
`dojozero play | analyze | selfplay | train | evaluate | benchmark | rules-test | tournament | report`. analyze 출력: best move, top-N(P, Q, visits, W/D/L), tactical status, PV, 금수점 + 이유. 별도 실행 파일 `pbrain-DojoZero.exe`(Piskvork 프로토콜 엔진, FR-21b).

## 24. File Formats
기보 chunk `.dzg`(action 수순 + 탐색 레코드), 체크포인트 `.pt`(format_version 포함), 추론 모델 ONNX + manifest(sha256), 벤치마크/호환성 JSON(schema_version), config(JSON: 규칙·탐색 / YAML: 학습·모델). 세부: ARCHITECTURE §5.

## 25. Configuration
`configs/{rules,models,training,search}`. 실행 시 병합 config를 run 디렉터리에 덤프. RuleProfile·FeatureConfig·SearchConfig는 정규화 해시를 산출해 기보·체크포인트·모델에 기록하고, 로드 시 불일치를 거부.

## 26. Test Strategy
- C++ GoogleTest 단위 테스트(카테고리: RULES §12 전부, 수백 개 이상 목표), JSON 데이터 구동 테스트, 1차원 전수 열거, oracle 차등 퍼징(≥10⁷), property(D4 가환성, apply/undo 왕복), opening 경로 열거, Python pytest.
- 순환 검증 방지: 금수·전술 테스트에 독립 출처(수작업 검토, 문헌, 교차 엔진, 앱 증거) 포함.
- 각 Phase 종료 = 해당 테스트 전부 통과 + 벤치마크 기록 + 문서 갱신 + 커밋.

## 27. Security / Integrity
- 앱·온라인 서비스 자동화 없음.
- 체크포인트 로드 시 `torch.load(weights_only=True)`, ONNX/체크포인트 sha256 검증. 신뢰할 수 없는 pickle 로드 금지.
- 외부 엔진은 별도 프로세스, 표준 입출력 프로토콜만, 사용자 승인된 바이너리만. 실행 전 메타데이터의 sha256과 대조해 다르면 실행 거부(버전 drift 방지). 외부 바이너리는 저장소·릴리스에 자동 포함하지 않는다.
- 외부 엔진 출력은 신뢰하지 않는 입력: 모든 착수는 canonical `GameState`로 합법성 검사, 응답 길이·형식 검증, 프로세스별 timeout.
- 입력(국면 JSON, CLI) 검증, malformed state 거부. 기보 파일 손상 감지(길이·magic·버전).

## 28. Legal / Data License Considerations
- 공개 기보 DB는 라이선스 확인 전 사용 금지(D-01).
- GPL 엔진(Rapfi 등)은 코드·가중치를 포함하지 않고 별도 프로세스로만 사용(D-03). 타 엔진 생성 기보를 학습에 쓸지는 OQ-05 — 결정 전까지 외부 대국 기보는 **평가 전용**(replay 투입 금지).
- 외부 엔진별 라이선스·배포 조건은 각 `external_engines/*.json`의 `license`에 기록하며, 확인 전에는 `UNVERIFIED`(D-05).
- 오목도장 앱: 이용약관 준수 — 자동화·스크래핑 없음, 사용자가 수동으로 얻은 소량의 규칙 검증 증거만 사용(D-04).
- 문헌 문제 국면은 좌표만 사용, 출처 기록.
- 프로젝트 자체 라이선스는 OQ-10.

## 29. Observability
구조화 JSON 로그, TensorBoard 메트릭(TRAINING §9), 평가 리포트(Markdown+JSON), 벤치마크 결과 버전별 저장, self-play 통계(게임 길이, 결정 분포, 흑/백 승률), 장시간 프로세스의 명령·플래그·PID 기록. **장기 strength history**: 모든 내부·외부 대국과 rating snapshot을 Strength DB(FR-24)에 누적, PROGRESS.md의 Current Strength Snapshot(측정 안 한 항목은 UNKNOWN), 외부 대국의 실패 사유(crash/timeout/malformed/illegal) 통계.

## 30. Failure Modes
| 실패 | 감지 | 대응 |
|---|---|---|
| 금수 판정 오류 | 단위·퍼징·교차 엔진·앱 증거 | 학습 금지 게이트, 회귀 테스트 추가 |
| 거짓금수 재귀 누락/무한 재귀 | 재귀 깊이 통계, 단계별 테스트 | 종료성 증명(돌 추가), 깊이 분포 기록 |
| opening 상태·색 배정 혼동 | 상태 머신 전 경로 테스트, PlayerId 기준 결과 기록 | 타입 분리(PlayerId/StoneColor/Role) |
| actor 연속 행동에서 부호 반전 오류 | 탐색 단위 테스트(고정 가치 트리) | 절대값 backup |
| solver가 WIN/LOSS 오판 | PV 재생 검증, 독립 검증 세트 | UNKNOWN 기본, 건전 축소만 |
| illegal action 학습/선택 | 런타임 assert, 기보 재생 검증 | logits 마스킹, target 검사 |
| 증강 변환 불일치 | 가환성 property 테스트 | C++ 단일 변환 경로 |
| NaN/발산 | non-finite 검사 | 즉시 중단 + 배치 덤프 |
| 모델·feature·profile 불일치 | 해시 검증 | 로드 거부 |
| self-play 다양성 붕괴(항상 같은 opening/swap) | 결정 분포·opening 엔트로피 로깅 | 탐험 노이즈, league |
| echo chamber (내부 Elo만 상승) | 외부 anchor 정기 대국 | 학습 방향 재검토 |
| 중복 워커 실행 | PID 파일, 프로세스 조회 | 재시작 전 정리 |
| 파일 손상/부분 쓰기 | magic/길이 검사 | tmp→rename 원자적 쓰기 |
| 외부 프로토콜 실패(crash, timeout, malformed 응답, illegal·금수 착수) | TournamentRunner 감지, canonical GameState 검증 | 일반 패배와 다른 `termination_reason`으로 기록, 실패율 별도 보고, 실패율이 임계 초과인 상대·설정은 ladder에서 제외 |
| 외부 바이너리 version drift | 실행 전 sha256 대조 | 불일치 시 실행 거부, 새 버전은 새 engine_id로 등록 |
| benchmark opening drift | opening set hash를 모든 report·DB에 기록 | 해시가 다른 결과끼리 비교 금지 |
| rating 척도 혼동 | 보고서 필드 이름 강제(`gomocup_reference_elo` / `dojo_external_elo` / `dojo_internal_elo`) | 공개 Elo는 메타데이터로만 |
| 특정 상대 과적합 | ladder 전체·보지 않은 상대(held-out)의 점수 추이 | 상대 다양화, Tier C로 교차 확인 |
| 정보 없는 매치(점수 <5% 또는 >95%만 반복) | adaptive selection 로그 | 점수 25–75% 상대에 compute 집중 |

## 31. Risks
| ID | 리스크 | 영향 | 가능성 |
|---|---|---|---|
| R-01 | 개발 머신이 CUDA 없는 내장 GPU → self-play RL compute 부족 | "최상급" 기력 미달 | 높음 |
| R-02 | 오목도장 실제 규칙이 가정과 다름 | 호환성 목표 미달 | 중간 |
| R-03 | 재귀 금수 판정 성능이 탐색 병목 | nodes/s 저하 | 중간 |
| R-04 | PyTorch XPU/ONNX 런타임 Windows+Intel GPU 호환성 문제 | 학습/추론 지연 | 중간 |
| R-05 | 전술 solver 비건전성(잘못된 증명) | 탐색·학습 오염 | 중간 |
| R-06 | opening 10제안 action space가 커서 학습 신호 희소 | opening 기력 저하 | 중간 |
| R-07 | 자체 테스트가 자체 정의를 순환 검증 | 숨은 규칙 오류 | 중간 |
| R-08 | 범위 과대(Phase 0–8 전체) | 일정 지연 | 높음 |
| R-09 | 외부 최상위 엔진과 compute 조건이 불공정(CPU 멀티스레드 엔진 vs GPU 배치 엔진, 동시 대국 수, 시간 관리 차이) → 잘못된 강도 결론 | 개발 방향 오판 | 높음 |
| R-10 | 외부 엔진의 Renju 규칙 구현 차이(금수·장목 처리) → 대국 결과 오염 | ladder 신뢰도 저하 | 중간 |

## 32. Mitigations
| 리스크 | 완화 |
|---|---|
| R-01 | 작은 모델로 시작, 전술 solver 비중 확대, Gumbel(저 sims) + playout cap randomization, 외부 anchor로 조기 측정, Milestone 3에서 실측 throughput 기반 compute 계획 재수립, 필요 시 외부 GPU(OQ-03) 또는 CPU 친화 대안 아키텍처 검토(OQ-06) |
| R-02 | 모든 불확실 동작을 profile 파라미터화, 앱 증거 JSON 회귀 테스트 |
| R-03 | 라인 코드 + 생성 테이블, hash 캐시, 벤치마크 기반 ADR-01 |
| R-04 | Evaluator 추상화로 런타임 교체 가능, CPU 경로 항상 유지 |
| R-05 | PV 재생 검증, 건전 축소 규칙 문서화·테스트, 독립 검증 세트 |
| R-06 | top-k 방식 기본(이론적 근거), Opening Value DB, opening 전용 커리큘럼 |
| R-07 | 독립 출처 테스트 의무화, 교차 엔진 차등 비교 |
| R-08 | Phase 게이트, 각 Milestone이 단독으로 유용한 산출물(규칙 엔진 → 분석기 → 엔진) |
| R-09 | compute 기록 의무(NFR-08), fixed-time과 fixed-visits 결과를 분리 보고, Tier C는 대국 1개씩 순차·고정 하드웨어, 여러 budget의 scaling curve로 결론 |
| R-10 | ladder 등록 전 규칙 적합성 사전 검사(금수 국면 세트, 흑 금수 착수 0건 확인), 금수 착수는 `FORBIDDEN_MOVE` 실패로 기록 |

## 33. Milestones
| Milestone | Phase | 내용 |
|---|---|---|
| M0 | 0–1 | 저장소 조사, 문서 6종 (이 문서) |
| M1 | 2 | 규칙 엔진 + opening 상태 머신 + 대칭 + 테스트 + 성능 벤치마크 |
| M2 | 3 | 전술 solver + 전술 벤치마크 |
| M3 | 4–5 | NN·데이터 파이프라인·사전학습, MCTS/Gumbel, 기준 봇 격파 |
| M4 | 6 | self-play RL 파이프라인, replay, 승격 |
| M5 | 7 | Taraguchi-10 opening RL, league |
| M6 | 8 | 하이브리드 vs 순수 비교, 최적화, 패키징 |
| Final | — | 재현 가능한 릴리스 |

**Evaluation infrastructure 트랙(v0.2 추가)**: 아래 E1–E4는 Phase 순서에 끼워 넣어 순차 구현한다(한 번에 전부 구현하지 않음). M2–M4는 이 트랙 이전에 완료되었으므로 해당 strength gate는 **소급(backfill)** 항목으로 E1에서 수행한다.

| 단계 | 시점 | 내용 | 선행 조건 |
|---|---|---|---|
| E0 | Phase 7 (현재) | 문서·아키텍처 준비만. Taraguchi-10 full-game 보고를 Standard Renju 보고와 분리 기록(이미 prefix-opening 매치로 측정 중) | — |
| E1 | Phase 8 착수 직후 (Ablation 전) | `cpp/protocol/`(Gomocup 클라이언트·좌표 변환·프로세스), ExternalEngine 메타 + sha256 검증, 시간 기반 SearchLimits, TournamentRunner, failure reason, opening corpus v1, `results.sqlite`, report v0, **첫 external calibration(M3 소급 gate)** | OQ-04 승인, OQ-11 버전 고정, OQ-13 시간 규정 |
| E2 | Phase 8 중 | Tier A/B 자동화, adaptive opponent selection, compute 기록 전 항목, FAST/STANDARD/STRONG scaling curve, M4 소급(세대별 internal vs external 상관) | E1 |
| E3 | Phase 8 말 | `pbrain-DojoZero.exe`(ADR-08), 패배 원인 분류 + reanalysis 큐, Diagnosis 자동 연결 | E2, ADR-02 |
| E4 | Final | Tier C reference evaluation, 최종 Strength Report | E3 |

## 34. Acceptance Criteria
- **M1**: 규칙 엔진 완성; exact-five / overline / 3-3 / 4-4 정확; 재귀 거짓금수 테스트 완비(재귀 1·2·3단 포함); Taraguchi-10 상태 머신 완성; 규칙 테스트 100%; 퍼징 ≥10⁷ 불일치 0; 1차원 전수 열거 통과; A-R3/A-R4 검증 결과 문서화.
- **M2**: VCF solver 동작; 고정 전술 벤치마크 통과(기준치는 세트 확정 시 설정), WIN 오답 0.
- **M3**: AlphaZero baseline이 합법적이고 유능한 대국; random·heuristic 봇 상대 SPRT 통과.
- **M4**: self-play 파이프라인 안정(24시간 연속 실행 실패 0); 후보 승격 동작; 세대별 Elo 상승(외부 anchor로 교차 확인).
- **M5**: Taraguchi-10 opening 전체 학습 가능; swap/제안/선택 action 지원; opening regret 감소.
- **M6**: 하이브리드 전술+신경망 탐색이 순수 baseline을 paired 통계로 유의하게 이김.
- **Final**: 재현 가능한 릴리스, 독립 엔진, 분석기, 학습된 체크포인트, 벤치마크 리포트, 완전한 문서.

**Strength gates (v0.2 추가, 위 기준에 더해짐)** — 절대 외부 Elo 목표는 정하지 않는다(OQ-03). 각 단계가 실제 대국 능력으로 무엇을 보여야 하는지만 정한다.
| Milestone | Strength gate | 현재 상태 |
|---|---|---|
| M2 | 전술 solver 기능 완성, 고정 전술 benchmark; external match infra skeleton 시작 | 전술 부분 충족(PROGRESS Phase 3), infra는 **미충족 → E1에서 소급** |
| M3 | 첫 neural/search engine; Gomocup 어댑터 + TournamentRunner 동작; external ladder 첫 calibration을 baseline으로 저장 | 엔진 충족, 어댑터·calibration **미충족 → E1 소급** |
| M4 | 의미 있는 세대마다 internal promotion; 정기 external ladder 비교; internal Elo와 external strength가 함께 오르는지 확인 | promotion 충족(내부 휴리스틱 anchor만), external 비교 **미충족 → E2 소급** |
| M5 | Taraguchi-10 full-game 별도 rating/report; opening 지표와 Standard Renju 강도를 둘 다 추적 | 부분 충족(Phase 7): prefix 오프닝 +123 [+22, +224] vs 무작위 오프닝 +7 [−88, +102]로 분리 측정. Strength DB·외부 비교는 E1 이후 |
| M6 | hybrid가 baseline보다 유의하게 강함; 주요 external anchor 대비 최종 격차 측정; top-engine comparison report | 미착수 |
| Final | report에 external ladder 위치, top anchor W/D/L, Elo delta + CI, compute 설정, tactical score, Taraguchi score, scaling curve, ablation 포함 | 미착수 |

## 35. Deliverables
C++ 라이브러리·`dojozero_engine.exe`, Python 패키지 `dojozero`(+`dojozero_core` 확장), 문서 6종 + 갱신 이력, 테스트 스위트, `benchmarks/` 고정 세트, 학습된 체크포인트 + ONNX + manifest, 평가·벤치마크 리포트, 재현 스크립트.

## 36. Open Questions
| ID | 질문 | 막는 Phase |
|---|---|---|
| OQ-01 | 오목도장 유단자 규칙이 실제로 RIF Renju + Taraguchi-10인가? 앱에서 C-01..C-20 증거를 수집할 수 있는가? | 없음(기본값 진행) / 호환성 확정은 M1 이후 |
| OQ-02 | 4수 후 Choice 1/2 결정자와 직접 5수 전 swap 결정자(A-T1, A-T2) — 앱 UI에서 누구에게 선택지가 뜨는가? | M5 정책 구조 (상태 머신은 profile로 양쪽 지원) |
| OQ-03 | 목표 기력과 compute 예산: "최상급"을 무엇으로 측정할까(예: Rapfi 대비 동일 시간 승률)? 외부(클라우드) GPU 사용 가능한가? | M3 이후 계획 |
| OQ-04 | 외부 기준 엔진(Rapfi, KataGomo, AlphaGomoku, JAX, Embryo, Gomocup 구버전 등) 바이너리를 어떻게 확보하는가(사용자 제공 vs 사용자 승인 후 다운로드)? 엔진별 라이선스 확인 주체 | E1 |
| OQ-05 | 타 엔진이 생성한 기보를 학습 데이터로 써도 되는가(Stage C)? | M3 |
| OQ-06 | compute 제약이 심할 경우 AlphaZero 대신(또는 병행) CPU 친화 alpha-beta + 경량 NN(NNUE류) 아키텍처를 검토할 것인가? 권장: 명세대로 AlphaZero로 진행하되 M3 실측 후 재평가 | M3 |
| OQ-07 | ~~PyTorch XPU(Intel GPU) 빌드 설치 허용?~~ **해결**: 사용자 승인으로 torch 2.14.1+xpu 설치(Phase 4). XPU 백엔드 우회 3건은 PROGRESS 참조 | — |
| OQ-08 | `pass_allowed` 기본값(C-06) | 없음 |
| OQ-09 | 10제안 대칭 모드 기본값 E1 vs E3 (C-10) | 없음(옵션) |
| OQ-10 | 프로젝트 라이선스 | Final |
| OQ-11 | reference engine **버전 고정** 정책: 각 엔진의 어느 버전(공개 날짜·sha256)을 ladder 기준으로 고정하고 언제 갱신하는가? 갱신 시 이전 버전도 유지하는가? | E1 |
| OQ-12 | 평가 하드웨어: 개발 노트북(E-01..E-08)에서 학습·self-play와 동시에 평가해도 되는가, 평가 전용 시간대를 둘 것인가? Tier C를 다른 머신에서 할 것인가? | E1 |
| OQ-13 | tournament 시간 규정: 수당 고정 시간(예: 50ms/100ms/500ms/1s/5s) vs 대국당 시간(Gomocup식 timeout_match), CPU 스레드 수, 동시 대국 수 | E1 |
| OQ-14 | Standard Renju opening corpus 출처: 자체 생성(다중 엔진 평가로 균형 판정) vs 공개 세트(RIF 26 + 대표 5수, 라이선스 확인) | E1 |
| OQ-15 | `pbrain-DojoZero.exe`의 추론 호스트: C++ ONNX/OpenVINO 추론(ADR-02) vs Python(PyTorch XPU) 프로세스를 감싸는 런처 (ADR-08) | E3 |

## 37. Research Questions
| 질문 | 대응 실험 |
|---|---|
| Transformer가 ResNet보다 Elo/compute가 높은가 | Ablation A |
| VCF/VCT가 MCTS에 어느 깊이부터 유효한가, 호출 임계값 | Ablation C (+ 임계값 sweep) |
| opening·midgame 네트워크 공유 vs 분리 | Stage F 변형 비교 |
| 10제안: autoregressive vs top-k | ADR-06, EVALUATION §7 |
| aux tactical heads의 sample efficiency | Ablation D |
| reanalysis의 동일 compute Elo 개선 | Ablation H |
| league의 순환 전략 감소 효과 | Ablation G + 비이행성 지표 |
| RQ-01 fixed visits와 fixed time에서 모델 순위가 달라지는가 | EVALUATION §23 |
| RQ-02 NN 규모 증가가 Elo/compute를 실제로 개선하는가 | EVALUATION §23 (Ablation A 확장) |
| RQ-03 tactical solver가 강한 외부 엔진 상대에서 어떤 유형의 패배를 줄이는가 | EVALUATION §21, §23 (Ablation C 확장) |
| RQ-04 solver 호출 임계값의 최적점이 상대 강도에 따라 달라지는가 | EVALUATION §23 |
| RQ-05 search budget이 커질수록 DojoZero의 상대 Elo는 어떻게 변하는가 | EVALUATION §18 scaling curve |
| RQ-06 self-play internal Elo와 external ladder Elo의 상관 | EVALUATION §23 |
| RQ-07 학습 세대가 진행되며 특정 외부 엔진에 과적합하는가 | EVALUATION §23 (held-out 상대) |
| RQ-08 Taraguchi-10 opening 개선이 balanced midgame Renju 강도를 희생하는가 | EVALUATION §15, §23 |

## 38. Future Work
GUI/웹 분석 보드, Gomocup 대회 실제 출전(대회 규정 대응; 프로토콜 엔진 자체는 v1 FR-21로 이동), gRPC/MQ 분산, NPU(OpenVINO) 추론, TensorRT(외부 GPU 시), 다른 opening rule(Soosõrv-8, Swap2, Super Taraguchi-10), 다른 보드 크기, Opening DB 공개 포맷, 앙상블 불확실성.

---
### 변경 이력
| 날짜 | 버전 | 변경 |
|---|---|---|
| 2026-10-18 | 0.1 | 초안 |
| 2026-10-19 | 0.2 | 지속적 Strength Evaluation System: G7 확장(External Strength Ladder), G9, FR-20..FR-27, NFR-08, §19 원칙, 실패 모드 6종, R-09/R-10, E0–E4 트랙과 strength gate, OQ-11..OQ-15, RQ-01..RQ-08. Gomocup 프로토콜을 Future Work에서 v1으로 이동 |
