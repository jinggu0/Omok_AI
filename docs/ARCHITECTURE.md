# DojoZero — ARCHITECTURE.md

Status: **Draft v0.1** · 2026-10-18
규칙 의미는 [RULES.md](RULES.md), 학습은 [TRAINING.md](TRAINING.md), 평가는 [EVALUATION.md](EVALUATION.md)가 규범이다.
이 문서는 모듈 경계, 인터페이스, 데이터 흐름, 파일 포맷, 아직 벤치마크로 결정해야 할 설계 선택(ADR)을 정의한다.

---

## 1. 전체 구조

```
                ┌──────────────────────── C++ core (dojozero_core 라이브러리) ────────────────────────┐
                │                                                                                     │
  RuleProfile ──▶ board/  ──▶ rules/  ──▶ opening/ ──▶ game/ (GameState, Action)                      │
                │  coord, Board,   ForbiddenChecker      OpeningSM,       │                              │
                │  Zobrist(8x)     ReferenceRules(oracle) Symmetry,       ▼                              │
                │                  PatternTables(gen)     ProposalValid.  tactics/ (Threat, VCF, VCT,    │
                │                                                          TSS, df-pn) ──▶ search/       │
                │                                                          features/ ─────▶ (PUCT, Gumbel,│
                │                                                          nn/Evaluator ──▶  MCTS-solver) │
                │                                                                             │         │
                │                                                          selfplay/ ◀────────┘         │
                └──────────────┬──────────────────────────────────────────────┬──────────────────────────┘
                               │ pybind11 (bindings/)                         │ dojozero_engine.exe (cli/)
                               ▼                                              ▼
   Python: model/ trainer/ replay/ league/ evaluation/ tools/        selfplay · analyze · benchmark · rules-test · play
                               │                                              ▲
                               └──── ONNX export (versioned, hashed) ─────────┘
```

### 1.1 의존 규칙 (MUST)
- 의존 방향은 위 그림의 왼쪽 → 오른쪽 한 방향. `rules/`는 `search/`, `nn/`을 알지 못한다.
- **게임 규칙은 C++ 한 곳에만** 존재한다. Python은 `bindings/`를 통해서만 규칙·feature·대칭·기보 재생을 사용한다. Python에 규칙 재구현 금지(예외: 없음. 테스트 oracle도 C++ `ReferenceRules`).
- Search는 NN을 `Evaluator` 인터페이스로만 안다. NN은 search를 모른다.
- Training(Python)과 Inference(C++ `Evaluator` 구현체)는 모델 파일(ONNX + manifest)로만 연결된다.
- Feature 인코딩은 C++ `features/` 단일 구현을 학습·추론 모두 사용한다.

---

## 2. 디렉터리 구조

```
/
├─ CMakeLists.txt, CMakePresets.json
├─ pyproject.toml
├─ configs/
│  ├─ rules/      renju_rif.json, taraguchi10.json, omokdojang_dan.json
│  ├─ models/     resnet_b6c64.yaml, resnet_b10c128.yaml, hybrid_attn_*.yaml
│  ├─ training/   stage_a_tactical.yaml ... stage_g_league.yaml
│  └─ search/     fast.json, standard.json, strong.json, analysis.json
├─ cpp/
│  ├─ board/      coord.h, board.{h,cpp}, zobrist.{h,cpp}, line_code.{h,cpp}
│  ├─ rules/      rule_profile.{h,cpp}, forbidden.{h,cpp}, reference_rules.{h,cpp},
│  │              pattern_table_gen.{h,cpp}, patterns.{h,cpp}
│  ├─ opening/    opening_phase.h, opening_sm.{h,cpp}, symmetry.{h,cpp}, proposals.{h,cpp}
│  ├─ game/       action.h, game_state.{h,cpp}, record.{h,cpp}
│  ├─ tactics/    threat.{h,cpp}, vcf.{h,cpp}, vct.{h,cpp}, tss.{h,cpp}, dfpn.{h,cpp}, tt.h
│  ├─ features/   features.{h,cpp}
│  ├─ nn/         evaluator.h, uniform_eval.cpp, heuristic_eval.cpp, onnx_eval.cpp
│  ├─ search/     node.h, puct.{h,cpp}, gumbel.{h,cpp}, search.{h,cpp}, batcher.{h,cpp}
│  ├─ selfplay/   worker.{h,cpp}, writer.{h,cpp}
│  ├─ bindings/   module.cpp
│  └─ cli/        main.cpp, commands/*.cpp
├─ python/dojozero/
│  ├─ model/ trainer/ replay/ league/ evaluation/ tools/ cli.py
├─ tests/
│  ├─ rules/ forbidden/ opening/ tactics/ search/ features/ python/
│  └─ compatibility/omokdojang/*.json
├─ benchmarks/   rules/ forbidden/ tactical/ vcf/ vct/ opening/ endgame/ adversarial/  (JSON)
├─ scripts/  checkpoints/  runs/  docs/
```
`checkpoints/`, `runs/`는 git 추적 제외(.gitignore). 대용량 산출물은 해시로 참조.

---

## 3. C++ Core

### 3.1 공통
- C++20, MSVC 14.43 (ASSUMPTIONS E-06). Linux/Clang 빌드 호환 유지(코드에 MSVC 전용 확장 금지).
- 예외는 API 경계(입력 파싱, 파일 IO)에서만. 핫패스(apply/undo/forbidden/search)는 예외·힙 할당 없음.
- 좌표: `using Point = uint8_t; constexpr Point kNoPoint = 255;` `(row, col)` 순서 고정 (RULES §1.2).
- 결정적 모드: `DeterministicMode` 플래그 — 모든 RNG는 명시적 seed의 `std::mt19937_64`/PCG, 스레드 1개, 해시 테이블 순회 순서 무관한 알고리즘.

### 3.2 Board — 표현 후보와 결정 절차 (ADR-01, 벤치마크 대기)
| 후보 | 내용 | 장점 | 단점 |
|---|---|---|---|
| A. compact array | `uint8_t cell[225]` + 경계 검사 | 단순, 정확성 검증 쉬움 | 패턴 판정 시 반복 스캔 |
| B. bitboard | 색별 225bit (4×uint64) | five 검출 빠름 | 패턴 window 추출 복잡 |
| C. hybrid | A + 4방향 **line code** (각 직선의 셀을 2bit로 패킹한 정수, 72개 직선) 증분 갱신 | p 주변 9칸 window를 shift 한 번으로 추출 → 생성된 테이블 조회 | 증분 갱신 비용 |

- Phase 2에서 **A로 먼저 구현**(oracle 겸 기준선) → C 구현 → 동일 테스트·차등 퍼징 통과 후 벤치마크(apply/undo/s, forbidden/s)로 기본값 결정. B는 C가 목표치 미달일 때만.
- `apply/undo`는 증분(돌 1개 + line code 4개 + Zobrist 8개 갱신). 보드 전체 복사 금지(MCTS 경로).

### 3.3 Zobrist / canonical hash
- 키: `stone[color][point]`, `phase[...]`, `offered[point]`, `side_to_move`. 64-bit, 고정 seed로 생성(재현성).
- **8개 변환별 해시를 동시 증분 유지**: `h[t] ^= key[color][T_t(p)]`. `canonical_hash = min_t h[t]` (+ 그 t). 착수당 XOR 8회로 D4 중복제거·NN 캐시·opening DB 키 제공.
- 해시 충돌 대비: TT/캐시 엔트리에 보조 32-bit 검증 키 저장.

### 3.4 Rules
- `ReferenceRules` (oracle): RULES §3–§4 정의를 **문장 그대로** 보드 스캔으로 구현. 테이블·캐시·최적화 없음. 테스트·퍼징 전용, 릴리스 경로에서 호출 금지.
- `PatternTables`: 1차원 window(길이 9~11, 셀 ∈ {E, own, opp, WALL})의 모든 경우를 `ReferenceRules`의 1차원 판정으로 **생성**. 출력: five/overline/four 개수, completion offset, apparent-three 여부와 extension offset. 생성 결과의 SHA-256을 빌드에 기록하고 테스트로 고정.
- `ForbiddenChecker` (fast): 테이블로 후보를 찾고 extension point 합법성은 **실제 보드에서 재귀**(RULES §4). `ForbiddenInfo` 생성 경로(느린 설명 모드)와 판정 경로(빠른 bool/verdict) 분리.
- Forbidden cache: `(board hash, point) → verdict`. 증분 무효화는 최적화일 뿐이며 차등 테스트로 동치가 증명된 경우에만 활성화.
- `RuleProfile` (RULES §7, ASSUMPTIONS):
```cpp
struct RuleProfile {
  std::string id;                     // "RENJU_RIF", "TARAGUCHI10", "OMOKDOJANG_DAN"
  int board_size = 15;                // 15만 지원 (컴파일 상수와 일치 검사)
  WinCondition black_win = EXACT_FIVE, white_win = FIVE_OR_MORE;
  ForbiddenSet black_forbidden = {OVERLINE, DOUBLE_FOUR, DOUBLE_THREE}, white_forbidden = {};
  OpeningRule opening_rule = TARAGUCHI_10;   // NONE | FIXED_POSITION | TARAGUCHI_10
  bool pass_allowed = false;
  IllegalMoveBehavior illegal_move_behavior = REJECT;  // REJECT | LOSS
  bool exact_five_precedence = true;          // RULES §6 (false는 테스트용 비교 모드)
  bool five_beats_overline_other_dir = true;  // A-R1 / C-02
  Taraguchi10Params taraguchi;                 // region sizes {1,3,5,7,9}, n_proposals=10,
                                               // choice_actor, swap_before_direct5_actor,
                                               // proposal_symmetry_mode, proposal_region
};
```
  Profile은 JSON(`configs/rules/*.json`)으로 로드, 정규화 후 해시를 모든 기보·체크포인트에 기록.

### 3.5 Opening
- `OpeningPhase` enum = RULES §8.3 표 그대로. 전이 함수는 `(phase, action) → phase` **표 기반** 단일 함수.
- `actor(phase, assignment) → PlayerId`, `role(phase) → Role`.
- `symmetry.{h,cpp}`: D4 변환(RULES §1.5), `stabilizer(position)`, `orbit(point, mode)`, `are_equivalent(q1,q2, mode)`.
- `proposals.{h,cpp}`: `validate_full(Q)`, `validate_partial(Q_k)`, `legal_next_proposals(Q_k)`.

### 3.6 Action, GameState
```cpp
enum class ActionType : uint8_t { PLACE, PASS, SWAP, KEEP_COLOR, CHOOSE_DIRECT_FIFTH,
                                  CHOOSE_TEN_PROPOSALS, PROPOSE_FIFTH, SELECT_PROPOSED_FIFTH };
struct Action { ActionType type; Point point = kNoPoint; };   // point: PLACE/PROPOSE/SELECT만

enum class TerminalResult : uint8_t { ONGOING, BLACK_WIN, WHITE_WIN, DRAW };
struct Outcome { TerminalResult result; PlayerId winner_or_none; WinReason reason; };

class GameState {
 public:
  explicit GameState(const RuleProfile&, SideAssignment initial);
  bool apply(Action);                 // false = 불법(REJECT). 상태 불변 보장
  void undo();                        // 마지막 apply 정확히 복원 (phase, assignment, offered 포함)
  void legal_actions(ActionList&) const;
  bool is_legal(Action) const;
  ForbiddenInfo forbidden(Point) const;   // 설명 포함 (느린 경로)
  MoveVerdict black_verdict(Point) const; // 빠른 경로
  Outcome result() const;
  OpeningPhase phase() const; PlayerId actor() const; StoneColor side_to_move() const;
  SideAssignment assignment() const;
  std::span<const Point> offered_fifths() const; Point chosen_fifth() const;
  Point last_move() const; uint64_t hash() const; CanonicalHash canonical_hash() const;
  GameState transformed(int t) const;     // D4 변환 (테스트·증강용, 핫패스 아님)
  static Expected<GameState> load(const PositionSpec&, const RuleProfile&); // malformed 거부
 private:
  Board board_; OpeningState opening_; SideAssignment assign_;
  std::vector<UndoRecord> history_;   // 미리 reserve(400); UndoRecord는 POD
  // forbidden cache, threat cache (hash-keyed, 선택적)
};
```
- 착수 수(ply)와 action 수는 다르다. history는 action 단위, 기보 표기는 ply 단위로 둘 다 제공.

### 3.7 Tactics (Phase 3)
```cpp
enum class ProofResult : uint8_t { WIN, LOSS, UNKNOWN };   // 탐색 시작 국면의 행위자 관점
struct ProofOutput { ProofResult result; std::vector<Action> pv; uint64_t nodes;
                     uint32_t elapsed_ms; StopReason stop; };   // stop: PROVEN, BUDGET, TIME, EXHAUSTED_NO_PROOF
```
- `ThreatDetector`: 점·색별 위협 등급(FIVE, STRAIGHT_FOUR, FOUR, THREE(흑: 진짜만), BROKEN_THREE …), 흑 금수 반영.
- `VCFSolver`: 공격자는 합법 four만, 방어자는 completion 차단(또는 자신의 five). 방어자가 흑이고 차단점이 금수면 차단 불가 → 공격 승(**금수 유도**). 방어자의 차단이 four를 만들면 공격자 응수 강제. TT + 반복 심화 + 노드/시간 budget.
- `VCTSolver` / `ThreatSpaceSearch`: four + three 위협. three에 대한 방어 집합 = three를 무력화하는 모든 점 ∪ 방어자의 모든 four(역공) ∪ 방어자의 VCF 존재 시 반박.
- `ProofNumberSolver`: df-pn(TT, threshold 제어). OR 노드 = 공격자 위협수, AND 노드 = 방어 집합.
- **건전성 규칙 (MUST)**:
  1. `WIN`은 PV가 실제 규칙 엔진으로 재생 검증 가능해야 반환(디버그 빌드에서 항상 재검증).
  2. 방어 집합 축소(threat-restricted)는 **증명된 건전한 축소**만 허용(four에 대한 응수, three에 대한 응수 집합). 근거를 코드 주석+테스트로.
  3. `LOSS`는 행위자의 **모든** 합법 수(또는 건전 축소 집합)가 상대 WIN으로 증명된 경우에만.
  4. budget 소진·탐색 범위 밖 → `UNKNOWN`. "VCF 없음"은 `LOSS`가 아니다.
- 의사결정 우선순위 (`AIEngine`): 즉시 5 → 상대 즉시 5 방어(강제) → 증명된 전술 WIN/LOSS → neural MCTS.

### 3.8 Features (NN 입력)
- `encode(const GameState&, const FeatureConfig&, float* out)` → `C×15×15`. `FeatureConfig`(plane 목록·history K·tactical planes on/off)는 해시되어 체크포인트·ONNX manifest에 기록. 로드시 불일치면 거부.
- Plane 목록 (기본, 행위자 관점):
  | # | plane | 비고 |
  |---|---|---|
  | 0 | actor 돌 | 결정 phase에서는 actor의 현재 색 |
  | 1 | 상대 돌 | |
  | 2,3 | 흑 돌 / 백 돌 | 색 절대값 |
  | 4 | 마지막 착수 | |
  | 5..5+K-1 | 최근 K ply 착수 (옵션) | |
  | +1 | legal mask (현재 phase의 point action) | |
  | +1 | 흑 금수 마스크 | 규칙 엔진 산출 |
  | +1 | actor 색 (상수 plane) | |
  | +1 | opening 허용 영역 | |
  | +1 | offered fifths | |
  | +1 | 이미 제안한 fifths (autoregressive) | |
  | +1 | chosen fifth | |
  | +N | phase one-hot (상수 plane) | |
  | (opt) | tactical planes: 양측 five/four/three 위협점 | `use_tactical_feature_planes` ablation |
- D4 증강은 `GameState::transformed` + 동일 encode로 수행(Python 쪽 배열 회전으로 하지 않음 → 회전 규약 불일치 원천 차단). 대안으로 Python 회전을 쓰려면 C++ 결과와 일치 테스트 필수.

### 3.9 NN Evaluator 인터페이스
```cpp
struct EvalRequest { const float* features; PhaseHeadMask heads; const uint8_t* legal_mask; };
struct EvalResult  { float policy[226];      // 현재 phase의 head 출력 (masked softmax 전 logits)
                     float wdl[3];           // actor 관점 W/D/L
                     float decision[2];      // swap/keep 또는 direct/ten (해당 phase만)
                     /* aux heads optional */ };
class Evaluator { public: virtual void evaluate(std::span<const EvalRequest>, std::span<EvalResult>) = 0;
                  virtual ModelInfo info() const = 0; virtual ~Evaluator() = default; };
```
구현체: `UniformEvaluator`(테스트), `HeuristicEvaluator`(기준 봇), `OnnxEvaluator`(ONNX Runtime; CPU / OpenVINO EP / DirectML EP), `PyBridgeEvaluator`(개발용, 프로세스 간 공유메모리). **ADR-02**: 추론 런타임은 Phase 4–5 벤치마크(eval/s, 지연, 배치 크기별)로 결정. libtorch는 Windows+Intel GPU 지원이 불확실하여 후순위.

### 3.10 Search (Phase 5)
- Node: `P, N, W (actor-독립 절대값: P0 관점), Q, children[], proven, phase, actor, virtual_loss`. 메모리 arena 할당, edge 배열 연속.
- **Actor-aware backup (MUST)**: opening에서는 같은 플레이어가 연속으로 행동한다(예: CHOOSE_TEN 후 PROPOSE×10). 따라서 negamax식 부호 반전 금지. 값은 P0 관점 절대값으로 누적하고 선택 시 노드의 actor 관점으로 변환한다. 정상 Renju 구간에서는 결과가 표준 negamax와 동일함을 테스트.
- Selection: PUCT `Q + c·P·√N_parent/(1+N)`, FPU reduction. Gumbel root: Gumbel-top-k 샘플 + Sequential Halving, completed-Q 기반 improved policy target (Danihelka et al. 2022). **ADR-03**: 둘 다 구현, 동일 compute에서 Elo로 선택.
- MCTS-solver: 자식 중 하나가 actor 승으로 증명 → 노드 증명 승; 모든 자식 증명 패 → 증명 패. 증명값은 NN 값보다 우선.
- 전술 통합: 확장 시 (1) 즉시 5 → 증명 승, (2) 상대 four → 자식을 차단점(+자기 5)으로 제한, (3) 조건부 VCF(노드 budget, 위협 존재 시), (4) 루트에서 VCF/VCT 큰 budget. 호출 임계값은 config + ablation(EVALUATION §Ablation C).
- Batching: self-play는 **game-parallel** (한 프로세스에서 G개 게임, 각자 트리, 공용 배치 큐). 분석 모드는 tree-parallel + virtual loss. 배치 크기·대기 timeout은 config.
- NN cache: canonical hash + phase/offered 키, LRU. 트리 재사용(다음 착수 시 서브트리 유지).
- `SearchLimits { max_visits, max_time_ms, max_nodes, solver_budget }`, profile FAST/STANDARD/STRONG/ANALYSIS는 `configs/search/*.json`.
```cpp
struct SearchResult { Action action; float win_p, draw_p, loss_p; uint32_t visits;
                      std::vector<Action> pv; ProofResult tactical_proof;
                      uint64_t search_nodes; uint32_t elapsed_ms;
                      std::vector<CandidateStat> top;  /* action, P, Q, N, WDL */ };
class AIEngine { public: SearchResult search(const GameState&, const SearchLimits&); };
class OpeningEngine {                     // opening 결정 전용 래퍼 (같은 search 사용)
 public: std::vector<Action> legal_opening_actions(const GameState&);
         OpeningDecision evaluate_opening(const GameState&, const SearchLimits&);
         ProposalSet generate_ten_fifths(const GameState&, const SearchLimits&); };
```

### 3.11 Self-play runtime (Phase 6)
- `dojozero_engine selfplay --config ... --model <manifest>`: G개 동시 게임, 결과를 chunk 파일로 기록(§5.1). 파일은 `tmp → rename` 원자적 쓰기.
- 모델 갱신: `runs/<run>/models/manifest.json`(현재 best/candidate 버전, hash) 폴링. 기보에 사용 모델 hash 기록.
- 초기 분산: 파일시스템 큐(단일 머신). 확장(gRPC/MQ)은 Future Work.

---

## 4. Python

- `dojozero_core` (pybind11): GameState, Action, encode_features, symmetry, solver, record reader/writer.
- `model/`: `ResNetPV`, `HybridAttnPV` (conv stem + residual + 축(axial)/전역 attention 블록). 헤드: policy(226), WDL(3), aux(마스크들, BCE), opening heads(swap 2, branch 2, direct-fifth = policy head 공유 + phase plane, propose 225, select 225-masked).
- `trainer/`: 학습 루프, 체크포인트, ONNX export + manifest, NaN 감지(손실·그래디언트 non-finite 시 즉시 중단 + 직전 배치 덤프; silent NaN 금지).
- `replay/`: SQLite 인덱스(게임 id, 모델 버전, profile hash, opening key, 결과, 길이, canonical hash 집합) + chunk 파일. 샘플링(FIFO window, 감쇠, prioritized 옵션), 중복 제거, reanalysis 큐.
- `league/`: opponent pool, PFSP 가중치, 모델 레지스트리.
- `evaluation/`: 대국 스케줄러(paired), 외부 엔진 어댑터(Gomocup/piskvork 프로토콜 클라이언트), Elo/SPRT 통계, 벤치마크 러너, 리포트 생성.
- `tools/`: 기보 변환, 국면 시각화, 호환성 테스트 JSON 작성 보조.
- 학습 장치: **ADR-04** PyTorch XPU(Intel Arc) vs CPU vs 외부 GPU — Phase 4 벤치마크. 코드는 `device` 추상화로 장치 무관.

---

## 5. 파일 포맷 (모두 버전 필드 필수)

### 5.1 Game record (`.dzg` chunk)
- 원칙: **보드가 아니라 action 수순을 저장**한다. 국면은 C++ 엔진 재생으로 복원(규칙 단일 구현 유지, 저장량 최소).
- 구조: `magic "DZG\0"`, `u16 format_version`, `u32 n_games`, 이후 게임별
  `u32 meta_len` + UTF-8 JSON 메타(`game_id, rule_profile_hash, initial_assignment, model_hashes{P0,P1}, search_config_hash, seed, opening_key, result, reason, created_at, engine_git_commit`)
  + `u16 n_actions` + action 배열 + action별 search 레코드(`root_value f32, wdl f32[3], n_visits u32, policy: u16 k + (u16 action_idx, f16 prob)[k], proof u8, solver_nodes u32, legal_count u16, flags u8`).
- 디버그 export: 같은 내용의 JSONL.

### 5.2 Checkpoint (`.pt`)
`{format_version, model_config, feature_config + hash, rule_profile_hash, state_dict, optimizer, scheduler, step, rng_states(python/numpy/torch), data_cursor, git_commit, env(torch/driver/hardware), metrics}`. 이어 학습 시 bit 단위 재현을 목표(결정적 모드에서 테스트).

### 5.3 Inference model
`model_<version>_<sha256[:12]>.onnx` + `manifest.json {version, sha256, parent_checkpoint, feature_config_hash, rule_profile_hash, heads, opset, created_at}`. 로더는 해시와 feature config를 검증.

### 5.4 Benchmark / compatibility case (JSON)
```json
{ "schema_version": 1, "id": "vcf-0042", "category": "VCF", "source": {"kind":"handmade|literature|differential|app", "ref":"..."},
  "profile": "RENJU_RIF", "position": {"moves": ["H8","..."]} ,
  "query": {"type": "vcf", "attacker": "BLACK", "budget_nodes": 1000000},
  "expected": {"result": "WIN", "first_moves": ["J9"]}, "notes": "" }
```

### 5.5 Config
- 규칙·탐색: JSON (C++과 Python 공통 로딩). 학습·모델: YAML (Python). 실행 시 최종 병합 config를 run 디렉터리에 덤프.

---

## 6. CLI

| 명령 | 구현 | 내용 |
|---|---|---|
| `dojozero rules-test` | C++ | 규칙 테스트 + 호환성 JSON + 차등 퍼징(짧은 모드) |
| `dojozero play` | C++ | 텍스트 대국(사람 vs 엔진, 엔진 vs 엔진), opening action 입력 지원 |
| `dojozero analyze` | C++ | best move, top-N(P,Q,N,WDL), tactical status, PV, 금수점과 이유(ForbiddenInfo) |
| `dojozero selfplay` | C++ | self-play 워커 |
| `dojozero benchmark` | C++/Py | 고정 벤치마크 + 성능 측정 |
| `dojozero train` | Python | 학습 |
| `dojozero evaluate` | Python | paired match, Elo/SPRT, 리포트 |

`dojozero`는 Python console script 하나로 노출하고 C++ 명령은 `dojozero_engine.exe`로 위임한다.

---

## 7. 테스트 구조

| 층 | 도구 | 내용 |
|---|---|---|
| C++ 단위 | GoogleTest (FetchContent, 고정 태그) | 카테고리별 (RULES §12) |
| 데이터 구동 | JSON 케이스 → C++/pytest 공용 러너 | `tests/**/cases/*.json`, `tests/compatibility/omokdojang/*.json` |
| 전수 열거 | 1차원 window 전체 | 테이블 생성기 = oracle 일치, A-R3/A-R4 검증 |
| 차등 퍼징 | fast vs `ReferenceRules` | 고정 seed, 흑 밀집 국면 편향 생성기, 불일치 국면 자동 최소화 → 회귀 JSON |
| Property | 무작위 apply/undo 왕복, D4 변환 불변량(금수 판정·합법수·결과가 변환과 가환), hash 일치 | |
| 상태 머신 | opening 전 경로 열거(제안 단계는 샘플), malformed 상태 거부 | |
| Python | pytest | binding, feature/증강 일관성, 손실 함수, NaN 감지 |

규칙 엔진 목표: 수백 개 이상의 명시적 단위 테스트 + 퍼징 ≥10⁷ 국면 불일치 0.

---

## 8. 관측성 · 실패 처리
- 로그: 구조화 JSON 로그(run id, component, level). 엔진 stderr 진단, stdout은 프로토콜 전용.
- 메트릭: TensorBoard (training), 평가 리포트(Markdown + JSON).
- 실패 모드와 대응은 PRD §Failure Modes. 핵심: 불법 action은 `apply()` 거부로 표면화(조용한 무시 금지), 모델/feature 해시 불일치 시 로드 거부, NaN 즉시 중단, 워커 크래시 시 부분 chunk 폐기(원자적 rename 전이므로 손상 없음).

---

## 9. 빌드 · 툴체인
- `CMakePresets.json`: `msvc-release`, `msvc-debug`, `msvc-asan`. Generator: Visual Studio 17 2022 (Ninja 없음).
- 서드파티(고정 버전): GoogleTest, pybind11, nlohmann/json, CLI11, (Phase 5+) ONNX Runtime.
- Python 패키징: `pyproject.toml` (scikit-build-core로 C++ 확장 빌드) — ADR-05, Phase 2 말 확정.

---

## 10. 결정 대기 (ADR 목록)
| ID | 결정 | 결정 시점 | 기준 |
|---|---|---|---|
| ADR-01 | Board 표현 A/B/C | Phase 2 말 → **부분 결정**: A(padded array + 8중 증분 Zobrist) 채택. 금수 판정 2.1M/s로 목표 충족, apply+undo(3.4M/s)·legal_actions(25k/s)는 미달. C(line code + 생성 테이블 + 증분 금수 캐시)는 Phase 2에서 **구현하지 않음** — MCTS nodes/s를 실측하는 Phase 5 전에 병목 여부로 재결정 | apply/undo/s, forbidden/s, 정확성 동일 |
| ADR-02 | 추론 런타임 (ORT CPU / OpenVINO / DirectML / PyBridge / libtorch) | Phase 4–5 | batched eval/s, 지연, 설치 용이성 |
| ADR-03 | PUCT vs Gumbel (self-play/대국 각각) | Phase 5–6 | 동일 compute Elo |
| ADR-04 | 학습 장치 | Phase 4 | step/s, 안정성 |
| ADR-05 | Python 빌드 방식 | Phase 2 → **Phase 4로 연기** (pybind11 바인딩은 학습 파이프라인에서 처음 필요) | Windows 재현성 |
| ADR-06 | 10제안 정책 방식 (top-k vs autoregressive vs beam) | Phase 7 | 제안 품질(regret) + Elo |
| ADR-07 | ResNet vs Hybrid attention | Phase 6–8 | Elo/compute |
