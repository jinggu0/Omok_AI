# DojoZero — TRAINING.md

Status: **Draft v0.1** · 2026-10-18
**선행 조건 (MUST)**: 규칙 테스트 100% 통과 + oracle 차등 퍼징 불일치 0 + 전술 벤치마크 기준 충족(EVALUATION §3) 전에는 어떤 학습도 시작하지 않는다.

---

## 1. 원칙
1. **보상은 승/무/패만**: win=+1, draw=0, loss=−1 (actor PlayerId 관점). 열린 삼·중앙·사 보상 등 shaping 금지(reward hacking 방지).
2. 전술 지식은 보상이 아니라 **정확한 solver가 만든 supervised/auxiliary target**으로 준다.
3. 순수 랜덤 self-play로 시작하지 않는다 — Stage A→G 커리큘럼.
4. 모든 실험은 재현 가능: git commit, 병합 config, seed, 데이터 생성 범위, 모델 hash 기록(EVALUATION §8).
5. 자원 현실(ASSUMPTIONS E-04): 내장 GPU 한 대. 기본 모델은 작게 시작하고(6 blocks × 64ch), 단계별로 "이 compute로 무엇을 얻었는가"를 Elo/compute로 보고한다.
6. "학습이 안 된다" 진단 순서(MUST): 규칙 정확성 → action 합법성 → target 정확성 → self-play 다양성 → search 정확성 → replay 분포 → gradient → optimizer → hyperparameter.

---

## 2. 커리큘럼 (Stage A–G)

| Stage | 목적 | 데이터 | 종료 기준 |
|---|---|---|---|
| A | 규칙/전술 데이터 생성 | exact solver로 자동 생성 | 각 카테고리 목표 수량 + 라벨 검증 샘플 100% 일치 |
| B | supervised 전술 사전학습 | Stage A | 전술 홀드아웃 정확도 목표(아래) 달성, 정체 시 종료 |
| C | (선택) 전문가 모방 | 라이선스 확인된 기보 + 자체 엔진 기보 | 정책 top-1 정확도 정체 |
| D | 기본 Renju self-play | 중반 balanced 국면에서 시작(opening 없음) | 연속 3세대 승격 실패 시 다음 단계 |
| E | 전체 게임 Renju | 빈 보드 / balanced 시작 국면 | Elo 정체 |
| F | Taraguchi-10 커리큘럼 | opening phase부터 전체 | opening 결정 지표 안정 |
| G | League | opponent pool | 지속 |

### 2.1 Stage A — 데이터 생성 (solver 라벨)
생성기는 C++ (규칙·solver 단일 구현) + 고정 seed. 카테고리:
- win-in-1, block-in-1 (상대 four 차단; 차단점이 흑 금수인 경우 포함)
- VCF (공격 흑/백 각각), VCT (budget 내 증명된 것만)
- 합법/불법 금수 국면(3-3, 4-4, 장목, 거짓금수 재귀 1–3단)
- 4-3 생성, 방어 희생수(defensive sacrifice: four로 시간 벌기), 금수 유도(백이 흑의 차단점을 금수로 만듦)
- 국면 생성법: (1) 자체 엔진 약한 self-play 기보에서 국면 샘플, (2) 무작위 국면 + solver 필터, (3) 증명된 수순의 중간 국면.
- 라벨: policy target = 증명된 승리 첫 수 집합의 균등 분포(승리 국면) / 차단수 집합(방어), value target = 증명 결과(WIN/LOSS)만, UNKNOWN 국면은 **value target 없음**(마스킹). UNKNOWN을 LOSS/DRAW로 라벨링 금지.
- 메타데이터: 생성기 버전, solver budget, seed, profile hash.

**Phase 4 구현 (`dojozero_datagen`, `data/stage_a_v1`, git 미추적)**: 무작위 NORMAL_RENJU 중반 국면(RENJU_RIF·OMOKDOJANG_DAN 반반, canonical D4 hash 중복 제거) 120,000개를 `analyze_tactics`(VCF 8k / VCT 3k / LOSS 1.5k 노드)로 라벨링. policy: 즉시승 → 모든 five point, VCF/VCT/강제차단 → 증명된 수 1개(다른 승리수가 있어도 1개만이라 편향 있음). value: 증명된 WIN/LOSS만, UNKNOWN은 마스킹. 실측 분포: IMMEDIATE_WIN 43.5%, VCF 22.9%, NONE 13.6%, PROVEN_LOSS 8.4%, NO_DEFENSE 6.3%, VCT 4.7%, FORCED_BLOCK 0.6% — **무작위 대국 국면이라 즉시승리에 치우침**(실전 국면 분포와 다름). 생성 속도 ~33 샘플/s(8스레드, 라벨링이 지배적).

### 2.2 Stage B — supervised 전술 사전학습
- 손실: §4 전체 (policy + value(증명된 것만) + aux).
- 홀드아웃: 생성 seed가 다른 별도 세트 + `benchmarks/tactical`(학습 금지 세트).
- 주의: 전술 국면만으로 학습하면 value가 "항상 결정적"으로 편향 → Stage D 진입 시 value head 보정 확인(calibration plot).

### 2.3 Stage C — (선택) 전문가 모방
- 허용 출처만: 라이선스 확인된 공개 기보(ASSUMPTIONS D-01 해결 후), 자체 solver/엔진 기보. 타 엔진 생성 기보 사용 여부는 사용자 결정(PRD OQ-05).
- 각 샘플에 `source, license, url, retrieved_at` 메타데이터.

### 2.4 Stage D/E — Renju self-play
- 시작 국면 분포: D는 balanced 중반 국면(자체 엔진 평가값 |v|<0.2), E는 빈 보드 + balanced 오프닝 세트 혼합.
- 탐색: Gumbel(저 simulation self-play에 유리, ADR-03) 또는 PUCT + Dirichlet noise. 온도: 초반 ply τ=1, 이후 τ→0 (config).
- Playout cap randomization(KataGo): 일부 수만 full search로 정책 target 기록, 나머지는 fast search로 진행 — target 기록 여부 플래그.
- Resign: 비활성 기본. 활성화 시 검증용으로 일정 비율(예 10%) 게임은 resign 금지하고 false-resign rate 측정(목표 <5%).

### 2.5 Stage F — Taraguchi-10
- 모든 opening action을 self-play에서 학습: swap 결정(1,2,3수 후, 4수 후, 직접 5수 후), 2/3/4수, Choice 1/2, 직접 5수, 10제안, 제안 선택.
- 탐색 트리는 opening 결정 노드를 포함(actor-aware backup, ARCHITECTURE §3.10).
- 탐험: swap/branch 결정에 ε-탐험 또는 Dirichlet noise, 결정별 분포를 로깅해 붕괴(항상 swap 등) 감지.
- 10제안(ADR-06) — 이론 근거: 백은 V_black이 최소인 제안을 고르므로, 정확한 V가 주어지면 최적 집합 = 비대칭 제약 하 V_black 상위 10 orbit. 따라서
  - **Approach B (기본)**: 후보 q마다 `P+q`(백 6수 차례) 국면 가치를 얕은 탐색으로 추정 → orbit 대표 상위 M개(예: 30)에 Sequential Halving으로 탐색 배분, **10위 경계 근처**에 집중 → 상위 10 orbit.
  - **Approach A**: PROPOSE_FIFTH를 10회 순차 action으로 보는 autoregressive 정책(이미 제안한 점 plane 사용). 학습 target = B 방식의 탐색 결과.
  - **Approach C**: A의 beam search.
  - 비교 지표: 제안 집합의 min-value regret(강한 분석 탐색 기준), self-play Elo.
- 제안 선택(백): 10개로 제한된 루트에서 일반 탐색.
- **Opening Value Database**: 키 = canonical opening state(canonical hash + phase + offered set 정규형), 값 = {policy, WDL, visits, model version, uncertainty(분산 또는 앙상블 차이)}. self-play와 분석 탐색 결과를 누적, 버전별 분리 저장. swap 결정은 이 DB 값을 prior로 사용 가능(옵션).
- 흑/백 이점 측정: opening self-play에서 최종 색 배정별 승률을 세대별로 기록. 균형 opening일수록 0.5 근처가 정상.

### 2.6 Stage G — League
- Pool: current, best, historical checkpoints(균등 간격 샘플), specialist(전술 solver 강화 봇), opening specialist(Stage F 특화), defensive specialist(방어 target 가중 학습본), 무작위화된 강한 상대(탐색 노이즈·온도 변형).
- 상대 선택: PFSP(prioritized fictitious self-play) — 현재 모델이 이기기 어려운 상대에 가중. 비율(config): self 50% / pool 50% 시작.
- League 대국 기록도 replay에 들어가되 `opponent_type` 메타데이터로 비율 제어. 정책 target은 학습 대상 모델 측 수만 사용.
- 순환 전략(rock-paper-scissors) 감지: pool 내 승률 행렬의 비이행성(intransitivity) 지표를 세대별 기록.

---

## 3. Self-play record (위치별 저장 항목)
ARCHITECTURE §5.1 포맷. 위치(action)별:
`state(=수순으로 복원), legal mask(복원 시 재계산 + 저장된 legal_count로 검증), rule phase, π (visit 분포 또는 Gumbel improved policy), root value/WDL, chosen action, final result z (actor PlayerId 관점), side/color mapping, opening metadata, solver proof result, search stats(visits, nodes, time), model version/hash, target_recorded flag`.
Training tuple: `(s, π, z, aux_targets)`. aux target은 학습 시 C++ 엔진으로 재계산(저장하지 않음 → 규칙 수정 시 자동 갱신, 단 규칙 버전 hash 불일치 시 경고).

---

## 4. 손실 함수

```
L = c_policy · L_policy + c_value · L_value + c_aux · Σ_k w_k L_aux_k + c_open · L_opening + λ ||θ||²
```
- `L_policy`: CE(π, softmax(masked logits)). **illegal action은 logits에서 −∞ 마스킹** 후 softmax — 학습 중에도 illegal에 확률을 주지 않음. π가 illegal에 질량이 있으면 데이터 오류로 즉시 실패.
- `L_value`: WDL CE (기본) 또는 scalar MSE (config). target z 또는 z와 root value 혼합(α, config, 기본 α=0).
- `L_aux`: immediate_win_mask, forced_block_mask, four_threat_mask, three_threat_mask(흑: 진짜 three만), forbidden_mask — BCE. proven_tactical_value(옵션) — 증명된 국면만 CE.
- `L_opening`: swap/branch 결정 CE (target = 탐색 결정 분포), propose/select 정책 CE.
- 정규화: AdamW weight decay(기본) 또는 L2 (SGD 실험) — config에서 택일, 중복 적용 금지.
- Optimizer: AdamW 기본(lr 1e-3, wd 1e-4, β=(0.9,0.999)), 대안 SGD+momentum 0.9. Scheduler: warmup + cosine decay. 값은 초기 추정치이며 Stage별 sweep 결과로 갱신.
- Gradient clipping(norm, config). AMP 사용 시 loss scaling + non-finite 검사.

---

## 5. Replay buffer
- 디스크 기반 chunk + SQLite 인덱스(ARCHITECTURE §4).
- 최대 게임 수/위치 수(config), **generation window**: 최근 W개 모델 버전 데이터만 기본 샘플, 그 이전은 감쇠 가중(지수, config) — 오래된 데이터 지배 방지.
- FIFO 기본, prioritized sampling 옵션(value 오차 또는 정책 KL 기반, 중요도 보정 포함).
- 균형 메타데이터: opening key, 최종 색 배정, 결과 → 흑/백·opening 균형 샘플링 가중.
- 중복 제거: canonical D4 hash. 동일 국면 다수 발생 시 π·z를 평균하는 병합 모드(옵션) vs 개별 유지(기본).
- 샘플 단위로 D4 무작위 변환(§6).
- **Reanalysis**: 과거 국면을 최신 모델+강한 탐색으로 재평가해 π/value target 갱신. 재분석 target에는 `reanalysis_model_hash`, 시점 기록. 비율 config(예 20%). Ablation H.

---

## 6. 데이터 증강 (D4)
- 8개 변환 ID·순서는 RULES §1.5 고정.
- 변환 대상 일체: 보드, last move/history planes, legal mask, 금수 마스크, opening 영역(중앙 정사각형이라 불변이지만 일반 profile 대비 변환 적용), offered set, 선택 fifth, π의 point action index, aux 마스크. PASS·SWAP 등 비-point action은 불변.
- 구현: C++ `GameState::transformed(t)` 후 재인코딩(ARCHITECTURE §3.8). 테스트: 임의 국면에서 `encode(transform(s)) == transform_planes(encode(s))`, `legal(transform(s)) == transform(legal(s))`, `forbidden` 가환성.

---

## 7. 네트워크
- Baseline: AlphaZero ResNet (conv stem, B residual blocks, C channels). 시작 크기: B6C64 → B10C128 (compute 측정 후).
- Strong 후보: conv stem + residual + 전역/축 attention 블록(15×15=225 토큰, 상대 위치 bias). (ADR-07)
- 헤드: policy(225+PASS), WDL(3), aux 마스크(5×225), opening(swap 2, branch 2, propose 225, select 225). phase별 legal mask 적용.
- Global pooling 브랜치(KataGo식)로 전역 정보(phase, 돌 수) 전달.
- 초기화·정규화: BatchNorm 기본, 추론 시 fused. 체크포인트 format_version.

---

## 8. 분산 학습 구조
- 1 Trainer + N self-play worker + M evaluation worker. 초기: 단일 머신 프로세스 + 파일시스템 큐.
- 모델 배포: trainer가 ONNX + manifest(version, sha256) 게시 → 워커 폴링. 기보마다 모델 hash.
- 확장: gRPC/MQ (Future Work).
- 자원 배분(이 머신 가정): self-play 워커 1프로세스(game-parallel, 배치 추론), trainer는 self-play와 시간 분할 또는 동시(벤치마크로 결정).

---

## 9. 재현성 · 실험 추적
- 각 run 디렉터리: `config.merged.yaml`, `git_commit`, `git_diff.patch`(dirty 시), `env.json`(Python/PyTorch/드라이버/하드웨어), `seeds.json`, 체크포인트, 메트릭.
- 결정적 테스트 모드: 고정 seed + 단일 스레드 + 결정적 커널 → 짧은 학습 2회 실행 손실 곡선 bit 일치 테스트(CPU).
- Resume: 체크포인트에 optimizer/scheduler/RNG/데이터 커서 저장 → 중단·재개 후 손실 곡선이 비중단 실행과 일치하는지 테스트.
- 추적: TensorBoard 기본(경량, 로컬). 기록: total/policy/value/aux loss, policy entropy, weight norm, grad norm, lr, Elo, tactical accuracy, 흑/백 승률, opening 통계, self-play 게임 길이 분포, 결정(swap/branch) 분포.
- 장시간 프로세스(CLAUDE.md §6): 시작 시 정확한 명령·플래그를 run 디렉터리에 기록, 재시작 전 기존 프로세스 확인(PID 파일 + 프로세스 조회).

---

## 10. 초기 수치 (모두 추정치, 측정 후 갱신)
| 항목 | 초기값 | 갱신 근거 |
|---|---|---|
| 모델 | ResNet B6C64 | eval/s 실측 |
| self-play 탐색 | Gumbel, 루트 16 후보, 64–200 sims | Elo/compute |
| batch size (train) | 256 | 장치 메모리 |
| replay window | 최근 25만 위치 | 학습 곡선 |
| 승격 | EVALUATION §5 SPRT | — |
