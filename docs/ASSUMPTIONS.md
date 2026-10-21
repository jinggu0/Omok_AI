# DojoZero — ASSUMPTIONS.md

Status: **Draft v0.1** · 2026-10-18

이 문서는 (1) 공식 규칙 문구가 모호해 엔진이 채택한 해석, (2) 오목도장 앱과의 호환성 중
확인되지 않은 항목, (3) 개발 환경·자원에 대한 가정을 **숨기지 않고** 기록한다.
코드에서 이 문서의 ID(`A-xx`, `C-xx`, `E-xx`)를 주석으로 참조한다.

각 가정의 상태:
- `OPEN` — 미검증. 기본값으로 구현하되 바꿀 수 있게 profile/config로 노출
- `VERIFIED(source)` — 1차 출처 또는 앱 실측으로 확인
- `REJECTED` — 검증 결과 틀림 → RULES.md 수정 이력 기록

---

## 1. 규칙 해석 가정 (Rule Interpretation)

| ID | 가정 | 근거 | 대안 | 영향 | 상태 |
|---|---|---|---|---|---|
| A-R1 | 흑이 한 방향에 정확한 5, 다른 방향에 장목을 동시에 만들면 흑 승 | RIF 9.2 문언 "without at the same time attaining five" | 장목 우선(흑 패) | 극히 드문 국면, 전술 엔진의 승리 판정 | OPEN (C-02) |
| A-R2 | 엔진은 5목을 자동 감지·즉시 종국 (RIF 9.7의 "알아채지 못함" 절차 무시) | 엔진/앱 공통 관행 | — | 없음 | OPEN |
| A-R3 | 한 방향에서 p가 FIVE/OVERLINE/FOUR를 만들면 그 방향은 THREE로 세지 않음 | RIF three = "row with three stones" | 같은 방향에 four와 three 동시 계수 | 4-3 판정 | VERIFIED(1D 전수 열거, 착수점 ±5칸, 모든 열 위치 354,051 패턴): "같은 직선에 p를 지나는 four와 겉보기 three가 동시에 존재"하는 패턴 **0건** → 이 가정은 판정 결과에 영향을 주지 않음. 범위 밖(±6 이상) 돌은 미열거 |
| A-R4 | 방향당 THREE는 최대 1개 | 한 직선에 two-three 패턴이 없다고 추정 | 방향당 2개 계수 | 3-3 판정 | VERIFIED(같은 열거): p를 지나는 서로 다른 three 돌 집합이 한 직선에 2개 이상인 패턴 **0건** |
| A-R5 | 흑 차례에 모든 빈 점이 금수이고 패스 불가면 무승부 | 규칙 공백 | 흑 패 / 강제 패스 | 극히 드묾 | OPEN (C-06) |
| A-R6 | Opening 단계(NORMAL_RENJU 이전) 패스 불가 | RIF 12.9 (첫 3수 패스 불가)의 확장 | — | 없음 | OPEN |
| A-R7 | 흑의 FOUR completion이 장목이 되는 점은 completion으로 치지 않음 | RIF "five in a row" = 정확히 5 (장목은 별도 정의) | — | 4-4, straight four 판정 | VERIFIED(RIF Art.3 정의 구조) |
| A-T1 | Choice 1/Choice 2는 4수 직후 **흑 holder**가 결정 | RenjuNet 포럼 Taraguchi 해설 "Black player decides"; 한국어 해설 "흑이 10개 후보를 둘 경우 이 시점에 흑백 확정" | 백 holder 결정 | opening 정책 구조 전체 | OPEN (C-07) |
| A-T2 | Choice 1의 첫 swap은 흑 holder가 결정 (≡ 흑 holder의 3지선다: 제안10 / swap / 유지 후 직접5) | "after move k, the other player may swap" 패턴 (4수=백 → other=흑 holder) | 백 holder가 결정 | opening 정책 | OPEN (C-08), profile `swap_before_direct5_actor` |
| A-T3 | "대칭 금지"는 E1(5돌 국면 D4 동치) — E2(stabilizer orbit)와 동치임을 증명(RULES §10.2) | RenjuNet 포럼: "can not be symmetrical in any aspect of the position" | E3 점 단위 D4 | 제안 생성기 | E1=E2는 VERIFIED(도달 가능한 4수 국면 8,464개 × 모든 빈 점 전수 비교). E1 vs E3 선택은 OPEN (C-10), profile `proposal_symmetry` |
| A-T4 | 대칭 제안이 입력되면 `OFFER_TEN` 전체를 REJECT (자동 제거하지 않음) | 엔진 무결성 | 자동 dedupe | 외부 입력 처리 | OPEN (C-10) |
| A-T5 | 10개 제안은 영역 제한 없음 | RIF-T10 "anywhere on board" | 9×9 제한 | 제안 정책 action space | VERIFIED(RIF-T10) / 앱은 OPEN (C-09) |
| A-T6 | 영역 경계 포함 (Central(N)은 N×N 정사각형 전체) | 일반 해석 | — | 없음 | OPEN (C-20) |
| A-T7 | 6수 이후 swap 없음, 7수부터 일반 Renju | RIF-T10 | — | — | VERIFIED(RIF-T10) |

### 1.1 외부 프로토콜 가정 (Gomocup/Piskvork, v0.2 추가)
| ID | 가정 | 근거 | 대안 | 영향 | 상태 |
|---|---|---|---|---|---|
| A-P1 | `INFO rule`은 비트마스크이며 Renju 값을 받은 엔진은 흑 금수(3-3·4-4·장목)를 RIF와 같게 처리 | Gomocup 프로토콜 관행 | 엔진별 해석 차이(재귀 거짓금수 미구현, 장목 처리 차이) | 외부 대국 결과 오염 | 부분 VERIFIED(2026-10-19, `rule 4`): 사용 가능한 10개 엔진 모두 장목·4-4·3-3 함정 회피, 백 장목 승리 인식, 짧은 대국 4판 실패 0 (`dojozero.evaluation.conformance`). 재귀 거짓금수는 미검사. 심판은 항상 canonical `GameState` |
| A-P4 | 외부 엔진은 `timeout_turn`만 주어도 수당 시간을 지킨다 | 프로토콜 문서 | 매치 시계(`timeout_match`/`time_left`)가 없으면 기본값(사실상 무제한)으로 오래 생각 | TIMEOUT 오판 | REJECTED(Embryo 2021/2026이 시계 없이 timeout) → 클라이언트가 항상 `timeout_match`(기본 250×turn)와 수마다 `time_left`를 보냄 |
| A-P2 | Gomocup 좌표 `x,y`는 왼쪽 위 원점(x = 열, y = 행, 아래로 증가) | Gomocup 프로토콜 문서 관행 | 다른 원점 | 모든 외부 착수 | OPEN — 변환은 어댑터 1곳(`gomocup_to_point`/`point_to_gomocup`), 전 점 왕복 테스트 통과. 실제 외부 엔진으로 방향 확인은 바이너리 확보 후. 방향이 틀려도 Renju는 D4 불변이라 게임은 동치(단 corpus 국면이 외부 엔진에게 대칭 변환되어 보임) |
| A-P3 | Renju 시작 국면을 `BOARD`로 주면 엔진은 돌 수로 자기 색을 판단 | Renju는 흑 선수 | 엔진이 색을 별도로 가정 | 색 혼동 | VERIFIED(10개 엔진): 흑·백 차례 BOARD 함정에서 모두 올바른 색으로 착수(즉승점·백 장목 승리) |

---

## 2. Compatibility Matrix — `OMOKDOJANG_DAN` vs RIF

**전제 자체도 가정이다**: "오목도장 유단자 규칙 = RIF Renju 금수 + Taraguchi-10". 공개 웹에서
오목도장 앱의 공식 규칙 문서를 찾지 못했다(2026-10-18 조사). 한국어 공개 해설은
"2024년부터 Taraguchi-10 채택" 같은 2차 언급만 있다. → `C-11`.

모든 행은 `UNVERIFIED_APP_BEHAVIOR` 이다(검증 전까지).

| ID | 항목 | RIF/공식 | 엔진 기본값 | 오목도장 | 검증 방법 | 회귀 테스트 위치 |
|---|---|---|---|---|---|---|
| C-01 | 정확한 5 + 4-4/3-3 동시 | 흑 승 | 흑 승 | UNVERIFIED_APP_BEHAVIOR | 앱에서 해당 국면 재현, 착수 허용·결과 스크린샷 | `compatibility/omokdojang/five_precedence_*.json` |
| C-02 | 정확한 5 + 장목 동시 | 흑 승(문언) | 흑 승 | UNVERIFIED_APP_BEHAVIOR | 동일 | 동일 |
| C-03 | 금수점 착수 시도 | 반칙패(RIF, 백 지적 필요) | REJECT | UNVERIFIED_APP_BEHAVIOR (금수점 표시·클릭 차단 추정) | 앱에서 금수점 클릭 | `forbidden_handling_*.json` |
| C-04 | 재귀 거짓금수 | 완전 재귀(RIF 9.3) | 완전 재귀 | UNVERIFIED_APP_BEHAVIOR (앱이 1단만 구현했을 가능성) | 1단/2단/3단 재귀 국면 세트를 앱에서 확인 | `false_forbidden_depth*.json` |
| C-05 | 한 직선 4-4 | 금수 | 금수 | UNVERIFIED_APP_BEHAVIOR | `B.BBB.B` 류 국면 | `line_double_four_*.json` |
| C-06 | 패스 / 둘 곳 없음 | 패스 허용, 연속 패스 무승부 | 패스 불가, 무승부 | UNVERIFIED_APP_BEHAVIOR | UI에 패스 버튼 존재 여부 | `pass_*.json` |
| C-07 | Choice 1/2 결정자 | 미명시 | 흑 holder | UNVERIFIED_APP_BEHAVIOR | 4수 후 UI에서 어느 쪽에 선택지가 뜨는지 | `opening/choice_actor.json` |
| C-08 | Choice 1 첫 swap 결정자 | "other player" | 흑 holder | UNVERIFIED_APP_BEHAVIOR | 동일 | `opening/swap_before_5.json` |
| C-09 | 10제안 영역 | 보드 전체 | 보드 전체 | UNVERIFIED_APP_BEHAVIOR | 9×9 밖 제안 시도 | `opening/proposal_region.json` |
| C-10 | 대칭 정의 / 처리 | "symmetrical not allowed" | E1, 입력 거부 | UNVERIFIED_APP_BEHAVIOR | 대칭 국면에서 거울상 쌍 제안 시도; 국면은 비대칭이지만 점 쌍이 D4 관계인 경우(E1≠E3) 시도 | `opening/proposal_symmetry.json` |
| C-11 | 유단자 규칙이 Taraguchi-10인가 | — | Taraguchi-10 | UNVERIFIED_APP_BEHAVIOR | 앱 규칙 안내 화면 / 유단자 대국 기보 | 문서 |
| C-12 | 1수 천원 자동 배치 | 흑이 둠 | action으로 표현(유일 합법수) | UNVERIFIED_APP_BEHAVIOR | UI 관찰 | `opening/move1.json` |
| C-13 | 6수 후 swap | 없음 | 없음 | UNVERIFIED_APP_BEHAVIOR | UI 관찰 | — |
| C-14 | 백 장목 | 승 | 승 | UNVERIFIED_APP_BEHAVIOR | 국면 재현 | `white_overline.json` |
| C-15 | 판 꽉 참 | 사실상 무승부 | 무승부 | UNVERIFIED_APP_BEHAVIOR | (재현 어려움) | — |
| C-16 | 시간 제한·초읽기 | 대회 규정 | 범위 밖 (엔진 SearchLimits로만) | UNVERIFIED_APP_BEHAVIOR | — | — |
| C-17 | 무승부 제안·기권 | 대회 규정 | 범위 밖 | — | — | — |
| C-20 | 중앙 N×N 경계 포함 | 포함(일반 해석) | 포함 | UNVERIFIED_APP_BEHAVIOR | 경계 점 착수 시도 | `opening/region_boundary.json` |

### 2.1 앱 증거 수집 절차 (사용자 수동)
자동 조작·온라인 대국 부착은 **하지 않는다**(PRD Non-goals). 증거는 사람이 직접 앱에서 수집한다.
1. 국면을 앱의 대국/복기/연습 기능으로 재현한다.
2. 스크린샷 + 관찰 결과(착수 허용 여부, 금수 표시, 결과)를 기록한다.
3. `tests/compatibility/omokdojang/<C-ID>_<slug>.json` 으로 저장:

```json
{
  "schema_version": 1,
  "compat_id": "C-04",
  "source": { "kind": "manual_app_test", "app_version": "x.y.z",
              "date": "2026-10-20", "evidence": ["screenshots/C-04_a.png"] },
  "profile": "OMOKDOJANG_DAN",
  "position": { "moves": ["H8", "I9", "..."] },
  "side": "BLACK",
  "query": { "point": "J10" },
  "expected_legal": false,
  "expected_forbidden": "DOUBLE_THREE",
  "expected_reason": "app marks J10 as forbidden",
  "notes": ""
}
```
- `position`은 착수 수순(`moves`) 또는 돌 목록(`black`/`white` + `phase` + `side_assignment`) 중 하나.
- 다른 쿼리 종류: `expected_legal_moves`(전체 합법수 집합), `expected_result`(종국 판정), `expected_phase`(opening 전이).
4. 테스트 러너는 이 디렉터리를 자동 수집한다. 엔진 기본값과 다르면 **실패**로 보고하고, 해당 `C-xx`를 VERIFIED/REJECTED로 갱신한 뒤 profile 기본값을 조정한다.

---

## 3. 환경·자원 가정 (Environment)

조사 시점(2026-10-18) 실측:

| ID | 항목 | 값 | 영향 |
|---|---|---|---|
| E-01 | OS | Windows 11 Home 10.0.26200 | 빌드/스크립트는 Windows 우선, Linux 호환 유지 |
| E-02 | CPU | Intel Core Ultra 7 258V (Lunar Lake, 8코어 = 4P + 4LP-E, SMT 없음) | 병렬 self-play 워커 수 상한 ≈ 6–8 |
| E-03 | RAM | 32 GB (공유 메모리) | replay buffer는 디스크 기반 |
| E-04 | GPU | Intel Arc 140V 내장 GPU (공유 메모리, **CUDA 없음**) | 학습/추론 백엔드 선택에 결정적. PyTorch XPU 또는 ONNX Runtime(OpenVINO/DirectML EP) 후보 |
| E-05 | NPU | Lunar Lake NPU 존재 (미검증) | OpenVINO NPU 추론은 Future Work |
| E-06 | C++ 컴파일러 | MSVC 14.43 (VS 2022 Community), C++20 사용 | MinGW GCC 6.3도 PATH에 있으나 **사용 금지**(C++17 미흡) |
| E-07 | CMake | 4.3.2 | |
| E-08 | Python | 3.11.9, numpy 2.4.6, pytest 8.4.2 설치됨, **PyTorch 미설치** | Phase 4 전에 설치 필요 (XPU 빌드) |
| E-09 | 저장소 | 빈 git 저장소 (`main`, 커밋 없음) | 기존 코드 제약 없음 |
| E-10 | 평가 하드웨어 | 외부 엔진 평가도 같은 노트북(E-01..E-08)에서 수행한다고 가정. 학습·self-play와 동시 실행 시 CPU/GPU 경합으로 시간 통제 대국 결과가 달라짐 | Tier B/C 외부 대국은 다른 무거운 작업과 동시에 실행하지 않음, match에 hardware_id·동시 작업 기록 (PRD OQ-12). **유휴 절전**: 세션이 백그라운드 대기 중이면 앱의 keep-awake가 풀려 노트북이 절전에 들어감(라운드로빈 v1 무효화) → 평가 진입점이 프로세스 단위 절전 방지 요청을 유지(덮개 닫힘은 막지 못함) |
| E-11 | Windows 전원 스로틀링(EcoQoS) | 창 없는 백그라운드 프로세스(에이전트 셸에서 띄운 벤치마크·평가)는 Windows가 저속 코어/저클럭에 둘 수 있음. 실측(2026-10-20, 배터리): 같은 ee1d011 솔버가 3.2k vs 24.8k nodes/s(1스레드), 몇 분 단위로 최대 5배 변동 | Phase 8 E3b부터 벤치마크 도구(`dojozero_tbench/tprof`), 평가 진입점(`keep_awake` → `disable_power_throttling`), C++ 엔진 실행기(외부 엔진 자식 프로세스)가 **프로세스 단위로** 스로틀링을 끔(시스템 설정은 바꾸지 않음). 그 이전의 시간 통제 평가(E1–E3a)가 스로틀링 아래 실행됐는지는 **UNKNOWN** — 같은 실행 안의 arm 비교는 영향을 같이 받았지만 절대 수치(sims/수, nodes/s)는 과소일 수 있음 |

### 3.1 자원 가정에서 나오는 핵심 리스크 (PRD Risks R-01)
AlphaZero 계열의 "최상급" 기력은 통상 수백~수천 GPU-일 규모 self-play를 필요로 한다. 노트북 내장 GPU 한 대로는
그 규모가 불가능하다. 따라서 다음을 **가정하지 않는다**: "이 머신에서 RL만으로 최상급 기력 도달".
대신 (a) 정확한 전술 solver 비중 확대, (b) 작은 네트워크 + 효율적 탐색, (c) 외부 기준 엔진과의 조기 비교,
(d) 필요 시 외부 GPU(클라우드) 사용 여부를 사용자가 결정 — 를 PRD Open Questions에 둔다.

---

## 4. 데이터·라이선스 가정

| ID | 가정 | 상태 |
|---|---|---|
| D-01 | 공개 렌주 기보 DB(RenjuNet 등)의 재사용 라이선스는 불명확 → 명시적 허가/라이선스 확인 전 사용하지 않음 | OPEN |
| D-02 | 자체 solver·자체 엔진 생성 기보는 자유롭게 사용 | 확정 |
| D-03 | 타 엔진(Rapfi 등, GPL-3.0) 은 **별도 프로세스로 대국 상대/평가 기준**으로만 사용, 코드·가중치 미포함 | OPEN — 타 엔진 생성 기보를 학습 데이터로 쓸지 사용자 결정 필요 (PRD OQ-05) |
| D-04 | 오목도장 앱 데이터(기보·스크린샷)는 사용자가 직접 수집한 규칙 검증용 소량만 사용, 학습 데이터로 대량 수집하지 않음 | 확정 |
| D-05 | 외부 엔진(KataGomo, AlphaGomoku, JAX, Embryo, Gomocup 구버전 등)의 라이선스·재배포 조건은 엔진별로 다르며 확인 전까지 `UNVERIFIED`. 바이너리는 저장소에 포함하지 않고 사용자가 제공·승인한 것만 실행. 외부 대국 기보는 평가 전용(학습 사용은 OQ-05) | OPEN |

---

## 5. 변경 이력
| 날짜 | ID | 변경 |
|---|---|---|
| 2026-10-18 | 전체 | 초안 작성 |
| 2026-10-18 | A-R3, A-R4, A-T3 | Phase 2 열거 결과로 갱신 (docs/PROGRESS.md) |
| 2026-10-19 | A-P1..A-P3, E-10, D-05 | 외부 Strength Evaluation System(PRD v0.2) 가정 추가 |
| 2026-10-20 | E-11 | Windows 전원 스로틀링(EcoQoS) 발견, 프로세스 단위 해제 (Phase 8 E3b) |
