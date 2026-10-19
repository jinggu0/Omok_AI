# DojoZero — RULES.md

Status: **Draft v0.1 (Phase 1, 리뷰 대기)** · 2026-10-18
이 문서는 엔진이 구현할 규칙의 **단일 규범 명세(normative spec)** 다. 코드와 이 문서가
다르면 코드가 버그다. 앱(오목도장) 동작과 다를 수 있는 항목은 `UNVERIFIED_APP_BEHAVIOR`
로 표시하고 [ASSUMPTIONS.md](ASSUMPTIONS.md)의 Compatibility Matrix(ID `C-xx`)와 연결한다.

규범 용어: **MUST / MUST NOT / SHOULD / MAY** (RFC 2119 의미).

출처 등급:
- `[RIF]` — RIF 공식 Renju 규칙 (renju.net/rifrules, Art. 3, 9, 10, 12)
- `[RIF-T10]` — RenjuNet Taraguchi-10 Rule 페이지 (renju.net/rule/25)
- `[INTERP]` — 공식 문구가 모호해 엔진이 채택한 해석 (ASSUMPTIONS.md `A-xx`에 근거 기록)
- `UNVERIFIED_APP_BEHAVIOR` — 오목도장 앱에서 실제로 확인되지 않은 동작

---

## 0. 문서 구성

| § | 내용 | 구분 |
|---|---|---|
| 1 | 보드·좌표계·대칭 변환 | 기본 |
| 2 | 기본 Renju 진행 | **1. 기본 Renju** |
| 3 | 패턴 정의 (FIVE … LEGAL FOUR-THREE) | 기본 / 금수 |
| 4 | 흑 금수 판정 알고리즘 | **2. 흑 금수** |
| 5 | 거짓금수 (false / pseudo forbidden) | **3. 거짓금수** |
| 6 | Exact-five precedence | **4. exact-five precedence** |
| 7 | 종국·무승부·패스·불법수 처리 | 기본 |
| 8 | Taraguchi-10 상태 머신 | **5. Taraguchi-10** |
| 9 | Swap 의미론 | **6. swap semantics** |
| 10 | 10개 5수 제안 / 대칭 판정 | **7. ten fifth proposals** |
| 11 | 오목도장 호환성 — 미검증 항목 요약 | **8. 미검증 항목** |
| 12 | 규칙 테스트 요구사항 연결 | |

---

## 1. 보드, 좌표계, 대칭

### 1.1 보드
- 15×15 교차점. 각 점의 상태 ∈ {EMPTY, BLACK, WHITE}.
- 보드 밖은 `WALL`(어떤 색의 돌로도 취급하지 않음, 연속을 끊는다).

### 1.2 좌표계 (프로젝트 전체 유일 좌표계)

**이 규약 외의 좌표 표현(x/y, (col,row) 순서 등)은 코드·문서·데이터 어디에도 사용하지 않는다.**

| 이름 | 정의 | 범위 |
|---|---|---|
| `row` | 아래쪽(1단)부터 0 | 0..14 |
| `col` | 왼쪽(A열)부터 0 | 0..14 |
| `Point` (index) | `row * 15 + col` | 0..224 |
| 표기 (notation) | 열 문자 `A..O` (I 포함, 건너뛰지 않음) + 단 번호 `1..15` | 예: `H8` |

- `H8` = (row 7, col 7) = index 112 = 천원(天元, 중앙).
- `A1` = 좌하단 = index 0. `O15` = 우상단 = index 224.
- 텍스트 보드 출력 시 15단이 맨 위에 오도록 출력한다(바둑판 기보 관례).
- 함수 시그니처는 항상 `(row, col)` 순서. `Point` 와 `(row,col)` 상호 변환은 `cpp/board/coord.h` 한 곳에서만 정의한다.
- 패스 액션은 Point가 아니다(§7.3).

### 1.3 중앙 N×N 영역
`Central(N)` (N 홀수) = { (r,c) : |r−7| ≤ (N−1)/2 ∧ |c−7| ≤ (N−1)/2 }.
경계 포함(inclusive). `Central(1)={H8}`, `Central(3)`은 G7..J9 9점, … `Central(9)`는 D4..L12 81점.
(경계 포함 해석은 `C-20`)

### 1.4 방향 (4개)
| id | 이름 | (Δrow, Δcol) |
|---|---|---|
| 0 | HORIZONTAL | (0, +1) |
| 1 | VERTICAL | (+1, 0) |
| 2 | DIAG (↗) | (+1, +1) |
| 3 | ANTI_DIAG (↖) | (+1, −1) |

### 1.5 D4 대칭 변환
중심 H8 기준 상대좌표 `(dr, dc) = (row−7, col−7)`. 8개 변환 `T0..T7`의 ID와 순서는 고정(재현성):

| id | 이름 | (dr, dc) → |
|---|---|---|
| 0 | identity | (dr, dc) |
| 1 | rot90 (반시계) | (dc, −dr) |
| 2 | rot180 | (−dr, −dc) |
| 3 | rot270 | (−dc, dr) |
| 4 | mirror (좌우 반전) | (dr, −dc) |
| 5 | mirror∘rot90 = T1(T4(·)) | (−dc, −dr) |
| 6 | mirror∘rot180 = T2(T4(·)) (상하 반전) | (−dr, dc) |
| 7 | mirror∘rot270 = T3(T4(·)) | (dc, dr) |

모든 변환은 H8을 고정한다. 변환은 보드·Point·action·마스크·제안 집합·opening region에 **동일하게** 적용되어야 한다(TRAINING.md §Augmentation).
Phase 2 테스트: 군 공리(합성 닫힘, 역원, `T1^4 = T0`, `T4^2 = T0`)를 전수 검증.

---

## 2. 기본 Renju 진행 `[RIF]`

1. 흑이 먼저 두고, 흑·백이 번갈아 빈 점에 돌 하나씩 둔다. (opening 단계는 §8이 이 규칙을 대체/확장한다.)
2. 놓인 돌은 움직이거나 제거하지 않는다.
3. **흑 승리**: 흑이 **정확히 5개**의 연속(§3.2 FIVE)을 만들면 즉시 승리. 동시에 금수 모양이 생겨도 승리(§6).
4. **백 승리**: 백이 5개 이상 연속(FIVE 또는 OVERLINE)을 만들면 즉시 승리. 백에게 금수는 없다.
5. **흑 금수**: 흑은 5를 동시에 만들지 않는 한 OVERLINE, DOUBLE FOUR, (진짜) DOUBLE THREE를 만드는 착수를 할 수 없다(§4). 금수 착수의 처리 방식은 profile 설정(§7.4).
6. **무승부**: 양측 연속 패스(RIF Art. 10) 또는 둘 곳이 없는 경우(§7.2).

---

## 3. 패턴 정의

### 3.0 공통 표기
- `c` ∈ {BLACK, WHITE}. 착수점 `p`, 방향 `d`.
- `L(p,d)` : `p`를 지나는 방향 `d`의 직선(보드 안쪽 점들의 수열).
- **run**: 같은 색 돌이 빈칸·상대 돌·WALL 없이 이어진 최대 연속 구간.
- 판정은 항상 "가상의 착수 후 보드" 기준이다. 예: "흑이 p에 두면 FOUR가 생기는가"는
  `board + {p:BLACK}` 에서 판정한다.
- 아래 정의에서 "five를 만든다"는 `c`별로 다르다:
  - BLACK: 착수 후 그 돌을 포함하는 run 길이가 **정확히 5**.
  - WHITE: 착수 후 run 길이 **≥ 5**.

### 3.1 FIVE
보드 상태 `B`에서 색 `c`의 길이 **정확히 5**인 run.
- 흑의 승리 조건은 FIVE만 해당한다.

### 3.2 OVERLINE (장목)
길이 **≥ 6** 인 run.
- 흑: 금수(§4), 단 동시에 다른 방향에 FIVE가 생기면 승리(§6).
- 백: 승리(5목 이상이면 승리).

### 3.3 FOUR
`[RIF Art.3]` "A row with four stones to which you can add one more stone to attain five in a row."

형식 정의 (착수점 `p`를 포함하는 four만 금수 판정에 쓰인다):
- 방향 `d`의 직선 위에서, 색 `c`의 돌 집합 `S` (|S|=4, `p ∈ S`)가 **FOUR**라는 것은
  어떤 빈 점 `e ∈ L(p,d)`가 존재하여 `S ∪ {e}`가 연속 5칸을 채우고, `e`에 `c`를 두면
  `c` 기준 "five를 만든다"(흑: 정확히 5, 백: ≥5)는 뜻이다. `e`를 그 four의 **completion point**라 한다.
- 흑의 경우 completion 결과가 장목이 되는 `e`는 completion point가 **아니다**
  (예: `B B B B . B` 에서 `.` 은 장목이므로 four가 아니다).
- **four의 개수 세기**: `p`를 포함하는 서로 다른 집합 `S`의 개수. 같은 `S`가 completion point 2개를 가지면 1개의 four(= STRAIGHT FOUR)다.
  - `. B B B B .` → S 하나, completion 2개 → four 1개 (straight four)
  - `B . B B B . B` (p가 가운데 셋 중 하나) → S가 2개 → **한 직선 안의 double four** `[RIF]`

### 3.4 STRAIGHT FOUR (열린 사)
`[RIF Art.3]` 두 가지 다른 방법으로 five를 만들 수 있는 four.
- 형식: completion point가 2개인 FOUR. 흑의 경우 두 completion 모두 **정확한 5**여야 한다.
- 구조적으로 항상 `. c c c c .` 형태(연속 4 + 양끝 빈칸)이며, 흑은 추가로 양쪽 빈칸 바깥이
  흑돌이 아니어야 한다(그렇지 않으면 그쪽 completion이 장목).

### 3.5 THREE (진짜 삼)
`[RIF Art.3 + Art.9.3]` 흑 기준 정의 (금수 판정에는 흑의 three만 필요).

방향 `d`에서 착수점 `p`를 포함하는 **THREE**가 존재한다 ⇔ 다음을 모두 만족하는 빈 점 `e ∈ L(p,d)` (**extension point**)가 하나 이상 존재:

1. `board + {p, e}` 에서 `p`와 `e`를 포함하는 방향 `d`의 **STRAIGHT FOUR**가 생긴다 (§3.4, 흑 기준).
2. `e`에 두는 순간 FIVE가 만들어지지 **않는다** ("without at the same time a five in a row is made").
3. `e`는 `board + {p}` 상태에서 **흑에게 합법인 착수**이다. 즉 `e`는
   (a) OVERLINE이 아니고, (b) DOUBLE FOUR가 아니고, (c) **금수인 DOUBLE THREE가 아니다** — (c)는 이 정의를 **재귀적으로** 적용해 판정한다 `[RIF Art.9.3(a)(b)]`.

조건 3을 빼고 1·2만 만족하는 것을 **APPARENT THREE (겉보기 삼)** 라 한다. 엔진은 디버그 출력에서 둘을 구분한다.

방향 우선순위 `[INTERP, A-R3]`: 한 방향 `d`에서 `p`가 이미 FIVE/OVERLINE/FOUR를 만들면 그 방향은 THREE로 세지 않는다(그 방향은 4개 이상의 돌을 포함하므로 RIF의 "row with three stones"가 아님).
방향당 THREE는 0 또는 1개로 센다 `[INTERP, A-R4]`. Phase 2의 1차원 전수 열거(착수점 ±5칸, 354,051 패턴)에서
"한 직선에 서로 다른 THREE 둘" 및 "같은 직선에 FOUR와 겉보기 THREE 동시" 패턴은 모두 **0건**이었다(A-R3/A-R4 VERIFIED).

예 (1차원, `|`=WALL, 흑만 표시, 다른 칸은 빈칸):
- `. . B B B . .` (양쪽 2칸 이상 열림) → THREE (연속 삼)
- `. B . B B . .` → THREE (띈 삼, extension = 가운데 빈칸)
- `| B B B . .` (가장자리에 붙음) → **THREE 아님**: 유일한 4 연속 `BBBB.`은 completion이 한쪽뿐
- `| . B B B . .` → THREE (extension `|.BBBB.` 의 오른쪽 점이 straight four를 만듦)
- `B . . B B B . . B` → **THREE 아님**: 어느 쪽으로 늘려도 반대쪽 completion이 장목이 되어 straight four가 불가 (§5.2)

### 3.6 DOUBLE FOUR (4-4, 사사)
흑이 `p`에 두어 `p`를 포함하는 FOUR가 **2개 이상** 생기는 착수 (방향이 같아도 됨, §3.3).
- 금수, 단 §6 예외.
- 4-4 판정에는 재귀가 없다: completion(=five)은 항상 합법이기 때문이다(§6).

### 3.7 DOUBLE THREE (3-3, 삼삼)
흑이 `p`에 두어 `p`를 포함하는 **진짜 THREE**(§3.5)가 **서로 다른 2개 이상의 방향**에서 생기는 착수.
- 금수, 단 §6 예외.
- APPARENT THREE가 2개 이상이지만 진짜 THREE가 1개 이하이면 **금수가 아니다** → 거짓금수(§5).

### 3.8 FALSE FORBIDDEN (거짓금수 / pseudo forbidden)
겉보기에는 3-3(또는 4-3-3 등)이지만, 하나 이상의 three의 모든 extension point가
OVERLINE / DOUBLE FOUR / (재귀적으로) 금수 DOUBLE THREE 이거나, 가장자리·장목 구조 때문에
straight four가 될 수 없어서 진짜 THREE가 1개 이하가 되는 착수. **합법수**다. 자세한 것은 §5.

### 3.9 LEGAL FOUR-THREE (4-3)
`p`를 포함하는 FOUR가 정확히 1개, 진짜 THREE가 정확히 1개(FOUR와 다른 방향)이고 OVERLINE이 없는 착수. **합법**이며 렌주에서 흑의 주된 승리 수단.
- 4-3-3 (진짜 THREE 2개 이상) → 3-3 금수. 4-4-3 → 4-4 금수.

---

## 4. 흑 금수 판정 알고리즘 (규범적 의사코드)

흑의 착수 `p`(빈 점)에 대한 판정 결과:

```
enum class MoveVerdict { WIN_FIVE, LEGAL, FORBIDDEN_OVERLINE,
                         FORBIDDEN_DOUBLE_FOUR, FORBIDDEN_DOUBLE_THREE };

verdict(B, p):                      # B: 현재 보드, p: 빈 점, 흑 착수
    B' = B + {p: BLACK}
    if any direction d: run_len(B', p, d) == 5:      return WIN_FIVE   # §6 precedence
    if any direction d: run_len(B', p, d) >= 6:      return FORBIDDEN_OVERLINE
    if count_fours(B', p) >= 2:                       return FORBIDDEN_DOUBLE_FOUR
    if count_real_threes(B', p) >= 2:                 return FORBIDDEN_DOUBLE_THREE
    return LEGAL

count_real_threes(B', p):
    n = 0
    for d in 4 directions:
        if direction_has_five_or_overline_or_four(B', p, d): continue    # A-R3
        for e in candidate_extensions(B', p, d):        # 빈 점, p로부터 |offset| <= 4
            B'' = B' + {e: BLACK}
            if makes_five(B'', e): continue             # 조건 2
            if not straight_four_through(B'', p, e, d): continue   # 조건 1
            if verdict(B', e) in {LEGAL}:               # 조건 3 (재귀). WIN_FIVE는 조건 2에서 이미 배제
                n += 1; break                           # 방향당 1개 (A-R4)
    return n
```

규범 사항:
- 재귀는 **깊이 제한 없이** 수행한다(MUST). 각 재귀 단계마다 흑돌이 하나 추가되므로 반드시 종료한다.
  실측 최대 재귀 깊이와 판정 시간은 벤치마크로 기록한다.
- 고정 패턴 테이블·정규식만으로 판정을 끝내면 안 된다(MUST NOT). 테이블은 "후보 탐지"
  가속에만 쓸 수 있으며, 그 테이블은 이 정의로부터 **기계적으로 생성**되어야 하고(손으로 작성 금지),
  extension point의 합법성 판정은 항상 실제 보드에서 재귀로 한다.
- 3-3 판정 시 `e`의 합법성은 `B'` (= `p`를 이미 둔 보드)에서 판정한다 `[RIF 9.3: "making the move mentally"]`.
- 백 착수에는 금수 판정을 적용하지 않는다. 백의 FOUR/THREE는 전술 엔진용 정의(§3.3과 동일하되 five 조건이 ≥5)만 쓴다.

### 4.1 ForbiddenInfo (설명 가능성)
`forbidden(p)`는 판정과 함께 다음을 반환한다(MUST):

```
ForbiddenInfo {
  verdict: MoveVerdict
  directions: [Direction]                    # 판정에 관여한 방향
  generated_fours:  [{dir, stones[4], completion_points[]}]
  generated_threes: [{dir, stones[3], extension_points[],
                      extension_verdicts[{point, verdict, reason}], is_real}]
  recursive_validation_path: tree of (point, verdict, depth)
}
```

출력 예 (진짜 금수):
```
BLACK H8: FORBIDDEN_DOUBLE_THREE
  HORIZONTAL: three {F8,G8,H8}; extension E8 -> straight four (E8 legal: LEGAL)  => real
  DIAG:       three {G7,H8,J10}; extension I9 -> straight four (I9 legal: LEGAL) => real
  real threes = 2  => forbidden
```
출력 예 (거짓금수):
```
BLACK H8: LEGAL (apparent DOUBLE_THREE)
  HORIZONTAL: three ... extension E8 -> straight four (LEGAL) => real
  VERTICAL:   three ... extension H6 -> FORBIDDEN_DOUBLE_FOUR; extension H10 -> FORBIDDEN_OVERLINE => not real
  real threes = 1  => legal
```
(위 두 예시는 형식 예시다.) 실제 국면의 출력 — `tests/forbidden/cases/generated_recursive.json`의
`gen-false33-depth2-D4` (흑 E2 C3 B4 E4 D5 E5 C6 E6 I8 F9, 착수 D4), `dojozero_engine forbidden --point D4`:
```
BLACK D4: LEGAL
  HORIZONTAL: three; extension C4 -> straight four (FORBIDDEN_DOUBLE_THREE); => not real
  DIAG: three; extension B2 -> straight four (LEGAL); extension F6 -> straight four (LEGAL); => real
  fours=0 apparent_threes=2 real_threes=1
  recursion path:
      C4 (extends D4) = FORBIDDEN_DOUBLE_THREE
        C5 (extends C4) = LEGAL
        B3 (extends C4) = LEGAL
      B2 (extends D4) = LEGAL
      F6 (extends D4) = LEGAL
```

---

## 5. 거짓금수 (False / Pseudo Forbidden)

진짜 THREE가 되지 못하는 이유를 다음 4종으로 분류하고, 각 종류별 테스트 카테고리를 둔다.

| 코드 | 이유 | 예시 구조 | 근거 |
|---|---|---|---|
| FF-EDGE | 보드 가장자리 때문에 straight four 불가 | `\|BBB..` | §3.4 정의 |
| FF-OVERLINE-SHAPE | straight four의 한쪽 completion이 장목 → straight four 아님 | `B..BBB..B` | §3.4 정의 |
| FF-BLOCKED | 상대 돌로 막혀 straight four 불가 | `WBBB.W` 등 | §3.4 정의 |
| FF-EXT-OVERLINE | extension point 착수가 OVERLINE 금수 | | RIF 9.3(a) |
| FF-EXT-44 | extension point 착수가 DOUBLE FOUR 금수 | | RIF 9.3(a) |
| FF-EXT-33 | extension point 착수가 (진짜) DOUBLE THREE 금수 — **재귀** | | RIF 9.3(b) |
| FF-EXT-33-FALSE | extension point의 3-3이 다시 거짓금수라서 extension이 합법 → three는 진짜 (**2단 재귀**) | | RIF 9.3(b) |

### 5.1 핵심 원칙
1. 3-3이 금수인지는 "각 three가 **합법적인 수로** straight four가 될 수 있는가"로 결정된다.
2. 합법성 판정 자체가 다시 3-3 판정을 포함하므로 재귀다.
3. 거짓금수의 판정 결과는 **LEGAL** 이다. AI·legal move mask·UI 모두 착수 가능으로 취급한다.

### 5.2 1차원 예시 검증 (FF-OVERLINE-SHAPE)
라인 `B . . B B B . . B` (index 0..8, 흑 = 0,3,4,5,8):
- extension 2 → `BBBB`(2..5): completion 1 → 0..5 = 장목 ✗, completion 6 → 2..6 정확히 5 ✓ → completion 1개 → straight four 아님
- extension 6 → `BBBB`(3..6): completion 2 → 2..6 ✓, completion 7 → 3..8 = 장목 ✗ → straight four 아님
- 다른 extension은 4 연속을 만들지 못함
- ⇒ 이 라인의 3,4,5는 **THREE가 아니다**.

### 5.3 테스트 국면의 출처 (순환 검증 방지)
"엔진이 생성한 국면을 엔진으로 검증"하는 것은 구현 간 일관성만 확인할 뿐 RIF 의미 정확성을 보장하지 못한다. 따라서 거짓금수 테스트는 반드시 다음 **독립 출처**를 포함한다:
1. 사람이 손으로 구성하고 이 문서 정의로 단계별 검토한 국면 (각 테스트에 검토 사유 주석)
2. 공개 렌주 교재/문제집에서 금수·거짓금수로 명시된 국면 (출처·라이선스 기록, 국면 좌표만 사용)
3. 타 엔진과의 차등 비교(differential) 결과 중 불일치 국면을 사람이 판정한 것
4. 오목도장 앱에서 직접 확인한 국면 (`tests/compatibility/omokdojang/`)

---

## 6. Exact-Five Precedence

`[RIF Art.9.2]` "The game is won for white if black **without at the same time attaining five in a row**: makes an overline / double-four / double-three". 따라서:

| 흑 착수 결과 | 판정 | 근거 |
|---|---|---|
| 정확한 FIVE + DOUBLE FOUR (다른 방향 4) | **흑 승** | RIF 9.2 |
| 정확한 FIVE + DOUBLE THREE | **흑 승** | RIF 9.2 |
| 정확한 FIVE (한 방향) + OVERLINE (다른 방향) | **흑 승** | RIF 9.2 문언 그대로 `[INTERP, A-R1]` |
| OVERLINE만 (어느 방향에도 정확한 5 없음) | 금수 | RIF 9.2 |
| 백: FIVE 또는 OVERLINE | 백 승 | RIF |

- 엔진은 `verdict()`에서 FIVE 검사를 **가장 먼저** 수행한다(§4). 
- "FIVE + OVERLINE 동시" 처리는 앱마다 다르게 구현될 가능성이 커서 `C-02`로 별도 검증한다.
- RIF Art.9.7(사람이 5를 알아채지 못한 경우 관련 조항)은 인간 대국 절차 규정이므로 엔진은 FIVE를 자동 감지해 즉시 종국한다 `[INTERP, A-R2]`.

---

## 7. 종국, 무승부, 패스, 불법수

### 7.1 종국 판정 순서 (매 착수 후)
1. 착수자가 five 조건을 만족 → 착수자 승.
2. (흑 착수이고 금수, `illegal_move_behavior = LOSS`인 경우) → 백 승. `REJECT`면 착수 자체가 거부되어 이 단계에 도달하지 않는다.
3. 빈 점이 없음 → 무승부.
4. 그 외 → 진행.

### 7.2 둘 곳이 없는 경우
- 빈 점이 0개: 무승부.
- 흑 차례인데 모든 빈 점이 금수이고 `pass_allowed = false`: **무승부로 처리** `[INTERP, A-R5]`, `UNVERIFIED_APP_BEHAVIOR (C-06)`.

### 7.3 패스
- `[RIF Art.5, 10]` 패스 허용, 양측 연속 패스 시 무승부. `[RIF 12.9]` 첫 3수 이내 패스 불가.
- Taraguchi-10 opening 단계(§8, `NORMAL_RENJU` 이전)에서는 패스 불가 `[INTERP, A-R6]`.
- 오목도장 앱의 패스 가능 여부: `UNVERIFIED_APP_BEHAVIOR (C-06)` → profile 값 `pass_allowed`.
- 패스는 별도 action(`PASS`, NN action index 225)로 표현한다.
- `RENJU_GOMOCUP` profile = `RENJU_RIF`와 동일하되 `pass_allowed = false`. Gomocup 프로토콜에 패스가 없으므로 **외부 엔진 평가 전용**(PRD FR-21/22). 금수·승리 규칙은 RIF 그대로이며, 둘 곳 없음은 §7.2에 따라 무승부.

### 7.4 불법수 처리 (`illegal_move_behavior`)
| 값 | 의미 |
|---|---|
| `REJECT` (기본) | `apply()`가 false 반환, 상태 불변. 금수점은 `legal_actions()`에 포함되지 않는다. |
| `LOSS` | 흑 금수 착수를 허용하고 즉시 백 승으로 종국 (RIF 원 규칙 재현용) |

- 빈 점이 아닌 곳, 보드 밖, opening 영역 밖, 현재 phase에 맞지 않는 action은 profile과 무관하게 항상 `REJECT`.
- AI는 어떤 profile에서도 금수를 선택하지 않는다(legal mask). 두 값은 AI 관점에서 동치(금수 = 즉시 패)이며 차이는 외부 입력 재생·검증에만 있다.
- 오목도장 앱이 금수점 착수를 막는지(REJECT) 반칙패 처리하는지(LOSS): `UNVERIFIED_APP_BEHAVIOR (C-03)`.

---

## 8. Taraguchi-10 Opening

### 8.1 공식 문구 요약 `[RIF-T10]`
1. 흑이 1수를 중앙(H8)에 둔다. 상대는 swap할 수 있다.
2. 백이 2수를 중앙 3×3 안에 둔다. 상대는 swap할 수 있다.
3. 흑이 3수를 중앙 5×5 안에 둔다. 상대는 swap할 수 있다.
4. 백이 4수를 중앙 7×7 안에 둔다.
5. 4수 이후 두 가지 선택:
   - **Choice 1**: 상대는 swap할 수 있다. 흑이 5수를 중앙 9×9 안에 둔다. 상대는 swap할 수 있다. 백이 6수를 보드 어디든 둔다.
   - **Choice 2**: 흑이 서로 대칭이 아닌 5수 10개를 **보드 어디든** 둔다. 백이 그중 하나를 고르고 6수를 보드 어디든 둔다.
6. 이후 Renju 규칙으로 진행.

공식 문구가 **명시하지 않는 것** (모두 `[INTERP]` + `UNVERIFIED_APP_BEHAVIOR`):
- Choice 1/2를 **누가** 고르는가 → 4수 직후 흑을 가진 플레이어 `[A-T1, C-07]` (RenjuNet 포럼의 Taraguchi 해설: "Black player decides")
- Choice 1의 첫 swap을 누가 결정하는가 → "직전 착수자(4수의 백)의 상대" = 흑을 가진 플레이어 `[A-T2, C-08]`
- "대칭"의 정확한 정의 → §10 `[A-T3, C-10]`
- 5×5 등 영역 경계 포함 여부 → 포함 `[C-20]`

### 8.2 Player / Color 분리 (MUST)
```
enum class PlayerId   : uint8_t { P0, P1 };          // 실제 대국자 (사람/엔진 인스턴스)
enum class StoneColor : uint8_t { BLACK, WHITE };
struct SideAssignment { PlayerId black_holder; PlayerId white_holder; };  // 항상 서로 다름
enum class Role : uint8_t { BLACK_HOLDER, WHITE_HOLDER };  // phase가 지정하는 행위자 역할
```
- 각 phase는 **역할(Role)** 로 행위자를 지정하고, `SideAssignment`로 실제 `PlayerId`가 결정된다.
- 게임 시작 시 `SideAssignment`(임시 흑 = tentative black)는 외부 입력이다(추첨 방식은 범위 밖).
- 승패·보상·가치(value)는 **PlayerId 기준**으로 기록한다. 색 기준 값은 파생값이다.
- `side_to_move`(색)는 착수 phase에서만 의미가 있다. 결정(decision) phase에서는 `actor()`가 행위자다.

### 8.3 상태 머신 (규범)

| Phase | 행위자 Role | 합법 action | 다음 Phase |
|---|---|---|---|
| `BLACK_MOVE_1` | BLACK_HOLDER | `PLACE(H8)` (유일) | `SWAP_AFTER_1` |
| `SWAP_AFTER_1` | WHITE_HOLDER | `SWAP`, `KEEP_COLOR` | `WHITE_MOVE_2` |
| `WHITE_MOVE_2` | WHITE_HOLDER | `PLACE(p)`, p ∈ Central(3), 빈 점 | `SWAP_AFTER_2` |
| `SWAP_AFTER_2` | BLACK_HOLDER | `SWAP`, `KEEP_COLOR` | `BLACK_MOVE_3` |
| `BLACK_MOVE_3` | BLACK_HOLDER | `PLACE(p)`, p ∈ Central(5), 빈 점 | `SWAP_AFTER_3` |
| `SWAP_AFTER_3` | WHITE_HOLDER | `SWAP`, `KEEP_COLOR` | `WHITE_MOVE_4` |
| `WHITE_MOVE_4` | WHITE_HOLDER | `PLACE(p)`, p ∈ Central(7), 빈 점 | `CHOICE_AFTER_4` |
| `CHOICE_AFTER_4` | BLACK_HOLDER `[A-T1]` | `CHOOSE_DIRECT_FIFTH`, `CHOOSE_TEN_PROPOSALS` | `SWAP_BEFORE_DIRECT_5` / `OFFER_TEN_5TH_MOVES` |
| `SWAP_BEFORE_DIRECT_5` | BLACK_HOLDER `[A-T2]` (profile로 WHITE_HOLDER 선택 가능) | `SWAP`, `KEEP_COLOR` | `BLACK_DIRECT_MOVE_5` |
| `BLACK_DIRECT_MOVE_5` | BLACK_HOLDER | `PLACE(p)`, p ∈ Central(9), 빈 점, 흑 합법 | `SWAP_AFTER_DIRECT_5` |
| `SWAP_AFTER_DIRECT_5` | WHITE_HOLDER | `SWAP`, `KEEP_COLOR` | `WHITE_MOVE_6` |
| `OFFER_TEN_5TH_MOVES` | BLACK_HOLDER | `PROPOSE_FIFTH(p)` ×10 (부분 제안 k=0..9), 또는 원자적 `OFFER_TEN(p[10])` | (10개 완료 시) `SELECT_ONE_OF_TEN` |
| `SELECT_ONE_OF_TEN` | WHITE_HOLDER | `SELECT_PROPOSED_FIFTH(p)`, p ∈ offered | `WHITE_MOVE_6` |
| `WHITE_MOVE_6` | WHITE_HOLDER | `PLACE(p)`, 보드 어디든 빈 점 | `NORMAL_RENJU` |
| `NORMAL_RENJU` | side_to_move의 holder | `PLACE(p)` (흑은 금수 제외), `PASS`(profile) | 종국까지 |

추가 규범:
- `SELECT_PROPOSED_FIFTH(p)`는 `p`에 흑돌을 놓는 효과(= 5수)를 가지며, 선택되지 않은 9개 제안은 제거된다. 이 action 이후 6수 전 swap은 없다.
- `OFFER_TEN` 단계 진입 후에는 색 배정이 확정된다(이후 swap 없음).
- `NORMAL_RENJU` 진입 시 다음 착수는 흑(7수).
- Opening 동안 흑의 금수 판정은 항상 적용한다(5수 시점까지 흑돌 ≤3이므로 실제로는 발생 불가하나, 검사를 생략하지 않는다).
- 각 phase 전이는 `undo()`로 정확히 되돌릴 수 있어야 한다(MUST): phase, SideAssignment, offered set, history 모두 복원.
- 영역 크기(3/5/7/9)와 제안 개수(10)는 `Taraguchi10Profile` 파라미터다(Taraguchi-N, Super Taraguchi 변형 확장 가능, 단 기본값만 테스트 보장).
- Malformed state 거부: phase와 보드 돌 수/색이 불일치하거나, 영역 위반 돌, offered 집합 크기 오류, SideAssignment 중복 등은 상태 로드 시 거부(MUST).

### 8.4 단순 Renju 모드
`opening_rule = NONE` 이면 `NORMAL_RENJU`에서 빈 보드로 시작한다(흑 선, swap 없음). `opening_rule = FIXED_POSITION` 이면 주어진 국면/색배정에서 `NORMAL_RENJU`로 시작한다(학습 커리큘럼, 평가용 balanced opening).

---

## 9. Swap 의미론

1. `SWAP` = 두 플레이어의 **색 배정을 교환**한다: `black_holder ↔ white_holder`. 보드의 돌·수순은 변하지 않는다.
2. `KEEP_COLOR` = 배정 유지.
3. Swap 결정권자 일반 규칙: **색 C로 k수가 놓인 직후, 색 ¬C를 가진 플레이어**가 결정한다(§8.3 표). 예외: `SWAP_BEFORE_DIRECT_5`(§8.1, A-T2).
4. Swap 이후 다음 착수는 phase가 요구하는 **색**의 holder가 한다. 예: 1수 후 P1(백 holder)이 SWAP → P1이 흑을 갖고, P0가 백으로 2수를 둔다.
5. Swap 결정은 "어느 색을 가질 것인가"의 선택이다. 따라서 결정자의 가치는
   `max(V_black(s), V_white(s))` 이며, `V_color(s)`는 swap 직후 국면 `s`(같은 보드, 다음 phase)에서 해당 색을 가질 때의 기대 결과다. AI 탐색은 swap 노드를 이 의미로 평가한다(ARCHITECTURE.md §Search actor-aware backup).
6. 기록: 모든 swap/keep 결정은 history에 action으로 남는다(재생 가능성).

---

## 10. 10개 5수 제안 (Ten Fifth Proposals)

### 10.1 유효성 조건 (`ProposalValidator`)
제안 집합 `Q = {q1..q10}`이 유효하려면:
1. |Q| = 10, 원소가 서로 다름.
2. 각 `qi`는 현재 보드(4수 국면)에서 빈 점.
3. 각 `qi`는 흑에게 합법인 착수(현재 국면에서는 항상 참이지만 검사함).
4. 영역 제한 없음 — 보드 어디든 `[RIF-T10]`. (앱이 영역을 제한하는지: `C-09`)
5. 어떤 두 원소도 서로 **대칭이 아니다** (§10.2).

부분 제안(autoregressive 생성 중)은 1~3, 5를 지금까지의 원소에 대해 만족해야 하며, 남은 슬롯을 채울 수 있는 비대칭 후보가 충분해야 한다(15×15에서는 항상 충족; 검사는 함).

### 10.2 대칭(symmetry) 정의 `[A-T3]`
4수 국면 `P` (흑 B = {H8, b3}, 백 W = {w2, w4}).

**정의 E1 (위치 동치, 기본값)**: `q1 ~ q2` ⇔ ∃ g ∈ D4 : g(B ∪ {q1}) = B ∪ {q2} ∧ g(W) = W.
즉 두 제안이 만드는 5돌 국면이 D4 변환으로 같은 국면이면 대칭.

**정의 E2 (stabilizer orbit)**: `q1 ~ q2` ⇔ ∃ g ∈ Stab(P) : g(q1) = q2, Stab(P) = { g ∈ D4 : g(B)=B ∧ g(W)=W }.

**정리**: Taraguchi-10 4수 국면에서 E1 = E2.
증명: D4의 모든 원소는 H8을 고정하므로 g(B∪{q1}) = B∪{q2} 이면 g({b3,q1}) = {b3,q2}.
(i) g(b3)=b3 이면 g(q1)=q2 이고 g(B)=B, g(W)=W → g ∈ Stab(P).
(ii) g(b3)=q2, g(q1)=b3 인 경우: g가 대합(involution: 반사 또는 rot180)이면 q1 = g⁻¹(b3) = g(b3) = q2 로 서로 다른 제안이 아니다. g가 rot90/rot270이면 g(W)=W 와 |W|=2, W∌H8 에서 g(w2)=w4, g(w4)=w2 → g²(w2)=w2 → rot180이 w2를 고정 → w2=H8 모순. ∎
기계 검증: 도달 가능한 4수 국면 8,464개(2수 8 × 3수 23 × 4수 46)의 모든 빈 점에서 E1 정규형 = E2 정규형
(`tests/opening/test_symmetry.cpp`, Release 전수).

**정의 E3 (점 대칭, 대안)**: `q1 ~ q2` ⇔ ∃ g ∈ D4 \ {id} : g(q1) = q2 (국면 무시). 더 많은 쌍을 대칭으로 거부하는 보수적 해석. 앱이 이 방식을 쓸 가능성을 배제할 수 없어 profile 옵션으로 둔다.

profile: `proposal_symmetry_mode ∈ {POSITION (E1, 기본), POINTWISE_D4 (E3)}`. 앱 실제 동작: `UNVERIFIED_APP_BEHAVIOR (C-10)`.

### 10.3 정규형
- 각 제안 `q`의 canonical id = 그 orbit(정의에 따른 동치류)에서 Point index가 최소인 원소.
- 대칭 판정 모듈은 `cpp/opening/symmetry.{h,cpp}` 단일 구현. Python은 바인딩으로만 사용.

### 10.4 선택
- 백 holder는 Q 중 하나를 `SELECT_PROPOSED_FIFTH(q)`로 고른다. `q ∉ Q`는 REJECT.
- 이론적 성질(설계 근거, 규칙 아님): 정확한 가치 함수 V_black(P+q)가 주어지면 흑의 최적 제안은 비대칭 제약 하에서 V_black 상위 10개 orbit이다(백은 최소값을 고르므로 흑은 10번째로 좋은 값을 최대화). TRAINING.md §Opening 참조.

---

## 11. 오목도장 호환성 — 미검증 항목 요약

`OMOKDOJANG_DAN` profile은 "RIF Renju + Taraguchi-10" 이라는 **가정** 위에 있다. 다음은 공개 자료로 확정할 수 없어 전부 `UNVERIFIED_APP_BEHAVIOR` 이다. 상세 및 검증 절차는 ASSUMPTIONS.md Compatibility Matrix.

| ID | 항목 | 엔진 기본값 |
|---|---|---|
| C-01 | 정확한 5 + 4-4/3-3 동시 → 흑 승 처리 여부 | 흑 승 |
| C-02 | 정확한 5 + 다른 방향 장목 동시 | 흑 승 |
| C-03 | 금수점 클릭 시 착수 거부 vs 반칙패 | REJECT |
| C-04 | 재귀적 거짓금수(2단 이상)까지 정확히 구현하는지 | 완전 재귀 |
| C-05 | 한 직선 안 4-4 금수 인정 | 금수 |
| C-06 | 패스 허용 / 둘 곳 없을 때 처리 | pass 불가, 무승부 |
| C-07 | Choice 1/2 결정자 | 4수 후 흑 holder |
| C-08 | Choice 1 첫 swap 결정자 | 흑 holder |
| C-09 | 10개 제안 영역 제한 | 없음(보드 전체) |
| C-10 | "대칭" 정의 및 대칭 제안 처리(거부/자동 제외) | E1, 거부 |
| C-11 | 유단자 대국이 실제로 Taraguchi-10인지(다른 opening rule 가능성) | Taraguchi-10 |
| C-12 | 1수 천원 자동 착수 여부 | 엔진은 action으로 표현 |
| C-13 | 6수 후 추가 swap 없음 | 없음 |
| C-14 | 백 장목 승리 | 승리 |
| C-15 | 판 꽉 참 → 무승부 | 무승부 |
| C-20 | 중앙 N×N 경계 포함 | 포함 |

---

## 12. 규칙 테스트 요구사항과의 연결

테스트 카테고리(ARCHITECTURE.md §Tests, EVALUATION.md §Rule correctness)와 이 문서 절의 대응:

| 카테고리 | 절 |
|---|---|
| 4방향 / five / overline / 백·흑 비대칭 | §3.1, §3.2, §6 |
| simple / straight / broken four, 한 직선 4-4 | §3.3, §3.4, §3.6 |
| simple / broken three, 가장자리 | §3.5 |
| double-three, false double-three, 재귀 거짓금수 | §3.7, §5 |
| exact-five + apparent forbidden | §6 |
| opening 영역, swap 전이, 10제안 검증, D4 | §1.5, §8, §9, §10 |
| malformed state 거부 | §8.3 |

AI 학습 착수 조건: 위 모든 카테고리 **통과율 100%** + 참조 구현(oracle)과의 차등 퍼징 불일치 0건.

---

### 참고 출처
- RIF Renju Rules — https://www.renju.net/rifrules/
- Taraguchi-10 Rule (RenjuNet) — https://www.renju.net/rule/25/
- An introduction of Taraguchi rule (RenjuNet forum) — https://www.renju.net/forum/thread/951/
- Taraguchi opening rule (Wikipedia) — https://en.wikipedia.org/wiki/Taraguchi_opening_rule
- Renju Caffe rules — https://www.renjucaffe.com/rules/
