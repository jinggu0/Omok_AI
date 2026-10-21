# GUI에서 DojoZero와 두기 (pbrain-DojoZero.exe)

`pbrain-DojoZero.exe`는 Gomocup/Piskvork 프로토콜 엔진 실행 파일이다. 실제 엔진은 Python 호스트
(`python -m dojozero.protocol.pbrain`, 코덱 = C++ `EngineSession`)이고, exe는 그 프로세스를 띄우고
stdin/stdout을 그대로 넘겨주는 런처다(ADR-08의 임시 형태: C++ 추론 exe는 아직 결정 전).
소스: `cpp/cli/pbrain_launcher.cpp`, `python/dojozero/protocol/pbrain.py`.

## 1. 바로 두기: Yixin Board (준비된 폴더)

이 PC에는 바로 쓸 수 있는 폴더가 준비되어 있다(2026-10-21, 실제 Yixin Board 창에서 엔진 대 엔진
한 판을 끝까지 두어 확인).

| 경로 | 내용 |
|---|---|
| `C:\dojozero_gui\YixinBoard-DojoZero\` | Rapfi-YixinBoard 복사본. `engine.exe` = DojoZero 런처, `pbrain-DojoZero.ini` |
| `C:\dojozero_gui\engine\` | DojoZero 코드 **고정 스냅샷**(커밋은 `SNAPSHOT.txt`) + 그 코드로 빌드한 `_core`. 작업 트리가 개발 중이어도 GUI 엔진은 바뀌지 않는다 |
| `C:\Users\jingg\Downloads\Rapfi-YixinBoard\` | 원본(Rapfi). **건드리지 않았다** |

순서:
1. 무거운 작업(학습, 벤치마크, 다른 엔진)을 끈다. 기동 시간과 수당 시간이 부하에 크게 좌우된다(4절).
2. `C:\dojozero_gui\YixinBoard-DojoZero\Yixin.exe`를 실행한다.
3. **10–20초 기다린다.** 그동안 DojoZero가 torch·모델을 올리고 XPU warm-up을 한다. 창은 먼저 뜨지만
   엔진은 아직 준비 중이다. 그 사이에 둔 수는 엔진이 준비된 뒤 처리될 것으로 추정(명령이 입력 버퍼에 쌓임, 사람 클릭으로는 미확인).
4. 기본 설정: **사람 = 흑, DojoZero = 백**, 렌주(free renju), 15×15, 수당 5초, 대국 1시간.
   흑으로 첫 수를 두면 DojoZero가 응수한다.
5. 색·시간 바꾸기: Yixin 메뉴에서 바꿀 수 있다(메뉴 이름은 확인하지 않음). 확실한 방법은 **Yixin을
   닫은 상태에서** 폴더의 `settings.txt`를 메모장으로 고치는 것이다(각 줄의 첫 숫자만 바꾼다):
   - 4번째 줄 `computer play black`, 5번째 줄 `computer play white`: 1 = DojoZero. DojoZero에게 흑을
     맡기면 첫 수는 중앙(5절의 오프닝 가드). 둘 다 1이면 DojoZero끼리 둔다(이렇게 확인함).
   - 6번째 줄 `level` = 1(custom), 7번째 줄 `time limit (turn)` = 수당 **초**(3 이상 권장),
     8번째 줄 `time limit (match)` = 대국 전체 **초**.
   - level 0(unlimited)이면 Yixin이 수당 2,000,000 ms를 보내므로 ini의 `max_move_ms=30000`(30초)이 상한이 된다.
6. `use database moves`(32번째 줄)는 1로 둔다. 0으로 두었을 때 Yixin이 시작 후 착수 요청을 보내지
   않는 것을 관찰했다(원인 미확인).
7. 끝내기: Yixin 창을 닫으면 DojoZero 프로세스도 함께 종료된다(확인함).

주의:
- 규칙은 **free renju**(settings 3번째 줄 `2`)만 쓴다. freestyle/standard는 엔진이 거부한다(ERROR).
  swap 계열(3/5/6)은 확인하지 않았다(UNKNOWN).
- Yixin의 분석·데이터베이스·hash 같은 확장 명령(`yx...`)에는 `UNKNOWN`으로 답한다. 착수에는 영향이
  없지만 Rapfi처럼 분석 정보가 표시되지는 않는다.
- 문제가 생기면 `C:\dojozero_gui\YixinBoard-DojoZero\pbrain-DojoZero.log`(Python 오류)와, settings의
  `record debug log`를 1로 바꾼 뒤 생기는 `debuglog.txt`(Yixin↔엔진 명령)를 본다.

### Yixin Board 방언 처리(`rule_dialect=yixin`)
Yixin Board는 렌주를 `INFO rule 2`로 보낸다(Gomocup 표준에서 2 = 연속 게임). ini에
`rule_dialect=yixin`이 있을 때만 호스트가 이를 `INFO rule 4`(렌주)로 바꿔 세션에 넘긴다. 그 밖의
판정은 그대로 C++ `EngineSession` + 정식 `GameState`(RENJU_GOMOCUP)가 한다. 기본값(gomocup)에서는
`rule 2`를 계속 거부한다. Yixin은 자기 폴더의 `engine.exe`만 실행하므로 런처를 그 이름으로 둔다.
좌표: Yixin의 좌표 변환은 판의 대칭이라 렌주 대국에는 영향이 없다(엔진 대 엔진 대국이 정식 규칙으로
정상 종료됨을 확인).

## 2. Piskvork

**검증 상태: UNKNOWN** — 이 PC에 Piskvork가 없다(다운로드는 사용자 승인 필요).
1. Piskvork에서 한쪽 대국자를 AI로 두고 경로에 `pbrain-DojoZero.exe`를 지정한다(이름이 `pbrain-`로
   시작해야 새 프로토콜 엔진으로 인식). 예: `C:\dojozero_gui\engine\build\cpp\Release\pbrain-DojoZero.exe`
   와 같은 폴더에 아래 ini를 두되 `rule_dialect` 줄은 빼거나 `gomocup`으로 둔다.
2. 보드 15, 규칙 Renju. 엔진은 렌주 비트가 켜진 `INFO rule 4`를 기대한다(INFO rule이 오지 않으면
   렌주로 간주). Piskvork가 Renju 선택 시 실제로 보내는 값은 미확인.
3. Timeout turn ≥ 3000 ms, timeout match 넉넉히. 엔진 시작 대기 시간 설정이 있으면 90초 이상.

## 3. 직접 빌드·설정 (다른 위치에 설치할 때)

```bash
cmake --build build/msvc --config Release --target pbrain_dojozero
```

exe 옆 `pbrain-DojoZero.ini`(모든 키 선택, 상대 경로는 repo 기준):

```ini
repo=C:/temp_git/Omok_AI                 # 기본값: exe를 빌드한 소스 트리
python=.venv/Scripts/python.exe
model=runs/taraguchi_v1/models/gen_001.pt
device=xpu                               # XPU가 없으면 cpu
rule_dialect=gomocup                     # Yixin Board는 yixin
opening_guard=1                          # 0 = 끔 (5절)
max_move_ms=0                            # 수당 탐색 시간 상한(ms), 0 = GUI 제한만
stderr=pbrain-DojoZero.log               # Python 경고·오류 로그(exe 폴더 기준) 또는 inherit
```

`repo`를 개발 중인 작업 트리로 두면, 다른 작업이 Python/C++을 바꾸는 도중에는 엔진이 시작 직후
죽을 수 있다(실제로 발생: 수정된 `driver.py` + 재빌드 전 `_core`). GUI용으로는 1절처럼 커밋
스냅샷(`git archive`)에서 `_core`를 따로 빌드해 쓴다.

## 4. 기동 시간과 시간 설정

- 엔진은 START에 `OK`를 보내기 전에 torch 로드 + 모델 로드 + XPU warm-up을 끝낸다.
  측정(2026-10-21, 이 PC): **가벼운 부하 6.9–10.5초, 다른 무거운 작업과 동시 실행 시 47–70초**
  (CPU 100%에서 `import torch`만 26초).
- 수당 시간은 GUI 제한의 85% − 30 ms를 쓴다(1초 제한 → 0.82–0.85초 실측). 부하가 있으면 1초 제한에서
  1.86초가 걸린 사례와 TIMEOUT 1건이 있었다 → **수당 3초 이상** 권장.
- 한 번 뜬 엔진은 새 게임에서 재사용되므로 기동 지연은 GUI가 엔진을 처음 띄울 때만 생긴다.

## 5. 오프닝 가드(사용 편의 장치, 기력 아님)

현재 모델(taraguchi_v1)은 Taraguchi-10 오프닝으로만 학습되어 빈 판·초반 자유 국면을 본 적이 없고,
가드 없이는 빈 판 흑 첫 수 `14,1`, 사람 중앙 착수 뒤 백 `2,0`처럼 가장자리에 둔다.
`opening_guard=1`(런처 기본값)이면:
- 빈 판: 중앙(H8 = `7,7`)에 둔다.
- 돌이 6개 미만: 탐색이 고른 수가 기존 돌에서 거리 2(가로·세로·대각 칸 수의 최댓값) 이내면 그대로,
  아니면 그 영역 안에서 방문 수가 가장 많은 수(없으면 policy가 가장 높은 수)를 둔다.
- 6개 이상: 탐색 결과 그대로. 학습에서 일반 렌주 국면은 항상 6수 오프닝 이후에 시작하므로(RULES §8)
  6개부터는 학습 분포 안이다.

이 가드는 GUI 호스트에서만 켜진다(평가·토너먼트 경로는 끔). 기력 측정 결과에는 영향이 없고, 기력
향상 장치도 아니다. 가드가 켜져 있어도 6수 이후의 수 품질은 모델 그대로다.

## 6. 알려진 한계

- 기력: 외부 엔진 기준으로 약하다(PROGRESS의 Current Strength Snapshot). 방어 전술이 주 약점.
- 첫 GUI 실행 직후 10–20초(부하 시 1분 이상)는 엔진이 응답하지 않는다.
- Yixin의 분석 표시·데이터베이스·swap 규칙은 지원하지 않는다.
- 스냅샷 엔진은 `C:\temp_git\Omok_AI\.venv`의 Python과 `runs/taraguchi_v1/models/gen_001.pt`를 그대로
  쓴다. 둘 중 하나가 옮겨지거나 지워지면 ini를 고쳐야 한다.
