# GUI에서 DojoZero와 두기 (pbrain-DojoZero.exe)

`pbrain-DojoZero.exe`는 Gomocup/Piskvork 프로토콜 엔진 실행 파일이다. 실제 엔진은 Python 호스트
(`python -m dojozero.protocol.pbrain`, 코덱 = C++ `EngineSession`)이고, exe는 그 프로세스를 띄우고
stdin/stdout을 그대로 넘겨주는 런처다(ADR-08의 임시 형태: C++ 추론 exe는 아직 결정 전).
소스: `cpp/cli/pbrain_launcher.cpp`.

## 1. 빌드

```bash
cmake --build build/msvc --config Release --target pbrain_dojozero
```

결과: `build/msvc/cpp/Release/pbrain-DojoZero.exe` (정적 CRT, 다른 폴더로 복사해도 됨).

## 2. 설정 (선택)

exe는 빌드한 저장소(`<repo>`)를 기본값으로 기억한다. exe를 다른 폴더로 옮겨도 저장소가 그대로면
설정 없이 동작한다. 바꾸려면 exe 옆에 `pbrain-DojoZero.ini`를 둔다(모든 키 선택):

```ini
repo=C:/temp_git/Omok_AI
python=.venv/Scripts/python.exe          # 상대 경로는 repo 기준
model=runs/taraguchi_v1/models/gen_001.pt
device=xpu                               # XPU가 없으면 cpu
stderr=pbrain-DojoZero.log               # Python 경고·오류 로그(exe 폴더 기준, 이어쓰기) 또는 inherit
```

시작이 안 되면 exe 옆 `pbrain-DojoZero.log`를 먼저 본다(시작 명령줄과 Python 오류가 남는다).

## 3. 기동 시간과 시간 설정 (중요)

- 엔진은 START에 `OK`를 보내기 전에 torch 로드 + 모델 로드 + XPU warm-up을 끝낸다.
  측정(2026-10-21, 이 PC): **가벼운 부하 6.9–10.5초, 다른 무거운 작업과 동시 실행 시 47–70초**
  (CPU 100% — 다른 Python 작업이 약 5.5코어 점유 — 에서 `import torch`만 26초). warm-up 이후 수당 시간은 제한을 지킨다(1초 제한에서 0.82–0.85초).
- 따라서 GUI에 "엔진 시작 대기 시간"이 있다면 **최소 90초**로 둔다. 첫 대국 전에 무거운 작업
  (학습, 다른 엔진, 벤치마크)을 끄는 것이 좋다.
- 수당 시간(timeout turn)은 **2초 이상** 권장. 부하가 있으면 1초 제한에서 1.86초가 걸린 사례가 있다.
- 한 번 뜬 엔진은 RESTART/새 게임에서 재사용되므로 기동 지연은 GUI가 엔진을 처음 띄울 때만 생긴다.

## 4. Piskvork

1. Piskvork 실행 → Players 설정에서 한쪽을 "AI"(Computer)로 두고 경로에
   `pbrain-DojoZero.exe`를 지정한다(파일 이름이 `pbrain-`로 시작해야 Gomocup 엔진으로 인식).
2. 보드 크기 **15**, 규칙(Rule) **Renju**. 엔진은 렌주 비트가 켜진 `INFO rule 4`를 기대한다
   (INFO rule이 오지 않으면 렌주로 간주). 다른 규칙·크기는 엔진이 `ERROR`로 거부한다(Freestyle/
   Standard, 연속 게임, RECTSTART 미지원). Piskvork가 Renju 선택 시 실제로 보내는 값은 미확인.
3. Timeout turn ≥ 2000 ms, timeout match는 넉넉히(예: 무제한 또는 30분).
4. 새 게임 시작 → 첫 수까지 위 3절의 기동 시간이 걸린다.

**검증 상태: UNKNOWN** — 이 PC에 Piskvork가 없어 실제 Piskvork 창으로는 확인하지 못했다.
같은 프로토콜을 쓰는 프로젝트 자체 Gomocup 클라이언트(TournamentRunner)로는 실패 0으로 확인했다
(PROGRESS 참조). Piskvork의 START 대기 시간 설정 이름·기본값도 미확인.

## 5. Rapfi-YixinBoard (Yixin Board)

**현재 상태: 사용 불가 (규칙 번호 불일치)** — 2026-10-21 실측.

- Yixin Board는 자기 폴더의 `engine.exe`만 실행한다(경로 설정 없음). 시험하려면 Rapfi가 든 원래
  폴더를 건드리지 말고 **폴더 전체를 복사**한 뒤, 복사본의 `engine.exe`를 `pbrain-DojoZero.exe`
  사본으로 바꾸고(이름을 `engine.exe`로), 같은 폴더에 `pbrain-DojoZero.ini`를 둔다(이름은 그대로
  `pbrain-DojoZero.ini`). 보드 크기 15.
- 실측: Yixin Board가 보내는 명령은 엔진에 정상 도달하고(`START 15`, `INFO ...`, 소문자
  `start`/`board`/`done`, `yx...` 확장 명령), Yixin을 닫으면 엔진도 종료된다. `yx...` 확장 명령에는
  `UNKNOWN`으로 답한다(무해).
- **막힌 점**: Yixin Board의 렌주 설정("free renju")은 `INFO rule 2`를 보낸다. Gomocup 표준에서
  2는 "연속 게임"이고 DojoZero 코덱은 렌주 = `INFO rule 4`만 받으므로, 착수 요청마다
  `ERROR DojoZero supports only Renju (INFO rule 4)`를 답한다. Yixin에서 두려면 코덱(EngineSession)이
  Yixin 방언의 `rule 2`를 렌주로 받아들이도록 바꾸는 결정이 필요하다(미결정, PROGRESS 참조).
- 좌표: Rapfi 설정에 "Yixin-Board는 flipY_X 변환"이라고 되어 있다. 15×15에서 이 변환은 판의 대칭
  (회전/반사)이고 렌주 규칙은 대칭에 불변이므로 대국 자체에는 문제가 없을 것으로 **추정**(미검증).

## 6. 알려진 한계

- **빈 판·초반 1–2수가 비정상**: 현재 모델(taraguchi_v1)은 Taraguchi-10 오프닝으로 학습되어
  빈 판에서 둔 경험이 없다. 흑 첫 수가 `14,1`(가장자리), 사람이 중앙(`7,7`)에 둔 뒤의 백 응수가
  `2,0`처럼 가장자리로 나온다(in-process 탐색에서도 같은 현상 → 런처 문제 아님). 평가 매치는
  5–6돌 오프닝 국면에서 시작하므로 이 결함이 수치에 드러나지 않았다. GUI에서 두려면 오프닝 몇 수를
  사람이 직접 놓거나(BOARD로 국면 지정) 모델 쪽 보완이 필요하다.
- 엔진 기력: 외부 엔진 기준으로 약하다(PROGRESS의 Current Strength Snapshot).
