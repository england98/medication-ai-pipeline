# 테스트 결과 폴더 작성 규칙

새 테스트 결과는 `outputs/<테스트목적>_YYYYMMDD_HHMMSS/`에 저장합니다. 날짜와 시각은 **테스트 결과 폴더를 생성한 한국시간(KST, UTC+09:00)**이며 폴더 이름의 맨 뒤에 붙습니다.

```text
outputs/
  webcam-q4-empty-hand_20260929_153000/
  webcam-q4-drinking-only_20260929_154500/
  video-yolo-mediapipe-detection_20260929_160000/
  video-qwen-q4-candidate-verification_20260929_161500/
```

위 날짜·시각은 형식 예시입니다. 실제 실행 시 자동으로 결정됩니다. 종료 시각, 영상 촬영 시각, 복약 예정 시각을 의미하지 않습니다.

## 테스트 목적 정하기

입력 방식, 확인할 기능, 비교 조건을 읽고 구분할 수 있게 작성합니다. `test1`, `new`, `final`처럼 목적을 알 수 없는 이름은 피합니다.

| 테스트 목적 | 이름 예시 |
| --- | --- |
| 웹캠 Q4 전체 연결 확인 | `webcam-pipeline-q4` |
| 빈손을 입에 대는 동작의 오탐 확인 | `webcam-q4-empty-hand` |
| 물만 마실 때 완료 오판 확인 | `webcam-q4-drinking-only` |
| 대상 이탈 후 추적 손실 확인 | `webcam-tracking-loss` |
| 저장 영상의 탐지 모듈 확인 | `video-yolo-mediapipe-detection` |
| Qwen 단일 프레임 추론 확인 | `video-qwen-single-frame` |
| Q4 후보 검증 확인 | `video-qwen-q4-candidate-verification` |

이름은 1~64자이며 한글·영문 등 문자, 숫자, `_`, `-`를 사용할 수 있습니다. 첫 글자는 문자·숫자·`_`로 시작합니다. 공백·경로 구분자·콜론은 사용할 수 없습니다. 예를 들어 `웹캠-빈손-오탐`도 가능합니다.

## 전체 파이프라인에서 지정하기

설정 파일에 `test_name`을 추가합니다. `output_dir`은 결과 폴더들이 들어갈 상위 디렉터리입니다.

```json
{
  "output_dir": "../outputs",
  "test_name": "webcam-pipeline-q4"
}
```

위 JSON은 추가·변경할 항목만 표시합니다. 나머지 모델·인증·입력 설정은 기존 파일에 유지합니다.

실행마다 목적을 바꾸려면 CLI 옵션을 사용합니다.

```powershell
.\runtime\.venv\Scripts\python.exe -X utf8 -m medication_pipeline run --config configs/pipeline.webcam.local.json --test-name webcam-q4-empty-hand
```

우선순위는 **`--test-name` → 설정의 `test_name` → 자동 기본 이름**입니다. CLI 옵션은 이번 실행에만 적용되고 원본 설정 파일을 수정하지 않습니다. 실제 적용된 설정은 결과 폴더의 `pipeline-config.json`에 저장됩니다.

이름을 지정하지 않거나 `null`이면 입력·실행 모드에 따라 다음 이름을 사용합니다.

| 실행 | 자동 기본 이름 |
| --- | --- |
| 파일 영상 전체 실행 | `video-pipeline` |
| 웹캠/캡처 전체 실행 | `webcam-pipeline` |
| 파일 영상 `--detect-only` | `video-detection` |
| 웹캠/캡처 `--detect-only` | `webcam-detection` |

테스트 시작 시 터미널에 `Test output: <실제 경로>`가 출력됩니다. 그 경로를 로그 확인과 이후 `verify --run-dir`에 사용합니다.

## 실영상 점검 스크립트에서 지정하기

세 점검 스크립트의 `--output`은 **날짜·시각을 붙이기 전 경로**입니다. 지정하지 않으면 다음 기본 경로를 사용하며, 모두 날짜·시각이 자동으로 붙습니다.

| 스크립트 | 기본 경로 접두부 |
| --- | --- |
| `scripts/smoke_real_detection.py` | `outputs/video-yolo-mediapipe-detection` |
| `scripts/smoke_qwen_model.py` | `outputs/video-qwen-single-frame` |
| `scripts/smoke_qwen_gguf.py` | `outputs/video-qwen-q4-candidate-verification` |

예를 들어 `--output outputs/video-q4-low-resolution`이면 `outputs/video-q4-low-resolution_YYYYMMDD_HHMMSS`에 결과가 생성됩니다. `--output`에는 날짜·시각을 직접 붙이지 않습니다.

탐지 결과를 새로 생성하고 Q4 검증에 연결할 때는 **터미널에 출력된 실제 탐지 결과 폴더**를 `--detections`에 전달합니다.

```powershell
# 기존 테스트 영상으로 탐지 기록 생성
.\runtime\.venv\Scripts\python.exe -X utf8 scripts/smoke_real_detection.py --video "../medication-local-assets/tests_Qwen3-VL-2B-Instruct-Q8_0.gguf/robot3d__A003_P002_G004_C004.mp4" --output outputs/video-robot3d-detection

# 아래 경로를 위 명령의 Test output 값으로 교체합니다.
$detectionDir = '.\outputs\video-robot3d-detection_YYYYMMDD_HHMMSS'
.\runtime\.venv\Scripts\python.exe -X utf8 scripts/smoke_qwen_gguf.py --detections $detectionDir --output outputs/video-robot3d-q4-verification
```

Q4 검증 설정의 `source`는 탐지에 사용한 원본 영상과 같아야 합니다. `--detections`를 생략하면 기존 `outputs/real-detection-smoke-robot` 기록을 사용합니다. 최신 결과를 자동으로 선택하지 않습니다.

## 중복 실행과 기존 기록

같은 목적의 테스트가 같은 초에 시작하면 `webcam-q4-empty-hand-02_YYYYMMDD_HHMMSS`처럼 목적 뒤에 번호를 추가합니다. 날짜·시각은 계속 맨 뒤에 유지하며 기존 결과를 덮어쓰거나 섞지 않습니다.

각 새 폴더의 `test-run.json`에는 테스트 목적, 밀리초를 포함한 시작 시각과 UTC 오프셋, 시간대, 실제 폴더 경로가 저장됩니다. 기존 단계별 JSON 기록의 UTC 시각은 그대로 사용합니다.

웹캠 시험 영상은 같은 결과 폴더의 `recording/input.mp4`에 저장합니다. 원본 프레임 시각 대응은 `recording/frames.jsonl`, 녹화 상태·프레임 수·종료 사유는 `recording/metadata.json`에서 확인합니다. 저장 범위와 중단 시 확인 방법은 [웹캠 테스트 절차](webcam-testing.md)를 참고합니다.

기존 결과 폴더는 내부 기록의 절대 경로와 참조를 보존하기 위해 이름을 유지합니다. `verify --run-dir`도 이미 저장된 폴더를 이어서 처리하므로 새 폴더나 새 타임스탬프를 만들지 않습니다. 모델·조건을 바꾸는 새 실험은 새 실행으로 진행합니다.

이 규칙은 전체 파이프라인 실행과 위 실영상 점검 스크립트의 최상위 결과 폴더에 적용합니다. 후보·클립·요청별 ID와 하위 폴더 구조는 그대로 유지합니다.
