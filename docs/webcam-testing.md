# 웹캠 전체 파이프라인 테스트 절차

Windows PowerShell과 현재 작업공간의 Qwen3-VL-2B Q4_K_M 설정을 기준으로 작성했습니다. 모든 명령은 작업공간 루트에서 실행합니다.

이 절차는 웹캠 입력부터 대상 추적, 후보 영상 생성, Qwen 검증, 세션 판단까지 확인하기 위한 것입니다. **짧은 실제 웹캠 실행에서 수신·추적·후보 탐지·영상 저장을 확인했지만, 세션 파일 저장 오류로 중단되어 Qwen 검증까지 포함한 전체 라이브 실행은 아직 완료하지 못했습니다.** 아래 준비 상태는 2026-09-29 점검 및 [기존 검증 기록](testing.md)을 기준으로 합니다.

해당 세션 저장 오류에는 Windows 파일 교체 재시도를 추가했습니다. 실제 파일 잠금 복구 테스트와 저장 영상의 탐지 전용 재처리가 통과했습니다. 상세 수치는 [저장 오류 수정 검증 기록](testing.md#windows-세션-저장-오류-수정-및-영상-재처리--2026-09-29)에 있습니다.

## 1. 준비 상태와 시험 범위

| 항목 | 현재 상태 |
| --- | --- |
| 실행 환경 | `runtime/.venv`에 탐지·추론 의존성 설치 완료 |
| 탐지 모델 | YOLO11n, MediaPipe 얼굴·손·포즈 모델 준비 완료 |
| VLM | Q4_K_M GGUF 언어 모델, F16 시각 인코더, llama.cpp 준비 완료 |
| 현재 설정 | `configs/pipeline.local.json`은 Q4를 사용하지만 입력은 MP4 |
| 웹캠 장치 | `Integrated Webcam`에서 1280×720 영상 96프레임 수신·저장 확인. 장시간 안정성·지연은 미검증 |
| 자동 검증 | Windows 파일 잠금 복구 검사를 포함한 225개 테스트 통과 |
| 실영상 Q4 점검 | 후보 3개 처리 성공, 모두 `UNCERTAIN`. 정확도·웹캠 성능 평가가 아님 |
| 진행 화면 | 미리보기·박스 오버레이·대시보드 없음. 저장 파일로 확인 |
| 웹캠 영상 저장 | 수신한 전체 프레임을 `recording/input.mp4`에 저장. 음성 제외 |
| 종료 기능 | 완료 판정 시 자동 종료. 시간 제한·정상 수동 종료 기능 없음 |

**종료 제약을 먼저 확인합니다.** 웹캠에는 일반적인 파일 종료 시점이 없습니다. 완료 근거가 나오지 않으면 계속 실행됩니다. `Ctrl+C`는 중단으로 처리되어 활성 후보 마감·대기 요청 처리·최종 요약 저장이 생략될 수 있습니다. 반복 대조군 시험을 안정적으로 운영하려면 정상 종료 기능을 보완해야 합니다. 현재 중단 후 확인 가능한 범위는 [7절](#7-종료와-중단-후-처리)에 설명합니다.

전체 입력 영상은 위 후보 검증과 별도로 저장합니다. 일반적인 `Ctrl+C` 중단이나 처리 오류에도 수신한 구간의 MP4를 마무리하도록 구현했습니다. 작업 관리자 강제 종료·전원 종료·저장 장치 오류에서는 파일 완성을 보장하지 않습니다.

첫 시험은 다음 항목을 확인하는 데 집중합니다.

- 영상 입력과 대상 추적이 유지되는가?
- 손·입·물체 관측으로 적절한 후보와 전후 클립이 만들어지는가?
- Qwen 입력에 핵심 장면이 포함되고 결과와 근거가 저장되는가?
- 빈손 동작이나 물만 마시는 장면을 복약 완료로 잘못 확정하지 않는가?

## 2. 촬영 환경과 장치 확인

1. 밝은 환경에서 얼굴·입·양손·다루는 물체가 함께 보이도록 구도를 잡습니다.
2. 첫 시험에서는 화면에 한 사람만 나오게 합니다. 첫 분석 프레임부터 대상자가 보여야 합니다. 사람이 들어올 때까지 기다리는 기능은 없습니다.
3. 필요하면 Windows 카메라 앱으로 구도를 확인한 뒤 앱을 닫습니다. Teams 등 다른 카메라 사용 프로그램도 종료합니다.
4. Windows의 카메라 접근 설정에서 데스크톱 앱의 접근을 허용했는지 확인합니다.

장치명이 다르거나 웹캠을 바꿨다면 다음 명령으로 목록을 다시 확인합니다. 별도 `ffmpeg.exe` 설치 없이 현재 Python의 PyAV를 사용합니다. 장치 목록만 조회하며 프레임을 저장하지 않습니다.

```powershell
@'
import av

av.logging.set_level(av.logging.INFO)
with av.logging.Capture() as logs:
    try:
        av.open("dummy", format="dshow", options={"list_devices": "true"})
    except av.error.FFmpegError:
        pass
for _, _, message in logs:
    print(message.strip())
'@ | .\runtime\.venv\Scripts\python.exe -X utf8 -
```

목록 조회는 출력 후 종료 과정에서 FFmpeg 오류를 반환할 수 있어 위 예제에서 예외를 처리합니다. 출력에 영상 장치가 없거나 `dshow` 관련 오류만 있다면 장치가 확인된 것이 아닙니다.

## 3. 웹캠 전용 설정 만들기

기존 MP4 설정을 보존하고 웹캠 설정을 별도로 만듭니다. 이미 파일이 있으면 복사하지 않습니다.

```powershell
if (-not (Test-Path -LiteralPath configs/pipeline.webcam.local.json)) {
    Copy-Item -LiteralPath configs/pipeline.local.json -Destination configs/pipeline.webcam.local.json
}
```

`configs/pipeline.webcam.local.json`에서 아래 항목을 변경하거나 추가합니다. **아래 JSON은 변경할 항목만 보여주는 예시입니다. 파일 전체를 교체하지 말고 `models`, `detection`, `video`, `vlm` 등 기존 설정을 유지합니다.**

```json
{
  "source": "video=Integrated Webcam",
  "input_format": "dshow",
  "capture_options": {},
  "record_input": true,
  "start_frame": 0,
  "output_dir": "../outputs",
  "test_name": "webcam-pipeline-q4",
  "occurrence": {
    "schema_version": "1.0",
    "scheduled_occurrence_id": "webcam-case-001",
    "user_id": "webcam-test-subject",
    "scheduled_at": "2026-09-29T09:00:00.000Z"
  },
  "authentication": {
    "mode": "operator",
    "operator_confirmed": false,
    "target_bbox": null
  }
}
```

- `source`의 이름은 실제 장치 목록과 일치시킵니다.
- `capture_options={}`는 장치 기본 설정을 사용합니다. 해상도·FPS를 강제로 지정하지 않고 먼저 입력 호환성을 확인합니다.
- `record_input=true`이면 입력 영상을 저장합니다. `false`는 저장하지 않으며, 생략 또는 `null`이면 웹캠/캡처 입력만 자동 저장합니다. 현재 `pipeline.webcam.local.json`에는 `true`를 설정했습니다.
- `test_name`은 시험 목적입니다. 예: `webcam-pipeline-q4`, `webcam-q4-empty-hand`. 결과는 `outputs/<목적>_YYYYMMDD_HHMMSS`에 KST 시작 시각을 붙여 저장합니다. [결과 폴더 작성 규칙](test-results.md)을 참고합니다.
- `scheduled_occurrence_id`는 시험별로 구분합니다. `user_id`도 시험 대상에 맞게 설정합니다.
- `scheduled_at`은 예시이므로 실제 시험 회차의 UTC 시각으로 변경합니다. `Z`는 UTC입니다. PowerShell에서 `(Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")`로 현재 UTC 값을 확인할 수 있습니다.
- **운영자가 촬영 대상자를 확인한 후에만** `operator_confirmed`를 `true`로 변경합니다. 운영자 확인은 생체인증이 아닙니다.
- 한 사람으로 시작하면 `target_bbox=null`을 사용합니다. 여러 사람 중 초기 대상을 지정하는 설정은 [실행 설정](../configs/README.md)을 참고합니다.

상대 파일 경로는 설정 파일의 디렉터리 기준입니다. 웹캠의 `source`는 `input_format="dshow"`일 때 파일 경로로 변환하지 않습니다.

현재 Q4 설정에서 다음 값도 확인합니다.

| 설정 | 사용할 값 |
| --- | --- |
| `vlm.backend` | `llama_cpp` |
| `vlm.model_id` | `../models/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-Q4_K_M.gguf` |
| `vlm.device` | `cuda` |
| `vlm.llama_cpp.mmproj_offload` | `false` — 시각 인코더는 CPU |
| `vlm.max_frames` | `4` |
| `vlm.include_rois` | `false` |

런타임·모델 경로나 설치 문제가 있으면 [4비트 실행 안내](qwen-4bit.md)를 참고합니다.

## 4. 실행 전 점검

```powershell
.\runtime\.venv\Scripts\python.exe -X utf8 -m medication_pipeline validate --config configs/pipeline.webcam.local.json
.\runtime\.venv\Scripts\python.exe -X utf8 -m medication_pipeline doctor --config configs/pipeline.webcam.local.json
```

`validate`는 설정 형식을 검사합니다. `doctor`는 라이브러리·모델 파일·운영자 확인 등 준비 항목과 녹화 사용 시 `recording_encoder`를 검사합니다. `ready=true`를 확인하고 실행합니다.

`operator_confirmed=false`로 실패하면 대상자 확인 후 설정을 변경합니다. **`doctor`의 통과는 웹캠 개방·프레임 수신·실제 추론 성공을 보장하지 않습니다.** 캡처 장치는 파일 입력처럼 존재 여부를 검사하지 않으며 실제 입력은 다음 단계에서 확인합니다.

## 5. 전체 파이프라인 실행과 동작 순서

대상자가 촬영 위치에 있는 상태에서 터미널 A에서 실행합니다.

```powershell
.\runtime\.venv\Scripts\python.exe -X utf8 -m medication_pipeline run --config configs/pipeline.webcam.local.json
```

Qwen 서버를 별도로 실행할 필요는 없습니다. 첫 검증 요청에서 자동으로 준비하며 이후 요청에서 재사용합니다.

실행 시 출력되는 `Test output: <실제 경로>`를 기록합니다. 이번 실행의 목적만 바꾸려면 위 명령에 `--test-name webcam-q4-empty-hand`를 추가합니다. 원본 설정 파일은 수정하지 않습니다.

실제 내부 흐름은 다음과 같습니다.

```text
웹캠 입력 → 운영자 확인 연결·대상 추적 → YOLO/MediaPipe 관측
         → E01/E02/E03 후보 생성 → 후보 전후 클립·검증 요청 저장
         → Qwen 검증 작업 → 결과 수신·세션 판단 → 완료 시 촬영 루프 종료
```

1. [6절](#6-진행-상황-확인)의 `observations.jsonl`이 갱신되고 추적 상태가 `TRACKED`인지 확인합니다.
2. 몇 초간 움직임을 줄여 기준 상태를 확보합니다.
3. 시험 동작을 한 번 수행합니다. 첫 시험에서는 짧은 간격으로 여러 동작을 반복하지 않습니다.
4. 손이나 용기를 입에서 내리고 약 3초 이상 동작을 마무리합니다. 현재 이탈 기준은 500ms, 후보 후행 영상은 1초이며 실제 처리는 지연될 수 있습니다.
5. `queue-events.jsonl`에 요청이 생성되고 Qwen 처리가 진행되는지 확인합니다.
6. 결과가 도착할 때까지 촬영 범위와 대상자를 유지합니다. 대기 중 추가 동작도 새로운 후보를 만들 수 있습니다.

E01은 준비 관련, E02는 손–입 접근 관련, E03는 용기–입 접근 관련 후보입니다. E02는 약이 검출되지 않아도 생성되며 E02와 E03가 함께 나올 수 있습니다. **후보 생성은 행동 확인이나 복약 완료가 아닙니다.**

VLM은 별도 단일 작업 스레드에서 처리합니다. 설정의 `max_pending=2`는 동시 추론 2개를 의미하지 않습니다. 기존 실영상 점검에서 후보당 입력 처리와 생성에 약 62~90초가 걸렸습니다. 서버 시작·앞선 요청 대기·캡처와 탐지 부하가 추가될 수 있으므로 웹캠 판정 시간의 보장값으로 사용하지 않습니다.

`analysis_interval_ms=100`도 실제 10fps 처리 성능을 보장하지 않습니다. 실시간 처리량과 입력 지연은 이번 시험에서 확인할 항목입니다.

## 6. 진행 상황 확인

터미널 B도 작업공간 루트에서 엽니다. 터미널 A의 `Test output` 경로를 사용하거나 최근 실행 폴더를 나열하고 방금 시작한 실행을 선택합니다.

```powershell
Get-ChildItem -LiteralPath .\outputs -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'pipeline-config.json') } |
    Sort-Object CreationTime -Descending |
    Select-Object -First 5 FullName, CreationTime

# 위 목록에서 이번 시험 폴더의 경로로 교체합니다.
$runDir = '.\outputs\webcam-pipeline-q4_YYYYMMDD_HHMMSS'
```

여러 실행이 있다면 폴더 안의 `pipeline-config.json`에서 `source`와 시험 회차도 확인합니다. 아래 경로는 모두 선택한 실행 폴더 기준입니다.

| 파일 | 확인 내용 |
| --- | --- |
| `test-run.json` | 테스트 목적·KST 시작 시각·실제 저장 경로 |
| `recording/input.mp4` | 수신한 전체 입력 프레임의 H.264 영상. 종료 후 재생 |
| `recording/frames.jsonl` | 녹화 프레임 번호·상대 시각과 원본 FrameRef의 대응 |
| `recording/metadata.json` | 녹화 상태·종료 사유·최종 프레임 수·해상도·경로 |
| `observations.jsonl` | 분석 프레임별 대상 추적·객체·손/입/포즈 관측 |
| `candidate-history.jsonl` | 후보 ID, 유형, revision, 행동 구간 |
| `queue-events.jsonl` | 요청 저장 `READY`, 결과 수신 `FINISHED` |
| `verification/<요청별 폴더>/states.jsonl` | 개별 검증의 `QUEUED`, `RUNNING`, `FINISHED` |
| `llama-runtime/server.log` | Qwen 서버 시작·입력 처리·응답 생성 로그 |
| `results.jsonl` | 파이프라인이 수신한 결과와 근거·처리 오류 |
| `session/session.json` | 현재 세션과 추적 상태 |
| `session/session-view.json` | 외부 조회용 상태와 완료 여부 |
| `session/persistence-events.jsonl` | 세션 JSON 교체 충돌 시 재시도·복구·최종 실패 내역. 충돌이 없으면 생성되지 않음 |

최근 관측의 추적 상태를 확인합니다. 파일이 만들어진 뒤 실행합니다.

```powershell
$observation = Get-Content -LiteralPath (Join-Path $runDir 'observations.jsonl') -Encoding UTF8 -Tail 1 | ConvertFrom-Json
$observation.tracking
```

요청 진행 로그를 계속 보려면 다음을 실행합니다.

```powershell
Get-Content -LiteralPath (Join-Path $runDir 'queue-events.jsonl') -Encoding UTF8 -Tail 20 -Wait
```

Qwen 로그는 위 모니터를 끝내거나 별도 터미널에서 확인합니다.

```powershell
Get-Content -LiteralPath (Join-Path $runDir 'llama-runtime/server.log') -Encoding UTF8 -Tail 20 -Wait
```

터미널 B의 `Ctrl+C`는 해당 로그 모니터를 끝냅니다. 파이프라인이 실행 중인 터미널 A의 `Ctrl+C`와 구분합니다.

파일은 해당 단계에 도달해야 생성됩니다. 후보가 없으면 큐·VLM 로그가 없는 것이 정상일 수 있습니다. 파일이 없다는 메시지가 나오면 먼저 이전 단계의 관측·후보 이력을 확인합니다.

현재 세션 상태는 파일을 다시 읽어 확인합니다. 이 파일은 덮어쓰기되므로 줄 추가 방식의 로그와 다릅니다.

`session.json`과 `session-view.json`에는 `Get-Content -Wait`를 사용하지 않습니다. 교체되는 JSON은 아래처럼 한 번 읽고 닫으며, 지속 관찰에는 `session/session-history.jsonl` 등 추가 방식의 로그를 사용합니다. Windows에서 교체를 허용하지 않는 읽기 핸들이 열려 있으면 파일 교체가 막힐 수 있습니다.

```powershell
Get-Content -LiteralPath (Join-Path $runDir 'session/session-view.json') -Encoding UTF8
```

후보 이력의 여러 행은 같은 후보의 revision 갱신일 수 있습니다. 행 수를 행동 수로 세지 말고 `candidate_id`로 구분합니다. `queue/*.json`도 처리 후 남는 보존 기록이므로 파일 수를 현재 대기 요청 수로 해석하지 않습니다.

### 저장되는 영상의 범위

- 첫 수신 프레임부터 입력 루프가 끝날 때까지 저장합니다. 분석 주기에 의해 건너뛴 프레임, `start_frame` 이전 프레임, 후보가 없는 장면도 포함합니다. 첫 프레임에서 사람 선택에 실패한 경우에도 이미 수신한 영상은 남습니다.
- 녹화는 파이프라인이 디코딩해 수신한 프레임을 대상으로 합니다. 카메라 드라이버에서 이미 유실된 프레임은 복원하지 않으며 카메라의 최대 FPS로 별도 촬영하는 기능은 아닙니다. 모델 로딩 전과 촬영 종료 후 VLM 대기 시간도 영상에 추가하지 않습니다.
- 마이크·음성·박스 오버레이는 저장하지 않습니다. 프레임은 H.264로 재인코딩되며 원본 비트스트림의 무손실 복사는 아닙니다. 홀수 해상도는 우측·하단에 최대 1픽셀을 패딩하고 메타데이터에 기록합니다.
- 영상 시각은 첫 수신 프레임을 0ms로 하고 원본 프레임 간격을 보존합니다. `frames.jsonl`로 녹화 프레임과 원본 시각·프레임 번호를 연결할 수 있습니다. 녹화 프레임 수는 분석 프레임 수와 다를 수 있습니다.
- 영상 인코딩에 CPU와 디스크가 추가로 사용됩니다. `video.max_spool_bytes`는 후보 버퍼 한도이며 전체 녹화 파일의 크기 제한은 아닙니다.

`metadata.json`의 최종 프레임 수·종료 사유는 녹화 마감 때 갱신됩니다. 진행 중에는 `RECORDING`으로 표시되며 해당 파일의 프레임 수를 실시간 카운터로 사용하지 않습니다.

## 7. 종료와 중단 후 처리

### 완료 판정으로 자동 종료한 경우

완료 결과를 수신하면 캡처 루프가 종료됩니다. 남은 후보 정리와 검증 요청 처리, 파일 정리를 수행한 후 명령이 끝납니다. 따라서 완료 표시 후에도 종료까지 시간이 걸릴 수 있습니다.

```powershell
Get-Content -LiteralPath (Join-Path $runDir 'summary.json') -Encoding UTF8
```

`run_status=FINISHED`, `processing_errors`, 요청·결과 수와 세션 상태를 확인합니다. `FINISHED`는 실행 처리가 끝났다는 뜻이며 복약 완료는 별도의 세션 값으로 확인합니다.

녹화 결과는 `summary.json`의 `recording`에도 포함됩니다. 다음 명령으로 메타데이터를 확인하고 영상을 엽니다. `status=FINALIZED`이고 `video_path`가 있을 때 재생합니다.

```powershell
Get-Content -LiteralPath (Join-Path $runDir 'recording/metadata.json') -Encoding UTF8
Invoke-Item -LiteralPath (Join-Path $runDir 'recording/input.mp4')
```

`FINALIZED`는 수신한 구간의 영상 파일 마무리를 의미하며 복약 완료나 전체 후보 처리 완료를 뜻하지 않습니다. 프레임을 받지 못했다면 `EMPTY`와 `video_path=null`, 녹화 오류가 있다면 `ERROR`를 기록합니다. MP4는 마감 전에는 재생되지 않을 수 있습니다.

### 완료 판정이 없어 시험을 중단해야 하는 경우

현재 `--duration`, `--max-frames`, 정상 종료 버튼은 없습니다. `UNCERTAIN`이나 비복약 대조군은 자동 종료되지 않을 수 있습니다. 중단이 필요하면 다음 순서로 처리합니다.

1. 가능하면 동작을 마친 뒤 후보의 후행 영상과 요청이 저장되었는지 확인합니다.
2. 터미널 A에서 `Ctrl+C`를 한 번 누릅니다.
3. 수신 영상 파일을 먼저 마무리하고 입력 장치를 해제합니다. 실행 중인 VLM 작업 정리로 반환이 늦어질 수 있으므로 프로세스 종료까지 기다립니다.
4. `run-error.json`의 `INTERRUPTED` 기록, `recording/metadata.json`, 이미 저장된 결과를 확인합니다. `summary.json`이 없어도 정상적으로 마무리된 `recording/input.mp4`는 재생할 수 있습니다.

**중단은 정상 마감과 다릅니다.** 활성 후보의 종료 처리, 아직 생성되지 않은 클립, 미전달 결과의 세션 반영이 생략될 수 있습니다. 작업 관리자 강제 종료나 반복 인터럽트는 정리도 끊을 수 있습니다.

프로세스가 끝났고 `session/session.json` 및 보존된 `queue/*.json`이 있으면 다음 명령으로 저장된 요청을 검증·반영할 수 있습니다.

```powershell
.\runtime\.venv\Scripts\python.exe -X utf8 -m medication_pipeline verify --run-dir $runDir
```

- 실행 중인 파이프라인과 동시에 수행하지 않습니다. `.pipeline-running`이 남아 있으면 실행 상태를 먼저 확인합니다. 실행 중인 잠금 파일을 삭제하지 않습니다.
- `verify`는 저장된 클립과 요청을 사용합니다. 완성되지 않은 후보나 저장되지 않은 웹캠 프레임을 복구하지 않습니다.
- 완료된 검증 결과는 재사용합니다. 미완결 검증 기록은 무조건 자동 재시도하지 않으므로 해당 요청의 `states.jsonl`과 오류를 확인합니다.
- 검증이 끝나면 `mode=VERIFY`의 요약이 저장됩니다. 기존 `run-error.json`은 앞선 캡처 중단 이력이므로 함께 보존합니다.
- 이 요약을 얻어도 중단 전 전체 촬영 구간이 빠짐없이 처리되었다는 의미는 아닙니다.

탐지만 분리하려면 `run`에 `--detect-only`를 붙일 수 있습니다. 이 경우 Qwen은 실행하지 않고 이후 `verify`로 연결합니다. **웹캠 종료 제약은 동일하며, 탐지만으로 복약 완료가 발생하지 않습니다.**

## 8. 결과와 근거 검토

| 값 | 의미 |
| --- | --- |
| `processing_status=OK` | 검증 처리가 성공함. 행동이 확인되었다는 뜻은 아님 |
| `processing_status=ERROR` 또는 `NOT_RUN` | 처리 오류 또는 미실행. 행동 부정 판정으로 해석하지 않음 |
| `verification=CONFIRMED` | 해당 이벤트의 확인 규칙 충족 |
| `verification=REFUTED` | 해당 이벤트에 대한 반증 판정 |
| `verification=UNCERTAIN` | 해당 이벤트의 확인·반증 근거가 충분하지 않음 |
| 세션 `CLOSED / COMPLETE` | 영상 기반 복약 완료 규칙을 충족하여 세션 종료 |
| 세션 `ACTIVE / null` | 완료 근거가 아직 없음. 최종 미복약 판정이 아님 |

현재 완료 판단은 E02의 약물 식별과 입안 전달 확인을 핵심으로 하며, 대상·문맥 및 같은 행동에 대한 수신된 상충 근거도 검사합니다. E01→E02→E03가 반드시 순서대로 확인되어야 하는 구조는 아닙니다.

후보별로 다음 순서로 검토합니다.

1. `queue/*.json`의 요청 ID·후보 ID·행동 구간·클립 참조를 연결합니다.
2. `media/clips/<클립ID>/video.mp4`를 재생해 동작 전후와 핵심 순간이 포함되는지 확인합니다.
3. `verification/<요청별 폴더>/model_input.json`에서 실제 입력 프레임과 순서를 확인합니다.
4. 같은 폴더의 `result.json`에서 `checks`, 설명, `evidence`, 입력 인덱스, 원문 응답과 오류를 확인합니다.
5. `decision/evaluations.jsonl`과 세션 기록에서 이 결과가 완료 판단에 어떻게 반영되었는지 확인합니다.

Qwen에 입력되는 프레임은 최대 4장이며 ROI 입력은 꺼져 있습니다. 원본 클립에 약이나 전달 순간이 보여도 선택된 입력에서 빠질 수 있으므로, `UNCERTAIN`의 원인을 판단할 때 두 자료를 함께 봅니다.

후보가 전혀 생성되지 않은 미탐 구간은 `recording/input.mp4`에서 확인합니다. `frames.jsonl`의 원본 프레임 번호와 `observations.jsonl`을 비교하면 관측·후보 생성 단계에서 빠진 지점을 조사할 수 있습니다. `verify`는 보존된 요청만 처리하므로, 전체 녹화에서 후보를 다시 탐지하려면 MP4를 입력으로 새 실행을 시작합니다.

녹화 MP4를 재시험할 때는 설정 사본의 `source`를 해당 파일 경로로 바꾸고 `input_format=null`, `capture_options={}`, `record_input=false`, `start_frame=0`으로 설정합니다. 시험 목적도 `video-replay-...`처럼 바꿉니다. 녹화 영상은 0ms부터 시작하므로 이전 세션의 시각·인증 결과를 그대로 재사용하지 않고 새 세션으로 진행합니다.

## 9. 권장 시험 항목과 기록

먼저 한 회차에 한 가지 동작을 시험하고, 조건을 바꿔 반복합니다. 아래 기대 결과는 검증 기준이며 현재 정확도를 보장하는 값이 아닙니다.

| 시험 | 확인할 기준 |
| --- | --- |
| 정지한 한 사람 | `TRACKED` 유지, 불필요한 후보와 완료 발생 여부 |
| 빈손을 입에 대기·얼굴 만지기 | E02 후보가 생겨도 복약 완료로 잘못 확정하지 않는지 |
| 물만 마시기 | E03 및 동시 E02 후보와 검증 결과 확인, 음수만으로 완료되지 않는지 |
| 목표 복약 과정이 보이는 장면 | 약 식별·입안 전달 순간이 클립 및 실제 입력에 포함되는지 |
| 손·입 가림, 화면 이탈, 다른 사람 등장 | 다른 사람으로 대상이 바뀌지 않는지, 불확실·추적 손실 처리 |
| 간격을 둔 반복 동작 | 같은 후보의 revision과 별도 행동의 새 ID가 구분되는지 |
| 연속 입력·후보 누적 | 요청 적체, 결과 지연, 처리 오류, 메모리·디스크 사용 증가 |

현재 구현의 평가 한계도 기록합니다.

- 기본 YOLO11n에는 약·포장 전용 클래스가 없어 E01을 본격적으로 평가하려면 해당 클래스를 학습한 가중치와 `class_map`이 필요합니다.
- 추적이 `UNRESOLVED`가 되면 자동 재연결하지 않습니다. 새 실행과 대상자 확인이 필요합니다.
- 요청 `READY`부터 결과 수신까지의 시간은 큐 대기와 처리를 포함하며 카메라 촬영부터의 전체 지연과 다릅니다. 연속 처리량·유실·전체 지연은 별도 측정이 필요합니다.
- 실제 삼킴, 처방 일치, 전체 복용량은 현재 출력 범위에 포함되지 않습니다.

시험별로 다음 양식을 복사해 기록합니다. 실제로 본 행동을 기대값으로 먼저 기록하고 모델 결과와 비교합니다.

```text
시험 회차 / 실행 폴더:
일시 / 대상자 식별값:
웹캠 / 해상도·FPS(확인한 경우):
설정 파일 / 기존 설정에서 바꾼 값:
시험 동작 / 사람이 확인한 실제 행동:
추적 상태 / 손·입 관측 상태:
후보 ID·유형 / 클립 경로:
핵심 순간이 클립과 Qwen 입력에 포함되는지:
요청 준비 시각 / 결과 수신 시각:
processing_status / verification / checks:
세션 상태 / 종료 방식 / 처리 오류:
오탐·미탐·불확실 원인 추정 / 다음 변경 항목:
```

## 10. 문제 발생 시 확인 순서

| 증상 | 확인 순서 |
| --- | --- |
| `ready=false` | 실패한 doctor 항목 → 운영자 확인 → 모델·런타임 경로 |
| 웹캠을 열 수 없음 | 장치명 → 다른 앱 점유 → Windows 접근 설정 → 캡처 옵션 |
| 첫 프레임에서 대상 선택 실패 | 실행 전 사람이 보이는지 → 한 사람인지 → 구도·밝기 |
| JSON 저장 중 `WinError 5/32/33` | 자동 재시도 후 `persistence-events.jsonl`의 `RECOVERED`/`FAILED` 확인 → 해당 JSON을 계속 열어 두는 프로그램·읽기 전용 속성·폴더 권한 확인 |
| PTS·프레임 시각 오류 | `run-error.json` 확인. 현재 입력은 유효하고 증가하는 원본 PTS가 필요하며 임의 시각으로 대체하지 않음 |
| 관측은 있지만 후보가 없음 | `TRACKED` 여부 → 손·입·물체 관측 → 동작과 탐지 임계값. E01은 모델 클래스도 확인 |
| 후보는 있지만 요청이 없음 | 동작 종료·후행 영상 확보 여부 → 클립 생성 오류·디스크 한도 |
| Qwen 결과가 늦음 | `states.jsonl` → 서버 로그 → 앞선 요청 수 → 입력 처리 시간 |
| 계속 `UNCERTAIN` | 원본 클립 → 실제 입력 4장 → 작은 물체·가림·핵심 순간 누락 → 모델 응답 근거 |
| `summary.json`이 없음 | 아직 실행 중인지 → `run-error.json` → 중단 후 보존 요청 검증 가능 여부 |

JSON 교체는 일시적인 Windows 접근·공유·잠금 오류에 한해 최대 7회 시도하며, 대기 시간 합계는 940ms입니다. `RECOVERED`면 실행이 계속됩니다. `FAILED`면 기존 JSON과 새 내용의 `.tmp`를 보존한 채 오류가 전달됩니다. 세션 오류의 진단 파일은 `session/persistence-events.jsonl`이며, 다른 JSON의 오류는 해당 JSON과 같은 디렉터리에 기록됩니다. 로그에는 실패 경로·WinError·시도 횟수·시간이 포함되지만 잠금을 건 외부 프로세스는 식별하지 않습니다. 진단 파일도 쓸 수 없으면 터미널 경고를 확인합니다.

해상도·FPS를 지정하려면 장치가 지원하는 모드를 먼저 확인합니다. [FFmpeg DirectShow 문서](https://ffmpeg.org/ffmpeg-devices.html#dshow)의 `video_size`, `framerate`, `list_options`를 참고합니다. 지원하지 않는 값을 지정하면 입력 개방에 실패할 수 있습니다.

관련 문서: [4비트 실행 안내](qwen-4bit.md) · [검증 기록](testing.md) · [실행 설정](../configs/README.md) · [런타임](../runtime/README.md)
