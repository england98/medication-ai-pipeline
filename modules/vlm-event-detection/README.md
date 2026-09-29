# VLM 이벤트 검증

현재 전체 파이프라인은 **Q4_K_M GGUF + F16 시각 인코더**를 선택할 수 있습니다. `backend="llama_cpp"` 설정과 자동 서버 관리는 [4비트 실행 안내](../../docs/qwen-4bit.md)를 참고합니다. 아래 CPU 설치 예시는 기존 Transformers 경로입니다.

전체 파이프라인 작업 공간의 `modules/vlm-event-detection` 모듈입니다. 전체 구조와 모듈별 개발 범위는 [상위 README](../../README.md)를 참고합니다. 이 문서의 명령은 이 모듈 폴더를 기준으로 실행합니다.

[설계 문서](https://medication-ai-vision-docs.vercel.app/architecture/event-detection)의 **E01 준비·취급, E02 약의 입 전달, E03 음수 후보 검증**을 독립적으로 실행합니다. 모델은 `Qwen/Qwen3-VL-2B-Instruct`를 사용합니다. 후보·세션·영상 연결 검사 → 실제 PTS 기반 프레임 선택 → 모델 고유 processor/chat template → 응답·근거 검사 → 규칙 판정 → 요청별 기록 순서로 처리합니다.

현재 범위는 후보별 VLM 검증입니다. 자동 이벤트 탐지, 사용자 인증, 세션 완료 판단은 후속 모듈입니다. `prepare`는 이벤트 탐지 연결 전 기능을 확인하기 위한 **운영자 입력 기반 수동 후보 생성** 명령입니다. 실제 복약 영상·모델 가중치는 포함하지 않으며, 실제 모델의 정확도·지연시간은 해당 영상과 PC에서 추가 측정해야 합니다.

## Windows 설치와 계약 검사

PowerShell에서 저장소 루트 기준으로 실행합니다. 현재 PC에서 사용한 `uv`와 `uv.lock`으로 Python 3.10 가상환경과 의존성을 맞춥니다. 한글 경로의 Python editable 설치를 위해 UTF-8 모드를 켭니다.

```powershell
$env:PYTHONUTF8 = "1"
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv sync --locked --extra dev --python 3.10
.\.venv\Scripts\python.exe -X utf8 -m pytest
.\.venv\Scripts\python.exe -X utf8 -m vlm_verification validate --request examples/request.e02.json --session examples/session.json --config examples/config.cpu.json
```

`validate`는 JSON 스키마와 세션·후보·프레임 시간축 연결을 확인하며, 모델을 로드하거나 영상을 디코딩하지 않습니다. 예제의 사용자·인증·시각·영상 메타데이터는 형식 설명용 가상 값입니다. `example-not-included.mp4`는 제공되지 않으므로 예제 그대로 `run`하면 `NOT_RUN / MISSING_VIDEO`를 기록합니다.

## 현재 PC에서 첫 모델 실행

Transformers 예제는 CPU `float32`를 사용합니다. Q4 로컬 설정은 GTX 1650 Ti 4GB에서 언어 모델을 CUDA로, 시각 인코더를 CPU로 실행합니다. 실제 점검 범위와 지연은 [검증 기록](../../docs/testing.md)을 참고합니다.

```powershell
uv sync --locked --extra dev --extra inference --python 3.10
```

`examples/config.cpu.json`은 2026-09-29에 공식 모델 메타데이터에서 확인한 [고정 commit `89644892e4d85e24eaac8bacfd4f463576704203`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)을 사용합니다. 메타데이터의 최종 수정일은 2025-10-23입니다. `main`과 같은 이동하는 이름은 허용하지 않습니다. 예제 설정은 `local_files_only: true`이므로 모델 파일을 미리 같은 revision으로 로컬 캐시에 준비해야 합니다. 아래 명령은 해당 revision의 모델 파일을 다운로드합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', revision='89644892e4d85e24eaac8bacfd4f463576704203')"
```

준비할 영상은 한 사람의 행동 전후가 포함된 짧은 MP4/MOV 클립입니다. `prepare`는 영상 전체를 클립으로 사용하고 모든 디코딩 프레임의 실제 PTS를 읽습니다. 명목 FPS로 시각을 계산하지 않으며, 영상 자르기·재인코딩은 수행하지 않습니다. 최초 파일 프레임을 원본 `frame_index=0`으로 두므로, 새 파일로 시간축을 재설정했다면 새로운 `stream_id`를 사용합니다.

`examples/session.json`을 복사해 실험의 세션·대상·사용자·예정 회차를 명시합니다. 독립 클립의 PTS가 0에서 시작하면 `source_time_origin_ms`는 보통 0입니다. 행동 시각은 세션 상대 밀리초이며, `session_ms = source_ms - source_time_origin_ms`를 만족해야 합니다. 인증을 대신 수행하는 명령은 없으므로 예제의 가상 인증 ID를 실제 인증 성공으로 취급하지 않습니다.

아래는 행동 구간이 1–3초이고, 정규화 좌표 `[0.1, 0.05, 0.9, 0.95]` 안에 같은 대상이 클립 내내 들어오는 경우의 예입니다. 실제 영상에 맞춰 시각과 좌표를 수정합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m vlm_verification prepare --video "D:\videos\trial01.mp4" --session examples/session.json --config examples/config.cpu.json --event E02 --action-start-ms 1000 --action-end-ms 3000 --target-bbox 0.1 0.05 0.9 0.95 --output-dir experiments/trial01
.\.venv\Scripts\python.exe -X utf8 -m vlm_verification validate --request experiments/trial01/request.json --session experiments/trial01/session.json --config experiments/trial01/config.json
.\.venv\Scripts\python.exe -X utf8 -m vlm_verification run --request experiments/trial01/request.json --session experiments/trial01/session.json --config experiments/trial01/config.json --output-dir outputs
```

`prepare`는 새 폴더에 요청·세션·설정·탐지 설정 JSON을 생성합니다. 원본 영상은 절대 경로로 참조하므로 기록 검토가 끝날 때까지 이동·덮어쓰기하지 않습니다. 기존 세션 추적 정보가 없으면 운영자가 입력한 대상 영역을 첫 프레임의 수동 추적 선언으로 기록하고 세션 revision을 증가시킵니다. 이미 제공된 추적 정보는 유지하며 `UNRESOLVED`이면 실행 입력으로 승인하지 않습니다. 모든 프레임에 같은 대상 bbox를 적용하므로 여러 사람이 겹치거나 대상이 영역을 벗어나는 영상에는 프레임별 `CandidateRegion`을 직접 구성해야 합니다. 탐지 항목은 `UNAVAILABLE`로 기록됩니다.

실제 이벤트 탐지와 연동할 때는 종료된 후보 버전과 `DetectionInfo`, `CandidateMedia`, `SessionContext`를 [예제 요청](examples/request.e02.json)의 계약으로 전달합니다. 전체 JSON 계약은 `src/vlm_verification/contracts.py`에 있습니다. 모델 입력에 정답, 기대 판정, 이전 모델 답변을 넣지 않습니다.

## 판정과 처리 기록

| 이벤트 | `CONFIRMED` 조건 | `REFUTED` 조건 | 나머지 유효한 답변 |
| --- | --- | --- | --- |
| E01 | 취급 `YES` | 취급 `NO` | `UNCERTAIN` |
| E02 | 같은 물체의 약 식별·입 전달 모두 `YES` | 두 항목 중 하나 `NO` | `UNCERTAIN` |
| E03 | 같은 용기의 입 도달·음수 형태 동작 모두 `YES` | 두 항목 중 하나 `NO` | `UNCERTAIN` |

E01 내용물 꺼내기 답변은 별도로 보존합니다. 필수값 누락·허용되지 않은 값·논리적 모순·실제 입력에 없는 근거 번호는 응답 오류입니다. `UNKNOWN`에는 가림·해상도 등 관찰 제한을 실제 입력 번호에 연결한 근거가 필요합니다. E03 확인은 실제 섭취량이나 삼킴, 선행 약 복용, 세션 완료를 의미하지 않습니다.

| CLI 종료 코드 | 의미 |
| --- | --- |
| `0` | `validate`/`prepare` 성공 또는 `run`의 정상 처리 `OK`. `UNCERTAIN`·`REFUTED`도 정상 처리입니다. |
| `1` | 잘못된 JSON·계약·입력 연결, 읽기/쓰기 실패 등 CLI 입력 오류 |
| `2` | `run` 결과 `ERROR`/`NOT_RUN` 또는 CLI 실행 실패. argparse 사용법 오류도 `2`입니다. |

`run`은 표준 출력에 `VerificationResult` JSON을 출력하고 `outputs/<request_id의 SHA-256>/`에 기록합니다. 파일은 처리 단계에 따라 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `request.json`, `session.json`, `execution_config.json` | 실행 시점 입력과 설정 사본 |
| `states.jsonl` | `QUEUED` → `RUNNING` → `FINISHED` 상태 이력 |
| `model_input.json` | 실제 렌더링된 프롬프트, 입력 순서·원본 프레임·ROI·리사이즈·패딩 대응 |
| `runtime.json` | Python·라이브러리·실행 환경 정보 |
| `result.json` | 처리 상태, 항목별 답변, 규칙 판정, 응답 원문, 실패 사유, 처리 시각 |

근거 시각은 응답의 `input_indices` → `model_input.json`의 프레임 → `request.json`의 클립 프레임 → `source_ms`/`session_ms` 순서로 조회합니다. 기본값은 영상 처음과 끝을 포함해 최대 8개 전체 프레임을 균등 선택합니다. `include_rois: true`이면 제공된 ROI도 대응 기록과 함께 사용합니다. 빠른 동작이 선택 프레임 사이에 있을 수 있으므로 영상 길이와 `max_frames`는 기능 검증 후 조정합니다.

같은 요청 ID와 같은 입력·설정의 재실행은 저장된 결과를 반환합니다. 다른 조건으로 재실험하거나 `ERROR`/`NOT_RUN`을 해결한 후 다시 실행할 때는 새 `request_id`를 부여합니다. 같은 ID의 기록을 덮어쓰지 않습니다. 중단된 요청을 동일 ID로 자동 재추론하지 않으며 새 요청으로 제출합니다. 모델 응답은 후보별로 독립적으로 생성하고, 다른 후보에 복사하지 않습니다.

## Python 연동과 검증 범위

```python
from pathlib import Path
from vlm_verification.backend import QwenBackend
from vlm_verification.config import ExecutionConfig
from vlm_verification.contracts import SessionContext, VerificationRequest
from vlm_verification.service import VerificationService, VerificationWorker

folder = Path("experiments/trial01").resolve()
config = ExecutionConfig.model_validate_json((folder / "config.json").read_text("utf-8"))
request = VerificationRequest.model_validate_json((folder / "request.json").read_text("utf-8"))
session = SessionContext.model_validate_json((folder / "session.json").read_text("utf-8"))
service = VerificationService(QwenBackend(folder), config, Path("outputs"), media_root=folder)
result = service.verify(request, session)

# 탐지 루프와 분리하려면 순차 실행 worker의 Future를 이용합니다.
worker = VerificationWorker(service)
future = worker.submit(request, session)
result = future.result()
worker.shutdown()
```

테스트는 가짜 모델 응답과 합성 영상으로 계약·판정·근거 연결·기록 동작을 검증합니다. 모델 가중치를 다운로드하거나 실제 복약 영상 정확도를 측정하지 않습니다. 첫 실제 모델 실행에서는 E01/E02/E03의 명확한 양성, 다른 행동, 가림 사례를 준비하고, 사람의 별도 정답과 저장된 결과를 비교해 정확도·불확실 비율·응답 형식 오류율·처리 시간을 측정할 수 있습니다. 사람 정답 파일은 검증 요청과 분리해 보관합니다.


## 전체 파이프라인 연결

현재 공통 계약의 단일 정의는 `../../shared/contracts/src/medication_contracts/models.py`입니다. 기존 `vlm_verification.contracts` import는 그대로 사용할 수 있습니다. 이 모듈은 로컬 `medication-contracts` 패키지를 의존성으로 설치합니다.

전체 영상 실행은 [루트 안내](../../README.md)의 runtime 환경을 사용합니다. Windows Python 3.10 한글 경로에서 uv의 인터프리터 검사에 인코딩 오류가 발생하면 `python.exe -X utf8 ../../scripts/fix_windows_pth.py .venv`를 실행한 후 동기화합니다. 통합 setup.ps1은 runtime 환경에 이 보정을 자동 적용합니다.
