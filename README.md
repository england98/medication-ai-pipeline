# Medication AI Pipeline

[확정 설계](https://medication-ai-vision-docs.vercel.app/architecture/event-detection)의 인증·추적 → 이벤트 탐지 → 후보 영상 → Qwen3-VL 검증 → 세션 판단·종료를 구현한 로컬 Python 파이프라인입니다.

## 시작하기

**4비트 전체 파이프라인 테스트:** 현재 로컬 설정은 Q4_K_M GGUF와 llama.cpp를 사용합니다. [4비트 준비·실행 안내](docs/qwen-4bit.md)를 따릅니다. 아래 설치 예시는 Transformers/BF16 경로도 함께 준비하는 방법입니다.

**웹캠 테스트:** [웹캠 전체 파이프라인 테스트 절차](docs/webcam-testing.md)에 장치·설정 준비, 실행, 진행 로그·결과 확인, 중단 후 처리와 시험 항목을 정리했습니다. 현재 시간 제한·정상 수동 종료 기능은 없어 문서의 종료 제약을 먼저 확인합니다.

웹캠 입력은 기본적으로 결과 폴더의 `recording/input.mp4`에 저장합니다. `record_input`으로 저장 여부를 지정할 수 있습니다. 정상 종료와 일반적인 `Ctrl+C` 중단 시 수신한 구간의 파일을 마무리하며, 프레임별 원본 시각은 `recording/frames.jsonl`에 남습니다. 음성은 저장하지 않습니다.

PowerShell에서 작업공간 루트를 기준으로 실행합니다. Python 3.10과 uv를 사용합니다.

```powershell
# 개발 환경: dev / 실제 탐지: detection / 전체 모델 실행: full
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Mode full
./runtime/.venv/Scripts/python.exe -X utf8 scripts/fetch_detection_models.py
./runtime/.venv/Scripts/python.exe -X utf8 scripts/fetch_qwen_model.py
if (-not (Test-Path configs/pipeline.local.json)) { Copy-Item configs/pipeline.example.json configs/pipeline.local.json }
```

로컬 설정의 영상·모델 경로, 사용자·예정 회차를 편집합니다. 상대 경로는 **설정 파일의 디렉터리 기준**입니다. 영상 대상자를 직접 확인한 오프라인 실험에서는 `authentication.operator_confirmed=true`로 설정합니다. 여러 사람이 있으면 `target_bbox`로 첫 분석 프레임의 대상을 지정합니다. 외부 인증 결과 입력도 지원합니다.

Transformers 백엔드는 BF16 체크포인트를 사용합니다. 위 다운로드 스크립트는 고정 revision의 공식 파일을 `models/Qwen3-VL-2B-Instruct`에 저장하고 크기·해시를 검증합니다. 해당 경로는 `backend="transformers"`와 로컬 체크포인트 경로를 지정합니다. GGUF는 `backend="llama_cpp"`와 별도의 실행 프로그램·시각 인코더 설정을 사용합니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline doctor --config configs/pipeline.local.json
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline run --config configs/pipeline.local.json
```

VLM 준비 전에는 `--detect-only`로 실제 탐지·클립·ROI·검증 요청까지 생성하고, 같은 모델 설정을 준비한 후 `verify`로 이어서 검증할 수 있습니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline run --config configs/pipeline.local.json --detect-only
# --run-dir은 위 실행에서 출력된 실제 결과 경로로 바꿉니다.
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline verify --run-dir outputs/video-detection_YYYYMMDD_HHMMSS
./runtime/.venv/Scripts/python.exe -X utf8 -m pytest -q
```

## 모듈 구성

| 위치 | 구현 |
| --- | --- |
| shared/contracts | 단일 공통 Pydantic 계약·인증·완료·종료·조회 형식 |
| modules/authentication-tracking | 운영자/외부 인증 연결, 대상 추적, UNRESOLVED 처리 |
| modules/event-detection | YOLO, MediaPipe 손·얼굴·포즈, 물체 연결, 시간·기하 규칙, E01/E02/E03 후보 |
| modules/session-video-management | 세션 수명, 디스크 순환 버퍼, 후보 구간 보존, 실제 PTS 클립·ROI |
| modules/vlm-event-detection | 기존 Qwen3-VL 검증·근거·오류 기록, 기존 import 호환 |
| modules/session-decision | 결과 누적·중복 제거·동일 물체/행동 상충 검사·E02 기반 완료 |
| runtime | 독립 실행 패키지, 디스크 요청 큐, VLM 작업 스레드, CLI |

실행마다 `outputs/<테스트목적>_YYYYMMDD_HHMMSS`에 설정·환경·모델/영상 해시, 관측, 후보 이력, 클립·ROI, 고정 요청, VLM 입력·응답·결과, 완료·종료·조회 결과를 보존합니다. 시각은 KST이며, 목적은 설정의 `test_name` 또는 `run --test-name webcam-pipeline-q4`로 지정합니다. 시작 시 실제 경로를 출력합니다. [결과 폴더 작성 규칙](docs/test-results.md)을 참고합니다. `summary.json`이 실행 요약이고 `session/session-view.json`이 외부 조회 형식입니다.

## 적용 범위

- 인증 알고리즘은 설계에서 미정입니다. 운영자 확인 또는 외부 AuthenticationResult를 받아 연결하며 생체인증으로 취급하지 않습니다.
- 기본 YOLO11n은 일반 객체 모델입니다. E01의 약·포장 탐지는 해당 클래스가 학습된 가중치가 필요합니다. E02는 약 검출 없이 손–입 접근으로 생성할 수 있습니다.
- 추적 손실·모호한 사람 겹침 이후에는 대상을 자동 교체하지 않습니다. 현재 추적기는 재인증한 새 세션이 필요합니다.
- 오류·미실행·불확실과 영상 종료는 최종 미완료 판정이 아닙니다. 완료 근거가 없으면 ACTIVE·null을 유지합니다.
- COMPLETE는 영상 기반 완료 규칙 충족입니다. 실제 삼킴·처방 일치·전체 복용량 확인값은 출력하지 않습니다.

[실행 설정](configs/README.md) · [구현·설계 대응](docs/development.md) · [검증 범위](docs/testing.md) · [기존 VLM 안내](modules/vlm-event-detection/README.md)
