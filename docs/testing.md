# 검증 기록과 실제 영상 테스트

2026-09-29 Windows / Python 3.10 환경에서 확인했습니다.

웹캠으로 직접 시험할 때는 [웹캠 전체 파이프라인 테스트 절차](webcam-testing.md)를 따릅니다. 장치 확인부터 Q4 실행·관측·결과 검토까지 다루며, 현재 종료 제약과 아직 검증하지 않은 웹캠 성능을 구분합니다.

새 실행 결과는 [테스트 결과 폴더 작성 규칙](test-results.md)에 따라 `<테스트목적>_YYYYMMDD_HHMMSS`로 저장합니다. 아래에 기재된 기존 실영상 결과 폴더는 당시 기록과 경로를 그대로 보존합니다.

## 자동 테스트

**225 passed**. 기존 테스트 180개에 GGUF 백엔드·오류 경계·기존 기록 호환성·전체 실행 백엔드 선택 테스트 18개, 결과 폴더의 KST 시각·동일 초 충돌 보존·이름 검증·CLI 목적 지정 테스트 9개, 전체 입력 영상 저장 테스트 8개, Windows JSON 교체 오류 처리 테스트 10개를 추가했습니다. 기존 통합 테스트에서 새 폴더의 결과 참조, 녹화 파일 생성, 저장 후 재검증도 확인했습니다. Ruff 검사와 CLI 설정 검사도 수행했습니다.

- E02는 알약·준비·음수 확인 없이 생성됩니다. E02/E03 동시 유지, 반복 행동 분리, 물체/입 단서의 손 가림 대응을 검사했습니다.
- 인증 불일치·대상 모호성·추적 손실 시 다른 사람으로 바뀌지 않는지 검사했습니다.
- VFR MP4를 실제 인코딩·디코딩하여 0/37/104/220/365ms 대응, 홀수 해상도, ROI 픽셀 경계를 검사했습니다.
- 전후 영상 부족, 활성 후보의 버퍼 보존, 디스크 한도, 임시 프레임 해제를 검사했습니다.
- 실제 영상 입출력 코드와 통제된 탐지/VLM 응답을 연결해 완료·종료 및 오류·불확실을 구분했습니다.
- 느린 VLM 중 지속 탐지, 별도 후보 보존, 수신 순서/행동 순서 분리, 동일 물체·행동 상충, 저장 후 검증·중복 재전달을 검사했습니다.
- 실제 Windows 파일 핸들로 교체를 차단한 뒤 잠금 해제 시 저장이 복구되는지 확인했습니다. 오류 코드별 재시도, 지속 실패 시 원본·임시 파일 보존, 다른 오류의 즉시 전달, 진단 로그 실패 격리, 세션 이력 중복 방지도 검사했습니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 -m pytest -q
./runtime/.venv/Scripts/python.exe -X utf8 -m ruff check shared/contracts/src modules/authentication-tracking/src modules/event-detection/src modules/session-decision/src modules/session-video-management/src runtime/src tests scripts
```

## 실제 모델 어댑터 점검

입력: `medication-local-assets/tests_Qwen3-VL-2B-Instruct-Q8_0.gguf/robot3d__A003_P002_G004_C004.mp4`.

공식 YOLO11n과 MediaPipe face/hand/pose 모델을 사용했습니다. 분석 프레임 58개, TRACKED 58개, UNRESOLVED 0개, 손·입·포즈 관측 각각 58개, 객체 관측 누계 162개, E02 후보 3개였습니다. E01/E03 후보는 0개입니다. 객체 누계는 프레임별 관측 수이며 고유 물체 수가 아닙니다.

결과는 `outputs/real-detection-smoke-robot/summary.json`, 원본 좌표·시각 관측은 `observations.jsonl`, 후보는 `candidates.json`에 있습니다. 라이브러리 버전은 `versions.json`, 모델 다운로드 URL·SHA256은 `models/download-manifest.json`에 보존했습니다. 설치 중 점검은 동일한 잠금 버전의 준비된 PyTorch wheel을 사용했으며 이후 탐지 의존성 설치도 완료했습니다.

Windows 한글 경로의 MediaPipe model_asset_path 열기 실패를 실제로 재현했고, model_asset_buffer로 바이트를 전달하도록 수정한 뒤 같은 영상으로 통과했습니다. Python 3.10/uv의 editable .pth 인코딩 문제도 보정 스크립트로 처리했습니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 scripts/smoke_real_detection.py --video "../medication-local-assets/tests_Qwen3-VL-2B-Instruct-Q8_0.gguf/robot3d__A003_P002_G004_C004.mp4" --output outputs/video-robot3d-detection
```

이 스크립트는 인증·복약 완료를 주장하지 않는 어댑터 점검입니다. 정답 라벨과 비교한 정확도/재현율은 측정하지 않았습니다. 이 테스트의 E02 3개는 검증된 복약 행동 수가 아닙니다.

## 실제 전체 실행 전 준비

`configs/pipeline.local.json`에 위 실제 영상과 내려받은 탐지 모델 경로를 마련했습니다. 먼저 대상자를 확인해 operator_confirmed를 설정하거나 외부 인증 결과를 연결합니다. 기본 YOLO11n에 없는 약·포장 클래스의 E01 평가는 사용자 학습 가중치로 진행합니다.

공식 Qwen3-VL-2B-Instruct Transformers 체크포인트를 `models/Qwen3-VL-2B-Instruct`에 준비했습니다. revision은 `89644892e4d85e24eaac8bacfd4f463576704203`이며 파일 11개, 총 4,266,647,442바이트를 받았습니다. 원격 메타데이터의 크기·LFS SHA256 또는 Git blob 해시와 대조했고, `download-manifest.json`에 파일별 SHA256을 보존했습니다. 이 BF16 설정은 `configs/pipeline.bf16.local.json`에 보존했고, 현재 `pipeline.local.json`은 아래 Q4 모델을 사용합니다.

`setup.ps1 -Mode full`로 추론 의존성을 설치했습니다. torch 2.14.0+cpu, transformers 4.57.6, accelerate 1.15.0, huggingface-hub 0.36.2, tokenizers 0.22.2, safetensors 0.8.0 환경입니다. 설치 후 자동 테스트 180개와 추가 스크립트 Ruff 검사가 통과했습니다. `doctor`는 라이브러리·모델·영상 항목이 모두 통과했고, 운영자가 아직 확인하지 않은 `operator_confirmed`만 false입니다.

`scripts/smoke_qwen_model.py`로 실제 영상의 첫 프레임을 224×126으로 줄여 네트워크 없이 추론했습니다. 기존 QwenBackend의 native processor와 동일한 모델 로딩 옵션(cpu, float32, sdpa)을 사용해 최대 32토큰의 장면 설명을 생성했습니다. 프로세서·모델 로딩 18.58초, 생성 25.59초로 성공했습니다. 결과·버전·설정은 `outputs/qwen-setup-smoke/summary.json`에 있습니다. 이 시간은 작은 이미지 한 장의 설치 점검 기록이며, 여러 프레임·ROI를 사용하는 전체 파이프라인의 지연을 나타내지 않습니다.

이 BF16 점검은 단일 프레임 모델 로딩·입출력 호환성 확인입니다. 대상자 확인 후 doctor/run으로 전체 실행할 수 있으며, run --detect-only는 클립과 요청까지만 생성합니다.

## Q4_K_M 실영상 후보 검증

공식 Q4_K_M 언어 모델과 F16 mmproj, llama.cpp b10991 CUDA 12.4 실행 프로그램을 고정 SHA256으로 검사했습니다. 기존 공용 로컬 자산을 현재 작업공간으로 복사했습니다. 언어 모델은 GTX 1650 Ti의 GPU, 시각 인코더는 CPU에서 실행했습니다. 실행 중 관찰된 장치 전체 GPU 메모리 사용량은 3,383MiB였으며 프로세스 단독 피크 측정값은 아닙니다.

실제 YOLO·MediaPipe 관측과 E02 후보 3개를 `scripts/smoke_qwen_gguf.py`로 재생했습니다. 원본 영상에서 전후 클립을 생성하고, 후보마다 전체 프레임 4장(각 672×384)을 입력하여 실제 Q4 추론·응답 파싱·근거 매핑·세션 판단기를 실행했습니다. 3개 모두 processing_status=OK, 판정은 UNCERTAIN이었습니다. 완료 근거가 생성되지 않았습니다. 인증은 테스트용 세션으로 대체해 생체인증·실사용자 확인·전체 라이브 캡처 검증으로 간주하지 않습니다.

서버 로딩을 포함한 검증 구간은 264.86초였습니다. 서버가 보고한 후보별 입력 처리+생성 시간은 약 90.45초, 72.52초, 61.73초입니다. 실시간 판정 속도를 달성했다는 의미가 아니며, 앞선 BF16 한 장/32토큰 점검과 입력·출력 예산이 달라 직접 속도 비교할 수 없습니다. 원본 서버 응답은 모두 eos로 끝났고 입력 잘림은 없었습니다. 프로세스는 완료 후 종료됨을 확인했습니다.

결과는 `outputs/q4-candidate-smoke/summary.json`, 입력·근거·판정은 `verification`, 서버 옵션·template·로그·시간은 `llama-runtime`에 있습니다. [4비트 실행 안내](qwen-4bit.md)를 참고합니다. 별도의 자동 통합 테스트는 실제 영상 입출력과 모의 서버 응답으로 GGUF 백엔드 선택부터 COMPLETE·세션 종료까지 검사합니다.

자동 테스트에서 쓰는 통제된 VLM 응답은 모델 정확도 평가가 아닙니다. 실제 Qwen 출력 품질, E01 사용자 학습 모델, 생체인증, Jetson, 실제 카메라의 연속 처리량·지연·프레임 유실은 아직 검증하지 않았습니다. 임계값과 전후 구간은 초기값이므로 라벨이 있는 실제 영상으로 조정해야 합니다.

실모델 API 기준: [Ultralytics 예측](https://docs.ultralytics.com/modes/predict/), [MediaPipe Face](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python), [Hand](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), [Pose](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).

## 전체 입력 영상 저장 점검

`InputRecorder`를 추가해 웹캠/캡처 입력에서 수신한 프레임 전체를 H.264 MP4로 기록합니다. 자동 테스트는 가변 PTS·홀수 해상도 패딩·원본 프레임 대응, 빈 입력, 시각 불연속, 분석 시작 전 프레임 포함, 녹화 비활성화, 첫 대상 선택 실패, `KeyboardInterrupt`·처리 오류 후 파일 마감과 재디코딩을 검사합니다. 느린 VLM 작업 정리 전에 파일이 마감되는지도 확인했습니다. 합성 입력과 통제된 탐지 응답을 사용한 검사이며 물리 웹캠 테스트는 아닙니다.

추가로 기존 `robot3d__A003_P002_G004_C004.mp4`(1920×1080, 20fps, 116프레임)를 녹화 코드로 다시 저장하고 재디코딩했습니다. 입력 116프레임과 출력 116프레임, 첫 프레임을 기준으로 한 전체 PTS 배열이 일치했습니다. 이 점검은 영상 저장 기능만 검사했으며 인증·VLM·웹캠 수신은 실행하지 않았습니다.

기록은 `outputs/video-recording-validation_20260929_151638/summary.json`에 있습니다. 같은 폴더의 `recording/input.mp4`는 저장 영상, `frames.jsonl`은 원본 프레임·시각 대응, `metadata.json`은 녹화 상태입니다.

현재 `configs/pipeline.webcam.local.json`에 `record_input=true`를 설정했습니다. `doctor`에서 `recording_encoder=true`를 포함해 `ready=true`를 확인했습니다. 실제 웹캠 연속 촬영과 녹화에 따른 처리량·지연 변화는 별도 검증 대상입니다.

## 웹캠 연속 입력 시험 — 2026-09-29 15:28:21 KST

`outputs/webcam-pipeline-test01_20260929_152821`의 사용자 실행 기록을 사후 검토했습니다. Integrated Webcam에서 1280×720 영상 96프레임을 수신·저장했고, 완료된 분석 관측 69개가 모두 TRACKED였습니다. 입·포즈는 69프레임, 오른손은 19프레임에서 관측됐으며 E02 후보 1개가 생성됐습니다. 저장된 약 10초 MP4는 96프레임 모두 재디코딩되고 프레임별 상대 PTS도 보존됨을 확인했습니다.

세션 상태 파일을 임시 파일로 교체하는 `os.replace()`가 Windows 접근 거부로 실패해 실행이 중단됐습니다. 후보 후행 영상 확보 전이어서 후보 클립·검증 요청은 생성되지 않았고 Qwen은 실행되지 않았습니다. 웹캠 수신·추적·후보 탐지·중단 시 녹화 마감은 확인했지만 전체 파이프라인 성공이나 모델 판정 정확도 검증으로 간주하지 않습니다.

오류 원인과 기록 간 revision 차이, 영상·후보 수치는 [이번 실행의 상세 검토](../outputs/webcam-pipeline-test01_20260929_152821/test-review.md)에 정리했습니다.

## Windows 세션 저장 오류 수정 및 영상 재처리 — 2026-09-29

공통 JSON 저장의 파일 교체 단계에 WinError 5/32/33 재시도를 추가했습니다. 최대 7회 시도하며 대기 시간 합계는 940ms입니다. 세션 이력을 다시 추가하지 않고 같은 완성된 임시 파일의 교체만 재시도합니다. 복구·실패 내역은 대상 파일과 같은 디렉터리의 `persistence-events.jsonl`에 남깁니다. 지속 실패 시 기존 JSON과 미반영 `.tmp`를 보존하고 오류를 전달합니다. 최초 실행에서 파일을 잠갔던 프로세스는 확인되지 않았습니다.

회귀 테스트 10개를 포함한 전체 225개 테스트와 Ruff 검사가 통과했습니다. `CreateFileW`로 삭제 공유를 허용하지 않는 실제 Windows 읽기 핸들을 열어 교체 오류를 재현하고, 핸들을 닫으면 저장이 완료됨을 확인했습니다. 통합 테스트에서는 `session.json`과 `session-view.json` 각각에 오류를 주입해 파이프라인 지속·검증 결과 수신·연속된 세션 revision을 확인했습니다. 해당 통합 테스트의 탐지·VLM 응답은 통제된 응답입니다.

기존 실패 실행의 `recording/input.mp4`를 실제 YOLO11n·MediaPipe로 탐지 전용 재처리했습니다. 96프레임을 읽어 70프레임을 분석했고, 세션 revision 71까지 저장한 뒤 `END_OF_INPUT`으로 정상 종료했습니다. 세션 이력 71행에 중복·누락이 없고 마지막 이력과 `session.json`이 일치했습니다. E02 클립·검증 요청은 각 1개이며 처리 오류와 잔여 `.tmp`는 없습니다. 이 재처리 중에는 파일 교체 충돌이 발생하지 않았으므로 충돌 복구 자체의 검증 근거는 위 회귀 테스트입니다.

결과는 [재처리 요약](../outputs/video-webcam-persistence-replay_20260929_154334/summary.json), [저장 일관성 확인](../outputs/video-webcam-persistence-replay_20260929_154334/persistence-validation.json)에 있습니다. 입력 영상이 끝나 EOF에서 후보를 마감했으며 실제 클립 범위는 4752~9968ms로 요청 범위 4724~10696ms보다 짧습니다. Qwen 추론과 신규 웹캠 촬영은 실행하지 않았고, 전체 라이브 테스트 및 장시간 안정성은 별도 확인 대상입니다.

## Livinglab A003 파일 영상 전체 실행 — 2026-09-29 16:37:55 KST

사용자가 지정한 `A003_복약/livinglab__A003_P201_G002_H120.mp4`(1920×1080, 20fps, 6.7초)를 실제 YOLO11n·MediaPipe·Qwen3-VL-2B Q4_K_M으로 실행했습니다. 영상의 단일 대상을 오프라인 테스트용 operator 모드로 연결했으며 생체인증 시험은 아닙니다. [전용 설정](../configs/pipeline.livinglab-A003-P201-G002-H120.local.json)에 입력 경로와 조건을 보존했습니다.

134프레임 전체를 읽고 67프레임을 분석했으며 추적은 모두 TRACKED였습니다. E02 후보·클립·요청·결과는 각각 4개이고, 결과를 세션 판단 단계까지 모두 수신한 뒤 `FINISHED / END_OF_INPUT / FULL`, 종료 코드 0으로 정상 종료했습니다. 처리 오류는 0건이며 세션 이력과 클립의 전체 프레임·PTS 일치도 확인했습니다.

판정은 REFUTED 2건·UNCERTAIN 2건으로 복약 완료는 확정되지 않았습니다. 전체 명령은 393.5초가 걸렸습니다. 약 식별 check와 근거 설명의 불일치, 용기와 복용 물체의 혼동 가능성, 모든 check의 근거가 첫 입력 프레임만 참조하는 문제가 관찰됐습니다. 연결 성공을 모델 정확도나 실시간 성능 검증으로 해석하지 않습니다.

결과는 [상세 검토](../outputs/video-livinglab-A003-P201-G002-H120-q4_20260929_163755/test-review.md), [실행 요약](../outputs/video-livinglab-A003-P201-G002-H120-q4_20260929_163755/summary.json), [집계·검증 기록](../outputs/video-livinglab-A003-P201-G002-H120-q4_20260929_163755/test-review-metrics.json)에 있습니다. 같은 폴더에 전체 콘솔 로그와 Qwen 서버 로그, 입력 프레임 비교 이미지, 후보별 영상·응답을 보존했습니다.
