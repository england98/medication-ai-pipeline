# Qwen3-VL-2B 4비트 실행

공식 [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/52d6c8ffea26cc873ac5ad116f8631268d7eb503)의 revision `52d6c8ffea26cc873ac5ad116f8631268d7eb503`을 사용합니다.

| 구성 | 파일 | 크기 |
| --- | --- | --- |
| 언어 모델 | Qwen3VL-2B-Instruct-Q4_K_M.gguf | 1,107,409,952바이트 |
| 시각 인코더·프로젝터 | mmproj-Qwen3VL-2B-Instruct-F16.gguf | 819,394,848바이트 |

합계 약 1.93GB입니다. Q4_K_M은 일부 텐서에 더 높은 정밀도를 사용하는 4비트 계열이며 시각 인코더는 F16입니다. 모든 텐서가 4비트라는 의미는 아닙니다. 실행 시 KV 캐시·연산 버퍼 메모리가 추가됩니다.

## 준비와 실행

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Mode detection
./runtime/.venv/Scripts/python.exe -X utf8 scripts/fetch_qwen_gguf.py
# 기존 공용 자산을 재사용할 경우:
# ./runtime/.venv/Scripts/python.exe -X utf8 scripts/fetch_qwen_gguf.py --reuse-assets ../medication-local-assets
```

모델은 `models/Qwen3-VL-2B-Instruct-GGUF`, Windows CUDA 12.4용 llama.cpp b10991은 `runtime/llama.cpp/b10991`에 준비됩니다. 모델·ZIP의 고정 SHA256을 검사하고 manifest를 남깁니다. `configs/pipeline.q4.example.json`이 전체 파이프라인 설정 예시입니다.

현재 `configs/pipeline.local.json`은 Q4 설정입니다. 기존 BF16 설정은 `configs/pipeline.bf16.local.json`에 보존되어 있습니다. 로컬 설정에서 영상 경로·대상자·회차를 지정하고, 영상 대상자를 확인한 뒤 `authentication.operator_confirmed=true`로 설정합니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline doctor --config configs/pipeline.local.json
./runtime/.venv/Scripts/python.exe -X utf8 -m medication_pipeline run --config configs/pipeline.local.json
```

실행기는 GGUF 서버를 자동으로 시작하고 같은 서버를 후보 요청들에 재사용합니다. 별도 서버를 띄울 필요가 없습니다. 종료 시 자신이 시작한 서버만 정리합니다. 기존 `--detect-only`, `verify --run-dir`도 선택된 백엔드를 유지합니다.

## 4GB GPU의 초기 설정

- `vlm.backend="llama_cpp"`: GGUF 백엔드 사용.
- `vlm.device="cuda"`, `dtype="auto"`: GGUF에 저장된 텐서 정밀도로 실행. PyTorch CUDA 빌드와 독립적인 llama.cpp CUDA 런타임 사용.
- `llama_cpp.gpu_layers=99`, `mmproj_offload=false`: 언어 모델은 GPU, 시각 인코더는 CPU.
- `llama_cpp.context_size=16384`: 입력 메타데이터·이미지와 출력의 토큰 한도.
- `max_frames=4`, `include_rois=false`: 후보당 전체 프레임 4장까지 사용. ROI 추가 입력은 현재 꺼져 있으며 미세한 약 식별에는 불리할 수 있음.
- `max_pixels=262144`, `max_new_tokens=768`: 기존 픽셀·응답 한도 유지.

CPU 전용 실행은 `vlm.device="cpu"`로 바꿉니다. GPU 메모리 부족 시 context_size/입력 예산을 명시적으로 조정하거나 CPU로 실행합니다. 메모리와 지연은 실제 입력 길이·해상도·장면에 따라 측정해야 합니다.

## 입력과 결과 기록

기존 클립 PTS·프레임 선택·TARGET 유지·ROI 원본 좌표 검사를 사용합니다. 이미지 크기를 32픽셀 격자와 native image token 한도에 맞춘 뒤 PNG로 전달하며 크기·입력 순서를 `model_input.json`에 기록합니다. 서버가 선언한 media marker와 native chat template으로 만든 문자열을 저장하고 동일한 문자열을 추론에 전달합니다.

기존 시스템 프롬프트와 E01/E02/E03 질문을 사용합니다. JSON 스키마 제약은 응답 형식만 제한합니다. YES/NO/UNKNOWN과 근거는 모델이 생성하고 기존 검증기·세션 판단 규칙이 검사합니다. 컨텍스트 초과, 응답 잘림, 모델/프로젝터 해시 불일치, 서버 종료는 처리 오류로 남습니다. 입력을 몰래 잘라내거나 오류를 판정값으로 바꾸지 않습니다.

`outputs/<테스트목적>_YYYYMMDD_HHMMSS/llama-runtime`에 서버 로그·native template 정보·모델 해시·실행 옵션·원본 서버 응답·처리 시간이 남습니다. 시각은 KST이며 목적 지정 방법은 [결과 폴더 작성 규칙](test-results.md)을 참고합니다. 서버는 임시 포트의 `127.0.0.1`에만 바인딩하며 실행마다 생성한 키를 사용합니다. API 키는 기록하지 않습니다. 영상·이미지를 외부 API에 보내지 않습니다.

## 실영상 후보 점검

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 scripts/smoke_qwen_gguf.py --output outputs/video-qwen-q4-candidate-verification
```

기존 `smoke_real_detection.py`에서 실제 영상·YOLO·MediaPipe로 얻은 관측과 후보를 재사용해 클립 생성 → 실제 Q4 검증 → 세션 판단기를 점검합니다. 테스트용 세션을 사용하므로 대상자 인증이나 실제 복약 완료를 주장하지 않습니다. `--output`에 지정한 경로 끝에 KST 날짜·시각을 붙여 매번 새 폴더를 만듭니다. 새 탐지 기록을 사용하려면 `--detections`에 실제 결과 폴더 경로를 지정합니다. 실시간 웹캠 처리량·모델 정확도 평가는 별도입니다.

백엔드 구현 기준: [llama.cpp b10991 server API](https://github.com/ggml-org/llama.cpp/blob/b10991/tools/server/README.md).
