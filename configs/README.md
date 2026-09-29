# 통합 실행 설정

현재 로컬 테스트 설정은 Q4_K_M입니다. `pipeline.q4.example.json`과 [4비트 실행 안내](../docs/qwen-4bit.md)를 참고합니다. `vlm.backend`는 `transformers` 또는 `llama_cpp`이며, GGUF 런타임·시각 인코더 경로도 이 설정 파일 기준으로 해석합니다.

`pipeline.example.json`을 `pipeline.local.json`으로 복사해 편집합니다. 상대 경로는 설정 파일 위치 기준이며 로컬 설정·모델·영상·출력은 버전 관리에서 제외합니다.

| 항목 | 내용 |
| --- | --- |
| source, start_frame | 원본 영상과 첫 인증·분석 프레임. 앞선 프레임은 분석에서 제외 |
| output_dir, test_name | 결과 상위 디렉터리와 테스트 목적. `<test_name>_YYYYMMDD_HHMMSS`를 KST 기준으로 생성. 이름 생략 시 입력·실행 모드에 따라 자동 지정 |
| input_format, capture_options | 파일은 null. FFmpeg 캡처는 Windows dshow, source는 video=장치명, options는 video_size/framerate 등 |
| record_input | true=수신한 전체 영상 저장, false=미저장, null/생략=캡처 입력만 자동 저장. `recording/input.mp4`, `frames.jsonl`, `metadata.json` 생성 |
| occurrence | 실험 대상 사용자·예정 회차·UTC 예정 시각 |
| authentication | operator와 명시적 확인, 또는 external과 result_path. bbox는 원본 0~1 좌표 |
| models | YOLO 객체/사람 모델, MediaPipe face/hand/pose task, 장치·크기·점수·class_map |
| tracking | IoU·모호성·사람 겹침·프레임 공백 허용값. 손실 후 자동 재연결 없음 |
| detection | 분석 주기, 접근 이력, 최소 지속·이탈 시간, 상대 거리·속도, 클래스 목록 |
| video | 유형별 선행·후행 ms, 순환 버퍼, 임시 디스크 제한, ROI 패딩·최대 수 |
| vlm | 기존 ExecutionConfig. 40자리 모델 revision, 장치·정밀도·이미지/픽셀/토큰 한도 |
| decision | 규칙 버전, 동일 물체 연결 시간, 명시적 비약물 설명의 상충 패턴 |
| max_pending | VLM worker에 제출하는 최대 요청 수. 동시 추론은 1개 |
| max_queued | 디스크 요청 큐 한도. 초과 시 요청을 보존하고 실행 오류로 종료 |

거리·속도는 대상 bbox의 픽셀 높이로 정규화하고 종횡비를 반영합니다. 시각은 원본 PTS를 사용합니다. 수치는 평가로 조정할 초기 구현값이며 확정된 성능 기준이 아닙니다.

버퍼는 `최대 선행 영상 + 접근 이력 + 후보 최소 지속 + 분석 주기` 이상이어야 합니다. 활성 후보가 참조하는 영상은 버퍼 범위를 벗어나도 보존합니다. 디스크 한도 초과는 프레임을 조용히 버리지 않고 오류로 기록합니다.

class_map은 모델 클래스 이름을 규칙 클래스 이름에 연결합니다. 예: medicine_pack → package. 일반 bottle을 약병으로 바꿔 관측되지 않은 의미를 주입하지 않습니다. 학습 모델의 실제 라벨에 맞춥니다.

외부 인증은 공통 AuthenticationResult 계약입니다. 성공 사용자·회차가 설정과 일치하고 기준 프레임의 frame_index/source_ms/크기/stream_id가 입력과 일치해야 합니다. 인증 frame.session_ms는 null입니다.

`doctor`는 의존성·자산 존재·오프라인 Qwen 캐시를 검사합니다. 가중치 호환성은 실모델 실행으로 확인해야 합니다. `validate`는 파일 존재와 독립적으로 설정 형식을 검사합니다.

실행별 목적은 `run --test-name webcam-q4-empty-hand`로 지정할 수 있으며 설정의 `test_name`보다 우선합니다. CLI 지정은 원본 설정을 수정하지 않습니다. 이름 규칙·점검 스크립트·중복 처리 방식은 [테스트 결과 폴더 작성 규칙](../docs/test-results.md)을 참고합니다.

`scripts/fetch_qwen_model.py`는 공식 `Qwen/Qwen3-VL-2B-Instruct`의 revision `89644892e4d85e24eaac8bacfd4f463576704203`을 내려받아 파일별 크기와 해시를 검증합니다. 기본 저장 위치를 사용하면 `vlm.model_id="../models/Qwen3-VL-2B-Instruct"`, `vlm.local_files_only=true`로 설정합니다. `download-manifest.json`에는 원본 모델 ID·revision·파일별 SHA256이 남습니다. CPU 실행의 기본 dtype은 float32입니다.

인증과 독립적으로 설치·모델 로딩을 점검하려면 다음 명령을 사용합니다. 설정의 실제 영상에서 첫 프레임 하나를 읽어 작은 이미지로 줄이고 최대 32토큰을 생성합니다. 복약 판정·전체 파이프라인 성능 평가는 수행하지 않습니다.

```powershell
./runtime/.venv/Scripts/python.exe -X utf8 scripts/smoke_qwen_model.py --config configs/pipeline.bf16.local.json
```
