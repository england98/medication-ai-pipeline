# 통합 실행기

src/medication_pipeline이 실제 영상 입력과 모든 모듈을 연결합니다. 설치는 루트 README를 참고합니다.

현재 로컬 설정은 Q4_K_M GGUF입니다. `vlm.backend=llama_cpp`이면 실행기가 로컬 llama.cpp 프로세스를 자동으로 시작·재사용·종료하며, `transformers`이면 기존 BF16 백엔드를 사용합니다. [4비트 실행 안내](../docs/qwen-4bit.md)를 참고합니다.

- validate --config: JSON·설정 관계 검사.
- doctor --config [--detect-only]: 의존성·영상·모델 자산 점검.
- run --config [--test-name 목적]: 탐지·클립 생성과 별도 VLM 스레드 실행.
- run --config --detect-only: 고정 요청 저장. 가짜 결과나 NOT_RUN을 생성하지 않음.
- verify --run-dir: 저장된 요청을 같은 모델 설정으로 검증. 모델 변경 실험은 새 run을 생성.

새 실행은 `outputs/<테스트목적>_YYYYMMDD_HHMMSS`에 저장하며 시각은 KST입니다. 목적은 CLI, 설정의 `test_name`, 입력·모드별 기본값 순으로 선택합니다. 시작 시 실제 경로를 출력하고 `test-run.json`에 목적·시작 시각을 기록합니다. 기존 폴더의 `verify`는 같은 위치에서 이어집니다. [결과 폴더 작성 규칙](../docs/test-results.md)을 참고합니다.

종료 코드: 0=실행 성공, 2=설정·실행 오류, 3=요청 ERROR/NOT_RUN 존재, 130=사용자 중단. 코드 0 자체가 복약 완료를 뜻하지 않으므로 summary.json의 session_result를 확인합니다.

queue/*.json에 요청·세션 사본이 저장됩니다. VLM admission 수를 제한하고 나머지는 디스크에 보존합니다. 검증 중에도 탐지하며 완료 시 신규 분석을 멈춥니다. 준비한 요청은 결과를 끝까지 보존합니다. EOF는 완료/최종 미완료 판정이 아닙니다.

VLM 시도가 불완전하면 같은 request_id의 자동 재추론을 거부합니다. 기록과 .running/states.jsonl을 조사하고 재실험은 새 run에서 수행합니다. 실행 중 프로세스의 잠금 파일을 삭제하지 않습니다.

파일/VFR 클립을 테스트합니다. FFmpeg input_format/options로 카메라 입력도 연결하지만 실제 장치 프레임 유실·저장 처리량·지연은 목표 환경에서 검증해야 합니다.

`record_input`을 생략하면 캡처 입력은 자동으로 전체 수신 영상을 저장하고 파일 입력은 중복 녹화하지 않습니다. true/false로 명시할 수 있습니다. `recording/input.mp4`와 원본 프레임 대응 `frames.jsonl`, 종료 상태 `metadata.json`을 남깁니다. 분석 시작 프레임과 주기에 무관하게 수신 프레임을 기록하며, 원본 PTS의 상대 간격을 유지합니다. 녹화는 캡처 루프 종료 시 마감하고, 인터럽트·오류 시에도 VLM 작업 정리 전에 파일을 마무리합니다. `Ctrl+C`의 후보 마감·결과 반영 제약은 그대로입니다.
