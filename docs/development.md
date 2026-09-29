# 구현 구조와 설계 대응

기준은 2026-09-29에 조회한 [설계 2-1~2-8](https://medication-ai-vision-docs.vercel.app/architecture/pipeline)입니다. 로컬 medication-ai-vision-design의 설계 원본과도 대조했습니다. 형제 프로젝트·데이터는 수정하지 않았습니다.

각 모듈은 독립 pyproject.toml·uv.lock을 유지합니다. runtime이 로컬 경로 의존성으로 연결합니다. 루트는 통합 테스트·문서 위치입니다. 현재 배치는 같은 프로세스의 탐지 흐름과 별도 VLM 작업 스레드이며 계약 객체와 ResourceRef가 모듈 경계입니다.

| 설계 | 구현 |
| --- | --- |
| 2-2 인증·추적 | authentication_tracking.tracking: 운영자/외부 인증, 초기 사람 선택, 공간 연속성, 손실 잠금 |
| 2-3 탐지 | event_detection.perception: 실모델·원본 좌표 변환·포즈 손목 연결. detector: 유형·손/물체별 상태 머신 |
| 2-4 VLM | vlm_verification: 고정 요청, Qwen3-VL, 입력·응답·근거 검사, 오류와 UNKNOWN 분리 |
| 2-5 판단 | session_decision: 수신 결과 누적, 행동 시각·물체 연결, 상충 검사, 완료 1회 |
| 2-6 관리 | session_video: 인증 후 시작, 완료 후 종료, JPEG 버퍼, 활성 후보 보존, VFR MP4·ROI |
| 2-7 계약 | medication_contracts: 단일 공통 정의. 기존 vlm_verification.contracts는 호환 import |
| 2-8 실행 | medication_pipeline: 입력 지속, 디스크 요청 큐, 비동기 VLM, 결과 수신·종료·기록 |

E02 후보는 알약 탐지와 E01/E03 확인을 요구하지 않습니다. E02/E03는 같은 시간에도 각각 보존합니다. 접근 이력을 시작에 연결하고 지속 관측은 같은 ID의 revision 증가로 전달합니다. 이탈 후 새 행동은 새 ID입니다. 좌우 손의 독립 움직임은 개별 채널, E01/E03는 물체별 채널입니다. 후보 수를 약 수량으로 해석하지 않습니다.

클립은 원본 시각을 millisecond PTS로 재인코딩합니다. 실제 확보 범위와 요청 범위를 구분합니다. ROI는 실제 픽셀 경계를 원본 정규화 bbox에 반영합니다. 대기·실행 중 클립과 결과 근거는 보존하고 참조가 끝난 JPEG 임시 프레임만 삭제합니다.

E02의 두 항목이 모두 YES인 OK·CONFIRMED가 핵심 근거입니다. 보조 확인은 필수가 아니며 다른 물체의 준비로 약 식별을 대신하지 않습니다. 같은 후보 또는 같은 물체의 겹치는 행동에서 NO 답변/명시적 비약물 설명이 나오면 해당 핵심 근거를 제외합니다. 자유문장 상충 검사는 설정 가능한 보수적 패턴이며 모든 의미적 모순을 판별하는 자연어 모델은 아닙니다.

실제 영상 테스트는 runtime 가상환경을 사용합니다. Windows Python 3.10의 한글 editable 경로는 setup.ps1에서 보정합니다. Jetson CUDA·MediaPipe 휠, 장치 캡처, 프로세스 분산은 목표 환경에서 별도 검증해야 합니다.
