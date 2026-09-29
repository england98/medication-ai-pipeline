# 공통 입출력 계약

src/medication_contracts/models.py가 모든 모듈의 단일 Pydantic 정의입니다. 기존 vlm_verification.contracts는 같은 클래스의 호환 import입니다.

설계 2-7의 인증·예정 회차·세션·추적·후보·탐지·클립·ROI·검증·완료·종료·조회 계약을 제공합니다. 교차 연결 검사는 서비스 경계에서 수행합니다. 완료 기록은 수신 결과의 핵심/보조/제외 분할과 일관성 참조를 검사합니다.

JSON은 UTF-8, schema_version은 1.0입니다. 원본 PTS·세션 상대 시각·클립 시각을 구분하고 좌표는 원본 기준 0~1입니다. 영상은 ResourceRef로 참조하며 UNKNOWN/UNCERTAIN과 ERROR/NOT_RUN을 구분합니다.

`records.write_json()`은 같은 디렉터리의 임시 파일에 완성된 JSON을 쓴 뒤 `os.replace()`로 교체합니다. Windows 접근·공유·잠금 오류(WinError 5/32/33)는 교체 단계만 최대 7회 시도합니다. 재시도 간 대기는 20/40/80/160/320/320ms, 누적 940ms이며 파일 입출력 시간은 별도입니다. 성공할 때까지 기존 JSON을 보존하고 세션 이력을 다시 추가하지 않습니다.

교체 오류가 발생하면 대상 파일과 같은 디렉터리의 `persistence-events.jsonl`에 `RETRY`, `RECOVERED`, `FAILED`와 오류 코드·시도 횟수·경로를 기록합니다. 계속 실패하면 원래 예외를 전달하고 기존 JSON과 새 내용의 `.tmp`를 보존합니다. 다른 입출력 오류는 재시도하지 않으며 진단 로그 자체의 저장 실패도 원래 오류를 덮어쓰지 않습니다. 이 처리는 일시적인 충돌을 위한 것이며 지속적인 파일 잠금이나 폴더 권한 문제를 해제하지는 않습니다.
