# 사용자 인증·추적

운영자가 확인한 오프라인 대상 또는 외부 AuthenticationResult를 세션에 연결합니다. 생체인증 알고리즘은 설계에서 미정입니다.

초기 사람이 하나이거나 지정 bbox가 하나의 사람을 선택할 때 연결합니다. 이후 공간 연속성·모호성·겹침·프레임 공백을 검사합니다. 불명확하면 UNRESOLVED를 유지하고 다른 사람으로 교체하지 않습니다. 복구는 새 인증·세션으로 수행합니다.

구현: src/authentication_tracking. 테스트: 루트 tests/integration/test_lifecycle.py.
