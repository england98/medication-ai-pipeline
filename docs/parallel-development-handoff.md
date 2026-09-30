# 병행 개발 인수인계: 1인 영상의 이벤트 탐지·영상 처리 → VLM 검증·세션 판단

## 이번 개발 범위

이번 고도화는 **한 사람만 나오는 영상의 이벤트 탐지 → 후보별 영상 구성 → VLM 검증 → 세션 최종 판단**에 집중한다. 인증·다중 인물 추적과 세션 시작·종료 관리는 이후 따로 개발해 통합한다. 이번 단계의 최종 산출은 후보별 `VerificationResult`와 해당 테스트 세션의 `CompletionRecord` 또는 완료 근거가 아직 없는 판단 상태다. 실제 사용자 인증과 세션 종료 연동의 정확도는 이 단계의 완료 기준에 포함하지 않는다.

이벤트별로 VLM이 어떤 영상·탐지 정보를 받으며 탐지 측이 무엇을 산출해야 하는지는 [이벤트 후보별 VLM 입력](event-candidate-vlm-inputs.md)을 따른다. 형식의 단일 기준은 [공통 계약 모델](../shared/contracts/src/medication_contracts/models.py)이며 교환 JSON은 UTF-8, `schema_version="1.0"`이다. 설계의 상세 의미는 [공통 입출력 계약](https://medication-ai-vision-docs.vercel.app/architecture/interfaces-results)을 참고한다.

## 역할과 인계 지점

| 담당 | 책임 | 인계 결과 |
| --- | --- | --- |
| 팀원: 이벤트 탐지·영상 처리 | E01/E02/E03 후보 생성·갱신, 실제 원본 프레임 관찰, 후보 전후 영상·ROI 구성 | 같은 후보 종료 버전의 `EventCandidate`, `DetectionInfo`, `CandidateMedia`와 읽을 수 있는 MP4·ROI·설정 참조 |
| 사용자: VLM 검증·세션 판단 | 고정 요청 입력 검사, 실제 모델 입력·응답·근거 기록, 결과 누적과 완료 규칙 | `VerificationResult`, 완료 시 `CompletionRecord` |
| 사용자: `runtime` 통합 | 1인 영상용 테스트 문맥 공급, `VerificationRequest` 조립·ID 발급, 큐·결과 전달과 통합 테스트 | 같은 `SessionKey`·`CandidateKey`로 묶인 요청과 세션 스냅샷 |

현재 전체 `runtime` 경로는 1차 구현의 인증·추적·`SessionManager`를 아직 호출한다. 위의 1인 영상용 문맥 공급과 완료 기록까지만 보는 흐름은 **이번 고도화의 통합 목표**이며 기존 호출을 이미 제거했다는 뜻은 아니다.

### 이번 단계의 테스트 문맥

계약에는 `SessionKey`·`SessionContext`와 탐지 입력의 `TrackingUpdate`가 필요하다. 통합 코드는 영상마다 테스트용 `session_id`·`user_id`·`scheduled_occurrence_id`·`auth_id`·`stream_id`·`target_track_id`와 `source_time_origin_ms`를 한 번 정한다. `SessionContext`에는 테스트용 `scheduled_at`·`started_at`, 1부터 시작하는 `session_revision`, `session_status="ACTIVE"`, `completion_id=null`, `closed_at=null`을 명시한다. 이 식별자는 실제 인증 성공이나 예정 복약 회차 확인의 증거가 아니다.

탐지 입력에는 영상의 유일한 사람 영역을 `TRACKED`인 `TrackingUpdate`로 제공한다. 사람이 없거나 한 명으로 특정할 수 없으면 `UNRESOLVED`로 처리하고 임의 대상을 선택하지 않는다. 이는 임시 대상 연결이며 재인식·인증·다중 인물 추적의 검증 결과가 아니다. 이후 실제 인증·추적·세션 관리 모듈을 붙일 때 공급 지점을 교체하되 아래 교환 객체의 의미는 유지한다.

## 고정할 데이터 계약

| 객체 | 핵심 조건 |
| --- | --- |
| `SessionKey` | `session_id`, `user_id`, `scheduled_occurrence_id`, `stream_id`, `target_track_id` 전체가 후보·탐지·영상·요청에서 같아야 한다. 이번 단계에서는 테스트 식별자다. |
| `EventCandidate` | E01/E02/E03 유형, `candidate_id`, 증가하는 `candidate_revision`, 행동 `action_range`, 물체 ID와 원본 프레임의 `regions`. 진행 중 `end_ms=null`은 허용하지만 VLM 요청에는 종료된 버전만 사용한다. |
| `DetectionInfo` | 같은 `context`·`CandidateKey`, 탐지 설정 `ResourceRef`, 1개 이상의 실제 프레임 `samples`. 객체·랜드마크·움직임은 참고 정보이며 VLM 답변을 미리 정한 값이 아니다. |
| `CandidateMedia` | 같은 `context`·`CandidateKey`, 종료된 `requested_range`, 실제 `clip` 또는 `unavailable_reason`. 클립의 `actual_range`는 확보한 첫·마지막 프레임의 시각이며 요청 범위와 구분한다. ROI가 없으면 `[]`다. |
| `SessionContext` | 같은 테스트 `SessionKey`와 원본 시간 원점·대상 연결 스냅샷. `tracking=UNRESOLVED`이면 현재 검증기는 `NOT_RUN/TARGET_UNRESOLVED`를 낸다. |
| `VerificationRequest` | 종료된 동일 후보 버전의 `event`·`detection`·`media`, `request_id`, 모델·프롬프트 버전, `execution_config`. 생성 후 ID와 내용은 변경하지 않는다. |

`ResourceRef`는 `{resource_id, locator, media_type}`이다. 영상 본문은 JSON에 넣지 않는다. 현재 VLM 로더는 로컬 경로나 `file:` URI를 읽는다. 요청을 `READY`로 기록하기 전에 MP4·ROI·설정 파일을 완성하고 VLM 쪽에서 읽을 수 있어야 한다. 검증 대기·실행 중과 결과 근거가 참조하는 자원은 보존한다.

### 시간·좌표·결측

- `*_at`은 UTC `YYYY-MM-DDTHH:mm:ss.SSSZ`. 행동 구간과 `session_ms`는 세션 기준 정수 ms, `source_ms`는 실제 원본 PTS, `clip_ms`는 클립 내부 PTS다. `session_ms = source_ms - source_time_origin_ms`이며 `Clip.frames`가 원본↔클립 시간축의 기준표다. FPS로 없는 시각·프레임을 채우지 않는다.
- bbox는 원본 이미지 기준 `[x_min, y_min, x_max, y_max]`, 왼쪽 위 원점, 값 0~1이다. ROI도 원본 기준 실제 크롭 영역을 기록한다. 크롭·리사이즈·패딩과 최종 이미지 순서는 VLM의 `ModelInputRecord`가 기록한다.
- 같은 행동은 같은 `candidate_id`의 revision을 올리고, 이탈 후 별도 행동은 새 ID로 만든다. E02와 E03가 겹쳐도 서로 다른 후보다. `object_track_ids=[]`는 물체 연결 미확보를 뜻하며 약의 부재 판정이 아니다.
- 분석했으나 탐지 항목이 없으면 `OBSERVED`·`items=[]`·`reason=null`; 정보를 얻지 못했으면 `UNAVAILABLE`·`items=[]`·구체적 사유다. 미확보 영역을 0 좌표나 가짜 프레임으로 채우지 않는다.

## 인계와 결과 처리 순서

1. 탐지 담당이 열린 후보와 같은 버전의 `DetectionInfo`를 갱신하고 영상 처리에 전달한다. 영상 처리는 선행 프레임을 유지한다.
2. 행동이 끝나면 `end_ms`를 정해 종료 버전을 고정한다. 영상 처리는 후행 구간 또는 EOF까지 확보한 뒤 `CandidateMedia`를 만든다. 확보하지 못한 영상은 실제 사유로 표시한다.
3. 사용자 담당 `runtime`이 동일 `SessionKey`·`CandidateKey`의 세 객체와 테스트 세션 스냅샷을 묶어 요청 ID를 한 번 발급한다. 현재 파일 형식은 `queue/<request_id의 SHA-256>.json`의 `{"request": ..., "session": ...}`이다.
4. VLM이 형식·연결·클립 PTS/ROI를 검사한 뒤 모델을 실행한다. 입력 연결 문제는 `NOT_RUN`, 처리 실패는 `ERROR`, 유효한 미확신 응답은 `OK+UNCERTAIN`으로 구분한다.
5. `VerificationResult`와 원본 요청을 세션 판단에 전달한다. 같은 ID·같은 내용은 중복 누적하지 않고 같은 ID의 다른 내용은 충돌로 거부한다. 결과 도착 순서를 행동 순서로 쓰지 않는다. 실제 세션 종료·알림 연동은 후속 단계다.

## 개발 전에 양쪽이 확인할 항목

| 항목 | 현재 기준·합의할 내용 |
| --- | --- |
| 요청 조립 소유권 | **사용자가 `runtime`을 담당**하며 현재 구현처럼 요청 ID를 한 번 발급·조립한다. 설계 문서의 VLM 측 ID 발급 문구와 다른 점은 추후 설계 문서에 반영한다. |
| 후보 종료와 대상 영역 손실 | 열린 후보 revision, 종료·EOF·대상 영역 손실 시 flush, 후행 프레임 대기, 검증에 사용할 `SessionContext.tracking` 스냅샷 시점을 고정한다. |
| 영상 접근·보존 | 1차 통합에서 같은 파일시스템을 쓸지, MP4·ROI·JSON의 읽기 가능 시점과 보존 기간·용량 상한을 정한다. 다른 장치로 나누면 경로 변환과 복사 완료 신호가 필요하다. |
| 완료 판단 시점 | 현재 설계와 구현은 **수신된 결과**에 `OK+CONFIRMED` E02 핵심 근거가 있고 관련 상충이 없으면 완료한다. 관련 E03까지 기다리는 요구가 생기면 대기 범위·마감 신호와 판단 규칙을 함께 정의한다. |
| 입력 품질·예산 | 현재 기본값은 선행 E01/E02/E03 각 1/1.5/1초, 후행 각 1초, 버퍼 5초, VLM 최대 8개 전체 프레임·ROI 미포함이다. 실제 영상에서 손–물체–입 관계가 선택 프레임에 남는지 보고 조정한다. |
| 모델·관찰 용어 | 탐지 클래스 맵, `part`·motion 이름/단위, 모델 파일·규칙 버전, 입력 해상도·분석 주기를 고정한다. 일반 객체 모델의 약·포장 검출 정확도를 가정하지 않는다. |
| 공통 실행 자료 | 양쪽에서 읽을 수 있는 짧은 실제 MP4와 요청·세션·설정 JSON을 공유한다. E01/E02/E03, 겹치는 후보, 영상 부족, 대상 영역 손실, VFR, VLM 불확실·오류를 포함한다. 기존 E02 예시 JSON의 MP4는 제공되지 않는다. |

## 이번 단계의 통합 완료 기준

- 한 사람만 나오는 실제 영상에서 후보 생성→종료→클립·ROI→검증 요청→VLM 결과→세션 판단을 재생하고 `candidate_id`·`request_id`·`clip_id`와 원본 프레임 시각을 역추적할 수 있다.
- E02 확인, E01/E03만 확인, E02 불확실·기각·기술 오류, 같은 물체의 상충, 겹치는 E02/E03, 중복 요청, 영상 미확보와 대상 영역 손실을 구분한다.
- 지연된 VLM 실행 중에도 영상 입력·탐지가 계속되고 같은 테스트 세션의 완료 기록을 한 번만 만든다. 실제 인증·세션 시작·종료 연동은 이 기준에 포함하지 않는다.

루트 통합 테스트 명령은 `runtime/.venv/Scripts/python.exe -X utf8 -m pytest -q`다. 합성 관측·응답 테스트 통과는 실모델의 탐지·검증 정확도를 뜻하지 않는다.
