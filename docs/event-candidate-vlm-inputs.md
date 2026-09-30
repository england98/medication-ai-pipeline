# 이벤트 후보별 VLM 입력과 탐지·영상 처리 산출물

## 이 문서의 목적과 범위

이 문서는 한 사람만 나오는 영상에서 **이벤트 탐지·영상 처리 담당자가 후보마다 무엇을 산출해야 VLM 검증이 가능한지** 정의한다. 이번 병행 개발의 흐름은 `이벤트 탐지 → 후보별 영상 구성 → VLM 검증 → 세션 최종 판단`이다. 인증·다중 인물 추적과 세션 시작·종료 관리는 추후 통합한다. 이번 단계의 테스트용 세션·대상 식별자와 `VerificationRequest` 조립은 통합 코드 담당자가 공급한다.

핵심은 **후보 유형만 넘기는 것으로는 검증할 수 없다는 것**이다. VLM에는 후보의 행동 구간을 보여 주는 실제 영상 프레임, 그 프레임의 원본 시각·대상/손/입/물체 영역, 탐지 관찰 정보가 함께 전달된다. 탐지 정보는 어디를 살펴볼지 알려 주는 가설이고, 약 식별·입 전달·음수 여부의 답은 VLM이 영상에서 판단한다.

| 후보 | VLM이 답할 질문 | 탐지·영상 처리에서 특히 남겨야 할 정보 |
| --- | --- | --- |
| **E01 준비·취급** | 약/포장을 취급하는가? 같은 동작에서 내용물을 꺼내는가? | 손–물체 상호작용의 구간·영역·샘플, 취급 전후의 클립 |
| **E02 입 전달** | 전달된 **같은 물체**가 약인가? 그 물체가 입 안으로 전달되는가? | 손/물체–입 접근의 구간·영역·샘플, 물체 식별과 전달 순간을 모두 담은 클립 |
| **E03 음수** | 컵/병이 입에 도달하는가? **같은 용기**로 마시는 동작이 보이는가? | 용기–입 관계의 구간·영역·샘플, 접근·입 도달·마시는 형태·이탈의 클립 |

각 행에서 탐지 측이 산출하는 것은 **질문에 답할 수 있는 후보와 영상 근거**다. 표의 질문에 대한 `YES`·`NO`·`UNKNOWN`은 탐지 출력 필드가 아니다.

## 후보 한 건이 VLM까지 가는 과정

| 단계 | 생산자 | VLM으로 이어지는 산출물 | 쓰임 |
| --- | --- | --- | --- |
| 1. 후보 탐지 | 이벤트 탐지 | `EventCandidate` | E01/E02/E03 유형, 행동 시작·끝, 대상·손·입·물체의 원본 프레임 영역, 관련 물체 ID |
| 2. 관찰 기록 | 이벤트 탐지 | `DetectionInfo` | 후보와 같은 버전의 객체·랜드마크·움직임 샘플과 탐지 설정 참조 |
| 3. 영상 구성 | 영상 처리 | `CandidateMedia`와 실제 MP4·ROI 파일 | 후보 전후 영상, 클립 프레임↔원본 프레임·시각 대응, 선택 가능한 ROI |
| 4. 요청 조립 | 사용자 담당 통합 코드 | `VerificationRequest` + 테스트 `SessionContext` | 위 세 객체를 같은 세션·후보 버전으로 묶고 모델/프롬프트 설정 및 요청 ID를 고정 |
| 5. 모델 입력 구성 | VLM 모듈 | `ModelInputRecord` | 실제 선택한 이미지 순서, 크기 변환, 렌더링된 프롬프트와 근거 인덱스를 기록 |

1~3은 팀원 담당이다. 4~5의 모델 선택·프레임 샘플링·리사이즈·질문 생성과 응답 판정은 팀원 산출물의 범위를 넘어선다. 공통 자료형의 단일 기준은 [계약 모델](../shared/contracts/src/medication_contracts/models.py)이다.

### VLM이 실제 모델에 넣는 것

VLM은 `VerificationRequest` 전체를 모델에 그대로 보내지 않는다. 먼저 요청의 세션·후보·시간·자원 연결을 검사하고 `CandidateMedia.clip.video`를 읽는다. 그 다음 `Clip.frames`에서 이미지를 선택해 **시간 순서의 전체 프레임**을 넣는다. `include_rois=true`일 때에만 선택된 프레임의 ROI 이미지가 전체 프레임 뒤에 추가된다. 현재 기본값은 최대 8개 전체 프레임, `include_rois=false`다. 이는 모델 입력 예산이며 원본 클립의 전체 프레임 수 제한은 아니다.

각 입력 이미지에는 `input_index`, `clip_frame_index`, `clip_ms`, 원본 `FrameRef`, `roi_id`/원본 좌표의 `source_bbox`가 텍스트로 붙는다. **그 원본 `FrameRef`와 정확히 일치하는** `EventCandidate.regions`와 `DetectionInfo.samples`도 참고 메타데이터로 붙는다. 선택되지 않은 프레임의 탐지 샘플은 모델 프롬프트에 직접 들어가지 않는다. `detector_config` 참조, 모델·설정 ID 등은 요청 연결과 재현을 위한 기록이지 탐지 결과를 정답으로 주는 필드가 아니다. 모델은 유형별 질문에 `YES`·`NO`·`UNKNOWN`과 실제 입력 이미지 번호에 연결된 근거로 답한다.

후보의 `object_track_ids` 목록 자체는 프롬프트에 직접 표시되지 않지만, 샘플·영역의 물체 ID는 해당 이미지의 참고 메타데이터에 나타날 수 있다. `NO`는 영상에서 부정 근거가 보일 때, `UNKNOWN`은 시야·해상도·가림·프레임 선택 때문에 판단할 수 없을 때 사용한다. 탐지 측은 어느 답을 낼지 미리 정하지 않는다.

영역 bbox와 탐지 샘플은 **이미지 위에 표시되지 않고 텍스트 JSON으로 전달**된다. 좌표만 산출해도 작은 약이 자동으로 확대되는 것은 아니다. ROI 파일을 만들어도 기본 설정인 `include_rois=false`에서는 VLM이 읽지 않는다. 필요한 장면이 전체 프레임에서 식별되지 않는다면 ROI 사용 여부와 크기를 양쪽이 함께 조정해야 한다. 또 클립에 장면이 존재해도 균등 샘플링한 최대 8개 전체 프레임에서 빠질 수 있으므로, 실제 `ModelInputRecord`로 질문에 필요한 시점이 남았는지 확인한다.

## 모든 이벤트 후보에 공통으로 필요한 출력

아래의 **계약 필수**와 **검증 실행 조건**을 구분한다. 계약상 허용하는 결측을 가짜 좌표·객체로 채우지 않는다.

| 구분 | 탐지·영상 처리 쪽에서 제공할 내용 | 경계 조건 |
| --- | --- | --- |
| 계약 필수 | `EventCandidate.context`와 `CandidateKey`(`candidate_id`, `candidate_type`, 증가하는 `candidate_revision`), 생성·갱신 일시, `action_range`, `object_track_ids`, `regions` | 진행 중 `end_ms=null`은 허용하지만 **VLM 요청에는 종료된 버전**만 사용한다. |
| 계약 필수 | 같은 `context`·`CandidateKey`의 `DetectionInfo`, 탐지 설정 `ResourceRef`, 1개 이상의 `DetectionSample` | 각 샘플은 실제 `FrameRef`와 `objects`·`landmarks`·`motions` 관찰 묶음을 갖는다. |
| 계약 필수 | 같은 `context`·`CandidateKey`의 `CandidateMedia` | 확보하려는 `requested_range`와 실제 `clip.actual_range`를 구분한다. 클립 미확보 시 `clip=null`과 사유를 전달한다. |
| 검증 실행 조건 | VLM 작업자가 읽을 수 있는 MP4, 실제 프레임 순서·PTS와 맞는 `Clip.frames`, 실제 파일이 있는 경우의 `RoiFrame` | 영상이 없으면 요청은 남기되 모델은 실행되지 않고 `NOT_RUN/MISSING_VIDEO`가 된다. |
| 검증 실행 조건 | 클립의 원본 프레임 중 적어도 하나와 **정확히 일치하는 `TARGET` 영역** | 해당 대상이 없거나 연결할 수 없으면 현재 검증기는 `NOT_RUN/TARGET_UNRESOLVED`로 처리한다. |

`FrameRef`는 같은 `stream_id`의 원본 `frame_index`, 원본 PTS의 `source_ms`, 세션 기준 `session_ms`, 원본 너비·높이를 보존한다. `session_ms = source_ms - source_time_origin_ms`다. 영역 bbox는 모두 원본 프레임 기준 `[x_min, y_min, x_max, y_max]`, 각 값 0~1이다. `Clip.frames`의 `clip_ms`는 클립 내부 시각으로, 원본·세션 시각과 혼용하지 않는다.

분석했지만 탐지된 항목이 없으면 `ObservationSet(status="OBSERVED", items=[], reason=null)`, 분석 자료 자체를 얻지 못했으면 `UNAVAILABLE`, 빈 목록, 구체적인 사유를 쓴다. `object_track_ids=[]`, `rois=[]`는 허용된다. E02 후보를 만들기 위해 알약 검출이나 E01 선행을 요구하지 않는다.

## E01: 약·포장 준비와 취급

| VLM이 영상에서 확인하는 항목 | 탐지 결과에 담아 전달할 관찰 정보 | 영상 처리에서 보존할 장면 |
| --- | --- | --- |
| `handling_visible`: 대상이 약 또는 관련 포장을 실제로 취급하는가 | E01 후보의 시작·종료 구간. 실제로 보인 `TARGET`·`HAND`·`OBJECT` 영역, 객체의 `class_label`·`score`·bbox, 손 랜드마크, 손–물체 거리(`hand_object_distance`). 안정적으로 이어진 물체라면 `object_track_id`를 후보와 영역·샘플에 일치시킨다. | 손이 물체에 접근하는 전후 맥락, 손과 약/포장의 상호작용, 물체를 알아볼 수 있는 전체 프레임. |
| `contents_removed_visible`: 같은 취급 동작에서 내용물을 꺼내는 장면이 보이는가 | 실제로 관찰된 내용물·손·포장 영역과 관련 샘플이 있으면 포함한다. 현재 탐지기가 별도의 ‘내용물 꺼냄 확정값’을 만들지는 않으며, 보이지 않으면 정보를 꾸며 넣지 않는다. | 꺼내기 전·중·후의 연속 장면. 취급 장면 하나만으로 꺼냄을 주장하지 않는다. |

현재 VLM 판정 규칙에서 **E01의 `CONFIRMED`는 `handling_visible=YES`로 결정**된다. `contents_removed_visible` 답변은 별도로 남지만 E01 확인의 필수 조건은 아니다. 따라서 탐지 쪽은 E01 후보를 꺼냄 확정으로 표시하지 않는다.

## E02: 약으로 식별한 같은 물체의 입 전달

| VLM이 영상에서 확인하는 항목 | 탐지 결과에 담아 전달할 관찰 정보 | 영상 처리에서 보존할 장면 |
| --- | --- | --- |
| `object_is_medication`: **입 전달 행동에 쓰인 바로 그 물체**가 영상에서 약으로 식별되는가 | E02 후보의 행동 구간, `TARGET`·`HAND`·`MOUTH` 영역, 보이는 경우 `OBJECT` 영역·객체 탐지 결과. 물체 연결이 신뢰할 만하면 같은 `object_track_id`를 구간 내에서 유지한다. 약이 안 보인다면 물체 ID·약 라벨을 지어내지 않는다. | 입 접근 이전에 물체 자체를 볼 수 있는 장면과 전달 장면을 같은 클립에 포함한다. 주변 포장이나 탐지 클래스 라벨만으로 약 식별을 대신할 수 없다. |
| `object_transfer_into_mouth_visible`: 같은 물체가 대상의 입 안으로 실제 전달되는가 | 손–입 거리(`hand_mouth_distance`), 접근 속도(`hand_mouth_approach_speed`), 관찰 가능하면 물체–입 거리(`object_mouth_distance`), 관련 손·입·물체의 프레임별 영역과 시각. 이 특징은 후보 단서이며 전달 확정값이 아니다. | 접근→입 주변 상호작용→손/물체 이탈까지의 장면. 손이 입 가까이에 있는 한 프레임만으로 입 전달을 확정하지 않는다. |

**E02의 `CONFIRMED`에는 두 항목 모두 `YES`가 필요**하다. 알약이 작거나 가려져도 손–입 움직임으로 후보를 만들 수 있지만, 영상에서 약 식별 또는 입 전달을 확인할 수 없으면 VLM은 `UNKNOWN`을 낼 수 있다. E02가 세션 완료의 핵심 근거이므로, 클립과 실제 선택 프레임에 물체 식별 장면과 전달 장면이 모두 남는지 통합 테스트로 확인한다.

## E03: 컵·물병을 이용한 음수 형태

| VLM이 영상에서 확인하는 항목 | 탐지 결과에 담아 전달할 관찰 정보 | 영상 처리에서 보존할 장면 |
| --- | --- | --- |
| `container_reaches_mouth`: 컵·물병이 대상의 입에 도달하는가 | E03 후보 구간, `TARGET`·`MOUTH`·`OBJECT` 및 가능하면 `HAND` 영역, 용기 탐지 클래스·bbox·점수, 신뢰 가능한 `object_track_id`, 물체–입 거리(`object_mouth_distance`). | 용기 접근과 입 도달 장면. 같은 용기를 프레임 사이에 따라갈 수 있도록 전체 프레임 맥락을 보존한다. |
| `drinking_motion_visible`: **같은 용기**를 사용해 마시는 형태의 움직임이 보이는가 | 실제로 관찰한 용기·손·입의 상대 위치와 움직임 샘플. 현재 탐지기는 기울기나 실제 음수 여부를 별도 확정 필드로 산출하지 않으므로, 해당 증거가 없으면 추가했다고 주장하지 않는다. | 입 도달 전후의 움직임과 이탈까지 포함한다. 용기가 입에 접근하거나 닿은 장면만으로 음수를 확정하지 않는다. |

**E03의 `CONFIRMED`에는 두 항목 모두 `YES`가 필요**하다. E03는 세션 판단에서 보조 근거이며 E02의 약 식별이나 입 전달을 대신하지 않는다.

## 후보를 넘기기 전 확인할 사항

1. 후보가 종료됐고, `event`·`detection`·`media`가 동일한 `context`와 **동일 `candidate_id + candidate_revision`**을 가리키는가?
2. `DetectionInfo.samples`와 `EventCandidate.regions`의 `FrameRef`가 원본 시각·해상도를 유지하고, 클립에 포함된 프레임과 일치하는가? 클립에 매핑되는 `TARGET` 영역이 있는가?
3. MP4의 실제 디코딩 프레임 수·PTS가 `Clip.frames`와 맞는가? ROI가 있다면 이미지 크기·원본 bbox·`clip_frame_index`가 맞는가?
4. 요청 전후 구간의 장면이 **각 이벤트의 질문을 답할 기회**를 주는가? 특히 E02의 물체 식별과 입 전달 장면이 실제 VLM 샘플링 이후에도 남는가?
5. 관찰 실패와 관찰 결과 없음, 영상 미확보를 서로 다른 상태로 기록했는가? 탐지 클래스·점수나 접근 신호를 VLM의 `YES` 답변으로 복사하지 않았는가?

형식 예시는 [E02 요청 JSON](../modules/vlm-event-detection/examples/request.e02.json)을 참고한다. 이 예시의 MP4는 포함되지 않으므로 실제 실행용 영상 자료로 사용하려면 읽을 수 있는 파일과 정확한 `Clip.frames`를 별도로 준비해야 한다. 더 넓은 담당 범위·통합 합의는 [병행 개발 인수인계](parallel-development-handoff.md)에 정리했다.

구현 확인 위치: [후보 생성](../modules/event-detection/src/event_detection/detector.py), [클립·ROI 구성](../modules/session-video-management/src/session_video/video.py), [VLM 프롬프트](../modules/vlm-event-detection/src/vlm_verification/prompts.py), [프레임 선택·로딩](../modules/vlm-event-detection/src/vlm_verification/media.py), [입력 연결 검사](../modules/vlm-event-detection/src/vlm_verification/service.py), [이벤트 판정](../modules/vlm-event-detection/src/vlm_verification/response.py).
