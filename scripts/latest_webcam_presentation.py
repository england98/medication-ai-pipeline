"""Webcam-specific slides using the previous MP4 presentation's visual style."""
import collections
import html
import json
import re
from datetime import datetime

from build_latest_mp4_demo import ROOT, read, rows


def build_presentation(d):
    out=d.out
    template=(ROOT/'outputs/demo-latest-mp4-20260929/presentation.html').read_text(encoding='utf8')
    style=re.search(r'<style>(.*?)</style>',template,re.S).group(1)
    script=re.search(r'<script>(.*?)</script>',template,re.S).group(1)
    slides=[];notes=[]
    def add(title,body,source,note,cls=''):
        n=len(slides)+1
        slides.append(f'<section class="slide {cls}" id="slide-{n}"><h2>{title}</h2><div class="content">{body}</div><footer>{source}<span>{n:02d} / 12</span></footer></section>')
        notes.append((title,note))

    add('웹캠 복약 이벤트 탐지와<br>VLM 검증', '''
      <div class="cover"><div><p class="lead">최신 웹캠 테스트의<br>탐지 과정과 판정 근거</p>
      <p>2026년 9월 29일 16:04 실행</p><p class="muted">Integrated Webcam · 1280 × 720</p>
      <p class="accent">E02 후보 2건<br>VLM 결과 모두 불확실</p></div>
      <img src="assets/candidate-1-input-1.png" alt="웹캠 후보 1의 0.528초 입력"></div>
      ''','test-run.json, recording/metadata.json, verification/*/result.json',
      '최신 웹캠 실행은 9월 29일 16시 04분입니다. 리빙랩 MP4 시험과 다른 실행이며 E02 후보는 두 건입니다. 개별 VLM 결과는 저장됐지만 중단 때문에 파이프라인 수신과 세션 판단은 완료하지 못했다는 점을 먼저 짚습니다.','title')
    add('저장 기록과 실행 범위', '''
      <div class="columns"><div><table><tr><th>녹화</th><td>약 23.9초, 154프레임<br>H.264, 음성 없음</td></tr>
      <tr><th>분석</th><td>126개 관측<br>YOLO11n + MediaPipe, CPU</td></tr>
      <tr><th>추적</th><td>TRACKED 97개<br>UNRESOLVED 29개</td></tr>
      <tr><th>후보·검증</th><td>E02 2건, 클립·요청 2건<br>VLM 결과 파일 2건</td></tr></table></div>
      <div><h3>원래 실행의 저장 상태</h3><p>입력 루프 <strong class="accent">INTERRUPTED</strong></p><p>녹화 메타데이터 <strong>ERROR</strong><br>154프레임은 모두 재생 가능</p>
      <p>results.jsonl과 summary.json 없음</p><p class="accent">개별 VLM 작업 완료와<br>전체 파이프라인 완료를 구분합니다.</p>
      <p class="small">운영자 확인으로 단일 대상을 연결했습니다.<br>분석은 설정상 100ms 간격이며 실제 수신 간격은 일정하지 않습니다.</p></div></div>
      ''','recording/metadata.json, observations.jsonl, run-error.json',
      '154는 녹화 프레임 수이고 126은 분석 관측 수입니다. 녹화 메타데이터는 정상 종료 상태가 아니지만 보존된 MP4의 154프레임은 모두 디코딩됐습니다. 이 사실은 중단 전후에 미기록 프레임이 전혀 없었다는 뜻은 아닙니다.')
    add('웹캠 탐지 과정과 후보 생성 순간','''
      <video controls preload="metadata" poster="assets/poster.jpg" src="event-vlm-demo.mp4"></video>
      <p class="small">전체 탐지 기록, 후보 구간 0.5배속, 생성 순간 정지, 추적 손실, VLM 입력과 결과</p>
      <p class="small">프레임 공백에서는 직전 영상을 유지하며 경과 시간을 표시합니다. VLM 결과는 사후 요약입니다.</p>
      ''','event-vlm-demo.mp4, recording/frames.jsonl, observations.jsonl',
      '영상은 전체 기록을 먼저 보여준 뒤 두 후보의 생성 순간을 느리게 다시 보여줍니다. 재생 중 프레임 유지 시간이 길어지는 구간은 사람의 정지 동작을 뜻하지 않고 저장된 다음 프레임이 없는 구간입니다. 마지막 VLM 화면은 당시 결과 파일을 요약한 것입니다.','video-slide')
    add('E02 후보를 만드는 조건','''
      <div class="columns"><div><p class="formula">d ≤ 0.16</p><p>손과 입의 거리가 진입 기준 이내</p>
      <p class="formula">또는 d ≤ 0.40, v ≥ 0.025</p><p>입으로 충분히 가까워지는 동작</p>
      <table class="compact"><tr><th>유지·종료</th><td>유지 거리 0.24<br>단서 소멸 500ms 후 종료</td></tr>
      <tr><th>발행</th><td>최소 활성 100ms<br>이전 관측 최대 600ms 포함</td></tr></table>
      <p class="small">d: 대상 키로 정규화한 거리<br>v: 대상 키/초</p></div>
      <div><img class="wide" src="assets/event-2-moment.jpg" alt="후보 2가 생성된 9.952초 화면">
      <p class="accent">최초 생성 #1 0.128초 / #2 9.952초</p><p>후보 #2의 행동 범위는<br>이전 관측을 포함해 9.280초부터입니다.</p>
      <p class="small">저장 관측으로 기하 규칙을 재현했습니다.<br>후보 이력 29행의 구간·revision이 모두 일치합니다.</p></div></div>
      ''','detector-config.json, candidate-history.jsonl, event_detection/detector.py',
      '별도 학습 기반 행동 분류기 대신 기하 특징을 사용합니다. 최초 후보 발행은 0.128초와 9.952초로 확인했습니다. 영상에 표시한 시각은 저장 관측값에 규칙을 재적용해 얻은 값입니다. 후보 발행 전의 관측을 포함하므로 행동 범위의 시작과 최초 발행 시각이 다릅니다.')
    add('후보 구간과 실제 확보한 클립','''
      <table><tr><th>후보</th><th>행동 범위</th><th>요청 클립</th><th>실제 클립</th><th>프레임</th></tr>
      <tr><td>#1 E02</td><td>0–2.704초</td><td>0–3.704초</td><td>0–3.664초</td><td>49</td></tr>
      <tr><td>#2 E02</td><td>9.280–12.096초</td><td>7.780–13.096초</td><td>8.608–13.024초</td><td>26</td></tr></table>
      <p class="lead">후보 #2는 요청보다 828ms 늦은 시점부터<br>선행 영상을 확보했습니다.</p>
      <p>7.152초 다음 저장 프레임은 8.608초입니다.<br>요청 시작 시각 7.780초에 대응하는 저장 프레임이 없습니다.</p>
      <p class="small">E02는 행동 전 1.5초, 종료 후 1초를 요청합니다. 클립의 마지막 프레임은 요청 끝과 각각 40ms·72ms 차이가 납니다.</p>
      ''','request.json의 media.requested_range·clip.frames, recording/frames.jsonl',
      '불규칙한 프레임 간격을 정확히 유지해야 합니다. 후보 2는 선행 구간이 828ms 부족합니다. 클립 49프레임과 26프레임 전체의 PTS를 대응표와 대조했습니다. 후보 수를 실제 복약 횟수로 해석하지 않습니다.')
    add('VLM에 전달한 이미지와 참고 정보','''
      <div class="columns"><div><h3>이미지 입력</h3><p>클립마다 시간순 전체 화면 4장</p>
      <p>1280 × 720을 <strong>672 × 384</strong>로 축소</p>
      <p>프레임 인덱스를 균등하게 선택하고<br>TARGET 연결 프레임을 확보합니다.</p>
      <p class="accent">ROI는 후보당 8장 생성했지만<br>이번 입력에서는 사용하지 않았습니다.</p>
      <p class="small">불규칙한 프레임 간격 때문에<br>선택 이미지의 시간 간격도 균등하지 않습니다.</p></div>
      <div><h3>텍스트 입력</h3><p>E02 유형, 행동 구간, 대상 track</p><p>원본 frame·시각, 대응 영역 좌표,<br>같은 프레임의 탐지 관측</p>
      <p>① 같은 물체를 약으로 식별하는가?<br>② 그 물체가 입으로 들어가는가?</p>
      <p class="small">감지 라벨·좌표는 참고 가설입니다.<br>정보가 부족하면 UNKNOWN을 사용하도록<br>프롬프트에서 지시합니다.</p></div></div>
      <p class="small">다음 이미지들은 저장 클립과 model_input.json의 인덱스·BICUBIC 설정으로 복원했습니다. 원본 HTTP 이미지 전송본은 별도 저장되지 않았습니다.</p>
      ''','model_input.json, request.json, vlm_verification/llama_cpp.py·media.py',
      '실제 입력의 기준은 model_input.json이며 이번에는 ROI를 포함하지 않았습니다. source_ms는 큰 원본 시계 값이므로 발표에는 첫 수신 프레임을 0으로 한 session_ms를 사용했습니다. 원본 시계와 상대 시각은 manifest에 모두 기록했습니다.')
    for e in d.events:
        pics=''.join(f'<figure><img src="{r["file"]}" alt="input {r["input_index"]}, 영상 {r["video_ms"]}ms"><figcaption>input {r["input_index"]} · {r["video_ms"]/1000:.3f}초 · 원본 #{r["source_frame_index"]}<br><span class="muted">탐지 관측 {r["sample_count"]}개 / 영역 {r["region_count"]}개</span></figcaption></figure>' for r in e['images'])
        add(f'E02 #{e["n"]} 입력 프레임과 사후 판정',f'''
          <div class="event-layout"><div class="frames">{pics}</div><div>
          <p class="small">행동 {e['start']/1000:.3f}–{e['end']/1000:.3f}초</p>
          <table class="compact"><tr><th>약 식별</th><td>UNKNOWN</td></tr><tr><th>입 전달</th><td>UNKNOWN</td></tr></table>
          <p class="verdict">UNCERTAIN</p><p class="event-text">{e['explanation']}</p>
          <p class="small">두 질문의 근거 모두 <strong class="accent">input 0</strong>을 명시합니다.</p>
          <p class="review">{e['caveat']}</p></div></div>
          ''',f'evidence/candidate-{e["n"]}/model_input.json, request.json, result.json',
          e['explanation']+' '+e['caveat']+' 이 결과는 입력 루프 중단 후 개별 검증 파일에 저장됐으며 원래 세션의 판단 결과가 아닙니다.')
    add('17.440초부터 대상 추적 손실','''
      <div class="columns"><div><img class="wide" src="assets/tracking-before.jpg" alt="17.312초 TRACKED 화면"><p>17.312초 · 마지막 TRACKED 관측</p></div>
      <div><img class="wide" src="assets/tracking-lost.jpg" alt="17.440초 UNRESOLVED 화면"><p class="accent">17.440초 · UNRESOLVED 전환</p></div></div>
      <p>이후 29개 관측에서 추적 미확정 상태가 유지됐고<br>대상 랜드마크는 UNAVAILABLE로 기록됐습니다.</p>
      <p class="small">전환 직전 분석 간격은 128ms입니다. 검출 누락·연결 실패·모호성 중 어떤 분기인지 기록이 없어 원인을 특정할 수 없습니다.</p>
      ''','observations.jsonl의 tracking·landmarks, recording/frames.jsonl',
      '추적 손실 전후에 인물이 화면에 보인다는 사실만으로 추적기의 내부 실패 원인을 단정할 수 없습니다. 큰 투명 용기가 전경에 있지만 그것이 원인이라는 증거는 없습니다. 2초 프레임 간격은 추적 손실 이후에 발생했습니다.')
    add('불규칙한 저장 프레임 간격','''
      <table><tr><th>직전 프레임</th><th>다음 프레임</th><th>간격</th><th>자료에서의 처리</th></tr>
      <tr><td>7.152초</td><td>8.608초</td><td class="accent">1.456초</td><td>직전 프레임 유지와 경과 시간 표시</td></tr>
      <tr><td>18.112초</td><td>20.112초</td><td class="accent">2.000초</td><td>동일 방식으로 표시</td></tr></table>
      <p class="lead">영상이 멈춰 보이는 구간을<br>사람이 정지한 행동으로 해석하면 안 됩니다.</p>
      <p>데모는 저장 PTS에 맞춰 프레임을 배치합니다.<br>원인 진단에는 캡처·추론 부하와 수신 시각을 추가로 확인해야 합니다.</p>
      <p class="small">자료의 MP4는 호환성을 위해 20fps로 출력했습니다. 실제 원본은 불규칙한 PTS를 가지며 원본 속도 영상은 마지막 표시 구간을 약 0.06초 연장합니다.</p>
      ''','recording/frames.jsonl, recording/metadata.json, manifest.json',
      '20fps는 데모 인코딩 속도이며 원래 웹캠 수신 성능을 뜻하지 않습니다. 원본은 약 23.889초, 데모 전체 기록 부분은 23.95초로 마지막 프레임을 포함하도록 반올림했습니다. 프레임 공백 원인은 이 기록만으로 알 수 없습니다.')
    add('입력 중단과 VLM 결과 저장 시점','''
      <table><tr><th>KST 시각</th><th>기록</th><th>의미</th></tr>
      <tr><td>16:04:45.049</td><td>#1 READY</td><td>검증 요청 준비</td></tr>
      <tr><td>16:04:56.303</td><td>#2 READY</td><td>첫 작업 뒤에서 대기</td></tr>
      <tr><td class="accent">16:05:03.192</td><td class="accent">입력 루프 중단</td><td>KeyboardInterrupt</td></tr>
      <tr><td>16:06:04.569</td><td>#1 결과 저장</td><td>OK / UNCERTAIN</td></tr>
      <tr><td>16:07:19.067</td><td>#2 결과 저장</td><td>OK / UNCERTAIN</td></tr></table>
      <p>READY부터 결과까지 #1 <strong>79.520초</strong>, #2 <strong>142.764초</strong><br>개별 작업 종료 후 파이프라인 수신·세션 반영 기록은 없습니다.</p>
      <p class="small">큐 대기를 포함한 시간입니다. 카메라 촬영부터 사용자에게 판정이 전달되기까지의 지연은 아닙니다.</p>
      ''','queue-events.jsonl, run-error.json, verification/*/states.jsonl·result.json',
      '입력이 중단된 시점에는 첫 VLM 작업이 아직 실행 중이었고 둘째는 대기 중이었습니다. 종료 정리 과정에서 개별 결과 파일은 남았지만 파이프라인 결과 수신은 수행하지 않았습니다. 결과 저장 시각을 실시간 판정 전달 시각으로 설명하지 않습니다.')
    add('확인한 결과와 다음 검증 범위','''
      <div class="columns"><div><h3>확인한 동작</h3><p>웹캠 입력·영상·관측 저장</p><p>E02 후보 2건과 클립·요청 생성</p>
      <p>Qwen 처리 2건 성공<br>두 질문 모두 UNKNOWN<br>규칙 판정 모두 UNCERTAIN</p>
      <p class="accent">복약 완료 미확정<br>세션 ACTIVE, session_result = null</p></div>
      <div><h3>다음 시험의 확인 항목</h3><p>중단 시 대기 작업 결과 수신과<br>세션 판단까지 마무리</p>
      <p>추적 손실 분기·연결 점수 기록</p><p>프레임 공백과 동시 추론 부하 분석</p><p>입·손 ROI와 핵심 시점 선택 비교</p>
      <p class="small">정답 라벨이 없어 정확도·미탐률은 계산하지 않았습니다. UNCERTAIN은 미복약 확정이 아닙니다.</p></div></div>
      ''','session/session-view.json, result.json, run-error.json',
      '두 질문이 모두 YES라면 CONFIRMED, 하나라도 NO면 REFUTED, NO 없이 UNKNOWN이 있으면 UNCERTAIN으로 코드에서 판정합니다. 이번에는 둘 다 UNKNOWN입니다. ACTIVE는 마지막 저장된 세션 상태이며 프로그램이 지금 실행 중이라는 뜻은 아닙니다. 이번 제작에서는 기존 결과를 세션에 반영하거나 모델을 재실행하지 않았습니다.')
    assert len(slides)==12
    doc=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>최신 웹캠 테스트: 이벤트 탐지와 VLM 검증</title><style>{style}</style></head><body><nav><button id="present">발표 모드 P</button><button id="prev">이전</button><button id="next">다음</button><button id="fullscreen">전체 화면</button><a href="technical-summary.html">상세 근거</a></nav>{"".join(slides)}<script>{script}</script></body></html>'
    (out/'presentation.html').write_text(doc,encoding='utf8')
    esc=html.escape
    content=['<h1>최신 웹캠 테스트 데모 자료</h1><p>2026.09.29 16:04 실행 · 개발·연구팀 발표용</p>',
      '<p><a href="presentation.html">발표 슬라이드</a> · <a href="presentation.pdf">PDF</a> · <a href="event-vlm-demo.mp4">데모 MP4</a> · <a href="detection-overlay.mp4">전체 탐지 영상</a> · <a href="clips/source-webcam.mp4">저장 원본 영상</a> · <a href="manifest.json">출처·프레임 매핑</a></p>',
      '<h2>사용 방법</h2><p>ZIP을 해제한 뒤 폴더 구조를 유지하세요. presentation.html을 Chrome/Edge로 열고 발표 모드(P), 방향키를 사용합니다. 동영상은 재생 버튼으로 시작합니다. PDF는 정적 자료이며 영상은 별도 MP4로 재생합니다. 영상은 한국어 설명을 포함한 무음 영상입니다.</p>',
      f'<h2>선정한 실행</h2><p><code>{esc(d.run.name)}</code><br>input_format=dshow인 실행 중 test-run.started_at이 가장 최근인 기록을 사용했습니다. 녹화 MP4의 제작 시점 SHA-256: <code>{d.source_sha}</code></p>',
      '<h2>핵심 결과</h2><p>E02 후보 2건, VLM OK / UNCERTAIN 2건. 16:05:03.192 입력 중단 후 결과 파일은 남았지만 파이프라인 결과 수신과 세션 판단에는 반영되지 않았습니다. 원래 실행은 INTERRUPTED이며 정상 종료된 실험으로 보고하지 않습니다.</p>',
      '<h2>판정 규칙</h2><p>VLM은 약 식별과 입 전달 여부에 YES/NO/UNKNOWN으로 답합니다. 두 답변 모두 YES면 CONFIRMED, 하나라도 NO면 REFUTED, NO 없이 UNKNOWN이 있으면 UNCERTAIN입니다. 이번 두 후보는 UNKNOWN + UNKNOWN입니다. 모델 내부 사고 과정을 추정하지 않고 실제 응답의 명시된 근거만 요약합니다.</p>',
      '<h2>제작·재현 방법</h2><ul><li>기존 결과·세션 기록은 변경하지 않았고 탐지 모델과 VLM은 재실행하지 않았습니다.</li><li>저장 관측에 당시 기하 규칙을 적용해 최초 후보 발행 시각을 재현했습니다. 원본 이력 29행의 구간·revision과 일치했습니다.</li><li>원본 154프레임의 PTS와 recording/frames.jsonl, 후보 클립 49·26프레임의 PTS와 Clip.frames를 모두 대조했습니다.</li><li>입력 PNG는 클립의 저장 인덱스와 672×384 BICUBIC 설정으로 복원했습니다. 원본 HTTP 전송 이미지가 보존된 것은 아닙니다.</li><li>source_ms는 원본 시계, video_ms는 session_ms에 대응하는 첫 프레임 기준 상대 시간입니다. 발표에는 video_ms를 표시합니다.</li><li>데모는 20fps로 출력하고 중간 공백에는 직전 영상·관측을 유지합니다. 화면에 프레임 유지·관측 경과 시간을 표시합니다.</li><li>약물 종류·실제 섭취·복약 횟수·정확도는 이 기록으로 확정하지 않았습니다.</li></ul>',
      '<h2>영상 챕터</h2><table><tr><th>시작</th><th>끝</th><th>내용</th></tr>']
    manifest=read(out/'manifest.json')
    for c in manifest['chapters']:content.append(f'<tr><td>{c["start_seconds"]:.2f}초</td><td>{c["end_seconds"]:.2f}초</td><td>{esc(c["title"])}</td></tr>')
    content.append('</table>')
    for e in d.events:
        n=e['n']
        content.append(f'<h2>E02 #{n}</h2><p>후보 <code>{e["request"]["event"]["candidate"]["candidate_id"]}</code><br>최초 생성 {e["emitted_ms"]/1000:.3f}초 · 행동 {e["start"]/1000:.3f}–{e["end"]/1000:.3f}초<br><a href="clips/candidate-{n}.mp4">클립</a> · <a href="evidence/candidate-{n}/request.json">요청</a> · <a href="evidence/candidate-{n}/model_input.json">입력</a> · <a href="evidence/candidate-{n}/prompt.txt">프롬프트</a> · <a href="evidence/candidate-{n}/result.json">결과</a></p>')
        content.append('<table><tr><th>input</th><th>클립 frame</th><th>원본 frame</th><th>영상 시각</th><th>원본 시계 ms</th><th>메타데이터</th></tr>')
        for r in e['images']:content.append(f'<tr><td><a href="{r["file"]}">{r["input_index"]}</a></td><td>{r["clip_frame_index"]}</td><td>{r["source_frame_index"]}</td><td>{r["video_ms"]/1000:.3f}초</td><td>{r["source_ms"]}</td><td>관측 {r["sample_count"]} / 영역 {r["region_count"]}</td></tr>')
        content.append(f'</table><p>{esc(e["explanation"])}</p><p><strong>검토:</strong> {esc(e["caveat"])}</p><details><summary>저장된 응답 원문</summary><pre>{esc(json.dumps(e["result"]["result"],ensure_ascii=False,indent=2))}</pre></details>')
    content.append('<h2>발표자 메모</h2>')
    for n,(title,note) in enumerate(notes,1):content.append(f'<h3>{n:02d}. {title}</h3><p>{esc(note)}</p>')
    content.append('<h2>생성 코드</h2><pre>runtime\\.venv\\Scripts\\python.exe -X utf8 scripts\\build_latest_webcam_demo.py</pre><p>출력은 원본 실행 폴더와 분리됩니다. 더 최근 웹캠 실행이 생기면 사례별 설명을 다시 검토하도록 생성을 중단합니다.</p>')
    report_style='body{font:18px/1.75 "Malgun Gothic",sans-serif;background:#f6f8fb;color:#152435;max-width:1120px;margin:50px auto;padding:0 28px}h1{font-size:36px}h2{margin-top:48px}a{color:#176590}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #c5d0db;text-align:left}code,pre{font-size:14px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#e8eef4;padding:20px}details{margin:22px 0}'
    (out/'technical-summary.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>웹캠 데모 상세 근거와 발표자 메모</title><style>{report_style}</style>{"".join(content)}</html>',encoding='utf8')
