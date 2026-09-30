"""Portable Korean HTML presentation and evidence appendix for the MP4 demo."""
from __future__ import annotations

import html
import json


def esc(value):
    return html.escape(str(value))


def build_presentation(demo):
    out = demo.out
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    slides = []
    notes = []

    def add(title, body, source, note, cls=''):
        n = len(slides) + 1
        slides.append(f'<section class="slide {cls}" id="slide-{n}"><h2>{title}</h2><div class="content">{body}</div><footer>{source}<span>{n:02d} / 12</span></footer></section>')
        notes.append((title, note))

    add('복약 이벤트 탐지와<br>VLM 검증', '''
      <div class="cover"><div><p class="lead">최신 MP4 테스트의<br>탐지 과정과 판정 근거</p>
      <p>2026년 9월 29일 16:37 실행</p><p class="muted">livinglab A003 / P201 / G002 / H120</p>
      <p class="accent">E02 후보 4건<br>반증 2건 · 불확실 2건</p></div>
      <img src="assets/candidate-3-input-2.png" alt="원본 4.500초의 VLM 입력 프레임"></div>
      ''', 'test-run.json, summary.json, verification/*/result.json',
      '이번 발표는 어제 실행한 최신 MP4 테스트입니다. 기존 웹캠 데모와 다른 실행입니다. 파이프라인은 정상 종료됐고 결과 4건을 수신했지만 복약 완료는 확정되지 않았습니다. 약 5분 발표를 기준으로 구성했습니다.', 'title')

    add('실행 조건과 처리 흐름', '''
      <div class="columns"><div><table><tr><th>원본</th><td>1920 × 1080, 20fps<br>6.7초, 134프레임</td></tr>
      <tr><th>관측</th><td>YOLO11n + MediaPipe<br>CPU, 100ms 간격, 67프레임</td></tr>
      <tr><th>이벤트</th><td>기하 규칙 기반 E02 후보 4건<br>E01·E03 후보 0건</td></tr>
      <tr><th>검증</th><td>Qwen3-VL-2B Q4_K_M<br>후보당 전체 프레임 4장</td></tr></table></div>
      <div><ol class="process"><li>객체·포즈·손·입 관측</li><li>거리·접근 속도로 후보 생성</li><li>행동 종료 후 전후 클립 확보</li><li>선택 이미지 + 메타데이터 + 질문</li><li>VLM 응답에 판정 규칙 적용</li></ol>
      <p class="small">대상은 operator 모드로 연결했습니다.<br>별도의 학습 기반 행동 분류기는 사용하지 않았습니다.</p></div></div>
      ''', 'pipeline-config.json, detector-config.json, summary.json',
      'YOLO는 일반 물체를, MediaPipe는 대상의 포즈·손·입을 관측합니다. 행동 후보는 별도 행동 신경망이 아니라 손과 입의 거리·속도와 상태 유지 규칙으로 생성합니다. 생체인증은 이번 범위에 포함되지 않습니다.')

    add('탐지 과정과 후보 생성 순간', '''
      <video controls preload="metadata" poster="assets/poster.jpg" src="event-vlm-demo.mp4"></video>
      <p class="small">원본 1배속, 탐지 재현 0.5배속, 생성 순간 정지, 후보별 VLM 입력과 결과</p>
      <p class="small">영상의 VLM 결과는 사후 요약입니다. 실제 실행은 약 6분 33.5초가 걸렸습니다.</p>
      ''', 'event-vlm-demo.mp4, observations.jsonl, candidate-history.jsonl',
      '데모를 재생합니다. 처음에는 원본을 보고, 다음에는 탐지 기록을 겹쳐 봅니다. 후보 생성 순간마다 1.8초 정지합니다. 노란 구간은 후보가 생성된 뒤의 활성 구간이고 회색은 이전 접근 동작을 포함한 행동 범위입니다. 영상의 결과 카드는 실제 추론 속도를 재현하지 않습니다.', 'video-slide')

    add('E02 후보를 만드는 조건', '''
      <div class="columns"><div><p class="formula">d ≤ 0.16</p><p>손과 입의 거리가 진입 기준 이내</p>
      <p class="formula">또는 d ≤ 0.40, v ≥ 0.025</p><p>입으로 충분히 가까워지는 동작</p>
      <table class="compact"><tr><th>유지·종료</th><td>유지 거리 0.24<br>단서 소멸 500ms 후 종료</td></tr>
      <tr><th>발행</th><td>최소 활성 100ms<br>이전 관측 최대 600ms 포함</td></tr></table>
      <p class="small">d: 대상의 픽셀 키로 정규화한 거리<br>v: 접근 속도, 대상 키/초</p></div>
      <div><img class="wide" src="assets/event-3-moment.jpg" alt="2.4초 E02 후보 3 생성 순간">
      <p class="accent">#3 최초 생성 2.400초</p><p>행동 범위는 앞선 접근 관측을 포함해<br>1.700초부터 시작합니다.</p>
      <p class="small">생성 시각은 저장 관측값으로 규칙을 재현했습니다.<br>원본 후보 이력 58행의 구간·revision이 모두 일치합니다.</p></div></div>
      ''', 'detector-config.json, event_detection/detector.py, candidate-history.jsonl',
      'd는 손의 관측점 중 입 중심과 가장 가까운 점의 거리입니다. v는 직전 관측과의 거리 감소를 시간으로 나눕니다. 한 번 생성된 후보는 거리 0.24까지 유지하며, 단서가 500ms 없어야 종료합니다. 후보의 시작 범위와 후보를 처음 내보낸 시각을 혼동하면 안 됩니다. 슬라이드 생성 시각은 저장된 값에 대한 규칙 재현으로 확인했습니다.')

    event_rows = ''.join(f'<tr><td>#{e["n"]} E02</td><td>{e["emitted_ms"]/1000:.3f}초</td><td>{e["start"]/1000:.3f}–{e["end"]/1000:.3f}초</td><td>{e["request"]["media"]["clip"]["actual_range"]["start_ms"]/1000:.3f}–{e["request"]["media"]["clip"]["actual_range"]["end_ms"]/1000:.3f}초</td><td class="accent">{e["result"]["verification"]}</td></tr>' for e in demo.events)
    add('후보 4건의 시간 관계', f'''
      <table><tr><th>후보</th><th>최초 생성¹</th><th>행동 범위</th><th>실제 클립</th><th>판정</th></tr>{event_rows}</table>
      <p class="lead">겹치는 후보 구간이 있으므로<br>후보 4건을 복약 4회로 해석할 수 없습니다.</p>
      <p class="small">E02 클립은 행동 시작 전 1.5초, 종료 후 1초를 요청합니다.<br>#4는 7.300초까지 요청했지만 마지막 원본 프레임은 6.650초입니다.</p>
      <p class="small">¹ 저장 관측과 당시 기하 규칙으로 재현한 영상 시각. 모든 시각은 원본 영상 기준입니다.</p>
      ''', 'request.json, media.clip.frames, result.json, manifest.json',
      '행동 범위, 후보 최초 생성, 영상 클립 범위를 구분합니다. 첫 후보의 행동 범위는 0초지만 후보 발행은 0.4초입니다. 클립은 문맥 확보를 위해 더 넓게 가져옵니다. 마지막 후보는 EOF로 뒤쪽 650ms를 확보하지 못했습니다. 각 후보를 별도의 복약 횟수로 합산하면 안 됩니다.')

    add('VLM에 실제 전달한 정보', '''
      <div class="columns"><div><h3>이미지 입력</h3><p>후보 클립에서 시간순으로 4장 선택</p>
      <p>1920 × 1080 전체 화면을<br><strong>672 × 384</strong>로 리사이즈</p>
      <p>균등 샘플링 + TARGET 영역이<br>연결되는 프레임 최소 1장 보장</p>
      <p class="accent">ROI는 후보당 8장 생성했지만<br>이번 VLM 입력에는 포함하지 않았습니다.</p></div>
      <div><h3>텍스트 입력</h3><p>후보 유형 E02, 행동 범위, 대상 track</p>
      <p>각 이미지의 원본 프레임·시각,<br>동일 프레임의 영역 좌표와 탐지 관측</p>
      <p>약 식별 및 입 전달 여부를 묻는 질문</p>
      <p class="small">좌표·검출 라벨은 참고 가설로 제시합니다.<br>프레임에 정확히 대응하는 관측이 없으면<br>빈 목록을 전달합니다.</p></div></div>
      <p class="small">다음 슬라이드의 이미지는 보존된 클립과 model_input.json의 설정으로 복원했습니다. 원본 HTTP 이미지 전송본은 별도 저장되지 않았습니다.</p>
      ''', 'model_input.json의 rendered_prompt·frames, llama_cpp.py, media.py',
      '프레임 번호는 추정하지 않고 저장된 Clip.frames 대응표와 model_input.json을 연결했습니다. 리사이즈는 당시와 같은 BICUBIC입니다. 후보 2·4는 균등 선택만으로 대상 영역 프레임이 남지 않아 TARGET 대응 프레임을 포함하도록 조정됐습니다. ROI 파일이 존재하는 것과 VLM에 전달한 것은 다릅니다. 이번 입력에는 ROI가 없습니다.')

    for e in demo.events:
        image_cells = ''.join(f'<figure><img src="{r["file"]}" alt="후보 {e["n"]} input {r["input_index"]}, 원본 {r["source_ms"]}ms"><figcaption>input {r["input_index"]} · {r["source_ms"]/1000:.3f}초 · 원본 #{r["source_frame_index"]}<br><span class="muted">탐지 관측 {r["sample_count"]}개 / 영역 {r["region_count"]}개</span></figcaption></figure>' for r in e['images'])
        c = e['result']['result']['checks']
        add(f'E02 #{e["n"]} 입력 프레임과 판정', f'''
          <div class="event-layout"><div class="frames">{image_cells}</div><div>
          <p class="small">행동 {e['start']/1000:.3f}–{e['end']/1000:.3f}초</p>
          <table class="compact"><tr><th>약 식별</th><td>{c['object_is_medication']}</td></tr>
          <tr><th>입 전달</th><td>{c['object_transfer_into_mouth_visible']}</td></tr></table>
          <p class="verdict">{e['result']['verification']}</p>
          <p class="event-text">{e['explanation']}</p>
          <p class="small">모델의 두 근거 모두 <strong class="accent">input 0</strong>을 명시합니다.</p>
          <p class="review">검토: {e['caveat']}</p></div></div>
          ''', f'evidence/candidate-{e["n"]}/model_input.json, request.json, result.json',
          f'네 이미지는 모델이 받은 순서입니다. {e["explanation"]} {e["caveat"]} 아래 탐지 관측과 영역 수는 실제 프롬프트에 들어간 같은 프레임 메타데이터의 수입니다. 모델이 다른 이미지를 내부적으로 사용하지 않았다고 단정할 수는 없고, 저장한 근거 인덱스가 0에 집중했다는 사실을 보여줍니다.')

    add('모델 응답과 규칙 판정의 구분', '''
      <div class="columns"><div><h3>VLM이 답하는 두 질문</h3><p>같은 복용 물체를 약으로 식별하는가?<br>그 물체가 입으로 들어가는가?</p>
      <table class="compact"><tr><th>답변 조합</th><th>규칙 결과</th></tr><tr><td>YES + YES</td><td>CONFIRMED</td></tr><tr><td>하나라도 NO</td><td>REFUTED</td></tr><tr><td>NO 없이 UNKNOWN 포함</td><td>UNCERTAIN</td></tr></table>
      <p class="small">최종 이벤트 판정은 코드가 계산합니다.<br>VLM은 checks와 관측 근거를 응답합니다.</p></div>
      <div><h3>이번 응답에서 확인한 문제</h3><p><strong>#1</strong> 약 식별 YES와 “식별 불가” 근거의 충돌</p>
      <p><strong>#2</strong> 약 용기의 외형을 복용 물체의 근거로 사용</p>
      <p><strong>#3·#4</strong> 행동 구간 이전의 input 0만 근거로 명시</p>
      <p class="accent">처리 상태 OK는 응답 형식 검사를<br>통과했다는 뜻입니다.<br>내용의 타당성까지 보장하지 않습니다.</p></div></div>
      ''', 'vlm_verification/response.py, verification/*/result.json',
      'YES·NO·UNKNOWN의 조합을 코드에서 이벤트 판정으로 바꿉니다. 이 실행에서는 NO가 포함된 두 후보가 REFUTED이고 나머지가 UNCERTAIN입니다. REFUTED는 해당 후보의 모델 응답에 대한 규칙 결과이며 세션 전체에서 약을 안 먹었다는 뜻은 아닙니다. 내부 추론 과정을 알 수는 없고 저장된 답변·명시된 근거·판정 코드만 설명합니다.')

    add('처리 결과와 후속 검증 과제', '''
      <div class="columns"><div><p class="big">4 / 4</p><p>요청 처리 성공, 결과 수신 완료<br>처리 오류 0건</p>
      <p class="big accent">완료 미확정</p><p>CONFIRMED 0건<br>세션 ACTIVE, session_result = null</p>
      <p class="small">프로그램은 FINISHED / END_OF_INPUT으로 정상 종료했습니다.</p></div>
      <div><h3>관측된 처리 시간</h3><table class="compact"><tr><th>후보</th><th>READY부터 결과까지</th></tr>
      <tr><td>#1 / #2</td><td>142.753초 / 209.539초</td></tr><tr><td>#3 / #4</td><td>271.008초 / 352.933초</td></tr></table>
      <p class="small">단일 worker 순차 처리, 큐 대기 포함<br>6.7초 영상의 전체 명령 실행 시간: 393.5초</p>
      <h3>다음 비교 실험</h3><p>핵심 시점 선택과 손·입 ROI 입력<br>약 용기와 복용 물체의 구분<br>답변과 근거 설명의 일치 여부</p>
      <p class="small">구간별 정답 라벨이 없어 정확도·미탐률은 계산하지 않았습니다.</p></div></div>
      ''', 'summary.json, test-execution.json, states.jsonl, queue-events.jsonl',
      '파이프라인 연결은 검증됐지만 복약 완료 판단과 실시간 성능은 달성하지 않았습니다. 후보 4건 모두 세션 판단에 반영됐습니다. 후속 비교는 정답 구간을 먼저 정하고 핵심 시점과 ROI, 복용 물체를 구분하는 질문, 근거 일관성을 평가하는 것이 적절합니다.')

    style = '''
    :root{--bg:#101c29;--fg:#f3f6fa;--muted:#a7b8c9;--accent:#ffca70;--cyan:#69d8ee}
    *{box-sizing:border-box}body{margin:0;background:#080f17;color:var(--fg);font-family:'Malgun Gothic','맑은 고딕',sans-serif}
    .slide{position:relative;width:1280px;height:720px;margin:26px auto;padding:42px 58px 46px;background:var(--bg);break-after:page;overflow:hidden}
    h2{font-size:40px;line-height:1.23;letter-spacing:-1.4px;margin:0 0 28px;font-weight:700}h3{font-size:25px;margin:0 0 17px;color:var(--cyan)}
    p{font-size:23px;line-height:1.48;margin:14px 0}.lead{font-size:30px;line-height:1.5}.small{font-size:17px;line-height:1.55;color:var(--muted)}
    .muted{color:var(--muted)}.accent{color:var(--accent)}.big{font-size:52px;font-weight:700;margin:12px 0}.formula{font-size:32px;color:var(--accent);font-weight:700}
    .columns{display:grid;grid-template-columns:1fr 1fr;gap:48px}.wide{width:100%;height:auto}.process{padding-left:34px;margin:0}.process li{font-size:24px;line-height:1.5;margin:19px 0;padding-left:8px}
    table{width:100%;border-collapse:collapse;font-size:22px;line-height:1.5}th,td{text-align:left;border-bottom:1px solid #34485b;padding:16px 10px;vertical-align:top}th{font-weight:400;color:var(--muted)}.compact{font-size:20px}.compact th,.compact td{padding:10px 7px}
    footer{position:absolute;bottom:18px;left:58px;right:58px;font-size:13px;color:#8ea5b9;border-top:1px solid #34485b;padding-top:9px}footer span{float:right}
    .title h2{font-size:48px;margin-bottom:0}.cover{display:grid;grid-template-columns:1fr 620px;gap:36px}.cover img{width:620px;margin-top:14px}.cover p{font-size:23px}.cover .lead{font-size:29px;margin-top:21px}
    video{width:850px;height:478.125px;display:block;margin:0 auto;background:#000}.video-slide h2{margin-bottom:16px}.video-slide p{margin:6px 0;text-align:center}
    .event-layout{display:grid;grid-template-columns:660px 1fr;gap:36px}.frames{display:grid;grid-template-columns:1fr 1fr;gap:22px 15px}.frames figure{margin:0}.frames img{width:100%;display:block}.frames figcaption{font-size:15px;margin-top:8px;line-height:1.5}.event-text{font-size:21px;line-height:1.55}.verdict{font-size:32px;color:var(--accent);font-weight:700;margin:16px 0}.review{border-left:3px solid var(--accent);padding-left:13px;font-size:17px;line-height:1.55;color:#d4dfeb}
    nav{position:fixed;z-index:10;top:8px;right:12px;display:flex;gap:8px}button,nav a{font:14px 'Malgun Gothic';color:white;background:#24384c;border:1px solid #536b83;padding:8px 12px;text-decoration:none;cursor:pointer}
    body.present{overflow:hidden}.present .slide{display:none;margin:0;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%) scale(var(--scale,1));transform-origin:center}.present .slide.current{display:block}
    @page{size:1280px 720px;margin:0}@media print{body{background:var(--bg);-webkit-print-color-adjust:exact;print-color-adjust:exact}nav{display:none}.slide,.present .slide{display:block!important;position:relative;left:auto;top:auto;transform:none;margin:0;width:1280px;height:720px;break-inside:avoid}.slide:last-child{break-after:auto}}
    '''
    script = '''
    const slides=[...document.querySelectorAll('.slide')];let current=0;
    function resize(){document.documentElement.style.setProperty('--scale',Math.min(innerWidth/1280,innerHeight/720))}
    function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.classList.toggle('current',i===current);if(i!==current)s.querySelectorAll('video').forEach(v=>v.pause())});if(!document.body.classList.contains('present'))slides[current].scrollIntoView({behavior:'smooth'})}
    function present(){document.body.classList.toggle('present');resize();show(current)}
    document.querySelector('#present').onclick=present;document.querySelector('#prev').onclick=()=>show(current-1);document.querySelector('#next').onclick=()=>show(current+1);
    document.querySelector('#fullscreen').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();
    addEventListener('resize',resize);addEventListener('keydown',e=>{if(e.target.tagName==='VIDEO')return;if(['ArrowRight','PageDown'].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1)}if(e.key==='p'||e.key==='P')present();if(e.key==='Escape')document.body.classList.remove('present')});
    show(0);resize();
    '''
    document = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>최신 MP4 테스트: 이벤트 탐지와 VLM 검증</title><style>{style}</style></head><body><nav><button id="present">발표 모드 P</button><button id="prev">이전</button><button id="next">다음</button><button id="fullscreen">전체 화면</button><a href="technical-summary.html">상세 근거</a></nav>{"".join(slides)}<script>{script}</script></body></html>'
    (out / 'presentation.html').write_text(document, encoding='utf-8')

    appendix = ['<h1>최신 MP4 테스트 데모 자료</h1><p>2026.09.29 16:37:55 KST · 개발·연구팀 발표용</p>',
        '<p><a href="presentation.html">발표 슬라이드</a> · <a href="presentation.pdf">공유용 PDF</a> · <a href="event-vlm-demo.mp4">전체 데모 MP4</a> · <a href="detection-overlay.mp4">원본 속도 탐지 영상</a> · <a href="manifest.json">출처·프레임 매핑</a></p>',
        '<h2>재생·공유 안내</h2><p>presentation.html을 Chrome 또는 Edge로 열고 발표 모드 버튼을 누릅니다. 방향키로 이동하며 동영상은 재생 버튼으로 시작합니다. HTML과 assets, clips, evidence를 같은 폴더 구조로 유지하세요. ZIP은 먼저 압축을 해제합니다. PDF에서는 영상이 재생되지 않습니다. MP4는 한국어 자막형 설명을 포함한 무음 영상입니다.</p>',
        '<h2>결과 요약</h2><p>E02 후보 4건, REFUTED 2건, UNCERTAIN 2건, CONFIRMED 0건. 검증 처리와 결과 수신은 4건 모두 성공했으며 세션의 복약 완료는 확정되지 않았습니다.</p>',
        f'<p>선정 실행: <code>{esc(demo.run.name)}</code><br>선정 기준: test-run.started_at 기준 가장 최근의 정상 완료 MP4 실행.<br>원본: {esc(demo.source.name)}<br>SHA-256: <code>{demo.source_sha}</code></p>',
        '<h2>영상 챕터</h2><table><tr><th>시작</th><th>끝</th><th>내용</th></tr>']
    for c in manifest['chapters']:
        appendix.append(f'<tr><td>{c["start_seconds"]:.2f}초</td><td>{c["end_seconds"]:.2f}초</td><td>{esc(c["title"])}</td></tr>')
    appendix.append('</table><h2>재현 방법과 해석 범위</h2><ul><li>원본 MP4 해시를 실행 당시 environment.json과 대조했습니다.</li><li>딥러닝 탐지와 VLM 추론을 다시 실행하지 않았습니다. 객체·랜드마크는 저장 관측값입니다. 기하 상태 기계를 재현해 후보 생성 영상 시각을 얻었고 기존 후보 이력 58행의 구간·revision과 대조했습니다.</li><li>관측 사이 영상 프레임에는 직전 관측을 50ms 유지합니다. 화면에 관측 시각과 경과 시간을 표시했습니다.</li><li>VLM 입력 PNG는 원래 HTTP 전송본이 아니라, 보존된 클립의 정확한 프레임과 기록된 672×384 BICUBIC 변환으로 복원한 이미지입니다. 클립의 모든 PTS와 Clip.frames를 대조했습니다.</li><li>모델의 내부 사고 과정은 기록돼 있지 않습니다. 자료는 실제 입력·응답·명시된 근거와 코드의 판정 규칙을 설명합니다.</li><li>후보 4건은 구간이 겹칩니다. 정답 라벨이 없어 복약 횟수·정확도·미탐률을 계산하지 않았습니다.</li></ul>')
    for e in demo.events:
        n=e['n']; clip=e['request']['media']['clip']; c=e['result']['result']['checks']
        appendix.append(f'<h2>E02 #{n}: {e["result"]["verification"]}</h2><p>후보 ID <code>{esc(e["request"]["event"]["candidate"]["candidate_id"])}</code><br>요청 ID <code>{esc(e["request"]["request_id"])}</code><br>최초 생성 {e["emitted_ms"]/1000:.3f}초 · 행동 {e["start"]/1000:.3f}–{e["end"]/1000:.3f}초<br>원본 클립 <a href="clips/candidate-{n}.mp4">재생</a> · <a href="evidence/candidate-{n}/request.json">요청</a> · <a href="evidence/candidate-{n}/model_input.json">모델 입력</a> · <a href="evidence/candidate-{n}/prompt.txt">전체 프롬프트</a> · <a href="evidence/candidate-{n}/result.json">결과</a></p>')
        appendix.append('<table><tr><th>input</th><th>클립 frame</th><th>원본 frame</th><th>원본 시각</th><th>실제 텍스트 메타데이터</th></tr>')
        for r in e['images']:
            appendix.append(f'<tr><td><a href="{r["file"]}">{r["input_index"]}</a></td><td>{r["clip_frame_index"]}</td><td>{r["source_frame_index"]}</td><td>{r["source_ms"]/1000:.3f}초</td><td>탐지 관측 {r["sample_count"]}, 영역 {r["region_count"]}</td></tr>')
        appendix.append(f'</table><p>{esc(e["explanation"])}</p><p><strong>검토:</strong> {esc(e["caveat"])}</p><details><summary>저장된 모델 응답 원문</summary><pre>{esc(json.dumps(e["result"]["result"],ensure_ascii=False,indent=2))}</pre></details>')
    appendix.append('<h2>발표자 메모</h2>')
    for n,(title,note) in enumerate(notes,1):
        appendix.append(f'<h3>{n:02d}. {title}</h3><p>{esc(note)}</p>')
    appendix.append('<h2>다시 생성하기</h2><pre>runtime\\.venv\\Scripts\\python.exe -X utf8 scripts\\build_latest_mp4_demo.py</pre><p>기본 실행은 outputs 아래에서 가장 최근 완료된 MP4 실행을 선택합니다. --run, --out으로 실행과 출력 폴더를 지정할 수 있습니다.</p>')
    report_style='body{font:18px/1.75 "Malgun Gothic",sans-serif;background:#f6f8fb;color:#152435;max-width:1080px;margin:50px auto;padding:0 28px}h1{font-size:36px}h2{margin-top:48px}a{color:#176590}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #c5d0db;text-align:left}code,pre{font-size:14px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#e8eef4;padding:20px}details{margin:22px 0}'
    (out / 'technical-summary.html').write_text(f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>MP4 데모 상세 근거와 발표자 메모</title><style>{report_style}</style>{"".join(appendix)}</html>', encoding='utf-8')
