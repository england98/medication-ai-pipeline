"""Build an evidence-based demo from the latest completed local MP4 run.

No detector/VLM model inference. Replays the deterministic event state machine
against saved observations and verifies its ranges/revisions against the run.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import shutil
from functools import lru_cache
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from event_detection.detector import EventDetector
from event_detection.settings import DetectionConfig
from medication_contracts import DetectedObject, Landmark, ObservationSet, ResourceRef, TrackingUpdate

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1600, 900, 20
BG, FG, MUTED = '#101c29', '#f3f6fa', '#a7b8c9'
CYAN, GOLD, GREEN, RED = '#69d8ee', '#ffca70', '#74e0af', '#ffa295'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(64)
def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf', size)


def text(draw, xy, value, size=26, color=FG, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def wrap(draw, xy, value, width, size=26, color=FG, line=1.5):
    x, y = xy
    current = ''
    for word in value.split(' '):
        trial = current + (' ' if current else '') + word
        if draw.textlength(trial, font=font(size)) > width and current:
            text(draw, (x, y), current, size, color)
            y += size * line
            current = word
        else:
            current = trial
    text(draw, (x, y), current, size, color)
    return y + size * line


def latest_run():
    options = []
    for p in (ROOT / 'outputs').glob('*/test-run.json'):
        run = p.parent
        if not (run / 'summary.json').exists() or not (run / 'pipeline-config.json').exists():
            continue
        cfg, summary = read(run / 'pipeline-config.json'), read(run / 'summary.json')
        source = cfg.get('source')
        if isinstance(source, str) and source.lower().endswith('.mp4') and summary['run_status'] == 'FINISHED':
            options.append((read(p)['started_at'], run))
    return max(options)[1]


class Demo:
    def __init__(self, run, out):
        self.run, self.out = run, out
        for folder in ('assets', 'clips', 'evidence'):
            (out / folder).mkdir(parents=True, exist_ok=True)
        self.config = read(run / 'pipeline-config.json')
        self.source = Path(self.config['source'])
        self.source_sha = sha(self.source)
        assert self.source_sha == read(run / 'environment.json')['source_sha256']
        # Korean response commentary below is specific to this audited experiment.
        # Never silently reuse it for a later run with different results.
        if run.name != 'video-livinglab-A003-P201-G002-H120-q4_20260929_163755':
            raise ValueError('A newer MP4 run was found. Review its responses and update the case-specific presentation before building.')
        self.summary = read(run / 'summary.json')
        self.obs = rows(run / 'observations.jsonl')
        self.obs_times = [r['tracking']['frame']['session_ms'] for r in self.obs]
        self.history = rows(run / 'candidate-history.jsonl')
        self.rules = read(run / 'detector-config.json')['rules']
        self.events = []
        for path in run.glob('verification/*/request.json'):
            request = read(path)
            result = read(path.parent / 'result.json')
            inp = read(path.parent / 'model_input.json')
            self.events.append(dict(request=request, result=result, input=inp, folder=path.parent,
                                    start=request['event']['action_range']['start_ms'],
                                    end=request['event']['action_range']['end_ms']))
        self.events.sort(key=lambda e: e['start'])
        self.replay()
        self.frames, self.times = [], []
        with av.open(str(self.source)) as container:
            self.source_duration = float(container.duration / av.time_base)
            for f in container.decode(video=0):
                self.times.append(round(float(f.pts * f.time_base) * 1000))
                self.frames.append(f.to_image().resize((1152, 648), Image.Resampling.LANCZOS))
        assert len(self.frames) == self.summary['frames_buffered']
        self.last = self.times[-1]
        self.extract_inputs()

    def replay(self):
        first = TrackingUpdate.model_validate(self.obs[0]['tracking'])
        detector = EventDetector(first.context, ResourceRef(resource_id='demo-config',
            locator=str(self.run / 'detector-config.json'), media_type='application/json'), DetectionConfig(**self.rules))
        generated, replayed, self.motions = {}, [], {}
        for row in self.obs:
            tracking = TrackingUpdate.model_validate(row['tracking'])
            updates = detector.process(tracking,
                ObservationSet[DetectedObject].model_validate(row['objects']),
                ObservationSet[Landmark].model_validate(row['landmarks']))
            if detector.history:
                self.motions[tracking.frame.session_ms] = detector.history[-1][0].motions.model_dump(mode='json')['items']
            for update in updates:
                event = update.event
                key = event.candidate.candidate_id
                if key not in generated:
                    generated[key] = len(generated)
                    self.events[generated[key]]['emitted_ms'] = tracking.frame.session_ms
                n = generated[key]
                replayed.append((n, event.candidate.candidate_revision, event.action_range.model_dump()))
        original_ids = {e['request']['event']['candidate']['candidate_id']: n for n, e in enumerate(self.events)}
        original = [(original_ids[r['candidate']['candidate_id']], r['candidate']['candidate_revision'], r['action_range']) for r in self.history]
        assert replayed == original, 'Rule replay must reproduce every saved revision and action range'
        self.replay_count = len(replayed)

    def extract_inputs(self):
        checks = ('object_is_medication', 'object_transfer_into_mouth_visible')
        explanations = [
            ('병을 들고 살펴보는 동작으로 설명했습니다. 약 식별 YES, 입 전달 NO이므로 규칙은 REFUTED를 반환했습니다.',
             '약 식별은 YES지만 근거에는 “약으로 명확히 식별되지 않는다”고 적었습니다. 답변과 설명이 충돌합니다.'),
            ('흰색 병의 외형을 약 용기의 근거로 사용하고, 뚜껑을 열려는 동작으로 설명했습니다. 입 전달 NO로 REFUTED입니다.',
             '용기 외형만으로 복용 물체를 식별했습니다. 입 가시성 설명도 응답 내부에서 일치하지 않습니다.'),
            ('병을 약으로 확정하지 못했고, 입으로 들어가는 장면도 판단하지 못했습니다. 두 질문 모두 UNKNOWN입니다.',
             '모든 근거가 input 0의 0.200초를 참조합니다. 행동 구간 안의 2.350초와 4.500초에 대한 근거 연결이 없습니다.'),
            ('bottle 라벨과 검출 점수를 반복해 언급하며 약 식별과 입 전달 모두 UNKNOWN으로 답했습니다.',
             '근거 input 0은 3.600초로 행동 시작 5.100초보다 앞섭니다. 후행 영상도 요청 범위보다 650ms 짧습니다.'),
        ]
        for n, e in enumerate(self.events, 1):
            e['n'] = n
            e['explanation'], e['caveat'] = explanations[n-1]
            clip = e['request']['media']['clip']
            src = Path(clip['video']['locator'])
            if not src.is_absolute():
                src = self.run / src
            shutil.copy2(src, self.out / f'clips/candidate-{n}.mp4')
            dest = self.out / f'evidence/candidate-{n}'
            dest.mkdir(exist_ok=True)
            for name in ('request.json', 'result.json', 'model_input.json'):
                shutil.copy2(e['folder'] / name, dest / name)
            (dest / 'prompt.txt').write_text(e['input']['rendered_prompt'], encoding='utf-8')
            selected = {f['clip_frame_index']: f for f in e['input']['frames']}
            e['images'] = []
            metadata = [json.loads(line[len('Input image metadata: '):]) for line in e['input']['rendered_prompt'].splitlines() if line.startswith('Input image metadata: ')]
            with av.open(str(src)) as container:
                count = 0
                for index, decoded in enumerate(container.decode(video=0)):
                    declared = clip['frames'][index]
                    assert abs(float(decoded.pts * decoded.time_base)*1000 - declared['clip_ms']) <= 2
                    count += 1
                    if index not in selected:
                        continue
                    record = selected[index]
                    assert record['roi_id'] is None and record['padding'] == [0,0,0,0]
                    pixels = decoded.to_image().convert('RGB').resize(tuple(record['input_size']), Image.Resampling.BICUBIC)
                    filename = f'assets/candidate-{n}-input-{record["input_index"]}.png'
                    pixels.save(self.out / filename)
                    m = metadata[record['input_index']]
                    assert m['source'] == declared['source']
                    e['images'].append(dict(file=filename, input_index=record['input_index'], clip_frame_index=index,
                        source_frame_index=declared['source']['frame_index'], source_ms=declared['source']['source_ms'],
                        input_size=record['input_size'], sha256=sha(self.out / filename),
                        region_count=len(m['candidate_regions_reference_only']), sample_count=len(m['detection_samples_reference_only'])))
                assert count == len(clip['frames'])
            assert len(e['images']) == 4
            values = [e['result']['result']['checks'][c] for c in checks]
            derived = 'CONFIRMED' if all(v == 'YES' for v in values) else 'REFUTED' if 'NO' in values else 'UNCERTAIN'
            assert derived == e['result']['verification']
        for name in ('summary.json', 'test-run.json', 'detector-config.json', 'pipeline-config.json',
                     'test-execution.json', 'test-review-metrics.json', 'candidate-history.jsonl', 'queue-events.jsonl',
                     'observations.jsonl', 'environment.json', 'execution-config.json'):
            shutil.copy2(self.run / name, self.out / 'evidence' / name)

    def annotated(self, index):
        im = self.frames[index].copy()
        d = ImageDraw.Draw(im)
        t = self.times[index]
        row = self.obs[max(0, bisect.bisect_right(self.obs_times, t)-1)]
        def box(b, color, label=''):
            b = [v*s for v, s in zip(b, (1152,648,1152,648))]
            d.rectangle(b, outline=color, width=2)
            if label:
                text(d,(b[0]+3,max(0,b[1]-23)),label,17,color,True)
        box(row['tracking']['target_bbox'], GREEN, 'TARGET')
        for obj in row['objects']['items']:
            box(obj['bbox'], GOLD, f"{obj['class_label']} {obj['score']:.2f}")
        points = row['landmarks']['items']
        pose = {p['index']:p['point'] for p in points if p['part']=='pose'}
        for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)):
            if a in pose and b in pose:
                d.line([(pose[k][0]*1152,pose[k][1]*648) for k in (a,b)],fill=CYAN,width=2)
        for p in points:
            if p['part']=='mouth' or p['part'].startswith('hand'):
                x,y=p['point'][0]*1152,p['point'][1]*648
                c=RED if p['part']=='mouth' else '#e09bf6'
                d.ellipse((x-2,y-2,x+2,y+2), fill=c)
        return im, row

    def scene(self, index, speed='1배속', freeze=None, original=False):
        im = Image.new('RGB',(W,H),BG)
        d = ImageDraw.Draw(im)
        t = self.times[index]
        if original:
            source = self.frames[index]
            row = self.obs[max(0,bisect.bisect_right(self.obs_times,t)-1)]
        else:
            source,row = self.annotated(index)
        im.paste(source,(0,93))
        text(d,(34,24),'원본 영상' if original else '객체·동작 관측과 이벤트 후보 생성',33,FG,True)
        text(d,(1168,27),f'원본 {t/1000:0.2f}초   {speed}',27,CYAN)
        x = 1180
        text(d,(x,116),'저장 관측 기반 재현',25,CYAN,True)
        text(d,(x,160),'YOLO11n + MediaPipe',22)
        age=t-row['tracking']['frame']['session_ms']
        text(d,(x,197),f'관측 시각 {t-age:04d}ms  /  경과 {age}ms',18,MUTED)
        counts=collections.Counter(o['class_label'] for o in row['objects']['items'])
        text(d,(x,242),'일반 객체',23,GOLD,True)
        wrap(d,(x,278),', '.join(f'{k} {v}' for k,v in counts.items()),385,20)
        text(d,(x,365),'손과 입의 기하 특징',23,CYAN,True)
        features=self.motions.get(t-age,[])
        for j,hand in enumerate(('hand_left','hand_right')):
            vals={m['name']:m['value'] for m in features if f'part:{hand}' in m['entity_refs']}
            dist=vals.get('hand_mouth_distance'); velocity=vals.get('hand_mouth_approach_speed')
            ds='미관측' if dist is None else f'{dist:.3f}'
            vs='없음' if velocity is None else f'{velocity:+.3f}'
            text(d,(x,404+j*36),f'{"왼손" if j==0 else "오른손"}  거리 {ds}  속도 {vs}',21)
        text(d,(x,485),'진입 d ≤ 0.16',22,GOLD)
        text(d,(x,523),'또는 d ≤ 0.40, v ≥ 0.025',22,GOLD)
        text(d,(x,560),'거리: 대상 키 기준 / 속도: 키/초',18,MUTED)
        active=[e for e in self.events if e['emitted_ms']<=t<e['end']]
        text(d,(x,612),'생성된 활성 후보',21,MUTED)
        text(d,(x,646),', '.join(f'E02 #{e["n"]}' for e in active) or '없음',29,GOLD,True)
        text(d,(28,751),'초록 대상   노랑 객체   하늘 포즈   분홍 손   주황 입',20,MUTED)
        for j,e in enumerate(self.events):
            y=790+j*22
            text(d,(30,y-11),f'#{e["n"]}',16)
            x0,x1=90,1118
            d.line((x0,y,x1,y),fill='#304459',width=2)
            # Thin line is retrospective action range; thick line starts at actual emitted candidate time.
            d.line((x0+e['start']/self.last*(x1-x0),y,x0+e['end']/self.last*(x1-x0),y),fill='#647b92',width=6)
            d.line((x0+e['emitted_ms']/self.last*(x1-x0),y,x0+e['end']/self.last*(x1-x0),y),fill=GOLD,width=8)
            px=x0+t/self.last*(x1-x0)
            d.line((px,y-7,px,y+7),fill=FG,width=2)
        text(d,(1180,757),'회색: 소급 포함한 행동 구간',18,MUTED)
        text(d,(1180,790),'노랑: 후보 생성 이후',18,GOLD)
        text(d,(1180,834),'VLM 결과는 뒤에서 사후 요약',19,MUTED)
        if freeze:
            d.rectangle((0,665,1152,739), fill='#52391b')
            text(d,(28,681),f'E02 #{freeze["n"]} 후보 생성  ·  원본 {t/1000:.3f}초',31,GOLD,True)
        return im

    def input_card(self,e):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(48,28),f'E02 #{e["n"]}  /  VLM에 전달한 전체 프레임 4장',38,FG,True)
        text(d,(48,87),'저장된 클립·인덱스·리사이즈 설정으로 복원한 입력   672 × 384   ROI 입력 없음',23,MUTED)
        for j,r in enumerate(e['images']):
            x=48+(j%2)*620;y=143+(j//2)*328
            with Image.open(self.out/r['file']) as img: im.paste(img.resize((500,286)),(x,y))
            text(d,(x,y+290),f'input {j}  /  원본 {r["source_ms"]/1000:.3f}초  /  frame {r["source_frame_index"]}',20,GOLD if j==0 else FG)
        x=1280
        text(d,(x,155),'모델 응답',28,CYAN,True)
        checks=e['result']['result']['checks']
        text(d,(x,219),'약 식별',22,MUTED);text(d,(x,254),checks['object_is_medication'],30)
        text(d,(x,322),'입으로 전달',22,MUTED);text(d,(x,357),checks['object_transfer_into_mouth_visible'],30)
        text(d,(x,434),'규칙 적용 결과',22,MUTED);text(d,(x,474),e['result']['verification'],28,GOLD,True)
        wrap(d,(x,570),'응답의 두 근거 모두 input 0만 명시',260,24)
        text(d,(48,837),'실제 추론은 어제 실행한 결과입니다. 이 장면은 발표를 위한 사후 요약입니다.',22,MUTED)
        return im

    def title_card(self, ending=False):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(64,62),'복약 이벤트 탐지와 VLM 검증' if not ending else '4개 후보의 검증 결과',51,FG,True)
        text(d,(66,142),'2026.09.29 16:37 실행   livinglab A003 / P201 / G002 / H120',25,CYAN)
        if ending:
            for j,(a,b) in enumerate((('후보 4건','E02 손·입 접근 관련 후보'),('REFUTED 2건','약 식별 YES + 입 전달 NO'),('UNCERTAIN 2건','약 식별 UNKNOWN + 입 전달 UNKNOWN'))):
                text(d,(70,240+j*135),a,41,GOLD,True);text(d,(650,252+j*135),b,30)
            text(d,(70,699),'복약 완료 미확정  /  session_result = null',35,FG,True)
            text(d,(70,775),'처리 성공 4/4와 판정의 타당성은 별도로 검토해야 합니다.',27,MUTED)
        else:
            im.paste(self.frames[90].resize((864,486)),(64,227))
            for j,line in enumerate(('6.7초 원본 영상','객체·손·입 관측','후보 생성 순간 정지','VLM 입력과 판정')):
                text(d,(990,266+j*98),line,34,FG if j!=2 else GOLD,True)
            text(d,(64,774),'저장된 탐지·검증 기록 기반 데모   모델 추론 재실행 없음',27,MUTED)
        return im

    def encode(self, path, frames):
        count=0
        with av.open(str(path),'w') as output:
            stream=output.add_stream('libx264',rate=FPS)
            stream.width,stream.height=W,H;stream.pix_fmt='yuv420p';stream.options={'crf':'21','preset':'fast'}
            for im in frames:
                for packet in stream.encode(av.VideoFrame.from_image(im)):
                    output.mux(packet)
                count+=1
            for packet in stream.encode():output.mux(packet)
        return count

    def videos(self):
        self.chapters=[]
        def demo_frames():
            cursor=0
            def chapter(name, count, source=None):
                nonlocal cursor
                self.chapters.append(dict(title=name,start_seconds=cursor/FPS,end_seconds=(cursor+count)/FPS,source=source))
                cursor+=count
            chapter('실행 개요',60)
            card=self.title_card()
            for _ in range(60):yield card
            chapter('원본 영상 1배속',len(self.frames),'original MP4')
            for i in range(len(self.frames)):yield self.scene(i,original=True)
            chapter('탐지 재현 0.5배속과 후보 생성 순간',len(self.frames)*2+4*36,'saved observations + validated rule replay')
            emitted={e['emitted_ms']:e for e in self.events}
            for i,t in enumerate(self.times):
                scene=self.scene(i,'0.5배속')
                for _ in range(2):yield scene
                if t in emitted:
                    still=self.scene(i,'생성 순간 정지',freeze=emitted[t])
                    still.save(self.out/f'assets/event-{emitted[t]["n"]}-moment.jpg',quality=94)
                    for _ in range(36):yield still
            for e in self.events:
                chapter(f'E02 #{e["n"]} 입력과 결과',120,'saved model_input.json and result.json')
                card=self.input_card(e)
                for _ in range(120):yield card
            chapter('결과 요약',120)
            card=self.title_card(ending=True)
            for _ in range(120):yield card
        print('Encoding presentation demo...',flush=True)
        self.demo_count=self.encode(self.out/'event-vlm-demo.mp4',demo_frames())
        print('Encoding full observation overlay...',flush=True)
        self.overlay_count=self.encode(self.out/'detection-overlay.mp4',(self.scene(i) for i in range(len(self.frames))))
        self.title_card().save(self.out/'assets/poster.jpg',quality=94)
        self.scene(90).save(self.out/'assets/overlay-preview.jpg',quality=94)

    def manifest(self):
        events=[]
        for e in self.events:
            events.append(dict(number=e['n'],candidate=e['request']['event']['candidate'],request_id=e['request']['request_id'],
                action_range=e['request']['event']['action_range'],first_emitted_ms=e['emitted_ms'],
                emitted_time_method='deterministic rule replay; all 58 saved revisions and ranges matched',
                clip_range=e['request']['media']['clip']['actual_range'],input_frames=e['images'],
                checks=e['result']['result']['checks'],verification=e['result']['verification'],
                evidence=e['result']['result']['evidence'],explanation_ko=e['explanation'],review_note_ko=e['caveat']))
        output=dict(source_run=str(self.run.relative_to(ROOT)),source_started_at=read(self.run/'test-run.json')['started_at'],
            source=dict(file=str(self.source),sha256=self.source_sha,frames=len(self.frames),duration_seconds=self.source_duration),
            method='Saved observations and deterministic rule replay; no model inference rerun. VLM images reconstructed from clips with recorded BICUBIC resize.',
            rule_replay=dict(matched_history_rows=self.replay_count),events=events,chapters=self.chapters,
            videos=[dict(file=name,frames=n,fps=FPS,duration_seconds=n/FPS,size=[W,H],codec='h264',sha256=sha(self.out/name)) for name,n in [('event-vlm-demo.mp4',self.demo_count),('detection-overlay.mp4',self.overlay_count)]],
            limitations=['Overlays hold the previous observation for intervening frames (50 ms).',
                'Event emission timestamps reconstructed from saved observations, not recorded wall-clock timestamps.',
                'Selected input pixels reconstructed; original HTTP image payloads were not archived.',
                'Demo durations do not represent actual inference latency. Original command took 393.5 seconds.',
                '4 overlapping event candidates do not imply 4 medication intakes.',
                'No medication completion confirmed; some model answers contradict their evidence.'])
        save(self.out/'manifest.json',output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/demo-latest-mp4-20260929')
    parser.add_argument('--skip-video',action='store_true')
    args=parser.parse_args()
    demo=Demo(args.run or latest_run(),args.out)
    if not args.skip_video:
        demo.videos()
        demo.manifest()
    from latest_mp4_presentation import build_presentation
    build_presentation(demo)
    print(args.out,flush=True)


if __name__=='__main__':
    main()
