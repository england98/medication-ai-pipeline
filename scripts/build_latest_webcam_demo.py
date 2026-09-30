"""Create a presentation package for the latest recorded webcam experiment.

Only deterministic event rules are replayed. No neural inference or original
session mutation is performed. Case commentary is pinned to the audited run.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import math
import shutil
from pathlib import Path

import av
from PIL import Image, ImageDraw

from build_latest_mp4_demo import Demo, ROOT, W, H, FPS, BG, FG, MUTED, CYAN, GOLD, GREEN, RED, read, rows, save, sha, text, wrap


def latest_webcam():
    found=[]
    for p in (ROOT/'outputs').glob('*/test-run.json'):
        config=p.parent/'pipeline-config.json'
        if config.exists() and read(config).get('input_format')=='dshow':
            found.append((read(p)['started_at'],p.parent))
    return max(found)[1]


class WebcamDemo(Demo):
    def __init__(self, run, out):
        self.run,self.out=run,out
        if run.name!='webcam-pipeline-test01_20260929_160431':
            raise ValueError('The latest webcam run changed. Audit its evidence before reusing this case-specific commentary.')
        for f in ('assets','clips','evidence'):(out/f).mkdir(parents=True,exist_ok=True)
        self.config=read(run/'pipeline-config.json')
        self.source=run/'recording/input.mp4'
        self.source_sha=sha(self.source)
        self.recording=read(run/'recording/metadata.json')
        self.frame_index=rows(run/'recording/frames.jsonl')
        self.obs=rows(run/'observations.jsonl')
        self.obs_times=[o['tracking']['frame']['session_ms'] for o in self.obs]
        self.history=rows(run/'candidate-history.jsonl')
        self.rules=read(run/'detector-config.json')['rules']
        self.session=read(run/'session/session-view.json')
        self.error=read(run/'run-error.json')
        self.events=[]
        for q in run.glob('verification/*/request.json'):
            request=read(q)
            self.events.append(dict(request=request,result=read(q.parent/'result.json'),input=read(q.parent/'model_input.json'),
                folder=q.parent,start=request['event']['action_range']['start_ms'],end=request['event']['action_range']['end_ms']))
        self.events.sort(key=lambda e:e['start'])
        self.replay()
        self.frames,self.times=[],[]
        with av.open(str(self.source)) as c:
            self.source_duration=c.duration/av.time_base
            for i,f in enumerate(c.decode(video=0)):
                t=round(float(f.pts*f.time_base)*1000)
                assert abs(t-self.frame_index[i]['recording_ms'])<=2
                assert self.frame_index[i]['source']['source_ms']-self.recording['source_time_origin_ms']==t
                self.times.append(t)
                self.frames.append(f.to_image().resize((1152,648),Image.Resampling.LANCZOS))
        assert len(self.frames)==len(self.frame_index)==self.recording['frame_count']==154
        self.last=self.times[-1]
        self.gaps=[dict(before_ms=a,after_ms=b,gap_ms=b-a) for a,b in zip(self.times,self.times[1:]) if b-a>1000]
        self.tracking=collections.Counter(o['tracking']['tracking_status'] for o in self.obs)
        self.lost_ms=next(o['tracking']['frame']['session_ms'] for o in self.obs if o['tracking']['tracking_status']=='UNRESOLVED')
        assert not (run/'results.jsonl').exists() and not (run/'summary.json').exists()
        assert [r['state'] for r in rows(run/'queue-events.jsonl')]==['READY','READY']
        self.extract_inputs()

    def extract_inputs(self):
        import json
        commentary=[
            ('모델은 손이 입 근처에 있고 입·물체가 잘 보이지 않는다고 설명했습니다. 약 식별과 입 전달 모두 UNKNOWN입니다.',
             '두 근거 모두 input 0을 참조합니다. 손이 입을 가리는 장면과 후속 프레임 사이의 변화가 근거에 충분히 연결됐는지 검토가 필요합니다.'),
            ('모델은 입 근처의 손을 설명하면서 약이나 입으로 전달되는 물체를 확인하지 못했다고 답했습니다. 두 질문 모두 UNKNOWN입니다.',
             '근거 input 0은 8.608초로 행동 시작 9.280초보다 앞섭니다. 다른 입력을 무시했다고 단정할 수는 없으나 핵심 행동 시점과 근거의 연결은 부족합니다.')]
        for n,e in enumerate(self.events,1):
            e['n']=n;e['explanation'],e['caveat']=commentary[n-1]
            clip=e['request']['media']['clip'];source=Path(clip['video']['locator'])
            shutil.copy2(source,self.out/f'clips/candidate-{n}.mp4')
            dest=self.out/f'evidence/candidate-{n}';dest.mkdir(exist_ok=True)
            for name in ('request.json','result.json','model_input.json','states.jsonl'):shutil.copy2(e['folder']/name,dest/name)
            (dest/'prompt.txt').write_text(e['input']['rendered_prompt'],encoding='utf8')
            metas=[json.loads(s[len('Input image metadata: '):]) for s in e['input']['rendered_prompt'].splitlines() if s.startswith('Input image metadata: ')]
            selected={f['clip_frame_index']:f for f in e['input']['frames']};e['images']=[]
            with av.open(str(source)) as c:
                count=0
                for i,f in enumerate(c.decode(video=0)):
                    count+=1;declared=clip['frames'][i]
                    assert abs(float(f.pts*f.time_base)*1000-declared['clip_ms'])<=2
                    if i not in selected:continue
                    r=selected[i];m=metas[r['input_index']]
                    assert m['source']==declared['source']
                    assert r['roi_id'] is None and r['padding']==[0,0,0,0]
                    filename=f'assets/candidate-{n}-input-{r["input_index"]}.png'
                    f.to_image().convert('RGB').resize(tuple(r['input_size']),Image.Resampling.BICUBIC).save(self.out/filename)
                    e['images'].append(dict(file=filename,input_index=r['input_index'],clip_frame_index=i,
                        source_frame_index=declared['source']['frame_index'],source_ms=declared['source']['source_ms'],
                        video_ms=declared['source']['session_ms'],input_size=r['input_size'],sha256=sha(self.out/filename),
                        sample_count=len(m['detection_samples_reference_only']),region_count=len(m['candidate_regions_reference_only'])))
                assert count==len(clip['frames'])
            assert len(e['images'])==4 and e['result']['verification']=='UNCERTAIN'
            assert set(e['result']['result']['checks'].values())=={'UNKNOWN'}
            assert all(x['input_indices']==[0] for x in e['result']['result']['evidence'])
        for name in ('test-run.json','pipeline-config.json','environment.json','execution-config.json','detector-config.json',
                     'observations.jsonl','candidate-history.jsonl','queue-events.jsonl','run-error.json'):
            shutil.copy2(self.run/name,self.out/'evidence'/name)
        for folder,names in [('recording',('metadata.json','frames.jsonl')),('session',('session.json','session-view.json'))]:
            dest=self.out/'evidence'/folder;dest.mkdir(exist_ok=True)
            for name in names:shutil.copy2(self.run/folder/name,dest/name)
        shutil.copy2(self.source,self.out/'clips/source-webcam.mp4')

    def annotated(self,index):
        row=self.obs[max(0,bisect.bisect_right(self.obs_times,self.times[index])-1)]
        if row['tracking']['target_bbox'] is None:
            im=self.frames[index].copy();draw=ImageDraw.Draw(im)
            # General object detections continue even when target landmarks stop.
            for obj in row['objects']['items']:
                b=[v*s for v,s in zip(obj['bbox'],(1152,648,1152,648))]
                draw.rectangle(b,outline=GOLD,width=2)
                text(draw,(b[0]+3,max(0,b[1]-23)),f"{obj['class_label']} {obj['score']:.2f}",17,GOLD,True)
            return im,row
        return super().annotated(index)

    def scene_at(self,t,mode='기록 시간 기준 1배속',freeze=None,original=False):
        index=max(0,bisect.bisect_right(self.times,t)-1)
        frame_time=self.times[index]
        source,row=self.annotated(index)
        if original:source=self.frames[index]
        im=Image.new('RGB',(W,H),BG);im.paste(source,(0,93));d=ImageDraw.Draw(im)
        text(d,(32,24),'웹캠 원본 기록' if original else '웹캠 객체·동작 관측과 이벤트 후보',34,FG,True)
        text(d,(1148,26),f'영상 {t/1000:06.3f}초',29,CYAN)
        text(d,(1180,111),mode,24,CYAN,True)
        age=t-row['tracking']['frame']['session_ms'];held=t-frame_time
        text(d,(1180,160),f'표시 프레임 #{index} / {frame_time/1000:.3f}초',20)
        text(d,(1180,195),f'프레임 유지 {held}ms / 관측 경과 {age}ms',18,MUTED)
        tracked=row['tracking']['tracking_status']=='TRACKED'
        text(d,(1180,245),'대상 연결됨' if tracked else '대상 추적 손실',29,GREEN if tracked else RED,True)
        parts=collections.Counter(p['part'] for p in row['landmarks']['items'])
        text(d,(1180,292),f'객체 {len(row["objects"]["items"])}개 / 포즈 {parts["pose"]}점',21)
        text(d,(1180,328),f'입 {parts["mouth"]}점 / 손 {parts["hand_left"]+parts["hand_right"]}점',21)
        features=self.motions.get(row['tracking']['frame']['session_ms'],[])
        vals={m['name']:m['value'] for m in features if 'part:hand_right' in m['entity_refs']}
        dist=vals.get('hand_mouth_distance');speed=vals.get('hand_mouth_approach_speed')
        text(d,(1180,387),'오른손·입 기하 특징',23,CYAN,True)
        text(d,(1180,424),'거리 '+(f'{dist:.3f}' if dist is not None else '미관측'),24)
        text(d,(1180,461),'접근 속도 '+(f'{speed:+.3f}' if speed is not None else '미관측'),24)
        text(d,(1180,512),'d ≤ 0.16 또는',22,GOLD)
        text(d,(1180,547),'d ≤ 0.40, v ≥ 0.025',22,GOLD)
        text(d,(1180,582),'거리: 대상 키 / 속도: 대상 키/초',18,MUTED)
        active=[e for e in self.events if e['emitted_ms']<=t<e['end']]
        text(d,(1180,637),'활성 후보: '+(', '.join(f'E02 #{e["n"]}' for e in active) or '없음'),27,GOLD,True)
        if held>300:
            d.rectangle((0,689,1152,741),fill='#4d3523')
            text(d,(25,698),f'저장된 다음 프레임 대기: 직전 프레임 {held/1000:.3f}초 유지',24,GOLD)
        if freeze:
            d.rectangle((0,665,1152,741),fill='#52391b')
            text(d,(26,682),freeze,30,GOLD,True)
        text(d,(28,755),'초록 대상   노랑 객체   하늘 포즈   분홍 손   주황 입',21,MUTED)
        for j,e in enumerate(self.events):
            y=808+j*35;x0,x1=95,1118
            text(d,(28,y-12),f'#{e["n"]}',18)
            d.line((x0,y,x1,y),fill='#304459',width=2)
            d.line((x0+e['start']/self.last*(x1-x0),y,x0+e['end']/self.last*(x1-x0),y),fill='#647b92',width=7)
            d.line((x0+e['emitted_ms']/self.last*(x1-x0),y,x0+e['end']/self.last*(x1-x0),y),fill=GOLD,width=9)
            px=x0+t/self.last*(x1-x0);d.line((px,y-9,px,y+9),fill=FG,width=2)
        text(d,(1180,749),'회색: 소급 포함한 행동 구간',18,MUTED)
        text(d,(1180,783),'노랑: 후보 생성 이후',18,GOLD)
        text(d,(1180,824),'사후 VLM: 2건 UNCERTAIN',21)
        text(d,(1180,858),'중단으로 세션 판단에는 미반영',18,RED)
        return im

    def input_card(self,e):
        # Base composition uses source_ms as a display label, so supply a view
        # with relative times while preserving the original source clock in evidence.
        view=dict(e,images=[dict(r,source_ms=r['video_ms']) for r in e['images']])
        im=super().input_card(view);d=ImageDraw.Draw(im)
        d.rectangle((0,814,W,H),fill=BG)
        text(d,(48,835),'중단 후 저장된 VLM 결과입니다. 이번 웹캠 실행의 세션 판단에는 반영되지 않았습니다.',23,RED)
        return im

    def title_card(self,ending=False):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        text(d,(64,55),'웹캠 복약 이벤트 탐지와 VLM 검증' if not ending else '웹캠 시험의 확인 범위',48,FG,True)
        text(d,(66,135),'2026.09.29 16:04 실행   Integrated Webcam   1280 × 720',26,CYAN)
        if not ending:
            im.paste(self.frames[20].resize((864,486)),(64,223))
            for j,line in enumerate(('저장 영상 약 23.9초','E02 후보 생성 순간','VLM 입력 8장과 근거','추적 손실·중단 상태')):
                text(d,(980,263+j*99),line,31,GOLD if j==1 else FG,True)
            text(d,(64,777),'저장 기록 재현   사후 VLM 2건 불확실   세션 결과 미반영',28,MUTED)
        else:
            for j,(a,b) in enumerate((('E02 후보 2건','클립·요청 2건 생성'),('VLM 2건 UNCERTAIN','처리 OK / 약 식별·입 전달 UNKNOWN'),('입력 루프 중단','16:05:03, KeyboardInterrupt'))):
                text(d,(68,237+j*136),a,40,GOLD,True);text(d,(690,247+j*136),b,28)
            text(d,(68,695),'결과 파일은 있지만 세션 판단에는 미반영',36,RED,True)
            text(d,(68,771),'복약 완료 미확정   사후 결과를 원래 세션에 추가하지 않았습니다.',27,MUTED)
        return im

    def videos(self):
        self.chapters=[]
        timeline=[min(n*1000//FPS,self.last) for n in range(math.ceil(self.last/1000*FPS)+1)]
        def demo_frames():
            cursor=0
            def mark(name,count,source):
                nonlocal cursor
                self.chapters.append(dict(title=name,start_seconds=cursor/FPS,end_seconds=(cursor+count)/FPS,source=source));cursor+=count
            mark('실행 개요',60,'saved recording')
            card=self.title_card()
            for _ in range(60):yield card
            mark('전체 탐지 재현 1배속',len(timeline),'recording PTS, saved observations')
            for t in timeline:yield self.scene_at(t)
            for e in self.events:
                start=max(0,e['emitted_ms']-1200);end=e['end']+700
                ts=sorted(set(range(start,end+1,50))|{e['emitted_ms']})
                mark(f'E02 #{e["n"]} 생성 순간 0.5배속',len(ts)*2+40,'saved observations, validated deterministic rule replay')
                for t in ts:
                    card=self.scene_at(t,'후보 구간 0.5배속')
                    for _ in range(2):yield card
                    if t==e['emitted_ms']:
                        card=self.scene_at(t,'생성 순간 정지',freeze=f'E02 #{e["n"]} 후보 생성 / 영상 {t/1000:.3f}초')
                        card.save(self.out/f'assets/event-{e["n"]}-moment.jpg',quality=94)
                        for _ in range(40):yield card
            mark('추적 상태 전환',80,'saved tracking observations')
            card=self.scene_at(self.lost_ms,'추적 전환 순간',freeze=f'{self.lost_ms/1000:.3f}초부터 UNRESOLVED')
            card.save(self.out/'assets/tracking-lost.jpg',quality=94)
            for _ in range(80):yield card
            for e in self.events:
                mark(f'E02 #{e["n"]} VLM 입력과 사후 결과',140,'saved model_input and result')
                card=self.input_card(e)
                for _ in range(140):yield card
            mark('결과 수신·세션 반영 상태',120,'run-error, queue-events, session-view')
            card=self.title_card(True)
            for _ in range(120):yield card
        print('Encoding webcam demo...',flush=True)
        self.demo_count=self.encode(self.out/'event-vlm-demo.mp4',demo_frames())
        print('Encoding full observation timeline...',flush=True)
        self.overlay_count=self.encode(self.out/'detection-overlay.mp4',(self.scene_at(t) for t in timeline))
        self.title_card().save(self.out/'assets/poster.jpg',quality=94)
        self.scene_at(17312,'추적 손실 직전').save(self.out/'assets/tracking-before.jpg',quality=94)
        self.manifest()

    def manifest(self):
        events=[]
        for e in self.events:
            events.append(dict(number=e['n'],candidate=e['request']['event']['candidate'],request_id=e['request']['request_id'],
                action_range=e['request']['event']['action_range'],first_emitted_ms=e['emitted_ms'],input_frames=e['images'],
                requested_range=e['request']['media']['requested_range'],actual_range=e['request']['media']['clip']['actual_range'],
                verification=e['result']['verification'],checks=e['result']['result']['checks'],evidence=e['result']['result']['evidence'],
                finished_at=e['result']['finished_at'],explanation_ko=e['explanation'],review_note_ko=e['caveat']))
        save(self.out/'manifest.json',dict(source_run=str(self.run.relative_to(ROOT)),source_started_at=read(self.run/'test-run.json')['started_at'],
            source=dict(file=str(self.source),sha256=self.source_sha,frames=len(self.frames),duration_seconds=self.source_duration,
                video_time_basis='recording_ms / session_ms, relative to first received frame',source_time_origin_ms=self.recording['source_time_origin_ms']),
            source_run_status=self.error['run_status'],pipeline_results_received=0,vlm_results_saved=2,session_result=self.session['session_result'],
            tracking=dict(self.tracking),tracking_lost_ms=self.lost_ms,source_frame_gaps=self.gaps,
            rule_replay=dict(matched_history_rows=self.replay_count,emission_time_method='reconstructed from saved observations; every saved range/revision matched'),
            events=events,chapters=self.chapters,videos=[dict(file=f,frames=n,duration_seconds=n/FPS,fps=FPS,size=[W,H],codec='h264',sha256=sha(self.out/f)) for f,n in [('event-vlm-demo.mp4',self.demo_count),('detection-overlay.mp4',self.overlay_count)]],
            limitations=['No neural inference rerun. Original interrupted session unchanged.',
                'VLM images reconstructed from saved clip frames and recorded BICUBIC resize, not archived HTTP payloads.',
                'Original variable timestamps retained as labels; CFR output holds the preceding source frame and reports its age. Final tail rounded up to a display frame.',
                'Gaps are missing saved observations, not evidence that the person stopped moving.',
                'Result files completed after input interruption and were not received by the session decision stage.']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/demo-latest-webcam-20260929');p.add_argument('--skip-video',action='store_true');args=p.parse_args()
    demo=WebcamDemo(latest_webcam(),args.out)
    if not args.skip_video:demo.videos()
    from latest_webcam_presentation import build_presentation
    build_presentation(demo)
    print(args.out,flush=True)


if __name__=='__main__':main()
