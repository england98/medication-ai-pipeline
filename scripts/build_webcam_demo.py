"""Render a narrated-by-labels demo from the saved 2026-09-29 webcam run.

Uses recorded frames and observations only; no inference is rerun.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/webcam-pipeline-test01_20260929_160431"
OUT = ROOT / "outputs/demo-webcam-20260929"
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
WIDTH, HEIGHT, FPS = 1600, 900, 10


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def label(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size=25, color="#EDF3FA"
) -> None:
    draw.text(xy, value, font=font(size), fill=color)


def bbox_pixels(bbox: list[float]) -> tuple[int, int, int, int]:
    return tuple(round(v * s) for v, s in zip(bbox, (1280, 720, 1280, 720)))


def render(source: np.ndarray, obs: dict, t_ms: int, last_ms: int, stale_ms: int) -> np.ndarray:
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 29)
    frame = source.copy()
    tracking = obs["tracking"]
    tracked = tracking["tracking_status"] == "TRACKED"
    target = tracking.get("target_bbox")
    if tracked and target:
        cv2.rectangle(frame, bbox_pixels(target)[:2], bbox_pixels(target)[2:], (65, 218, 92), 3)
    for obj in obs["objects"]["items"]:
        b = bbox_pixels(obj["bbox"])
        cv2.rectangle(frame, b[:2], b[2:], (255, 190, 65), 2)
        cv2.putText(
            frame,
            obj["class_label"],
            (b[0], max(25, b[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 190, 65),
            2,
            cv2.LINE_AA,
        )
    points = obs["landmarks"]["items"]
    for item in points:
        if item["part"] not in {"mouth", "hand_left", "hand_right"}:
            continue
        x, y = item["point"]
        color = (40, 100, 255) if item["part"] == "mouth" else (230, 70, 210)
        cv2.circle(frame, (round(x * 1280), round(y * 720)), 5, color, -1, cv2.LINE_AA)
    canvas[:720, :1280] = frame
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    draw.rectangle((1280, 0, 1599, 899), fill="#111820")
    draw.rectangle((0, 720, 1279, 899), fill="#111820")
    label(draw, (1310, 28), "웹캠 이벤트 탐지", 30)
    label(draw, (1310, 83), f"영상 {t_ms / 1000:05.1f}초", 26, "#7EDCE8")
    label(draw, (1310, 139), "대상 추적", 22, "#A9BBCB")
    label(
        draw,
        (1310, 172),
        "연결됨" if tracked else "추적 손실",
        27,
        "#53DC88" if tracked else "#FF9285",
    )
    label(draw, (1310, 230), "관측 레이어", 22, "#A9BBCB")
    label(draw, (1310, 265), f"객체  {len(obs['objects']['items'])}개", 24)
    label(draw, (1310, 303), f"포즈  {sum(x['part'] == 'pose' for x in points)}점", 24)
    label(draw, (1310, 341), f"입  {sum(x['part'] == 'mouth' for x in points)}점", 24)
    label(draw, (1310, 379), f"손  {sum(x['part'].startswith('hand') for x in points)}점", 24)
    label(draw, (1310, 437), "도형 기반 행동 후보", 21, "#A9BBCB")
    if 0 <= t_ms <= 2704:
        phase = "E02 후보 1 활성"
        phase_color = "#FFC45B"
    elif 9280 <= t_ms <= 12096:
        phase = "E02 후보 2 활성"
        phase_color = "#FFC45B"
    else:
        phase = "활성 후보 없음"
        phase_color = "#A9BBCB"
    label(draw, (1310, 472), phase, 24, phase_color)
    label(draw, (1310, 533), "E02: 손·입 접근", 22)
    label(draw, (1310, 566), "약물 식별은 VLM 확인", 20, "#A9BBCB")
    label(draw, (1310, 625), f"최근 분석: {stale_ms / 1000:.2f}초 전", 17, "#A9BBCB")
    label(draw, (30, 739), "탐지 기록 재현  |  초록: 대상  주황: 일반 객체  빨강: 입  분홍: 손", 25)
    x0, x1, y = 70, 1190, 815
    draw.line((x0, y, x1, y), fill="#50606C", width=8)
    for start, end in ((0, 2704), (9280, 12096)):
        draw.line(
            (x0 + int(start / last_ms * (x1 - x0)), y, x0 + int(end / last_ms * (x1 - x0)), y),
            fill="#FFC45B",
            width=12,
        )
    marker = x0 + int(t_ms / last_ms * (x1 - x0))
    draw.ellipse((marker - 9, y - 9, marker + 9, y + 9), fill="#FFFFFF")
    label(draw, (68, 840), "0초", 18, "#A9BBCB")
    label(draw, (1115, 840), f"{last_ms / 1000:.1f}초", 18, "#A9BBCB")
    label(draw, (1310, 747), "E02 후보 2건", 23, "#FFC45B")
    label(draw, (1310, 785), "사후 VLM: 2건 UNCERTAIN", 19)
    label(draw, (1310, 835), "실시간 세션 판정 미반영", 17, "#FF9285")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = read_jsonl(RUN / "recording/frames.jsonl")
    times = [row["recording_ms"] for row in index]
    observations = read_jsonl(RUN / "observations.jsonl")
    obs_by_frame = {row["tracking"]["frame"]["frame_index"]: row for row in observations}
    observed = sorted(obs_by_frame)
    decoder = av.open(str(RUN / "recording/input.mp4"))
    frames = [frame.to_ndarray(format="bgr24") for frame in decoder.decode(video=0)]
    decoder.close()
    assert len(frames) == len(times) == 154
    duration = 23856
    output = av.open(str(OUT / "webcam-event-demo.mp4"), "w")
    stream = output.add_stream("libx264", rate=FPS)
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "22", "preset": "medium"}
    for n in range(round(duration / 1000 * FPS) + 1):
        t = min(n * 1000 // FPS, duration)
        i = max(0, bisect.bisect_right(times, t) - 1)
        source_index = index[i]["source"]["frame_index"]
        observed_index = observed[max(0, bisect.bisect_right(observed, source_index) - 1)]
        obs = obs_by_frame[observed_index]
        stale = max(0, t - obs["tracking"]["frame"]["session_ms"])
        annotated = render(frames[i], obs, t, duration, stale)
        video_frame = av.VideoFrame.from_ndarray(annotated, format="bgr24")
        for packet in stream.encode(video_frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    print(OUT / "webcam-event-demo.mp4")


if __name__ == "__main__":
    main()
