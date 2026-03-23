import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import subprocess
from collections import defaultdict

# ========== 配置区 ==========
VIDEO_PATH = r"C:\RailJam_clipper\videos\input.mp4"
OUTPUT_DIR = r"C:\RailJam_clipper\output"
MIN_CLIP_SECONDS = 2
MAX_GAP_SECONDS = 3
CONF_THRESHOLD = 0.5

# ROI 区域（比例，相对于画面宽高）
ROI_X1 = 0.0
ROI_X2 = 0.82  # 排除右侧观众区
ROI_Y1 = 0.15  # 排除顶部远景
ROI_Y2 = 1.0
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在加载AI模型...")
model = YOLO("yolo11n.pt")
tracker = DeepSort(
    max_age=60,
    n_init=3,
    max_cosine_distance=0.4,
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height} @ {fps:.1f}fps, 共{total_frames}帧 ({total_frames/fps:.1f}秒)")

roi_x1 = int(ROI_X1 * width)
roi_x2 = int(ROI_X2 * width)
roi_y1 = int(ROI_Y1 * height)
roi_y2 = int(ROI_Y2 * height)
print(f"ROI区域: x({roi_x1}~{roi_x2}), y({roi_y1}~{roi_y2})")

track_segments = defaultdict(list)
track_active = {}
track_start = {}

frame_idx = 0
max_gap_frames = int(MAX_GAP_SECONDS * fps)

print("正在分析视频，请稍候...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 2 == 0:
        results = model(frame, classes=[0], verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                    continue

                box_height = y2 - y1
                if box_height < height * 0.04:
                    continue

                if conf > CONF_THRESHOLD:
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = str(track.track_id)
            active_ids.add(tid)

            if tid not in track_start:
                track_start[tid] = frame_idx

            track_active[tid] = frame_idx

        for tid in list(track_start.keys()):
            if tid not in active_ids:
                last = track_active.get(tid, 0)
                if frame_idx - last > max_gap_frames:
                    start = track_start.pop(tid)
                    end = last
                    duration = (end - start) / fps
                    if duration >= MIN_CLIP_SECONDS:
                        track_segments[tid].append((start, end))

    frame_idx += 1
    if frame_idx % 300 == 0:
        pct = 100 * frame_idx // total_frames
        print(f"  进度: {frame_idx}/{total_frames} ({pct}%) | 当前追踪ID数: {len(track_start)}")

for tid, start in track_start.items():
    end = track_active.get(tid, frame_idx)
    duration = (end - start) / fps
    if duration >= MIN_CLIP_SECONDS:
        track_segments[tid].append((start, end))

cap.release()

total_clips = sum(len(v) for v in track_segments.values())
print(f"\n检测完成！共识别 {len(track_segments)} 个滑手，{total_clips} 个片段")
print("正在按滑手剪辑视频...\n")

for tid, segments in track_segments.items():
    rider_dir = os.path.join(OUTPUT_DIR, f"rider_{str(tid).zfill(3)}")
    os.makedirs(rider_dir, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        t_start = max(0, start / fps - 0.5)
        duration = (end - start) / fps + 1.0
        out_path = os.path.join(rider_dir, f"clip_{str(i+1).zfill(2)}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t_start:.2f}",
            "-i", VIDEO_PATH,
            "-t", f"{duration:.2f}",
            "-c:v", "libx264", "-c:a", "aac",
            "-loglevel", "error",
            out_path
        ]
        subprocess.run(cmd)
        print(f"  已保存: rider_{str(tid).zfill(3)}/clip_{str(i+1).zfill(2)}.mp4  ({duration:.1f}秒)")

print(f"\n✅ 全部完成！请查看输出文件夹: {OUTPUT_DIR}")
print(f"共 {len(track_segments)} 个滑手文件夹，{total_clips} 个视频片段")
