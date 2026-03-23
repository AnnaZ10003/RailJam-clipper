import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from collections import defaultdict
import random

# ========== 配置区 ==========
VIDEO_PATH = r"C:\RailJam_clipper\videos\input.mp4"
OUTPUT_PATH = r"C:\RailJam_clipper\preview.mp4"
MIN_CLIP_SECONDS = 2
MAX_GAP_SECONDS  = 3
CONF_THRESHOLD   = 0.5

ROI_X1, ROI_X2 = 0.0,  0.82
ROI_Y1, ROI_Y2 = 0.15, 1.0

# 主滑手过滤（主滑手在场时压制小目标）
DOMINANT_HEIGHT_RATIO = 0.18
SMALL_TARGET_RATIO    = 0.55

# 速度过滤
MIN_SPEED_PX       = 3.0
SPEED_CHECK_FRAMES = 20

# 幽灵框过滤
MIN_HIT_STREAK = 2

# 入场/出场区
ENTRY_X1, ENTRY_X2 = 0.00, 0.45
ENTRY_Y1, ENTRY_Y2 = 0.30, 0.65
EXIT_X1,  EXIT_X2  = 0.45, 0.92
EXIT_Y1,  EXIT_Y2  = 0.50, 0.85

# ── 轨迹形态过滤（核心新逻辑）──────────────────────
# 最小总位移：轨迹起点到终点的直线距离（占画面宽度比例）
# 主滑手会穿越画面，背景人物原地打转
MIN_DISPLACEMENT_RATIO = 0.25   # 至少穿越画面宽度的25%

# 最小直线度：总位移 / 总路程，越接近1越直
# 主滑手方向一致，背景人物随机游走
MIN_LINEARITY = 0.45

# 轨迹至少要有这么多帧才做判断（太短的不可信）
MIN_TRAJ_FRAMES = 15
# ───────────────────────────────────────────────────
# ============================

def get_color(tid):
    random.seed(hash(tid) % 10000)
    return (random.randint(60,255), random.randint(60,255), random.randint(60,255))

def point_in_zone(cx, cy, x1r, x2r, y1r, y2r, W, H):
    return (x1r*W <= cx <= x2r*W) and (y1r*H <= cy <= y2r*H)

def calc_trajectory_features(positions, W):
    """计算轨迹的总位移和直线度"""
    if len(positions) < 2:
        return 0.0, 0.0
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    # 总位移（起点到终点直线距离）
    displacement = np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2)
    # 总路程（逐帧累计）
    total_path = sum(np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
                     for i in range(1, len(xs)))
    linearity = displacement / total_path if total_path > 0 else 0.0
    displacement_ratio = displacement / W
    return displacement_ratio, linearity

def draw_timeline(frame, total_frames, current_frame, clip_segments, width, height):
    tl_x   = width - 22
    tl_top = int(height * 0.04)
    tl_bot = int(height * 0.96)
    tl_h   = tl_bot - tl_top
    cv2.rectangle(frame,(tl_x-5,tl_top),(tl_x+5,tl_bot),(30,30,30),-1)
    cv2.rectangle(frame,(tl_x-5,tl_top),(tl_x+5,tl_bot),(100,100,100),1)
    for (sf, ef, tid) in clip_segments:
        color = get_color(tid)
        ys = tl_top + int((sf/total_frames)*tl_h)
        ye = tl_top + int((ef/total_frames)*tl_h)
        cv2.rectangle(frame,(tl_x-5,ys),(tl_x+5,ye),color,-1)
        cv2.line(frame,(tl_x-16,ys),(tl_x+16,ys),color,2)
        cv2.putText(frame,f"S:{tid}",(tl_x-52,ys+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1)
        cv2.line(frame,(tl_x-16,ye),(tl_x+16,ye),color,2)
        cv2.putText(frame,f"E:{tid}",(tl_x-52,ye+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,color,1)
    cy2 = tl_top + int((current_frame/total_frames)*tl_h)
    pts = np.array([[tl_x-16,cy2-6],[tl_x-16,cy2+6],[tl_x-6,cy2]],np.int32)
    cv2.fillPoly(frame,[pts],(255,255,255))
    cv2.putText(frame,"CUTS",(tl_x-18,tl_top-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1)

print("Loading model...")
model   = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=45, n_init=3, max_cosine_distance=0.4)

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

roi_x1,roi_x2 = int(ROI_X1*W), int(ROI_X2*W)
roi_y1,roi_y2 = int(ROI_Y1*H), int(ROI_Y2*H)

fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

track_segments  = defaultdict(list)
track_active    = {}
track_start     = {}
track_positions = defaultdict(list)   # (cx, cy)
track_is_slow   = {}
track_hit_entry = set()
track_hit_exit  = set()
track_confirmed = set()   # 同时满足：穿越ENTRY+EXIT + 轨迹形态合格
track_rejected  = set()   # 明确判定为背景人物
clip_segments   = []

frame_idx      = 0
max_gap_frames = int(MAX_GAP_SECONDS * fps)

print("Analyzing video...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    draw = frame.copy()

    # ROI
    ov = draw.copy()
    cv2.rectangle(ov,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,255),-1)
    cv2.addWeighted(ov,0.04,draw,0.96,0,draw)
    cv2.rectangle(draw,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,255),2)
    cv2.putText(draw,"ROI",(roi_x1+8,roi_y1+28),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)

    # ENTRY区（红色）
    ex1,ey1,ex2,ey2 = int(ENTRY_X1*W),int(ENTRY_Y1*H),int(ENTRY_X2*W),int(ENTRY_Y2*H)
    ov2 = draw.copy()
    cv2.rectangle(ov2,(ex1,ey1),(ex2,ey2),(0,0,180),-1)
    cv2.addWeighted(ov2,0.12,draw,0.88,0,draw)
    cv2.rectangle(draw,(ex1,ey1),(ex2,ey2),(0,0,255),2)
    cv2.putText(draw,"ENTRY",(ex1+4,ey1+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,80,255),2)

    # EXIT区（绿色）
    gx1,gy1,gx2,gy2 = int(EXIT_X1*W),int(EXIT_Y1*H),int(EXIT_X2*W),int(EXIT_Y2*H)
    ov3 = draw.copy()
    cv2.rectangle(ov3,(gx1,gy1),(gx2,gy2),(0,160,0),-1)
    cv2.addWeighted(ov3,0.12,draw,0.88,0,draw)
    cv2.rectangle(draw,(gx1,gy1),(gx2,gy2),(0,255,0),2)
    cv2.putText(draw,"EXIT",(gx1+4,gy1+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,80),2)

    if frame_idx % 2 == 0:
        results = model(frame, classes=[0], verbose=False)
        valid_dets, max_box_h = [], 0

        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                bh = y2-y1
                if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2):
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(70,70,70),1)
                    continue
                if bh < H*0.04:
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(40,40,160),1)
                    continue
                if conf > CONF_THRESHOLD:
                    valid_dets.append((x1,y1,x2,y2,conf))
                    max_box_h = max(max_box_h, bh)

        dominant = max_box_h >= H * DOMINANT_HEIGHT_RATIO
        detections = []
        for (x1,y1,x2,y2,conf) in valid_dets:
            bh = y2-y1
            if dominant and bh < max_box_h * SMALL_TARGET_RATIO:
                cv2.rectangle(draw,(x1,y1),(x2,y2),(180,0,180),1)
                cv2.putText(draw,"BG",(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,0,180),1)
                continue
            detections.append(([x1,y1,x2-x1,y2-y1],conf,"person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            if hasattr(track,'hit_streak') and track.hit_streak < MIN_HIT_STREAK:
                continue

            tid  = str(track.track_id)
            ltrb = track.to_ltrb()
            tx1,ty1,tx2,ty2 = map(int,ltrb)
            cx,cy = (tx1+tx2)//2,(ty1+ty2)//2
            bw,bh = tx2-tx1,ty2-ty1

            if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
            if tx1<0 or ty1<0 or tx2>W or ty2>H: continue
            if bw<=0 or bh<=0 or bw>W*0.5 or bh>H*0.8: continue

            active_ids.add(tid)
            track_positions[tid].append((cx, cy))
            if len(track_positions[tid]) > 120:
                track_positions[tid].pop(0)

            # 速度
            pts = track_positions[tid]
            is_slow = False
            if len(pts) >= SPEED_CHECK_FRAMES:
                rec   = pts[-SPEED_CHECK_FRAMES:]
                dists = [np.sqrt((rec[i][0]-rec[i-1][0])**2+(rec[i][1]-rec[i-1][1])**2)
                         for i in range(1,len(rec))]
                is_slow = np.mean(dists) < MIN_SPEED_PX
            track_is_slow[tid] = is_slow

            # 穿越区检测（不再要求入场时框的大小）
            if point_in_zone(cx,cy,ENTRY_X1,ENTRY_X2,ENTRY_Y1,ENTRY_Y2,W,H):
                track_hit_entry.add(tid)
            if point_in_zone(cx,cy,EXIT_X1,EXIT_X2,EXIT_Y1,EXIT_Y2,W,H):
                track_hit_exit.add(tid)

            # 轨迹形态判断（满足穿越条件后再检验轨迹）
            if (tid in track_hit_entry and tid in track_hit_exit
                    and tid not in track_confirmed and tid not in track_rejected):
                if len(pts) >= MIN_TRAJ_FRAMES:
                    disp_ratio, linearity = calc_trajectory_features(pts, W)
                    if disp_ratio >= MIN_DISPLACEMENT_RATIO and linearity >= MIN_LINEARITY:
                        track_confirmed.add(tid)
                    else:
                        track_rejected.add(tid)

            if tid not in track_start:
                track_start[tid] = frame_idx
            track_active[tid] = frame_idx

            color = get_color(tid)

            if is_slow or tid in track_rejected:
                # 橙色虚线：慢速或轨迹不合格
                reason = "SLOW" if is_slow else "TRAJ"
                for s in range(0,max(bw,bh),12):
                    if tx1+s<tx2:
                        cv2.line(draw,(tx1+s,ty1),(min(tx1+s+6,tx2),ty1),(0,130,255),1)
                        cv2.line(draw,(tx1+s,ty2),(min(tx1+s+6,tx2),ty2),(0,130,255),1)
                    if ty1+s<ty2:
                        cv2.line(draw,(tx1,ty1+s),(tx1,min(ty1+s+6,ty2)),(0,130,255),1)
                        cv2.line(draw,(tx2,ty1+s),(tx2,min(ty1+s+6,ty2)),(0,130,255),1)
                cv2.putText(draw,f"{reason}:{tid}",(tx1,ty1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,130,255),1)
            elif tid not in track_confirmed:
                # 白色细框：已穿越入场或出场，但轨迹还不够长做判断
                status = ""
                if tid in track_hit_entry: status += "E"
                if tid in track_hit_exit:  status += "X"
                cv2.rectangle(draw,(tx1,ty1),(tx2,ty2),(200,200,200),1)
                cv2.putText(draw,f"?{status}:{tid}",(tx1+2,ty1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            else:
                # 彩色粗框：确认滑手
                cv2.rectangle(draw,(tx1,ty1),(tx2,ty2),color,2)
                label = f"ID:{tid}"
                lw = len(label)*13
                cv2.rectangle(draw,(tx1,ty1-26),(tx1+lw,ty1),color,-1)
                cv2.putText(draw,label,(tx1+2,ty1-7),
                            cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
                # 轨迹
                vpts = pts[-60:]
                for j in range(1,len(vpts)):
                    a = j/len(vpts)
                    c = tuple(int(x*a) for x in color)
                    cv2.line(draw,vpts[j-1],vpts[j],c,2)

        # 检查消失的track
        for tid in list(track_start.keys()):
            if tid not in active_ids:
                last = track_active.get(tid,0)
                if frame_idx - last > max_gap_frames:
                    start = track_start.pop(tid)
                    end   = last
                    dur   = (end-start)/fps

                    # 消失时再做一次最终轨迹判断（给轨迹较短的track最后一次机会）
                    if (tid not in track_confirmed and tid not in track_rejected
                            and tid in track_hit_entry and tid in track_hit_exit):
                        pts = track_positions[tid]
                        if len(pts) >= MIN_TRAJ_FRAMES:
                            dr, lin = calc_trajectory_features(pts, W)
                            if dr >= MIN_DISPLACEMENT_RATIO and lin >= MIN_LINEARITY:
                                track_confirmed.add(tid)

                    if (dur >= MIN_CLIP_SECONDS
                            and tid in track_confirmed
                            and not track_is_slow.get(tid,False)):
                        track_segments[tid].append((start,end))
                        clip_segments.append((start,end,tid))

    draw_timeline(draw, total_frames, frame_idx, clip_segments, W, H)
    t_sec = frame_idx/fps
    cv2.putText(draw,f"{int(t_sec//60):02d}:{t_sec%60:05.2f}",
                (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    confirmed_active = sum(1 for t in track_start
                           if t in track_confirmed and not track_is_slow.get(t,False))
    cv2.putText(draw,f"Confirmed: {confirmed_active} | Rejected: {len(track_rejected)}",
                (10,62),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,220,255),2)

    out_writer.write(draw)
    frame_idx += 1
    if frame_idx % 300 == 0:
        print(f"  Progress: {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)"
              f" | confirmed={len(track_confirmed)} rejected={len(track_rejected)}")

# 收尾
for tid, start in track_start.items():
    end = track_active.get(tid, frame_idx)
    dur = (end-start)/fps
    if (tid not in track_confirmed and tid not in track_rejected
            and tid in track_hit_entry and tid in track_hit_exit):
        pts = track_positions[tid]
        if len(pts) >= MIN_TRAJ_FRAMES:
            dr, lin = calc_trajectory_features(pts, W)
            if dr >= MIN_DISPLACEMENT_RATIO and lin >= MIN_LINEARITY:
                track_confirmed.add(tid)
    if (dur >= MIN_CLIP_SECONDS
            and tid in track_confirmed
            and not track_is_slow.get(tid,False)):
        track_segments[tid].append((start,end))

cap.release()
out_writer.release()

total_clips = sum(len(v) for v in track_segments.values())
print(f"\nDone! Output: {OUTPUT_PATH}")
print(f"Confirmed riders: {len(track_segments)} | Clips: {total_clips}")
print("\nLegend:")
print("  Yellow box      = ROI")
print("  Red box         = ENTRY zone")
print("  Green box       = EXIT zone")
print("  Colored box+ID  = Confirmed rider (crossed ENTRY+EXIT, good trajectory)")
print("  White box ?EX   = Pending (E=hit entry, X=hit exit, waiting for trajectory)")
print("  Orange dashed   = Rejected (SLOW=too slow, TRAJ=bad trajectory)")
print("  Purple BG       = Suppressed by dominant rider logic")
