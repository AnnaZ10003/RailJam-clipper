import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import torch
import torchreid
import os
import argparse
from collections import defaultdict
import random

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=r"C:\RailJam_clipper\videos\input.mp4")
parser.add_argument("--out-dir", default=r"C:\RailJam_clipper\output")
args = parser.parse_args()

VIDEO_PATH = args.video
_stem      = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
CLIPS_DIR  = args.out_dir
PREVIEW_PATH = os.path.join(CLIPS_DIR, f"{_stem}_preview.mp4")
SHEET_PATH   = os.path.join(CLIPS_DIR, f"{_stem}_sheet.jpg")
os.makedirs(CLIPS_DIR, exist_ok=True)

# ========== 配置区 ==========

ROI_X1, ROI_X2 = 0.0,  0.82
ROI_Y1, ROI_Y2 = 0.15, 1.0

# 主滑手从左侧边缘入画，偶尔从底部入画
# ENTRY区：左侧较宽区域（覆盖入画后前几帧）
ENTRY_X1, ENTRY_X2 = 0.00, 0.30   # 左侧30%
ENTRY_Y1, ENTRY_Y2 = 0.15, 1.00   # 全高度
# 底部入场区（兜底）
ENTRY2_X1, ENTRY2_X2 = 0.00, 0.70
ENTRY2_Y1, ENTRY2_Y2 = 0.80, 1.00
# EXIT区：道具核心区域（画面中部）
EXIT_X1,  EXIT_X2  = 0.25, 0.85
EXIT_Y1,  EXIT_Y2  = 0.25, 0.80

MIN_CLIP_SECONDS   = 0.8
MAX_GAP_SECONDS    = 3
CONF_THRESHOLD     = 0.5
DOMINANT_HEIGHT_RATIO = 0.18
SMALL_TARGET_RATIO    = 0.55
MIN_SPEED_PX          = 3.0
SPEED_CHECK_FRAMES    = 20
MIN_HIT_STREAK        = 2

# Re-ID 参数
REID_FRAMES_PER_TRACK = 5    # 每个track保存多少帧
DBSCAN_EPS            = 0.5  # 聚类半径（越小越严格）
DBSCAN_MIN_SAMPLES    = 2    # 最小聚类成员数
PRE_ROLL_SECONDS      = 3.0  # 剪辑开始前的余量（从入画前回溯）
POST_ROLL_SECONDS     = 2.0  # 剪辑结束后的余量（主滑手离画后继续录制）

# 截图表格参数
THUMB_W, THUMB_H = 120, 160  # 每张截图的尺寸
# ============================

def get_color(cluster_id):
    cluster_id = int(cluster_id)
    if cluster_id == -1:
        return (100, 100, 100)  # 离群点：灰色
    random.seed(cluster_id * 137)
    return (random.randint(60,255), random.randint(60,255), random.randint(60,255))

def point_in_zone(cx, cy, x1r, x2r, y1r, y2r, W, H):
    return (x1r*W <= cx <= x2r*W) and (y1r*H <= cy <= y2r*H)

def draw_timeline(frame, total_frames, current_frame, clip_segments, W, H, cluster_map):
    tl_x   = W - 22
    tl_top = int(H * 0.04)
    tl_bot = int(H * 0.96)
    tl_h   = tl_bot - tl_top
    cv2.rectangle(frame,(tl_x-5,tl_top),(tl_x+5,tl_bot),(30,30,30),-1)
    cv2.rectangle(frame,(tl_x-5,tl_top),(tl_x+5,tl_bot),(100,100,100),1)
    for (sf, ef, tid) in clip_segments:
        cid   = cluster_map.get(tid, -1)
        color = get_color(cid)
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

# ─────────────────────────────────────────────
# 第一遍：追踪，收集track数据
# ─────────────────────────────────────────────
print("="*50)
print("PASS 1: Tracking...")
print("="*50)
model   = YOLO("yolo11n.pt")
tracker = DeepSort(max_age=45, n_init=3, max_cosine_distance=0.4)

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H} @ {fps:.1f}fps  {total_frames}frames  {total_frames/fps:.1f}s")

# ── 预扫描：自动检测 ROI（基于实际人体出现位置）──
print("Pre-scanning to auto-detect ROI...")
cap_pre = cv2.VideoCapture(VIDEO_PATH)
scan_cx, scan_cy = [], []
scan_idx = 0
while True:
    ret, frm = cap_pre.read()
    if not ret:
        break
    if scan_idx % 15 == 0:
        res = model(frm, classes=[0], verbose=False)
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                bh = y2 - y1
                if conf > CONF_THRESHOLD and bh > H * 0.06:
                    scan_cx.append((x1 + x2) // 2)
                    scan_cy.append((y1 + y2) // 2)
    scan_idx += 1
cap_pre.release()

if len(scan_cx) >= 3:
    cx_arr = np.array(scan_cx)
    cy_arr = np.array(scan_cy)
    pad_x = int(W * 0.07)
    pad_y = int(H * 0.04)
    roi_x1 = max(0,   int(np.percentile(cx_arr,  3)) - pad_x)
    roi_x2 = min(W-1, int(np.percentile(cx_arr, 97)) + pad_x)
    roi_y1 = max(0,   int(np.percentile(cy_arr,  3)) - pad_y)
    roi_y2 = min(H-1, int(np.percentile(cy_arr, 97)) + pad_y)
    print(f"Auto-ROI: x={roi_x1/W:.2f}~{roi_x2/W:.2f}  y={roi_y1/H:.2f}~{roi_y2/H:.2f}  ({len(scan_cx)} detections)")
else:
    roi_x1, roi_x2 = int(ROI_X1*W), int(ROI_X2*W)
    roi_y1, roi_y2 = int(ROI_Y1*H), int(ROI_Y2*H)
    print("Not enough detections for auto-ROI, using default.")

max_gap_frames = int(MAX_GAP_SECONDS * fps)

# track数据结构
track_start     = {}
track_start_archive = {}   # 永久保存每个tid的首次出现帧（不会被pop）
track_active    = {}
track_positions = defaultdict(list)
track_is_slow   = {}
track_hit_entry     = set()
track_hit_exit      = set()
track_last_exit_frame = {}   # {tid: last frame seen inside EXIT zone}
track_segments  = defaultdict(list)   # {tid: [(start,end),...]}
track_crops     = defaultdict(list)   # {tid: [crop_img,...]} 代表帧
clip_segments   = []                  # (start,end,tid) 用于时间轴

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 2 == 0:
        results = model(frame, classes=[0], verbose=False)
        valid_dets, max_box_h = [], 0

        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                bh = y2-y1
                if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
                if bh < H*0.04: continue
                if conf > CONF_THRESHOLD:
                    valid_dets.append((x1,y1,x2,y2,conf))
                    max_box_h = max(max_box_h, bh)

        dominant = max_box_h >= H * DOMINANT_HEIGHT_RATIO
        detections = []
        for (x1,y1,x2,y2,conf) in valid_dets:
            if dominant and (y2-y1) < max_box_h * SMALL_TARGET_RATIO: continue
            detections.append(([x1,y1,x2-x1,y2-y1],conf,"person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        active_ids = set()
        for track in tracks:
            if not track.is_confirmed(): continue

            tid  = str(track.track_id)
            ltrb = track.to_ltrb()
            tx1,ty1,tx2,ty2 = map(int,ltrb)
            cx,cy = (tx1+tx2)//2,(ty1+ty2)//2
            bw,bh = tx2-tx1,ty2-ty1

            # 已建立的 track：bbox 还与画幅有任何重叠就持续延伸结束时间
            # 完全离开画幅（bbox 不再与画幅相交）才停止
            # （hit_streak 在 coasting 时会归零，不能用于已确认 track 的出画判断）
            if tid in track_start and tx2 > 0 and tx1 < W and ty2 > 0 and ty1 < H:
                track_active[tid] = frame_idx

            # 以下是正常 ROI 内处理（检测匹配、速度、裁剪等）
            if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
            if hasattr(track,'hit_streak') and track.hit_streak < MIN_HIT_STREAK: continue
            if tx1<0 or ty1<0 or tx2>W or ty2>H: continue
            if bw<=0 or bh<=0 or bw>W*0.5 or bh>H*0.8: continue

            active_ids.add(tid)
            track_positions[tid].append((cx,cy))

            # 速度
            pts = track_positions[tid]
            is_slow = False
            if len(pts) >= SPEED_CHECK_FRAMES:
                rec   = pts[-SPEED_CHECK_FRAMES:]
                dists = [np.sqrt((rec[i][0]-rec[i-1][0])**2+(rec[i][1]-rec[i-1][1])**2)
                         for i in range(1,len(rec))]
                is_slow = np.mean(dists) < MIN_SPEED_PX
            track_is_slow[tid] = is_slow

            # 穿越区
            if (point_in_zone(cx,cy,ENTRY_X1,ENTRY_X2,ENTRY_Y1,ENTRY_Y2,W,H) or
                    point_in_zone(cx,cy,ENTRY2_X1,ENTRY2_X2,ENTRY2_Y1,ENTRY2_Y2,W,H)):
                track_hit_entry.add(tid)
            if point_in_zone(cx,cy,EXIT_X1,EXIT_X2,EXIT_Y1,EXIT_Y2,W,H):
                track_hit_exit.add(tid)
                track_last_exit_frame[tid] = frame_idx

            # 保存代表帧裁剪图（均匀采样）
            if len(track_crops[tid]) < REID_FRAMES_PER_TRACK:
                pad = 10
                cx1 = max(0, tx1-pad); cy1 = max(0, ty1-pad)
                cx2 = min(W, tx2+pad); cy2 = min(H, ty2+pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    track_crops[tid].append(crop.copy())
            elif frame_idx % 30 == 0:
                # 定期替换，保持均匀分布
                idx_replace = frame_idx % REID_FRAMES_PER_TRACK
                pad = 10
                cx1 = max(0, tx1-pad); cy1 = max(0, ty1-pad)
                cx2 = min(W, tx2+pad); cy2 = min(H, ty2+pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    track_crops[tid][idx_replace] = crop.copy()

            if tid not in track_start:
                track_start[tid] = frame_idx
                track_start_archive[tid] = frame_idx
            track_active[tid] = frame_idx

        # 消失的track → 记录片段（暂不过滤entry/exit，交给Re-ID决定）
        for tid in list(track_start.keys()):
            if tid not in active_ids:
                last = track_active.get(tid,0)
                if frame_idx - last > max_gap_frames:
                    start = track_start.pop(tid)
                    end   = last
                    dur   = (end-start)/fps
                    if (dur >= MIN_CLIP_SECONDS
                            and not track_is_slow.get(tid,False)):
                        track_segments[tid].append((start,end))
                        clip_segments.append((start,end,tid))

    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)"
              f" | tracks={len(track_start)} clips={len(clip_segments)}")

# 收尾
for tid, start in track_start.items():
    end = track_active.get(tid, frame_idx)
    dur = (end-start)/fps
    if (dur >= MIN_CLIP_SECONDS
            and not track_is_slow.get(tid,False)):
        track_segments[tid].append((start,end))
        clip_segments.append((start,end,tid))

# also check archive for any tids missed by track_start
for tid, start in track_start_archive.items():
    if tid in track_segments: continue
    end = track_active.get(tid, frame_idx)
    dur = (end-start)/fps
    if (dur >= MIN_CLIP_SECONDS
            and not track_is_slow.get(tid,False)):
        track_segments[tid].append((start,end))
        clip_segments.append((start,end,tid))

cap.release()
print(f"Pass 1 done. Candidate tracks: {len(track_segments)}  Clips: {len(clip_segments)}")

# ── 调试：打印每个track的区域通过情况 ──
print("\n[DEBUG] All tracks summary:")
print(f"{'ID':>6} {'dur':>6} {'entry':>6} {'exit':>6} {'slow':>6} {'start_sec':>10} {'clips':>6}")
all_tids = set(list(track_segments.keys()) + list(track_hit_entry) + list(track_hit_exit))
for tid in sorted(all_tids, key=lambda x: track_start_archive.get(x,0)):
    start_f = track_start_archive.get(tid, 0)
    end_f   = track_active.get(tid, 0)
    dur     = (end_f - start_f) / fps
    has_entry = tid in track_hit_entry
    has_exit  = tid in track_hit_exit
    is_slow   = track_is_slow.get(tid, False)
    n_clips   = len(track_segments.get(tid, []))
    selected  = "✓ SELECTED" if n_clips > 0 else ("✗ no_entry" if not has_entry else ("✗ no_exit" if not has_exit else ("✗ slow" if is_slow else "✗ too_short")))
    print(f"  {tid:>4}  {dur:>5.1f}s  {str(has_entry):>6}  {str(has_exit):>6}  {str(is_slow):>6}  {start_f/fps:>8.1f}s  {selected}")

# ─────────────────────────────────────────────
# Re-ID：提取特征 + 聚类
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Re-ID: Extracting features...")
print("="*50)

# 加载轻量Re-ID模型
reid_model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1000,
    pretrained=True
)
reid_model.eval()
if torch.cuda.is_available():
    reid_model = reid_model.cuda()
    print("Re-ID running on GPU")
else:
    print("Re-ID running on CPU")

def extract_feature(crop_imgs):
    """从一组裁剪图提取平均特征向量"""
    feats = []
    for crop in crop_imgs:
        if crop is None or crop.size == 0:
            continue
        img = cv2.resize(crop, (128, 256))
        img = img[:,:,::-1].astype(np.float32) / 255.0
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        tensor = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        with torch.no_grad():
            feat = reid_model(tensor).cpu().numpy().flatten()
        feats.append(feat)
    if not feats:
        return None
    return np.mean(feats, axis=0)

# 只对有片段的track提取特征
valid_tids = list(track_segments.keys())
features   = []
valid_tids_with_feat = []

for tid in valid_tids:
    crops = track_crops.get(tid, [])
    if not crops:
        continue
    feat = extract_feature(crops)
    if feat is not None:
        features.append(feat)
        valid_tids_with_feat.append(tid)

print(f"Extracted features for {len(features)} tracks")

# DBSCAN 聚类
cluster_map = {}   # {tid: cluster_id}
main_cluster = -999
labels = []
if len(features) >= 2:
    feat_matrix = normalize(np.array(features))
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
    labels = db.fit_predict(feat_matrix)
    for tid, label in zip(valid_tids_with_feat, labels):
        cluster_map[tid] = int(label)

    unique_labels = set(int(l) for l in labels)
    print(f"Clusters found: {unique_labels}")

    # ── 最终判断：完全抛弃DBSCAN标签，对每个track单独用avg_h判断 ──
    from collections import Counter, defaultdict as _dd
    cnt = Counter(int(l) for l in labels if l >= 0)
    # 原理：主滑手框大（离镜头近），背景人物框小（离镜头远）
    # 先过滤幽灵框，再用K-means自动找大/小框的分界线

    # Step1：计算每个track的avg_h和dark_ratio
    def calc_dark_ratio_local(crops):
        ratios = []
        for crop in crops:
            if crop is None or crop.size == 0: continue
            resized = cv2.resize(crop, (64, 128))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            ratios.append(np.sum(gray < 120) / gray.size)
        return np.mean(ratios) if ratios else 0.0

    all_tids_for_reid = list(set(list(valid_tids_with_feat) +
                                 [t for t in track_crops if track_crops[t]]))
    per_track_h = {}
    per_track_dark = {}
    for tid in all_tids_for_reid:
        crops = track_crops.get(tid, [])
        valid_crops = [c for c in crops if c is not None and c.size > 0]
        if valid_crops:
            per_track_h[tid]    = np.mean([c.shape[0] for c in valid_crops])
            per_track_dark[tid] = calc_dark_ratio_local(valid_crops)

    # Step2：过滤幽灵框（dark_ratio极低 = 裁剪图里没有真实人物）
    GHOST_THRESHOLD = 0.06
    ghost_tids  = {t for t,r in per_track_dark.items() if r < GHOST_THRESHOLD}
    valid_h_tids = [t for t in per_track_h if t not in ghost_tids]
    print(f"  Ghost tracks removed ({len(ghost_tids)}): {sorted(ghost_tids, key=lambda x:int(x))}")

    # Step3：K-means对avg_h自动二分
    from sklearn.cluster import KMeans
    if len(valid_h_tids) >= 2:
        h_arr = np.array([per_track_h[t] for t in valid_h_tids]).reshape(-1,1)
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km_labels = km.fit_predict(h_arr)
        c0, c1 = float(km.cluster_centers_[0]), float(km.cluster_centers_[1])
        main_km_lbl = 0 if c0 > c1 else 1   # 高度大的那组是主滑手
        threshold = (c0 + c1) / 2
        h_ratio = min(c0, c1) / max(c0, c1) if max(c0, c1) > 0 else 1.0
        print(f"  KMeans avg_h: {c0:.0f}px vs {c1:.0f}px → threshold={threshold:.0f}px  ratio={h_ratio:.2f}")
        if h_ratio > 0.72:
            # 两组高度太接近 → 无背景人物，全部视为主滑手
            print(f"  Height ratio {h_ratio:.2f} > 0.72 → no background detected, all treated as MAIN RIDERS")
            main_tids_set = set(valid_h_tids)
            bg_tids_set   = set()
        else:
            main_tids_set = {t for t,lbl in zip(valid_h_tids, km_labels) if lbl==main_km_lbl}
            bg_tids_set   = {t for t,lbl in zip(valid_h_tids, km_labels) if lbl!=main_km_lbl}
    else:
        main_tids_set = set(valid_h_tids)
        bg_tids_set   = set()
        threshold = 0

    bg_tids_set.update(ghost_tids)
    print(f"  MAIN RIDERS ({len(main_tids_set)}): {sorted(main_tids_set, key=lambda x:int(x))}")
    print(f"  BACKGROUND  ({len(bg_tids_set)}):  {sorted(bg_tids_set,  key=lambda x:int(x))}")

    # Step4：用结果覆盖cluster_map（main_cluster=2保持不变作为标记）
    main_cluster = 2
    for tid in all_tids_for_reid:
        cluster_map[tid] = main_cluster if tid in main_tids_set else 0

    if not cnt:
        print("  DBSCAN found no clusters (all noise) — K-means result is used instead.")
else:
    print("Not enough tracks for clustering, skipping.")

# ─────────────────────────────────────────────
# 第二遍：渲染预览视频
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("PASS 2: Rendering preview video...")
print("="*50)

cap2       = cv2.VideoCapture(VIDEO_PATH)
fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(PREVIEW_PATH, fourcc, fps, (W, H))

# 重建tracker（第二遍用于可视化）
tracker2 = DeepSort(max_age=45, n_init=3, max_cosine_distance=0.4)
model2   = YOLO("yolo11n.pt")

track_pos2  = defaultdict(list)
track_start2 = {}
track_active2 = {}
track_hit_entry2 = set()
track_hit_exit2  = set()
track_is_slow2   = {}

frame_idx = 0
while True:
    ret, frame = cap2.read()
    if not ret:
        break

    draw = frame.copy()

    # ROI
    ov = draw.copy()
    cv2.rectangle(ov,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,255),-1)
    cv2.addWeighted(ov,0.04,draw,0.96,0,draw)
    cv2.rectangle(draw,(roi_x1,roi_y1),(roi_x2,roi_y2),(0,255,255),2)
    cv2.putText(draw,"ROI",(roi_x1+8,roi_y1+28),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)

    # ENTRY/EXIT区仅用于Pass1调试统计，不在预览视频中标注（区域坐标依赖视频方向，不通用）

    if frame_idx % 2 == 0:
        results = model2(frame, classes=[0], verbose=False)
        valid_dets2, max_box_h2 = [], 0
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
                    valid_dets2.append((x1,y1,x2,y2,conf))
                    max_box_h2 = max(max_box_h2, bh)

        dominant2 = max_box_h2 >= H * DOMINANT_HEIGHT_RATIO
        detections2 = []
        for (x1,y1,x2,y2,conf) in valid_dets2:
            if dominant2 and (y2-y1) < max_box_h2 * SMALL_TARGET_RATIO:
                cv2.rectangle(draw,(x1,y1),(x2,y2),(180,0,180),1)
                continue
            detections2.append(([x1,y1,x2-x1,y2-y1],conf,"person"))

        tracks2 = tracker2.update_tracks(detections2, frame=frame)

        for track in tracks2:
            if not track.is_confirmed(): continue
            if hasattr(track,'hit_streak') and track.hit_streak < MIN_HIT_STREAK: continue

            tid  = str(track.track_id)
            ltrb = track.to_ltrb()
            tx1,ty1,tx2,ty2 = map(int,ltrb)
            cx,cy = (tx1+tx2)//2,(ty1+ty2)//2
            bw,bh = tx2-tx1,ty2-ty1

            if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
            if tx1<0 or ty1<0 or tx2>W or ty2>H: continue
            if bw<=0 or bh<=0 or bw>W*0.5 or bh>H*0.8: continue

            track_pos2[tid].append((cx,cy))
            if len(track_pos2[tid])>90: track_pos2[tid].pop(0)

            pts2 = track_pos2[tid]
            is_slow2 = False
            if len(pts2) >= SPEED_CHECK_FRAMES:
                rec   = pts2[-SPEED_CHECK_FRAMES:]
                dists = [np.sqrt((rec[i][0]-rec[i-1][0])**2+(rec[i][1]-rec[i-1][1])**2)
                         for i in range(1,len(rec))]
                is_slow2 = np.mean(dists) < MIN_SPEED_PX
            track_is_slow2[tid] = is_slow2

            if point_in_zone(cx,cy,ENTRY_X1,ENTRY_X2,ENTRY_Y1,ENTRY_Y2,W,H):
                track_hit_entry2.add(tid)
            if point_in_zone(cx,cy,EXIT_X1,EXIT_X2,EXIT_Y1,EXIT_Y2,W,H):
                track_hit_exit2.add(tid)

            # 决定颜色和标签
            cid   = cluster_map.get(tid, -1)
            is_main = (cid == main_cluster)
            has_segment = tid in track_segments

            if is_slow2:
                color = (0,130,255)
                label = f"SLOW:{tid}"
                thickness = 1
            elif has_segment and is_main:
                color = get_color(cid)
                label = f"RIDER:{tid}"
                thickness = 3
            elif has_segment and not is_main:
                color = (80,80,80)
                label = f"BG:{tid}"
                thickness = 1
            else:
                color = (200,200,200)
                label = f"?:{tid}"
                thickness = 1

            cv2.rectangle(draw,(tx1,ty1),(tx2,ty2),color,thickness)
            lw = len(label)*12
            cv2.rectangle(draw,(tx1,ty1-24),(tx1+lw,ty1),color,-1)
            cv2.putText(draw,label,(tx1+2,ty1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

            if has_segment and is_main:
                vpts = track_pos2[tid][-60:]
                for j in range(1,len(vpts)):
                    a = j/len(vpts)
                    c = tuple(int(x*a) for x in color)
                    cv2.line(draw,vpts[j-1],vpts[j],c,2)

    draw_timeline(draw, total_frames, frame_idx, clip_segments, W, H, cluster_map)
    t_sec = frame_idx/fps
    cv2.putText(draw,f"{int(t_sec//60):02d}:{t_sec%60:05.2f}",
                (10,32),cv2.FONT_HERSHEY_SIMPLEX,0.85,(255,255,255),2)
    main_count = sum(1 for t in track_segments if cluster_map.get(t,-1)==main_cluster)
    cv2.putText(draw,f"Main riders: {main_count}",
                (10,62),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,220,255),2)

    out_writer.write(draw)
    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"  Rendering: {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)")

cap2.release()
out_writer.release()
print(f"Preview video saved: {PREVIEW_PATH}")

# ─────────────────────────────────────────────
# 输出代表性截图表格
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Generating thumbnail sheet...")
print("="*50)

# 按聚类分组，主滑手群在前
from collections import defaultdict as dd
groups = dd(list)
for tid in track_segments:
    cid = cluster_map.get(tid, -1)
    groups[cid].append(tid)

ordered_groups = []
if main_cluster in groups:
    ordered_groups.append((main_cluster, groups[main_cluster]))
for cid, tids in sorted(groups.items()):
    if cid != main_cluster:
        ordered_groups.append((cid, tids))

# 计算画布大小
COLS       = REID_FRAMES_PER_TRACK
INFO_W     = 180
ROW_H      = THUMB_H + 10
HEADER_H   = 36
GROUP_GAP  = 20
total_rows = sum(len(tids) for _,tids in ordered_groups)
total_h    = HEADER_H * len(ordered_groups) + total_rows * ROW_H + GROUP_GAP * len(ordered_groups) + 60
total_w    = INFO_W + COLS * (THUMB_W + 4) + 20

sheet = np.ones((total_h, total_w, 3), dtype=np.uint8) * 30

y = 20
for cid, tids in ordered_groups:
    color = get_color(cid)
    tag   = "MAIN RIDERS" if cid == main_cluster else (f"GROUP {cid}" if cid >= 0 else "NOISE/BG")

    # 组标题
    cv2.rectangle(sheet,(0,y),(total_w,y+HEADER_H),
                  tuple(int(c*0.4) for c in color),-1)
    cv2.putText(sheet,f"Cluster {cid}: {tag}  ({len(tids)} tracks)",
                (10,y+24),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)
    y += HEADER_H + 4

    for tid in sorted(tids, key=lambda t: track_segments[t][0][0]):
        segs = track_segments[tid]
        start_sec = segs[0][0]/fps
        end_sec   = segs[-1][1]/fps
        total_dur = sum((e-s)/fps for s,e in segs)

        # 左侧信息栏
        cv2.putText(sheet,f"ID: {tid}",(6,y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        cv2.putText(sheet,f"{int(start_sec//60):02d}:{start_sec%60:04.1f}",(6,y+38),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
        cv2.putText(sheet,f"~{int(end_sec//60):02d}:{end_sec%60:04.1f}",(6,y+54),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,180,180),1)
        cv2.putText(sheet,f"{total_dur:.1f}s",(6,y+70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,220,100),1)

        # 代表帧截图
        crops = track_crops.get(tid, [])
        for i, crop in enumerate(crops[:COLS]):
            if crop is None or crop.size == 0:
                continue
            thumb = cv2.resize(crop, (THUMB_W, THUMB_H))
            tx = INFO_W + i*(THUMB_W+4)
            sheet[y:y+THUMB_H, tx:tx+THUMB_W] = thumb
            cv2.rectangle(sheet,(tx,y),(tx+THUMB_W,y+THUMB_H),color,2)

        # 行分隔线
        cv2.line(sheet,(0,y+ROW_H-2),(total_w,y+ROW_H-2),(60,60,60),1)
        y += ROW_H

    y += GROUP_GAP

cv2.imwrite(SHEET_PATH, sheet)
print(f"Thumbnail sheet saved: {SHEET_PATH}")

# ─────────────────────────────────────────────
# 第三遍：输出剪辑视频
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("PASS 3: Exporting clip videos...")
print("="*50)

main_segs = []
for tid in track_segments:
    if cluster_map.get(tid, -1) == main_cluster:
        for (s, e) in track_segments[tid]:
            main_segs.append((s, e, tid))
main_segs.sort(key=lambda x: x[0])

if not main_segs:
    print("No main rider segments found, skipping clip export.")
else:
    print(f"Exporting {len(main_segs)} clips for "
          f"{len(set(t for _,_,t in main_segs))} main rider track(s)")
    cap3 = cv2.VideoCapture(VIDEO_PATH)
    pre_roll_frames  = int(PRE_ROLL_SECONDS * fps)
    ux_tail_frames   = int(0.15 * fps)   # 体验感用途的硬性延长
    for clip_n, (start_f, end_f, tid) in enumerate(main_segs, 1):
        clip_start = max(0, start_f - pre_roll_frames)
        # 结束：bbox 完全出画 + 0.15s 体验延长，下一位进入 ROI 时强制截止
        if clip_n < len(main_segs):
            next_start_f = main_segs[clip_n][0]
            clip_end = min(end_f + ux_tail_frames, next_start_f - 1, total_frames - 1)
        else:
            clip_end = min(end_f + ux_tail_frames, total_frames - 1)
        clip_path  = os.path.join(CLIPS_DIR, f"{_stem}_clip{clip_n:03d}_t{tid}.mp4")
        out_clip   = cv2.VideoWriter(clip_path, fourcc, fps, (W, H))
        cap3.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for _ in range(clip_end - clip_start + 1):
            ret, frm = cap3.read()
            if not ret:
                break
            out_clip.write(frm)
        out_clip.release()
        dur = (clip_end - clip_start) / fps
        print(f"  [{clip_n:03d}] Rider {tid}: {clip_start/fps:.1f}s ~ {clip_end/fps:.1f}s  ({dur:.1f}s)  → {clip_path}")
    cap3.release()

print(f"\nDone!")
print(f"  Preview : {PREVIEW_PATH}")
print(f"  Sheet   : {SHEET_PATH}")
print(f"  Clips   : {CLIPS_DIR}/")
main_count = sum(1 for t in track_segments if cluster_map.get(t,-1)==main_cluster)
print(f"  Main riders confirmed: {main_count}")
print(f"  Total tracks: {len(track_segments)}")
