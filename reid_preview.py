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

MIN_CLIP_SECONDS   = 0.5    # 特征提取/救援用；短于此的接近段也能参与 Re-ID
MIN_OUTPUT_SECONDS = 0.8    # 最终剪辑输出的最短时长（独立片段；归并组不受此限）
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
track_boxes     = defaultdict(dict)   # {tid: {frame_idx: (tx1,ty1,tx2,ty2)}}，用于空间IoU计算
track_is_slow   = {}
track_hit_entry     = set()
track_hit_exit      = set()
track_last_exit_frame = {}   # {tid: last frame seen inside EXIT zone}
track_frame_exit = {}        # {tid: last frame where bbox still partially in full frame (ROI外延伸)}
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

            # 已建立的 track：bbox 还与画幅有任何重叠就记录出画帧（仅供剪辑结束点用）
            # 注意：不更新 track_active，保持其只反映 ROI 内的最后出现时间（供时序间隔计算）
            if tid in track_start and tx2 > 0 and tx1 < W and ty2 > 0 and ty1 < H:
                track_frame_exit[tid] = frame_idx

            # 以下是正常 ROI 内处理（检测匹配、速度、裁剪等）
            if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
            if hasattr(track,'hit_streak') and track.hit_streak < MIN_HIT_STREAK: continue
            if tx1<0 or ty1<0 or tx2>W or ty2>H: continue
            if bw<=0 or bh<=0 or bw>W*0.5 or bh>H*0.8: continue

            active_ids.add(tid)
            track_positions[tid].append((cx,cy))
            track_boxes[tid][frame_idx] = (tx1, ty1, tx2, ty2)

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
    name='osnet_ain_x1_0',
    num_classes=1000,
    pretrained=True
)
reid_model.eval()
if torch.cuda.is_available():
    reid_model = reid_model.cuda()
    print("Re-ID running on GPU")
else:
    print("Re-ID running on CPU")

CROP_QUALITY_MIN = 0.12   # 低于此值的 crop 视为空帧/漂移框，跳过

def crop_quality(crop):
    """对比度评分：人物 crop 纹理丰富，漂移到背景的框对比度极低"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = gray.mean()
    std  = gray.std()
    score = std / (mean + 1e-8)
    # 极暗 crop（框住地面/阴影）额外降分
    if mean < 20:
        score *= 0.3
    return score

def filter_crops(crops):
    """保留质量合格的 crop；若全部不合格则 fallback 到原始列表"""
    good = [c for c in crops if c is not None and c.size > 0
            and crop_quality(c) >= CROP_QUALITY_MIN]
    return good if good else [c for c in crops if c is not None and c.size > 0]

def extract_feature(crop_imgs):
    """从一组裁剪图提取平均特征向量"""
    feats = []
    for crop in filter_crops(crop_imgs):
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

# 颜色直方图特征（上半身/外套区域 HSV）
# 与 OSNet 特征加权拼接，增强颜色区分能力
COLOR_WEIGHT = 0.35   # 35% 颜色，65% 外观形态

def extract_color_hist(crops):
    """提取上体（外套）+ 下体（固定器/靴子）HSV 颜色直方图，返回归一化向量"""
    hists = []
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        h = crop.shape[0]

        def zone_hist(zone):
            if zone.size == 0:
                return np.zeros(48)
            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            hh = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            sh = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
            vh = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
            hist = np.concatenate([hh, sh, vh])
            return hist / (hist.sum() + 1e-8)

        upper = crop[:max(1, int(h * 0.55)), :]     # 上 55%：外套
        lower = crop[max(0, int(h * 0.70)):, :]     # 下 30%：靴子/固定器
        # 上体权重 0.6，下体权重 0.4
        hist = np.concatenate([zone_hist(upper) * 0.6, zone_hist(lower) * 0.4])
        hists.append(hist)
    return np.mean(hists, axis=0) if hists else None

color_features = []
valid_tids_color = []
for tid in valid_tids:
    crops = track_crops.get(tid, [])
    if not crops:
        continue
    cfeat = extract_color_hist(filter_crops(crops))
    if cfeat is not None:
        color_features.append(cfeat)
        valid_tids_color.append(tid)

# 构建每个 tid 的组合特征（OSNet + 颜色直方图加权拼接）
def make_combined_feat(osnet_feat, color_feat):
    o = osnet_feat / (np.linalg.norm(osnet_feat) + 1e-8)
    c = color_feat  / (np.linalg.norm(color_feat)  + 1e-8)
    return np.concatenate([(1 - COLOR_WEIGHT) * o, COLOR_WEIGHT * c])

color_feat_by_tid = {t: color_features[i] for i, t in enumerate(valid_tids_color)}
combined_feat_by_tid = {}
for tid, osnet_f in zip(valid_tids_with_feat, features):
    if tid in color_feat_by_tid:
        combined_feat_by_tid[tid] = make_combined_feat(osnet_f, color_feat_by_tid[tid])
    else:
        combined_feat_by_tid[tid] = osnet_f / (np.linalg.norm(osnet_f) + 1e-8)

print(f"Combined features (OSNet + color) built for {len(combined_feat_by_tid)} tracks")

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

    # ── 时序约束常量（供 K-means 救援步骤和 Re-ID 归并共用）──
    MERGE_SIM_THRESHOLD  = 0.90
    SPLIT_GAP_SECONDS    = 1.0
    MIN_GAP_SECONDS      = 25.0
    IMPOSSIBLE_SIM       = 0.97
    DOUBLE_DETECTION_SIM = 0.82

    def cosine_sim(a, b):
        a_n = a / (np.linalg.norm(a) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_n, b_n))

    def track_gap_seconds(t1, t2):
        segs1 = track_segments.get(t1, [])
        segs2 = track_segments.get(t2, [])
        if not segs1 or not segs2:
            return float('inf')
        end1   = max(e for _, e in segs1)
        start1 = min(s for s, _ in segs1)
        end2   = max(e for _, e in segs2)
        start2 = min(s for s, _ in segs2)
        gap_frames = max(start2 - end1, start1 - end2, 0) if not (start1 <= end2 and start2 <= end1) else -1
        return gap_frames / fps

    SPATIAL_CENTER_DIST_MAX = 0.20   # 中心距 > 20% 帧宽 → 两人同时在场

    def spatial_center_dist_during_overlap(t1, t2):
        segs1, segs2 = track_segments.get(t1, []), track_segments.get(t2, [])
        if not segs1 or not segs2:
            return float('inf')
        s_start = max(min(s for s,_ in segs1), min(s for s,_ in segs2))
        s_end   = min(max(e for _,e in segs1), max(e for _,e in segs2))
        if s_start > s_end:
            return float('inf')
        boxes1, boxes2 = track_boxes.get(t1, {}), track_boxes.get(t2, {})
        dists = []
        for f in range(s_start, s_end + 1, 2):
            if f in boxes1 and f in boxes2:
                b1, b2 = boxes1[f], boxes2[f]
                cx1 = (b1[0]+b1[2])/2;  cy1 = (b1[1]+b1[3])/2
                cx2 = (b2[0]+b2[2])/2;  cy2 = (b2[1]+b2[3])/2
                dists.append(((cx1-cx2)**2+(cy1-cy2)**2)**0.5 / W)
        return float(np.mean(dists)) if dists else float('inf')

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
            per_track_h[tid]    = np.percentile([c.shape[0] for c in valid_crops], 75)
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

    # Step4：背景救援 — 对背景组里的 track 计算与主滑手组的 Re-ID 相似度，
    # 分级阈值：
    #   ≤ SPLIT_GAP_SECONDS 内的背景 track → 视为"接近道具阶段"被误分，用 DOUBLE_DETECTION_SIM
    #   其余 → 用 RESCUE_SIM_THRESHOLD
    RESCUE_SIM_THRESHOLD = 0.87
    all_feat_by_tid = combined_feat_by_tid   # 已包含 OSNet + 颜色直方图
    rescued = set()
    for bg_tid in list(bg_tids_set):
        if bg_tid not in all_feat_by_tid:
            continue
        best_sim, best_main = 0.0, None
        for m_tid in main_tids_set:
            if m_tid not in all_feat_by_tid:
                continue
            sim = float(np.dot(
                all_feat_by_tid[bg_tid]  / (np.linalg.norm(all_feat_by_tid[bg_tid])  + 1e-8),
                all_feat_by_tid[m_tid] / (np.linalg.norm(all_feat_by_tid[m_tid]) + 1e-8)))
            if sim > best_sim:
                best_sim, best_main = sim, m_tid
        if best_main is None:
            continue
        gap = track_gap_seconds(bg_tid, best_main)
        # 接近段（gap ≤ SPLIT_GAP）用宽松阈值；其余用标准阈值
        # 不做空间中心距检测——跳跃动作会导致同一人 bbox 中心点大幅偏移
        threshold = DOUBLE_DETECTION_SIM if gap <= SPLIT_GAP_SECONDS else RESCUE_SIM_THRESHOLD
        if best_sim >= threshold:
            rescued.add(bg_tid)
            print(f"  [Rescue] t{bg_tid} 从背景组拉回 → 与 t{best_main} 相似度={best_sim:.3f}  gap={gap:.1f}s")
    main_tids_set |= rescued
    bg_tids_set   -= rescued
    if rescued:
        print(f"  共救援 {len(rescued)} 个 track: {sorted(rescued, key=lambda x:int(x))}")

    print(f"  MAIN RIDERS ({len(main_tids_set)}): {sorted(main_tids_set, key=lambda x:int(x))}")
    print(f"  BACKGROUND  ({len(bg_tids_set)}):  {sorted(bg_tids_set,  key=lambda x:int(x))}")

    # Step5：用结果覆盖cluster_map（main_cluster=2保持不变作为标记）
    main_cluster = 2
    for tid in all_tids_for_reid:
        cluster_map[tid] = main_cluster if tid in main_tids_set else 0

    if not cnt:
        print("  DBSCAN found no clusters (all noise) — K-means result is used instead.")
else:
    print("Not enough tracks for clustering, skipping.")

# ─────────────────────────────────────────────
# Re-ID 归并：把同一滑手的多个 track ID 合并
# 条件：时间不重叠 + 余弦相似度 >= 阈值
# ─────────────────────────────────────────────
MERGE_SIM_THRESHOLD  = 0.90   # 正常时间间隔下的合并阈值
SPLIT_GAP_SECONDS    = 1.0    # 间隔 ≤ 此值：追踪分割（遮挡/姿态），按宽松阈值合并
MIN_GAP_SECONDS      = 25.0   # 间隔在 SPLIT~MIN 之间：物理上不可能是同人返回，需极高阈值
IMPOSSIBLE_SIM       = 0.97   # "不可能返回区间"内仍允许合并的最低相似度
DOUBLE_DETECTION_SIM = 0.82   # 时间重叠时允许合并的最低相似度（同趟误分割）

def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_n, b_n))

def track_gap_seconds(t1, t2):
    """两个 track 之间的时间间隔（秒），重叠时为负值"""
    segs1 = track_segments.get(t1, [])
    segs2 = track_segments.get(t2, [])
    if not segs1 or not segs2:
        return float('inf')
    end1   = max(e for _, e in segs1)
    start1 = min(s for s, _ in segs1)
    end2   = max(e for _, e in segs2)
    start2 = min(s for s, _ in segs2)
    gap_frames = max(start2 - end1, start1 - end2, 0) if not (start1 <= end2 and start2 <= end1) else -1
    return gap_frames / fps

def spatial_center_dist_during_overlap(t1, t2):
    """计算两 track 时间重叠区间内的平均中心点距离（归一化到帧宽）。
    距离小 → 同一人姿态变化导致的双重检测；距离大 → 两人同时在场。"""
    segs1, segs2 = track_segments.get(t1, []), track_segments.get(t2, [])
    if not segs1 or not segs2:
        return float('inf')
    s_start = max(min(s for s,_ in segs1), min(s for s,_ in segs2))
    s_end   = min(max(e for _,e in segs1), max(e for _,e in segs2))
    if s_start > s_end:
        return float('inf')
    boxes1, boxes2 = track_boxes.get(t1, {}), track_boxes.get(t2, {})
    dists = []
    for f in range(s_start, s_end + 1, 2):
        if f in boxes1 and f in boxes2:
            b1, b2 = boxes1[f], boxes2[f]
            cx1 = (b1[0] + b1[2]) / 2;  cy1 = (b1[1] + b1[3]) / 2
            cx2 = (b2[0] + b2[2]) / 2;  cy2 = (b2[1] + b2[3]) / 2
            dists.append(((cx1-cx2)**2 + (cy1-cy2)**2)**0.5 / W)
    return float(np.mean(dists)) if dists else float('inf')

SPATIAL_CENTER_DIST_MAX = 0.20  # 中心距离 > 20% 帧宽 → 两人同时在场，阻止合并

def can_merge(t1, t2, sim):
    """
    三段式时序约束（纯相似度驱动，不依赖空间位置）：
    - 重叠或极短间隔（≤ SPLIT_GAP_SECONDS）：视为同趟误分割，按宽松阈值
      注：不做空间中心距检测——跳跃动作会导致同一人的 bbox 位置大幅偏移
    - 短间隔（SPLIT_GAP ~ MIN_GAP_SECONDS）：物理上不可能是同人返回，需极高阈值
    - 长间隔（> MIN_GAP_SECONDS）：正常归来，按正常阈值
    """
    gap = track_gap_seconds(t1, t2)
    if gap <= SPLIT_GAP_SECONDS:
        return sim >= DOUBLE_DETECTION_SIM
    elif gap < MIN_GAP_SECONDS:
        return sim >= IMPOSSIBLE_SIM
    else:
        return sim >= MERGE_SIM_THRESHOLD

main_feat_tids = [t for t in combined_feat_by_tid if t in main_tids_set]
feat_by_tid    = {t: combined_feat_by_tid[t] for t in main_feat_tids}

# Complete-linkage 聚类：合并两组时要求所有跨组对均 >= 阈值，避免传递性误连
groups = [[t] for t in main_feat_tids]  # 初始每人一组

def group_overlap(g1, g2):
    for t1 in g1:
        for t2 in g2:
            if tracks_overlap(t1, t2):
                return True
    return False

def group_can_merge(g1, g2, sim):
    """complete-linkage + 双重检测豁免：所有跨组对均可合并（时间不重叠或相似度极高）"""
    for t1 in g1:
        for t2 in g2:
            if not can_merge(t1, t2, sim):
                return False
    return True

def group_min_sim(g1, g2):
    """complete-linkage：取所有跨组对的最小相似度"""
    return min(cosine_sim(feat_by_tid[t1], feat_by_tid[t2])
               for t1 in g1 for t2 in g2
               if t1 in feat_by_tid and t2 in feat_by_tid)

# 诊断：打印关注配对的相似度明细（OSNet分量 vs 颜色分量）
DEBUG_PAIRS = [
    ('15','55'),('15','80'),('55','80'),
    ('34','35'),('34','76'),('35','76'),
    ('42','43'),('42','100'),('43','100'),('100','101'),
    ('100','102'),('101','102'),
    ('122','123'),
]
if any(t in feat_by_tid for pair in DEBUG_PAIRS for t in pair):
    print("\n--- 关注配对相似度明细 ---")
    osnet_dim = features[0].shape[0] if len(features) > 0 else 512
    for ta, tb in DEBUG_PAIRS:
        if ta not in feat_by_tid or tb not in feat_by_tid:
            continue
        fa, fb = feat_by_tid[ta], feat_by_tid[tb]
        sim_total = cosine_sim(fa, fb)
        # 拆分 OSNet 和颜色两段
        fa_o, fa_c = fa[:osnet_dim], fa[osnet_dim:]
        fb_o, fb_c = fb[:osnet_dim], fb[osnet_dim:]
        sim_o = cosine_sim(fa_o, fb_o) if fa_o.size > 0 else 0
        sim_c = cosine_sim(fa_c, fb_c) if fa_c.size > 0 else 0
        gap = track_gap_seconds(ta, tb)
        print(f"  t{ta}+t{tb}: total={sim_total:.3f}  osnet={sim_o:.3f}  color={sim_c:.3f}  gap={gap:.1f}s")

print("\n--- Re-ID 归并 (complete-linkage, 阈值={}) ---".format(MERGE_SIM_THRESHOLD))
merged_any = True
while merged_any:
    merged_any = False
    best_sim, best_i, best_j = -1, -1, -1
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            sim = group_min_sim(groups[i], groups[j])
            # 快速过滤：低于所有场景中最宽松阈值的直接跳过
            if sim < min(DOUBLE_DETECTION_SIM, MERGE_SIM_THRESHOLD):
                continue
            if not group_can_merge(groups[i], groups[j], sim):
                continue
            if sim > best_sim:
                best_sim, best_i, best_j = sim, i, j
    if best_i >= 0:
        ti_repr = groups[best_i]
        tj_repr = groups[best_j]
        print(f"  合并: {[f't{t}' for t in ti_repr]} + {[f't{t}' for t in tj_repr]}  min_sim={best_sim:.3f}")
        groups[best_i] = groups[best_i] + groups[best_j]
        groups.pop(best_j)
        merged_any = True

if len(groups) == len(main_feat_tids):
    print("  无需归并（所有主滑手 track 均为独立个体）")

def _first_frame(tid):
    segs = track_segments.get(tid, [(0, 0)])
    return segs[0][0]

# 按首次出场时间排序，构建 person_groups
person_groups = {}
for pid, grp in enumerate(
        sorted(groups, key=lambda g: min(_first_frame(t) for t in g)), 1):
    person_groups[pid] = sorted(grp, key=_first_frame)

print(f"\n归并后共 {len(person_groups)} 位独立滑手:")
for pid, tids in person_groups.items():
    segs_info = "  ".join(
        f"t{t}({_first_frame(t)/fps:.1f}s)" for t in tids)
    print(f"  P{pid:02d}: {segs_info}")

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
# 输出分组截图表格（按 person_groups 归组）
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Generating thumbnail sheet...")
print("="*50)

# 布局参数
COLS     = REID_FRAMES_PER_TRACK   # 每 track 显示几帧
INFO_W   = 190                     # 左侧信息栏宽度
ROW_H    = THUMB_H + 10            # 每个 track 行高
HEADER_H = 40                      # 每个 person 组的标题栏高度
GROUP_GAP = 16                     # 组间距

# 构建待渲染列表：主滑手按 person 分组，背景人物统一放最后
bg_tids = sorted(
    [t for t in track_segments if cluster_map.get(t, -1) != main_cluster],
    key=lambda t: track_segments[t][0][0])

total_rows = sum(len(tids) for tids in person_groups.values()) + len(bg_tids)
n_groups   = len(person_groups) + (1 if bg_tids else 0)
total_h    = HEADER_H * n_groups + total_rows * ROW_H + GROUP_GAP * n_groups + 60
total_w    = INFO_W + COLS * (THUMB_W + 4) + 20

sheet = np.ones((total_h, total_w, 3), dtype=np.uint8) * 22

def draw_person_group(sheet, y, pid, tids, color):
    n_passes = len(tids)
    label = (f"P{pid:02d}  {n_passes} pass{'es' if n_passes>1 else ''}"
             if pid > 0 else f"Background  ({len(tids)} tracks)")
    cv2.rectangle(sheet, (0, y), (total_w, y + HEADER_H),
                  tuple(int(c * 0.35) for c in color), -1)
    cv2.putText(sheet, label, (10, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    y += HEADER_H + 2

    for tid in tids:
        segs      = track_segments.get(tid, [(0, 0)])
        start_sec = segs[0][0] / fps
        end_sec   = segs[-1][1] / fps
        dur       = sum((e - s) / fps for s, e in segs)

        cv2.putText(sheet, f"t{tid}", (6, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        cv2.putText(sheet, f"{int(start_sec//60):02d}:{start_sec%60:04.1f}", (6, y + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
        cv2.putText(sheet, f"~{int(end_sec//60):02d}:{end_sec%60:04.1f}", (6, y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
        cv2.putText(sheet, f"{dur:.1f}s", (6, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 210, 90), 1)

        crops = track_crops.get(tid, [])
        for i, crop in enumerate(crops[:COLS]):
            if crop is None or crop.size == 0:
                continue
            thumb = cv2.resize(crop, (THUMB_W, THUMB_H))
            tx = INFO_W + i * (THUMB_W + 4)
            sheet[y:y + THUMB_H, tx:tx + THUMB_W] = thumb
            cv2.rectangle(sheet, (tx, y), (tx + THUMB_W, y + THUMB_H), color, 2)

        cv2.line(sheet, (0, y + ROW_H - 2), (total_w, y + ROW_H - 2), (50, 50, 50), 1)
        y += ROW_H

    return y + GROUP_GAP

y = 20
for pid, tids in person_groups.items():
    color = get_color(pid * 31)   # 每个 person 用固定颜色
    y = draw_person_group(sheet, y, pid, tids, color)

if bg_tids:
    y = draw_person_group(sheet, y, 0, bg_tids, (80, 80, 80))

cv2.imwrite(SHEET_PATH, sheet)
print(f"Thumbnail sheet saved: {SHEET_PATH}")

# ─────────────────────────────────────────────
# 第三遍：输出剪辑视频
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("PASS 3: Exporting clip videos...")
print("="*50)

# 按 person_groups 构建待导出片段列表（若无归并结果则回退到 cluster_map）
if person_groups:
    main_segs = []   # (start_f, end_f, tid, person_id)
    for pid, tids in person_groups.items():
        for tid in tids:
            for (s, e) in track_segments.get(tid, []):
                main_segs.append((s, e, tid, pid))
    main_segs.sort(key=lambda x: x[0])
else:
    main_segs = []
    for tid in track_segments:
        if cluster_map.get(tid, -1) == main_cluster:
            for (s, e) in track_segments[tid]:
                main_segs.append((s, e, tid, 0))
    main_segs.sort(key=lambda x: x[0])

if not main_segs:
    print("No main rider segments found, skipping clip export.")
else:
    n_persons = len(set(pid for _,_,_,pid in main_segs))
    print(f"Exporting {len(main_segs)} clips for {n_persons} unique rider(s)")
    cap3 = cv2.VideoCapture(VIDEO_PATH)
    pre_roll_frames  = int(PRE_ROLL_SECONDS * fps)
    ux_tail_frames   = int(0.15 * fps)
    for clip_n, (start_f, end_f, tid, pid) in enumerate(main_segs, 1):
        # 独立短片段（未与其他 track 归并的单条 track）跳过，避免输出太短的接近段
        pid_tids = person_groups.get(pid, [tid]) if person_groups else [tid]
        raw_dur  = (end_f - start_f) / fps
        if len(pid_tids) == 1 and raw_dur < MIN_OUTPUT_SECONDS:
            print(f"  [skip] t{tid}: {raw_dur:.1f}s < MIN_OUTPUT_SECONDS, 已归并入组则不单独输出")
            continue
        clip_start = max(0, start_f - pre_roll_frames)
        # 剪辑结束用 track_frame_exit（bbox 完全出画），fallback 到 end_f（ROI 内最后帧）
        frame_exit_f = track_frame_exit.get(tid, end_f)
        # 下一位（不同 person）进入时截止
        next_diff_start = None
        for nxt in main_segs[clip_n:]:
            if nxt[3] != pid:
                next_diff_start = nxt[0]
                break
        if next_diff_start is not None:
            clip_end = min(frame_exit_f + ux_tail_frames, next_diff_start - 1, total_frames - 1)
        else:
            clip_end = min(frame_exit_f + ux_tail_frames, total_frames - 1)
        clip_path = os.path.join(CLIPS_DIR, f"{_stem}_p{pid:02d}_clip{clip_n:03d}_t{tid}.mp4")
        out_clip  = cv2.VideoWriter(clip_path, fourcc, fps, (W, H))
        cap3.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for _ in range(clip_end - clip_start + 1):
            ret, frm = cap3.read()
            if not ret:
                break
            out_clip.write(frm)
        out_clip.release()
        dur = (clip_end - clip_start) / fps
        print(f"  [{clip_n:03d}] P{pid:02d}/t{tid}: {clip_start/fps:.1f}s ~ {clip_end/fps:.1f}s  ({dur:.1f}s)  → {clip_path}")
    cap3.release()

print(f"\nDone!")
print(f"  Preview : {PREVIEW_PATH}")
print(f"  Sheet   : {SHEET_PATH}")
print(f"  Clips   : {CLIPS_DIR}/")
main_count = sum(1 for t in track_segments if cluster_map.get(t,-1)==main_cluster)
print(f"  Main riders confirmed: {main_count}")
print(f"  Total tracks: {len(track_segments)}")
