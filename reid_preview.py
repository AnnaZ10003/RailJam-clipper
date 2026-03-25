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
parser.add_argument("--confirm", default=None,
                    help="用户确认文件路径（JSON），填写后重新运行以应用分组决策")
args = parser.parse_args()

VIDEO_PATH = args.video
_stem      = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
CLIPS_DIR  = args.out_dir
PREVIEW_PATH        = os.path.join(CLIPS_DIR, f"{_stem}_preview.mp4")
SHEET_PATH          = os.path.join(CLIPS_DIR, f"{_stem}_sheet.jpg")
CONFIRM_SHEET_PATH  = os.path.join(CLIPS_DIR, f"{_stem}_confirm_sheet.jpg")
CONFIRM_TMPL_PATH   = os.path.join(CLIPS_DIR, f"{_stem}_confirm_template.json")
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
scan_cx, scan_cy, scan_ty1 = [], [], []
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
                    scan_ty1.append(y1)          # bbox 顶边，用于上缘扩展
    scan_idx += 1
cap_pre.release()

if len(scan_cx) >= 3:
    cx_arr  = np.array(scan_cx)
    cy_arr  = np.array(scan_cy)
    ty1_arr = np.array(scan_ty1)
    pad_x     = int(W * 0.07)
    pad_y_bot = int(H * 0.04)
    pad_y_top = int(H * 0.04)
    roi_x1 = max(0,   int(np.percentile(cx_arr,   3)) - pad_x)
    roi_x2 = min(W-1, int(np.percentile(cx_arr,  97)) + pad_x)
    # 上缘用 bbox 顶边的低百分位，确保跳跃时上半身也在 ROI 内
    roi_y1 = max(0,   int(np.percentile(ty1_arr,  3)) - pad_y_top)
    roi_y2 = min(H-1, int(np.percentile(cy_arr,  97)) + pad_y_bot)
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

    # Step2b：计算每个 track 的净水平位移（方向判断用）和有符号速度（K-means 用）
    # track_net_vx      : 净水平位移（px），符号代表方向，用于方向过滤
    # track_signed_speed: 净水平位移 / 轨迹帧数（px/frame），同时编码方向和速度
    track_net_vx     = {}
    track_signed_speed = {}
    # track_dominant_dir: 所有相邻帧间 X 位移的中位数方向（-1/0/+1）
    # 比 net_vx 符号更可靠——即使起终点相近，若全程主要向某方向运动也能正确反映
    # 典型反例：先向右走一段再向左返回，net_vx≈0 但 dominant_dir=-1
    track_dominant_dir = {}
    for tid, positions in track_positions.items():
        if len(positions) < 4:
            track_dominant_dir[tid] = 0
            continue
        n = max(2, len(positions) // 5)
        first_x = np.mean([p[0] for p in positions[:n]])
        last_x  = np.mean([p[0] for p in positions[-n:]])
        net_vx = last_x - first_x
        track_net_vx[tid]      = net_vx
        track_signed_speed[tid] = net_vx / len(positions)
        # 主导方向：统计"快速移动帧"正/负计数比，而非所有帧的中位数
        # 解决问题：主滑手快速通过（+10px/帧）后若被缓慢漂移（-3px/帧）长时间追踪，
        # 若用中位数则漂移帧数多会压倒正向 → 误判为反向
        # 方向一致性规则：
        #   |dx| >= FAST_DX → 快速帧（有明确方向意义）
        #   pos_count >= 2 * neg_count → 主导正向 (+1)
        #   neg_count >= 2 * pos_count → 主导反向 (-1)
        #   否则 → 混合/不明确 (0)
        FAST_DX = 5.0   # px/frame；快速移动阈值（约 150px/s @ 30fps）
        diffs_x  = [positions[i+1][0] - positions[i][0] for i in range(len(positions) - 1)]
        pos_count = sum(1 for dx in diffs_x if dx >=  FAST_DX)
        neg_count = sum(1 for dx in diffs_x if dx <= -FAST_DX)
        total_fast = pos_count + neg_count
        if total_fast < 3:
            track_dominant_dir[tid] = 0   # 快速帧太少，方向不明确
        elif pos_count >= 2 * neg_count:
            track_dominant_dir[tid] = +1
        elif neg_count >= 2 * pos_count:
            track_dominant_dir[tid] = -1
        else:
            track_dominant_dir[tid] = 0   # 正反向快速帧数量相当，混合运动

    # Step2c：轨迹稳定性过滤 — 持续性大跳变 → 漂移框（在多人身上跳来跳去）
    # 主滑手跳跃只有 1-2 帧大位移，漂移框则多帧持续跳变
    # 使用欧氏距离（X+Y 两方向）：漂移框可能在水平或垂直方向跳变，只查 X 会漏检
    STABILITY_JUMP_MAX      = 0.20   # 单步欧氏跳变 > 20% 帧对角线算"大跳"
    STABILITY_JUMP_FRACTION = 0.08   # 大跳帧占比超过 8% → 判定为漂移框
    diag = (W**2 + H**2) ** 0.5
    unstable_tids = set()
    for tid in list(valid_h_tids):
        positions = track_positions.get(tid, [])
        if len(positions) >= 5:
            jumps = [
                ((positions[i+1][0] - positions[i][0])**2 +
                 (positions[i+1][1] - positions[i][1])**2) ** 0.5 / diag
                for i in range(len(positions) - 1)
            ]
            frac = sum(1 for j in jumps if j > STABILITY_JUMP_MAX) / len(jumps)
            if frac > STABILITY_JUMP_FRACTION:
                unstable_tids.add(tid)
    if unstable_tids:
        valid_h_tids = [t for t in valid_h_tids if t not in unstable_tids]
        print(f"  [稳定性过滤] 漂移框移除 ({len(unstable_tids)}): "
              f"{sorted(unstable_tids, key=lambda x:int(x))}")

    # Step3：K-means 三分（avg_h + net_vx）
    # 场景实际存在三类人群：
    #   [main]   主滑手     — 框大、净位移高正值（快速横穿道具）
    #   [walker] 同向围观者 — 框中小、净位移小正值（慢速同向移动）
    #   [drag]   拖牵返回者 — 框小、净位移负值（反向返回）
    # 3-cluster 能明确分离三类，避免主滑手（框偏小时）被归入背景大组。
    # walker 组与主滑手方向相同，救援时使用较低阈值；drag 组方向预检后基本不被救援。
    # 样本不足 15 条时退化为 2-cluster（简单视频）。
    from sklearn.cluster import KMeans
    bg_drag_tids_set = set()   # 拖牵/反向背景（严格过滤）
    bg_same_tids_set = set()   # 同向围观者（软过滤，可低阈值救援）

    if len(valid_h_tids) >= 2:
        h_vals  = np.array([per_track_h[t]          for t in valid_h_tids], dtype=float)
        vx_vals = np.array([track_net_vx.get(t, 0.) for t in valid_h_tids], dtype=float)
        # 各特征独立标准化，使两维度量级相当
        h_std  = h_vals.std()  + 1e-8
        vx_std = vx_vals.std() + 1e-8
        feats_2d = np.column_stack([
            (h_vals  - h_vals.mean())  / h_std,
            (vx_vals - vx_vals.mean()) / vx_std,
        ])

        use_3cluster = len(valid_h_tids) >= 15
        n_clusters   = 3 if use_3cluster else 2
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = km.fit_predict(feats_2d)

        c_h  = np.array([h_vals[km_labels == k].mean()  for k in range(n_clusters)])
        c_vx = np.array([vx_vals[km_labels == k].mean() for k in range(n_clusters)])

        if use_3cluster:
            main_km_lbl = int(np.argmax(c_h))          # 框最高 = 主滑手
            drag_km_lbl = int(np.argmin(c_vx))          # 最负向 = 拖牵返回者
            # 保证三个 label 各不相同（边界情况处理）
            if main_km_lbl == drag_km_lbl:
                drag_km_lbl = int(np.argsort(c_vx)[1])
            same_km_lbl = [k for k in range(3)
                           if k != main_km_lbl and k != drag_km_lbl][0]
            print(f"  KMeans 3-cluster:")
            print(f"    Main   (c{main_km_lbl}): h={c_h[main_km_lbl]:.0f}px  vx={c_vx[main_km_lbl]:+.0f}px")
            print(f"    Drag   (c{drag_km_lbl }): h={c_h[drag_km_lbl ]:.0f}px  vx={c_vx[drag_km_lbl ]:+.0f}px")
            print(f"    Walker (c{same_km_lbl }): h={c_h[same_km_lbl ]:.0f}px  vx={c_vx[same_km_lbl ]:+.0f}px")
            main_tids_set    = {t for t, lbl in zip(valid_h_tids, km_labels) if lbl == main_km_lbl}
            bg_drag_tids_set = {t for t, lbl in zip(valid_h_tids, km_labels) if lbl == drag_km_lbl}
            bg_same_tids_set = {t for t, lbl in zip(valid_h_tids, km_labels) if lbl == same_km_lbl}
            bg_tids_set      = bg_drag_tids_set | bg_same_tids_set
            # 若拖牵组 avg_vx 也是正值，说明本视频无明显反向群体，降级为全部同向背景
            if c_vx[drag_km_lbl] >= 0:
                print(f"  拖牵组 vx={c_vx[drag_km_lbl]:+.0f}px ≥ 0，无明显反向群体 → 合并两背景组为 walker")
                bg_same_tids_set |= bg_drag_tids_set
                bg_drag_tids_set  = set()
        else:
            # 2-cluster 退化模式
            main_km_lbl = int(np.argmax(c_h))
            h_ratio = (min(c_h) / max(c_h)) if max(c_h) > 0 else 1.0
            print(f"  KMeans 2-cluster: h=[{c_h[0]:.0f}, {c_h[1]:.0f}]px  "
                  f"vx=[{c_vx[0]:+.0f}, {c_vx[1]:+.0f}]px  ratio={h_ratio:.2f}")
            if h_ratio > 0.72 and abs(c_vx[0] - c_vx[1]) < W * 0.05:
                print(f"  h_ratio={h_ratio:.2f}>0.72 且 vx 差异小 → 无背景人物，全部视为主滑手")
                main_tids_set = set(valid_h_tids)
                bg_tids_set   = set()
            else:
                main_tids_set = {t for t, lbl in zip(valid_h_tids, km_labels) if lbl == main_km_lbl}
                bg_tids_set   = {t for t, lbl in zip(valid_h_tids, km_labels) if lbl != main_km_lbl}
                bg_same_tids_set = bg_tids_set  # 2-cluster 时无法区分，统一视为 walker
    else:
        main_tids_set = set(valid_h_tids)
        bg_tids_set   = set()
    bg_tids_set.update(unstable_tids)
    bg_tids_set.update(ghost_tids)

    # Step4：背景救援 — 对背景组里的 track 计算与主滑手组的 Re-ID 相似度
    # 两档阈值：
    #   drag 组（反向）：方向预检几乎不会进入救援；进入的用严格阈值
    #   walker 组（同向围观者）：方向相同，用较低阈值，避免漏抓小框主滑手
    RESCUE_SIM_DRAG   = 0.87   # 拖牵/反向组救援阈值
    RESCUE_SIM_WALKER = 0.87   # 同向围观者组阈值：与 drag 相同（冬运服装相似度天然高，0.82 太宽松）
    all_feat_by_tid = combined_feat_by_tid

    # ── 救援前预清洗：从 K-means 主组中移除疑似漂移框 ──
    # 漂移框特征：在主组中（大框/大正位移），但 dominant_dir 反向
    # 说明它短暂锁定在主滑手身上（获得大h和大net_vx），然后漂移回来
    # 若保留在主组，将作为错误的救援锚点，引入大量误救
    # 从 K-means 主组计算初步主流方向（先用 net_vx，dominant_dir 还未完全验证）
    prelim_vx = [track_net_vx[t] for t in main_tids_set
                 if t in track_net_vx and abs(track_net_vx[t]) > W * 0.04]
    main_dir_prelim = int(np.sign(np.median(prelim_vx))) if len(prelim_vx) >= 2 else 0

    drift_removed = set()
    if main_dir_prelim != 0:
        for tid in list(main_tids_set):
            dom = track_dominant_dir.get(tid, 0)
            if dom != 0 and dom != main_dir_prelim:
                # dominant_dir 反向 → 疑似漂移框，移出主组
                drift_removed.add(tid)
    if drift_removed:
        main_tids_set -= drift_removed
        bg_same_tids_set |= drift_removed   # 允许后续低阈值救援（万一是真实主滑手）
        bg_tids_set      |= drift_removed
        print(f"  [主组预清洗] 移除疑似漂移框 ({len(drift_removed)}): "
              f"{sorted(drift_removed, key=lambda x:int(x))}")

    rescued = set()
    for bg_tid in list(bg_tids_set):
        if bg_tid not in all_feat_by_tid:
            continue
        bg_vx = track_net_vx.get(bg_tid, 0)
        is_walker = bg_tid in bg_same_tids_set  # 同向围观者组（宽松救援）

        # 拖牵组：dominant_dir=-1（方向确认反向）→ 直接阻断
        # dominant_dir=0（方向未知）+ 极高相似度(≥0.925) → 允许第二轮救援（见下方 rescued2）
        if bg_tid in bg_drag_tids_set:
            bg_dominant_pre = track_dominant_dir.get(bg_tid, 0)
            if bg_dominant_pre == -1:
                continue   # 方向确认反向 → 硬阻断
            # 方向未知的拖牵轨迹：由第二轮救援（rescued2）处理，使用扩大后的主组更准确
            continue

        # 方向预检（dominant_dir + net_vx 双重确认才阻断）：
        # 仅 dominant_dir 反向但 net_vx 明确同向 → 可能是轨迹噪声，允许救援
        # 两者都反向 → 确定为反向，阻断
        bg_dominant = track_dominant_dir.get(bg_tid, 0)
        vx_clearly_main = abs(bg_vx) > W * 0.06 and int(np.sign(bg_vx)) == main_dir_prelim
        if main_dir_prelim != 0 and bg_dominant != 0 and bg_dominant != main_dir_prelim:
            if not vx_clearly_main:   # net_vx 也不是明确同向 → 双重确认反向，阻断
                continue
            # else: dominant 反向但 net_vx 明确同向 → 轨迹噪声，允许通过（下面的救援阈值会再把关）

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
        if gap <= 0:
            # 时间实际重叠（gap≤0）：两条轨迹同时存在，必然是不同人
            # 注意：Python 中 -0.0 < 0 为 False，所以用 ≤ 0 而非 < 0
            # Walker 组 + 同时存在 → 双重否定，直接阻断
            if is_walker:
                continue
            # 若方向也相反，更不应该救援
            if bg_dominant != 0 and bg_dominant != main_dir_prelim:
                continue
            threshold = RESCUE_SIM_DRAG             # 0.87 — 比双检测阈值更严格
        elif gap <= SPLIT_GAP_SECONDS:
            threshold = DOUBLE_DETECTION_SIM        # 极短间隔：可能是同人轨迹断裂
        elif is_walker:
            threshold = RESCUE_SIM_WALKER           # 同向围观者组：宽松
        else:
            threshold = RESCUE_SIM_DRAG             # 拖牵/其他背景：严格

        if best_sim >= threshold:
            rescued.add(bg_tid)
            tag = "walker" if is_walker else "drag"
            print(f"  [Rescue/{tag}] t{bg_tid} 从背景组拉回 → 与 t{best_main} "
                  f"相似度={best_sim:.3f}  gap={gap:.1f}s  vx={bg_vx:+.0f}px")
    main_tids_set |= rescued
    bg_tids_set   -= rescued
    if rescued:
        print(f"  共救援 {len(rescued)} 个 track: {sorted(rescued, key=lambda x:int(x))}")

    # ── 第二轮救援：拖牵组 dominant_dir=0 (方向未知) tracks ──
    # 第一轮救援扩大了 main_tids_set（如 t114 等），此时再检查拖牵组相似度更准确
    rescued2 = set()
    for bg_tid in list(bg_drag_tids_set - main_tids_set):
        if track_dominant_dir.get(bg_tid, 0) == -1:
            continue   # 方向确认反向 → 硬阻断
        if bg_tid not in all_feat_by_tid:
            continue
        best_sim2, best_main2 = 0.0, None
        for m_tid in main_tids_set:
            if m_tid not in all_feat_by_tid: continue
            s = float(np.dot(
                all_feat_by_tid[bg_tid] / (np.linalg.norm(all_feat_by_tid[bg_tid]) + 1e-8),
                all_feat_by_tid[m_tid] / (np.linalg.norm(all_feat_by_tid[m_tid]) + 1e-8)))
            if s > best_sim2: best_sim2, best_main2 = s, m_tid
        if best_main2 is None: continue
        gap2 = track_gap_seconds(bg_tid, best_main2)
        if gap2 <= 0: continue   # 同时存在 → 不可能同人
        if best_sim2 >= 0.925:
            rescued2.add(bg_tid)
            print(f"  [Rescue2/drag-unk] t{bg_tid} → t{best_main2} sim={best_sim2:.3f} gap={gap2:.1f}s")
    if rescued2:
        main_tids_set |= rescued2
        bg_tids_set   -= rescued2
        bg_drag_tids_set -= rescued2
        print(f"  第二轮救援 {len(rescued2)} 个 drag-unknown track: {sorted(rescued2, key=lambda x:int(x))}")

    # ── 方向过滤：主导运动方向与主滑手相反 → 直接移入背景（无需高并发条件）──
    # 使用 dominant_dir（逐帧位移中位数方向），比 net_vx 更能反映真实运动方向。
    # 先前版本要求"反向 AND 高并发"，导致单独反向者漏网；现改为反向即过滤。
    # 高并发条件保留作为补充：同向但高密度者也有嫌疑（同向围观者群体）。

    # 从主滑手组中确定主流方向（dominant_dir 中位数）
    main_dom_vals = [track_dominant_dir[t] for t in main_tids_set
                     if track_dominant_dir.get(t, 0) != 0]
    main_dir = int(np.sign(np.median(main_dom_vals))) if len(main_dom_vals) >= 3 else 0

    # 统计每帧所有 track（含背景）的并发数，反映真实人群密度
    MAX_CONCURRENT = 4
    frame_all_active = defaultdict(set)
    for tid, segs in track_segments.items():
        for sf, ef in segs:
            for f in range(sf, ef + 1, 3):
                frame_all_active[f].add(tid)

    track_max_concurrent = {}
    for tid, segs in track_segments.items():
        max_conc = max(
            (len(frame_all_active[f]) - 1 for sf, ef in segs
             for f in range(sf, ef + 1, 3) if frame_all_active[f]),
            default=0
        )
        track_max_concurrent[tid] = max_conc

    # 后置过滤：dominant_dir 反向 AND 高并发 → 漏网的背景人物
    # 仅凭方向（没有密度验证）过滤风险高（主滑手轨迹可能因噪声方向误判），
    # 结合高并发可大大降低误删主滑手的概率（主滑手出现时并发数通常较低）
    combined_filtered = set()
    if main_dir != 0:
        for tid in list(main_tids_set):
            dom = track_dominant_dir.get(tid, 0)
            is_wrong_dir    = dom != 0 and dom != main_dir
            is_high_density = track_max_concurrent.get(tid, 0) > MAX_CONCURRENT
            if is_wrong_dir and is_high_density:
                combined_filtered.add(tid)
    if combined_filtered:
        main_tids_set -= combined_filtered
        bg_tids_set   |= combined_filtered
        print(f"  [联合过滤] 移入背景 {len(combined_filtered)} 个 (dominant反向+高并发): "
              f"{sorted(combined_filtered, key=lambda x:int(x))}")

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

# ── 交互式确认 ──────────────────────────────────────────────────────────
# 相似度区间：< CONFIRM_LOW → 自动分开  |  > CONFIRM_HIGH → 自动合并
#             CONFIRM_LOW ~ CONFIRM_HIGH → 生成确认 sheet，等待用户决策
CONFIRM_LOW  = 0.85
CONFIRM_HIGH = 0.95
import json

# 1. 加载用户确认文件（若提供）
user_confirm_sim = {}   # {(min_tid, max_tid): 1.0 (同一人) | 0.0 (不同人)}
if args.confirm and os.path.exists(args.confirm):
    with open(args.confirm, 'r', encoding='utf-8') as f:
        confirm_data = json.load(f)
    for pair_key, decision in confirm_data.get('pairs', {}).items():
        if decision is None:
            continue
        ta, tb = pair_key.split('-')
        key = (min(ta, tb), max(ta, tb))
        user_confirm_sim[key] = 1.0 if decision else 0.0
    print(f"已加载确认文件：{len(user_confirm_sim)} 条决策")

def pair_sim(ta, tb):
    """相似度查询：优先使用用户确认，否则计算特征余弦相似度"""
    key = (min(ta, tb), max(ta, tb))
    if key in user_confirm_sim:
        return user_confirm_sim[key]
    if ta in feat_by_tid and tb in feat_by_tid:
        return cosine_sim(feat_by_tid[ta], feat_by_tid[tb])
    return 0.0

# 2. 找出待确认配对（只考虑时间约束允许合并的配对）
sorted_tids = sorted(main_feat_tids, key=lambda x: int(x))
uncertain_pairs = []
for i, ta in enumerate(sorted_tids):
    for tb in sorted_tids[i+1:]:
        sim = pair_sim(ta, tb)
        if not (CONFIRM_LOW <= sim <= CONFIRM_HIGH):
            continue
        gap = track_gap_seconds(ta, tb)
        # 只有时间约束可能放行的配对才需要确认
        in_split   = gap <= SPLIT_GAP_SECONDS
        in_normal  = gap >= MIN_GAP_SECONDS
        if not (in_split or in_normal):
            continue
        uncertain_pairs.append((ta, tb, sim, gap))
uncertain_pairs.sort(key=lambda x: -x[2])

# 3. 生成确认 sheet（每对滑手各取 3 张代表帧并排展示）
def _pick_crops(tid, n=3):
    """从 track_crops 中均匀选取 n 张质量合格的代表帧"""
    crops = filter_crops(track_crops.get(tid, []))
    if not crops:
        return [np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)] * n
    idxs = [int(i * (len(crops) - 1) / max(n - 1, 1)) for i in range(n)]
    picked = []
    for idx in idxs:
        c = brighten_crop(cv2.resize(crops[idx], (CROP_W, CROP_H)))
        picked.append(c)
    return picked

CROP_W, CROP_H = 120, 240
N_CROPS        = 3
INFO_W         = 220
PAD            = 10
ROW_H          = CROP_H + PAD * 2
ROW_W          = N_CROPS * CROP_W + INFO_W + N_CROPS * CROP_W + PAD * 4

def brighten_crop(img):
    """CLAHE 自适应对比度增强，让深色衣物细节更可见"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

if uncertain_pairs:
    sheet_rows = []
    for ta, tb, sim, gap in uncertain_pairs:
        key    = (min(ta, tb), max(ta, tb))
        status = user_confirm_sim.get(key)   # 1.0 / 0.0 / None
        row    = np.ones((ROW_H, ROW_W, 3), dtype=np.uint8) * 240

        # 左侧：ta 的 crops
        x = PAD
        for crop in _pick_crops(ta, N_CROPS):
            row[PAD:PAD+CROP_H, x:x+CROP_W] = crop
            x += CROP_W

        # 右侧：tb 的 crops
        x = PAD * 2 + N_CROPS * CROP_W + INFO_W
        for crop in _pick_crops(tb, N_CROPS):
            row[PAD:PAD+CROP_H, x:x+CROP_W] = crop
            x += CROP_W

        # 中间信息面板
        ix = N_CROPS * CROP_W + PAD * 2
        cv2.rectangle(row, (ix, PAD), (ix + INFO_W - PAD, ROW_H - PAD),
                      (220, 220, 220), -1)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        label_color = (50, 50, 50)
        cv2.putText(row, f"t{ta}  vs  t{tb}",
                    (ix + 6, PAD + 30), font, 0.7, label_color, 1)
        cv2.putText(row, f"sim = {sim:.3f}",
                    (ix + 6, PAD + 62), font, 0.65, label_color, 1)
        gap_str = f"gap = {gap:.1f}s" if gap >= 0 else "gap = overlap"
        cv2.putText(row, gap_str,
                    (ix + 6, PAD + 92), font, 0.65, label_color, 1)

        if status is None:
            decision_str, dc = "?  pending", (0, 140, 200)
        elif status == 1.0:
            decision_str, dc = "YES same", (0, 160, 0)
        else:
            decision_str, dc = "NO  diff", (0, 0, 200)
        cv2.putText(row, decision_str, (ix + 6, PAD + 130), font, 0.75, dc, 2)

        # ta / tb 标注
        cv2.putText(row, f"t{ta}", (PAD, PAD + 16), font, 0.65, (30, 30, 180), 2)
        cv2.putText(row, f"t{tb}",
                    (PAD * 2 + N_CROPS * CROP_W + INFO_W, PAD + 16),
                    font, 0.65, (30, 30, 180), 2)

        # 分隔线
        cv2.line(row, (0, ROW_H - 1), (ROW_W, ROW_H - 1), (180, 180, 180), 1)
        sheet_rows.append(row)

    confirm_sheet = np.vstack(sheet_rows)
    cv2.imwrite(CONFIRM_SHEET_PATH, confirm_sheet)
    print(f"确认 sheet 已生成：{CONFIRM_SHEET_PATH}  ({len(uncertain_pairs)} 对待确认)")

    # 4. 生成确认模板 JSON（仅首次，不覆盖已有文件）
    if not os.path.exists(CONFIRM_TMPL_PATH):
        template = {'pairs': {}}
        for ta, tb, sim, gap in uncertain_pairs:
            template['pairs'][f'{ta}-{tb}'] = None
        with open(CONFIRM_TMPL_PATH, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        print(f"确认模板已生成：{CONFIRM_TMPL_PATH}")
        print("  → 将 null 改为 true（同一人）或 false（不同人），")
        print("     然后用 --confirm 参数重新运行。")
else:
    print("无待确认配对（所有配对相似度均超出不确定区间）")
# ────────────────────────────────────────────────────────────────────────

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
    """complete-linkage：取所有跨组对的最小相似度（含用户确认覆盖）"""
    return min(pair_sim(t1, t2) for t1 in g1 for t2 in g2)

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
