import cv2
import numpy as np
from ultralytics import YOLO

# ========== 配置区 ==========
VIDEO_PATH    = r"C:\RailJam_clipper\videos\input.mp4"
OUTPUT_CHART  = r"C:\RailJam_clipper\size_distribution.png"

ROI_X1, ROI_X2 = 0.0,  0.82
ROI_Y1, ROI_Y2 = 0.15, 1.0

# 只统计ENTRY区内的框
ENTRY_X1, ENTRY_X2 = 0.00, 0.45
ENTRY_Y1, ENTRY_Y2 = 0.30, 0.65

CONF_THRESHOLD = 0.5
# ============================

print("Loading model...")
model = YOLO("yolo11n.pt")

cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {W}x{H}, {total} frames, {total/fps:.1f}s")

roi_x1,roi_x2 = int(ROI_X1*W), int(ROI_X2*W)
roi_y1,roi_y2 = int(ROI_Y1*H), int(ROI_Y2*H)
ex1,ey1 = int(ENTRY_X1*W), int(ENTRY_Y1*H)
ex2,ey2 = int(ENTRY_X2*W), int(ENTRY_Y2*H)

all_heights   = []   # ROI内所有框
entry_heights = []   # 只在ENTRY区内的框

frame_idx = 0
print("Scanning...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 5 == 0:
        results = model(frame, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                bh = y2-y1
                if not (roi_x1<=cx<=roi_x2 and roi_y1<=cy<=roi_y2): continue
                if bh < H*0.02: continue
                if conf > CONF_THRESHOLD:
                    ratio = bh / H
                    all_heights.append(ratio)
                    if ex1<=cx<=ex2 and ey1<=cy<=ey2:
                        entry_heights.append(ratio)

    frame_idx += 1
    if frame_idx % 500 == 0:
        print(f"  {frame_idx}/{total} ({100*frame_idx//total}%) "
              f"| ROI={len(all_heights)} ENTRY={len(entry_heights)}")

cap.release()
print(f"\nTotal: ROI={len(all_heights)}, ENTRY zone={len(entry_heights)}")

if entry_heights:
    arr = np.array(entry_heights)
    print(f"ENTRY zone box heights:")
    print(f"  min={arr.min():.3f}  max={arr.max():.3f}  "
          f"mean={arr.mean():.3f}  median={np.median(arr):.3f}")
    for p in [25,50,75,90,95]:
        print(f"  P{p} = {np.percentile(arr,p):.3f}")

# ── 画双直方图 ────────────────────────────────────
chart_w, chart_h = 1060, 680
ml, mr, mt, mb = 80, 40, 80, 90
pw = chart_w - ml - mr
ph = chart_h - mt - mb

img = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 245

cv2.putText(img, "Box Height Distribution  (blue=all ROI  green=ENTRY zone only)",
            (ml, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2)

cv2.line(img,(ml,mt),(ml,mt+ph),(50,50,50),2)
cv2.line(img,(ml,mt+ph),(ml+pw,mt+ph),(50,50,50),2)

bins     = 40
max_val  = 0.6
all_hist,   edges = np.histogram(all_heights,   bins=bins, range=(0, max_val))
entry_hist, _     = np.histogram(entry_heights, bins=bins, range=(0, max_val))
max_count = max(max(all_hist), max(entry_hist)) if all_hist.size else 1
bar_w = pw // bins

# 画全ROI柱（蓝色，半透明底）
for i, count in enumerate(all_hist):
    x1b = ml + i*bar_w
    x2b = x1b + bar_w - 1
    bh2 = int((count/max_count)*ph)
    y1b = mt + ph - bh2
    cv2.rectangle(img,(x1b,y1b),(x2b,mt+ph),(180,140,80),-1)

# 画ENTRY区柱（绿色，叠加）
for i, count in enumerate(entry_hist):
    x1b = ml + i*bar_w
    x2b = x1b + bar_w - 1
    bh2 = int((count/max_count)*ph)
    y1b = mt + ph - bh2
    cv2.rectangle(img,(x1b,y1b),(x2b,mt+ph),(60,160,60),-1)
    cv2.rectangle(img,(x1b,y1b),(x2b,mt+ph),(180,220,180),1)

# X轴刻度
for i in range(0, bins+1, bins//8):
    x = ml + i*bar_w
    val = i/bins * max_val
    cv2.line(img,(x,mt+ph),(x,mt+ph+6),(50,50,50),1)
    cv2.putText(img,f"{val:.2f}",(x-18,mt+ph+22),cv2.FONT_HERSHEY_SIMPLEX,0.42,(50,50,50),1)
cv2.putText(img,"box height / frame height",(ml+pw//2-100,mt+ph+50),
            cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,80,80),1)

# Y轴刻度
for i in range(0,6):
    y   = mt + ph - int(i/5*ph)
    val = int(i/5*max_count)
    cv2.line(img,(ml-5,y),(ml,y),(50,50,50),1)
    cv2.putText(img,str(val),(ml-48,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.42,(50,50,50),1)

# 百分位参考线（绿色区）
if entry_heights:
    arr = np.array(entry_heights)
    for pct, label, color in [
        (25, "P25", (200,160,0)),
        (50, "P50", (0,180,0)),
        (75, "P75", (0,120,200)),
    ]:
        val = np.percentile(arr, pct)
        lx  = ml + int((val/max_val)*pw)
        cv2.line(img,(lx,mt),(lx,mt+ph),color,1)
        cv2.putText(img,f"{label}:{val:.2f}",(lx+3,mt+20+(pct//25-1)*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)

cv2.imwrite(OUTPUT_CHART, img)
print(f"\nChart saved: {OUTPUT_CHART}")
print("\nHow to use this chart:")
print("  Look at the GREEN bars (ENTRY zone only).")
print("  Find where the green bars show a natural gap or valley.")
print("  Set ENTRY_MIN_HEIGHT_RATIO to that value.")
print("  P75 of ENTRY zone is usually a good starting threshold.")
