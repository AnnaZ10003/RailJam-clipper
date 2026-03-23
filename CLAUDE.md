# RailJam Clipper — 项目背景

滑雪公园（Rail Jam）赛事视频自动剪辑工具。输入一段完整赛事录像，自动检测并剪出每位主滑手过道具的片段。

## 技术栈

| 组件 | 用途 |
|------|------|
| YOLO11n | 人体检测 |
| DeepSort (`max_age=45, n_init=3`) | 多目标追踪 |
| torchreid OSNet x0.25 | Re-ID 外观特征提取 |
| DBSCAN | 外观聚类（目前被 K-means 结果覆盖） |
| K-means (k=2) on avg_h | 主滑手 vs 背景人物二分 |

## 文件结构

```
reid_preview.py      主脚本（追踪 → Re-ID → 预览视频 → 截图表格 → 剪辑视频）
analyze_sizes.py     辅助：分析检测框大小分布
videos/input.mp4     测试视频1（有背景人物，较长）
videos/input2.mp4    测试视频2（无背景人物，约20s，入画方向与input不同）
output/              所有输出文件（预览视频、截图表格、剪辑视频）
venv/                Python 3.11 虚拟环境
```

## 运行方式

```bash
# 默认跑 input.mp4
venv/Scripts/python reid_preview.py

# 指定视频
venv/Scripts/python reid_preview.py --video videos/input2.mp4

# 指定输出目录
venv/Scripts/python reid_preview.py --video videos/input2.mp4 --out-dir output
```

输出文件以视频名为前缀，统一写入 `--out-dir`：
- `{stem}_preview.mp4`   — 带追踪标注的预览视频
- `{stem}_sheet.jpg`     — 各 track 代表帧截图表格
- `{stem}_clip001_t{tid}.mp4` ... — 主滑手剪辑片段

## 识别逻辑（Pass 1）

1. 每隔1帧跑 YOLO 检测（`frame_idx % 2 == 0`）
2. ROI 过滤（默认：横向0~82%，纵向15~100%）
3. dominant 框过滤：当场内有大框时，过小的框被丢弃
4. DeepSort 追踪，`n_init=3`，`MIN_HIT_STREAK=2`
5. 速度过滤：近20帧平均位移 < 3px 视为静止背景
6. 片段记录：消失超过 `MAX_GAP_SECONDS=3s` 或视频结束时封存，持续时间 ≥ `MIN_CLIP_SECONDS=0.8s` 才保留

## 主滑手识别逻辑（Re-ID）

1. 对有片段的 track 用 OSNet 提取外观特征
2. DBSCAN 聚类（结果仅用于参考，实际被 K-means 覆盖）
3. 过滤幽灵框：`dark_ratio < 0.06` 的 track 剔除
4. K-means (k=2) 对 `avg_h`（裁剪图平均高度）二分：
   - 若两组高度比 > 0.72 → 无背景人物，全部视为主滑手
   - 否则：高度大的组 = 主滑手，高度小的组 = 背景
5. `main_cluster = 2`，背景 `cluster_id = 0`

**已知 bug 修复历史：**
- `if not cnt: main_cluster = -999` 会在 DBSCAN 全噪声时覆盖 K-means 结果 → 已改为仅打印提示

## 剪辑输出（Pass 3）

- 只输出 `cluster_map[tid] == main_cluster` 的片段
- 每条片段向前回溯 `PRE_ROLL_SECONDS=1.5s`（入画前余量），弥补 DeepSort confirm 延迟
- 原始画质，无标注

## 关键参数速查

```python
MIN_CLIP_SECONDS   = 0.8    # 片段最短时长（太短 = 误检）
MAX_GAP_SECONDS    = 3      # track 断开多久算结束
PRE_ROLL_SECONDS   = 1.5    # 剪辑开始前的回溯余量
CONF_THRESHOLD     = 0.5    # YOLO 置信度阈值
MIN_SPEED_PX       = 3.0    # 低于此速度视为静止
DOMINANT_HEIGHT_RATIO = 0.18  # 触发 dominant 模式的最小框高（相对画面）
SMALL_TARGET_RATIO    = 0.55  # dominant 模式下，小于最大框 55% 的框被丢弃
GHOST_THRESHOLD    = 0.06   # dark_ratio 低于此值 = 幽灵框
h_ratio threshold  = 0.72   # K-means 两组高度比，高于此值 = 无背景人物
```

## 下一步任务

1. **[进行中]** 验证 input2.mp4 泛化能力，修复剪辑开始时刻（pre-roll）和漏检问题
2. **[待做]** 输出剪辑视频（PASS 3 已实现，待验证效果）
3. **[待做]** 同一人身份归组：同一场景里同一滑手被 DeepSort 分配了多个 track ID（每次离画重入都会新建 ID），需要用 Re-ID 特征把它们归并，合成一条连贯剪辑
