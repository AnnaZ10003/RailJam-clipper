"""
GBT Track Classifier for Rail Jam Auto-Clipper
Replaces heuristic K-means + threshold rescue with a learned classifier.
"""
import json
import os
import numpy as np
from collections import defaultdict

# ── Feature Extraction ──────────────────────────────────────────────────

FEATURE_NAMES = [
    "h_normalized",          # 0  per_track_h / H
    "net_vx_normalized",     # 1  track_net_vx / W
    "signed_speed",          # 2  track_signed_speed (px/frame)
    "dominant_dir",          # 3  {-1, 0, +1}
    "mean_cx_norm",          # 4  mean(positions.x) / W
    "mean_cy_norm",          # 5  mean(positions.y) / H
    "trajectory_length",     # 6  sum(inter-frame dist) / diag
    "peak_dx",               # 7  max 5-frame avg |dx| / W
    "duration_seconds",      # 8  track duration in seconds
    "max_concurrent",        # 9  track_max_concurrent
    "reverse_neighbor_count",# 10 opposite-dir spatially-close overlapping tracks
    "hit_entry",             # 11 bool
    "hit_exit",              # 12 bool
    "is_slow",               # 13 bool
    "dark_ratio",            # 14 per_track_dark
    "crop_quality_mean",     # 15 mean crop quality (h of valid crops)
    "sim_to_main_centroid",  # 16 cos_sim to K-means main centroid
    "sim_to_nearest_main",   # 17 max cos_sim to any main track
    "stability_jump_frac",   # 18 fraction of large jumps
    "num_positions",         # 19 len(track_positions)
]

NUM_FEATURES = len(FEATURE_NAMES)


def compute_track_features(tid, *, W, H, fps,
                           track_positions, track_segments, track_net_vx,
                           track_signed_speed, track_dominant_dir,
                           per_track_h, per_track_dark,
                           track_hit_entry, track_hit_exit,
                           track_max_concurrent,
                           combined_feat_by_tid,
                           main_tids_set,
                           main_centroid=None,
                           track_crops=None):
    """Extract a feature dict for one track from existing data structures."""
    diag = (W**2 + H**2) ** 0.5
    positions = track_positions.get(tid, [])
    n_pos = len(positions)

    # Basic features
    h_norm = per_track_h.get(tid, 0) / H if H > 0 else 0
    vx_norm = track_net_vx.get(tid, 0) / W if W > 0 else 0
    speed = track_signed_speed.get(tid, 0)
    dom = track_dominant_dir.get(tid, 0)

    # Spatial position
    if n_pos > 0:
        mean_cx = np.mean([p[0] for p in positions]) / W if W > 0 else 0
        mean_cy = np.mean([p[1] for p in positions]) / H if H > 0 else 0
    else:
        mean_cx, mean_cy = 0.5, 0.5

    # Trajectory length (normalized)
    traj_len = 0.0
    if n_pos >= 2:
        for i in range(n_pos - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            traj_len += (dx**2 + dy**2) ** 0.5
        traj_len /= diag if diag > 0 else 1

    # Peak dx: max 5-frame average |dx|
    peak_dx = 0.0
    if n_pos >= 6:
        dxs = [abs(positions[i+1][0] - positions[i][0]) for i in range(n_pos - 1)]
        window = 5
        for i in range(len(dxs) - window + 1):
            avg = np.mean(dxs[i:i+window])
            if avg > peak_dx:
                peak_dx = avg
    elif n_pos >= 2:
        peak_dx = max(abs(positions[i+1][0] - positions[i][0]) for i in range(n_pos - 1))
    peak_dx /= W if W > 0 else 1

    # Duration
    segs = track_segments.get(tid, [])
    if segs:
        s_min = min(s for s, _ in segs)
        e_max = max(e for _, e in segs)
        duration = (e_max - s_min) / fps if fps > 0 else 0
    else:
        duration = 0

    # Max concurrent
    max_conc = track_max_concurrent.get(tid, 0)

    # Reverse neighbor count
    rnc = _reverse_neighbor_count(tid, track_positions=track_positions,
                                  track_segments=track_segments,
                                  track_dominant_dir=track_dominant_dir,
                                  fps=fps)

    # Entry/exit/slow
    hit_entry = 1.0 if tid in track_hit_entry else 0.0
    hit_exit = 1.0 if tid in track_hit_exit else 0.0
    is_slow = 1.0 if abs(track_net_vx.get(tid, 0)) < W * 0.04 and n_pos >= 10 else 0.0

    # Dark ratio
    dark = per_track_dark.get(tid, 0)

    # Crop quality (mean height of valid crops)
    crop_q = 0.0
    if track_crops:
        crops = track_crops.get(tid, [])
        valid = [c for c in crops if c is not None and c.size > 0]
        if valid:
            crop_q = np.mean([c.shape[0] for c in valid]) / H if H > 0 else 0

    # Re-ID similarity to main group
    sim_centroid = 0.0
    sim_nearest = 0.0
    feat = combined_feat_by_tid.get(tid)
    if feat is not None:
        feat_n = feat / (np.linalg.norm(feat) + 1e-8)
        if main_centroid is not None:
            mc_n = main_centroid / (np.linalg.norm(main_centroid) + 1e-8)
            sim_centroid = float(np.dot(feat_n, mc_n))
        for m_tid in main_tids_set:
            m_feat = combined_feat_by_tid.get(m_tid)
            if m_feat is not None:
                m_n = m_feat / (np.linalg.norm(m_feat) + 1e-8)
                s = float(np.dot(feat_n, m_n))
                if s > sim_nearest:
                    sim_nearest = s

    # Stability jump fraction
    jump_frac = 0.0
    if n_pos >= 5:
        jumps = [
            ((positions[i+1][0] - positions[i][0])**2 +
             (positions[i+1][1] - positions[i][1])**2) ** 0.5 / diag
            for i in range(n_pos - 1)
        ]
        jump_frac = sum(1 for j in jumps if j > 0.20) / len(jumps)

    return {
        "h_normalized": h_norm,
        "net_vx_normalized": vx_norm,
        "signed_speed": speed,
        "dominant_dir": dom,
        "mean_cx_norm": mean_cx,
        "mean_cy_norm": mean_cy,
        "trajectory_length": traj_len,
        "peak_dx": peak_dx,
        "duration_seconds": duration,
        "max_concurrent": max_conc,
        "reverse_neighbor_count": rnc,
        "hit_entry": hit_entry,
        "hit_exit": hit_exit,
        "is_slow": is_slow,
        "dark_ratio": dark,
        "crop_quality_mean": crop_q,
        "sim_to_main_centroid": sim_centroid,
        "sim_to_nearest_main": sim_nearest,
        "stability_jump_frac": jump_frac,
        "num_positions": n_pos,
    }


def _reverse_neighbor_count(tid, *, track_positions, track_segments,
                            track_dominant_dir, fps):
    """Count tracks with opposite direction, spatially close (<200px),
    and temporally overlapping."""
    dom = track_dominant_dir.get(tid, 0)
    if dom == 0:
        return 0
    segs = track_segments.get(tid, [])
    if not segs:
        return 0
    t_start = min(s for s, _ in segs)
    t_end = max(e for _, e in segs)
    positions = track_positions.get(tid, [])
    if not positions:
        return 0
    cx = np.mean([p[0] for p in positions])

    count = 0
    for other_tid, other_segs in track_segments.items():
        if other_tid == tid:
            continue
        o_dom = track_dominant_dir.get(other_tid, 0)
        if o_dom == 0 or o_dom == dom:
            continue
        o_start = min(s for s, _ in other_segs)
        o_end = max(e for _, e in other_segs)
        if o_start > t_end or o_end < t_start:
            continue
        o_pos = track_positions.get(other_tid, [])
        if o_pos:
            o_cx = np.mean([p[0] for p in o_pos])
            if abs(cx - o_cx) < 200:
                count += 1
    return count


def features_dict_to_array(feat_dict):
    """Convert feature dict to numpy array in canonical order."""
    return np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=float)


def compute_all_features(tids, **kwargs):
    """Compute features for all tids, return {tid: feature_dict}."""
    result = {}
    for tid in tids:
        result[tid] = compute_track_features(tid, **kwargs)
    return result


# ── GBT Classifier ──────────────────────────────────────────────────────

class TrackClassifier:
    """Gradient Boosted Trees classifier for main rider vs background."""

    def __init__(self):
        self.model = None
        self.feature_names = FEATURE_NAMES

    def fit(self, X, y, sample_weight=None):
        """Train GBT on feature matrix X (n_samples, 20) and labels y (0/1)."""
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, features_by_tid):
        """Predict P(main_rider) for each track.
        Args:
            features_by_tid: {tid: feature_dict}
        Returns:
            {tid: float probability}
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first or load a saved model.")
        tids = list(features_by_tid.keys())
        if not tids:
            return {}
        X = np.array([features_dict_to_array(features_by_tid[tid]) for tid in tids])
        probs = self.model.predict_proba(X)[:, 1]
        return {tid: float(p) for tid, p in zip(tids, probs)}

    def save(self, path):
        """Save model to disk."""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path):
        """Load model from disk. Returns None if file doesn't exist."""
        if not os.path.exists(path):
            return None
        import joblib
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.feature_names = data.get("feature_names", FEATURE_NAMES)
        return obj


# ── Label I/O ────────────────────────────────────────────────────────────

def bootstrap_labels_from_kmeans(main_tids, bg_drag_tids, bg_same_tids,
                                 ghost_tids, unstable_tids):
    """Generate initial labels from K-means clustering results.
    Returns: {tid: {"label": 0|1, "confidence": float, "source": str}}
    """
    labels = {}
    for tid in main_tids:
        labels[str(tid)] = {"label": 1, "confidence": 0.8, "source": "kmeans_main"}
    for tid in bg_drag_tids:
        labels[str(tid)] = {"label": 0, "confidence": 0.9, "source": "kmeans_drag"}
    for tid in bg_same_tids:
        labels[str(tid)] = {"label": 0, "confidence": 0.7, "source": "kmeans_walker"}
    for tid in ghost_tids:
        labels[str(tid)] = {"label": 0, "confidence": 0.95, "source": "ghost"}
    for tid in unstable_tids:
        labels[str(tid)] = {"label": 0, "confidence": 0.95, "source": "unstable"}
    return labels


def save_labels(labels, features_by_tid, path):
    """Save labels and features to JSON for user correction and retraining."""
    data = {
        "version": 1,
        "tracks": {}
    }
    for tid, label_info in labels.items():
        feat = features_by_tid.get(tid) or features_by_tid.get(str(tid))
        data["tracks"][str(tid)] = {
            **label_info,
            "features": feat if feat else {},
        }
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_labels(path):
    """Load labels from JSON. Returns {tid: {"label": int, "confidence": float}}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    for tid, info in data.get("tracks", {}).items():
        labels[tid] = {
            "label": info["label"],
            "confidence": info.get("confidence", 1.0),
        }
    return labels


def save_label_template(all_tids, main_tids_set, features_by_tid, path):
    """Generate a user-editable label template JSON.
    Users can change 'label' field: 1=main rider, 0=background."""
    template = {
        "_instructions": "Edit 'label' field: 1=main rider, 0=background. "
                         "Set 'corrected'=true for tracks you manually verified.",
        "tracks": {}
    }
    for tid in sorted(all_tids, key=lambda x: int(x)):
        feat = features_by_tid.get(tid, {})
        template["tracks"][str(tid)] = {
            "label": 1 if tid in main_tids_set else 0,
            "corrected": False,
            "h_normalized": round(feat.get("h_normalized", 0), 3),
            "dominant_dir": feat.get("dominant_dir", 0),
            "sim_to_nearest_main": round(feat.get("sim_to_nearest_main", 0), 3),
            "duration_seconds": round(feat.get("duration_seconds", 0), 1),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    return path


def train_from_label_files(labels_dir, features_dir=None):
    """Aggregate labels from multiple video label files and train a model.
    Args:
        labels_dir: directory containing {video}_labels.json files
        features_dir: same directory (features are embedded in label files)
    Returns:
        Trained TrackClassifier or None if insufficient data.
    """
    all_X, all_y, all_w = [], [], []

    label_files = [f for f in os.listdir(labels_dir) if f.endswith("_labels.json")]
    if not label_files:
        return None

    for fname in label_files:
        fpath = os.path.join(labels_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tid, info in data.get("tracks", {}).items():
            feat = info.get("features", {})
            if not feat or len(feat) < NUM_FEATURES:
                continue
            vec = [feat.get(name, 0) for name in FEATURE_NAMES]
            all_X.append(vec)
            all_y.append(info["label"])
            # User-corrected labels get highest weight
            conf = info.get("confidence", 0.8)
            if info.get("corrected", False):
                conf = 1.0
            all_w.append(conf)

    if len(all_X) < 10:
        print(f"  [Classifier] 训练样本不足 ({len(all_X)} < 10)，跳过训练")
        return None

    # Need at least 2 classes
    unique_labels = set(all_y)
    if len(unique_labels) < 2:
        print(f"  [Classifier] 仅有单一类别，跳过训练")
        return None

    X = np.array(all_X, dtype=float)
    y = np.array(all_y, dtype=int)
    w = np.array(all_w, dtype=float)

    clf = TrackClassifier()
    clf.fit(X, y, sample_weight=w)
    n_main = sum(y == 1)
    n_bg = sum(y == 0)
    print(f"  [Classifier] 训练完成: {len(all_X)} 样本 ({n_main} main + {n_bg} bg) "
          f"from {len(label_files)} video(s)")
    return clf
