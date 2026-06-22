"""
=============================================================================
UNIFORM PERSON TRACKING PIPELINE — v2 (Improved Re-ID Stability)
=============================================================================
Cải tiến so với v1:

[FIX-1] Tentative Re-ID Buffer (giải quyết vấn đề #1)
    - Khi một detection muốn nhận lại một lost-ID, KHÔNG commit ngay.
    - Tích lũy vote qua N frame liên tiếp (TENTATIVE_FRAMES).
    - Chỉ commit khi: (a) vote liên tục >= TENTATIVE_FRAMES, VÀ
                     (b) score trung bình >= ngưỡng cao hơn (recover_confirm_score).
    - Trong lúc chờ: detection được gán ID tạm thời (negative), không hiển thị
      ID cũ ra màn hình → tránh nhảy ID nhìn thấy.

[FIX-2] Profile Freeze khi lost (giải quyết vấn đề #2)
    - Khi tracklet đang lost, tuyệt đối không cập nhật appearance/biometric/shape
      profile từ bất kỳ detection nào (kể cả detection đang được tentative).
    - Profile chỉ được cập nhật sau khi re-ID đã được CONFIRMED.
    - Bổ sung "clean_profile" = snapshot profile tại lúc trước khi lost,
      dùng riêng cho việc re-ID matching thay vì profile đang dần bị ô nhiễm.

[FIX-3] Size-Aware Re-ID Gate (giải quyết vấn đề #2 một phần)
    - Tính size_ratio = area_det / median_area_tracklet.
    - Nếu size_ratio > SIZE_EXPAND_THRESH (e.g. 1.8): detection đang to hơn hẳn
      so với profile → đây không phải người đang xuất hiện lại từ bị che, mà
      là người khác → raise min_score cho trường hợp này.
    - Nếu size_ratio < SIZE_SHRINK_THRESH (e.g. 0.4): box nhỏ dị thường →
      không dùng để update profile.

[FIX-4] Velocity Direction Check
    - Với tracklet có >= MIN_VEL_FRAMES frame lịch sử, tính velocity vector.
    - Nếu detection ở phía ngược chiều di chuyển dự báo → penalty score.
    - Tránh track "nhảy" sang người đi ngược chiều trong vùng giao nhau.

[FIX-5] Appearance Staleness Decay
    - Appearance embedding cũ dần mất độ tin cậy theo thời gian.
    - Nếu tracklet lost lâu hơn APPEARANCE_STALE_FRAMES thì giảm trọng số
      appearance trong fusion score (nó có thể đã thay đổi góc nhìn).

[FIX-6] Đồng phục giống nhau → tăng trọng số spatial cue
    - Khi tất cả appearance score trong cost matrix đều cao và gần nhau
      (uniform crowd), appearance không còn phân biệt được → tự động giảm
      w_appearance và tăng w_center + w_shape để tránh nhầm.

=============================================================================
"""

import cv2
import numpy as np
import time
import collections
import warnings
import os
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH TOÀN CỤC
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Đường dẫn
    "video_path":   r"/mnt/nvme/opt/Tracking_research/assets/ps_Long/1_2026-04-23_142309.mp4",
    "output_path":  r"/mnt/nvme/opt/Tracking_research/assets/video_results/result_v2_1_2026-04-23_142309.mp4",
    "yolo_model":   "yolo26s.pt",
    "pose_model":   None,
    "tracker_cfg":  r"/mnt/nvme/opt/Tracking_research/bytetrack.yaml",
    "device":       "auto",
    "require_gpu":  True,
    "allow_cpu_fallback": False,

    # Detection
    "conf_thresh":  0.60,
    "iou_thresh":   0.55,
    "imgsz":        1152,
    "use_half":     True,

    # Tracklet lifecycle
    "tracklet_window":      95,
    "tracklet_max_lost":    180,
    "reid_interval":        4,
    "recover_lost_window":  180,

    # ── [FIX-1] Tentative Re-ID ───────────────────────────────────────────
    # Số frame liên tiếp cần vote trước khi xác nhận re-ID
    "tentative_frames":         5,
    # Score tối thiểu trong mỗi frame tentative (cao hơn recover_min_score)
    "recover_min_score":        0.52,
    # Score trung bình cần đạt để confirm sau N frame tentative
    "recover_confirm_score":    0.58,
    # Score tối thiểu để BẮT ĐẦU tentative (filter rác sớm)
    "tentative_entry_score":    0.48,

    # ── [FIX-3] Size-Aware Gate ───────────────────────────────────────────
    # Box detection lớn hơn median tracklet bao nhiêu lần thì tăng ngưỡng
    "size_expand_thresh":       1.7,
    # Penalty khi size ratio vượt ngưỡng mở rộng (cộng vào min_score)
    "size_expand_penalty":      0.10,
    # Box nhỏ hơn median bao nhiêu lần thì không update profile
    "size_shrink_thresh":       0.45,

    # ── [FIX-4] Velocity Direction Check ─────────────────────────────────
    # Cần tối thiểu bao nhiêu frame để tính velocity
    "min_vel_frames":           5,
    # Penalty score khi detection ở sai hướng velocity (0 = tắt)
    "velocity_direction_penalty": 0.06,
    # Góc tối đa (radian) giữa velocity và hướng đến detection để penalty
    "velocity_angle_thresh":    1.65,   # ~94 độ

    # ── [FIX-5] Appearance Staleness ─────────────────────────────────────
    # Số frame lost sau đó appearance weight bắt đầu giảm
    "appearance_stale_frames":  20,
    # Frame lost tối đa để appearance còn có ý nghĩa
    "appearance_max_stale":     90,

    # Spatial-Temporal constraints
    "max_pixel_speed":              90,
    "max_lost_pixel_speed":         55,
    "max_center_dist_ratio":        1.35,
    "max_lost_center_dist_ratio":   2.35,
    "min_iou_for_direct":           0.35,
    "crowded_iou_thresh":           0.08,
    "crowded_center_dist_ratio":    0.70,
    "crowded_min_score":            0.57,
    "crowded_min_margin":           0.12,
    "yolo_continuity_bonus":        0.03,

    # Trọng số fusion cơ bản
    "w_iou":            0.18,
    "w_center":         0.28,   # tăng lên vì spatial quan trọng hơn với đồng phục
    "w_appearance":     0.24,   # giảm xuống một chút
    "w_biometric":      0.10,
    "w_shape":          0.20,   # tăng shape
    "w_gait":           0.0,

    # Ngưỡng gán ID
    "match_threshold":                  0.42,
    "active_min_score":                 0.46,
    "lost_min_score":                   0.54,   # tăng lên để tránh recover sai
    "same_tracker_lost_min_score":      0.46,
    "lost_recover_delay_on_new_tracker": 6,
    "lost_in_crowd_min_score":          0.70,   # tăng lên
    "min_appearance_for_recover":       0.42,
    "min_assignment_margin":            0.09,
    "new_track_min_conf":               0.62,
    "use_crop_appearance":              True,

    # Hiển thị
    "show_debug_info":  True,
    "show_skeleton":    True,
    "show_window":      False,
    "save_video":       True,
}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: BODY BIOMETRICS
# ─────────────────────────────────────────────────────────────────────────────
class BodyBiometrics:
    @staticmethod
    def from_box(box):
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        aspect_ratio = h / w
        torso_ratio  = (y2 - y1) * 0.45 / h
        return np.array([aspect_ratio, torso_ratio], dtype=np.float32)

    @staticmethod
    def from_keypoints(kpts):
        if kpts is None or len(kpts) < 17:
            return None
        def pt(i):
            return kpts[i, :2]
        try:
            shoulder_w = np.linalg.norm(pt(5) - pt(6))
            hip_w      = np.linalg.norm(pt(11) - pt(12))
            torso_h    = np.linalg.norm((pt(5)+pt(6))/2 - (pt(11)+pt(12))/2)
            leg_h      = np.linalg.norm((pt(11)+pt(12))/2 - (pt(15)+pt(16))/2)
            body_h     = torso_h + leg_h
            if body_h < 1e-3:
                return None
            return np.array([
                shoulder_w / (hip_w + 1e-5),
                torso_h / (body_h + 1e-5),
                leg_h / (body_h + 1e-5),
                shoulder_w / (body_h + 1e-5),
                hip_w / (body_h + 1e-5),
            ], dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def similarity(feat_a, feat_b):
        if feat_a is None or feat_b is None:
            return 0.5
        norm_a = np.linalg.norm(feat_a)
        norm_b = np.linalg.norm(feat_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.5
        return float(np.dot(feat_a, feat_b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: GAIT SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────
class GaitSignature:
    def __init__(self, window=30):
        self.window = window
        self.history = collections.deque(maxlen=window)

    def update(self, kpts):
        if kpts is not None and len(kpts) >= 17:
            angles = self._compute_joint_angles(kpts)
            if angles is not None:
                self.history.append(angles)

    def get_signature(self):
        if len(self.history) < 10:
            return None
        arr = np.stack(list(self.history), axis=0)
        return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])

    def get_mean_angles(self):
        if len(self.history) < 5:
            return None
        arr = np.stack(list(self.history), axis=0)
        return arr.mean(axis=0)

    def _compute_joint_angles(self, kpts):
        try:
            def angle(a, b, c):
                ba = a - b; bc = c - b
                cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-5)
                return np.arccos(np.clip(cos_a, -1, 1))
            p = kpts[:, :2]
            return np.array([
                angle(p[5], p[7],   p[9]),
                angle(p[6], p[8],  p[10]),
                angle(p[11],p[13], p[15]),
                angle(p[12],p[14], p[16]),
                angle(p[5], p[11], p[13]),
                angle(p[6], p[12], p[14]),
            ], dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def similarity(sig_a, sig_b):
        if sig_a is None or sig_b is None:
            return 0.5
        norm_a = np.linalg.norm(sig_a)
        norm_b = np.linalg.norm(sig_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.5
        return float(np.dot(sig_a, sig_b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# Appearance embedding từ crop
# ─────────────────────────────────────────────────────────────────────────────
def extract_crop_appearance(frame, box):
    if frame is None or box is None:
        return None
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w_frame - 1, x1))
    y1 = max(0, min(h_frame - 1, y1))
    x2 = max(0, min(w_frame, x2))
    y2 = max(0, min(h_frame, y2))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (48, 96), interpolation=cv2.INTER_AREA)
    crop = crop[:, 5:43]
    regions = (crop, crop[:40, :], crop[40:72, :], crop[72:, :])
    feats = []
    for region in regions:
        if region.size == 0:
            continue
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
        feats.append(hist.astype(np.float32))
        lab_float = lab.reshape(-1, 3).astype(np.float32) / 255.0
        moments = np.concatenate([lab_float.mean(axis=0), lab_float.std(axis=0)])
        feats.append(moments.astype(np.float32))
    if not feats:
        return None
    feat = np.concatenate(feats).astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-6:
        return None
    return feat / norm


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: TRACKLET
# ─────────────────────────────────────────────────────────────────────────────
class Tracklet:
    """
    Lịch sử của một track. Có thêm:
    - clean_profile: snapshot profile TRƯỚC khi lost (không bị ô nhiễm)
    - tentative_votes: dict {tracker_id -> deque of scores} (FIX-1)
    """

    def __init__(self, track_id, tracker_id, box, frame_idx, appearance_feat=None, kpts=None):
        self.id          = track_id
        self.tracker_id  = tracker_id
        self.boxes       = collections.deque(maxlen=CFG["tracklet_window"])
        self.appearances = collections.deque(maxlen=CFG["tracklet_window"])
        self.frame_idxs  = collections.deque(maxlen=CFG["tracklet_window"])
        self.tracker_ids = collections.deque(maxlen=CFG["tracklet_window"])
        self.lost_frames = 0
        self.is_active   = True
        self.hit_streak  = 0

        # Biometric & Gait
        self.biometric_feat = None
        self.gait           = GaitSignature()

        # ── [FIX-2] Clean profile snapshot ───────────────────────────────
        # Được đóng băng tại thời điểm tracklet bắt đầu lost.
        self.clean_appearances: list = []       # list of np.array
        self.clean_biometric   = None
        self._profile_frozen   = False
        self._frames_since_lost_started = 0

        # ── [FIX-1] Tentative buffer ──────────────────────────────────────
        # gid_candidate -> {'scores': deque, 'det_ids': deque (yolo_id)}
        self.tentative_candidates: dict[int, dict] = {}

        self.update(tracker_id, box, frame_idx, appearance_feat, kpts)

    def freeze_profile(self):
        """
        Lưu snapshot profile sạch ngay khi lost bắt đầu.
        Gọi 1 lần duy nhất khi lost_frames chuyển từ 0 → 1.
        """
        if self._profile_frozen:
            return
        if len(self.appearances) > 0:
            self.clean_appearances = list(self.appearances)
        self.clean_biometric = (
            self.biometric_feat.copy()
            if self.biometric_feat is not None else None
        )
        self._profile_frozen = True

    def thaw_profile(self):
        """Xóa snapshot sau khi re-ID được confirmed."""
        self.clean_appearances = []
        self.clean_biometric   = None
        self._profile_frozen   = False

    def update(self, tracker_id, box, frame_idx, appearance_feat=None,
               kpts=None, update_profile=True):
        self.tracker_id = tracker_id
        self.boxes.append(box)
        self.frame_idxs.append(frame_idx)
        self.tracker_ids.append(tracker_id)
        self.lost_frames = 0
        self.is_active   = True
        self.hit_streak += 1
        self._profile_frozen = False
        self.tentative_candidates.clear()   # committed → clear all tentative

        if update_profile and appearance_feat is not None:
            feat = np.asarray(appearance_feat, dtype=np.float32)
            feat_norm = np.linalg.norm(feat)
            if feat_norm > 1e-6:
                self.appearances.append(feat / feat_norm)

        if update_profile:
            bio_kpts = BodyBiometrics.from_keypoints(kpts) if kpts is not None else None
            bio_box  = BodyBiometrics.from_box(box)
            bio      = bio_kpts if bio_kpts is not None else bio_box
            if bio is not None:
                if self.biometric_feat is None or self.biometric_feat.shape != bio.shape:
                    self.biometric_feat = bio
                else:
                    self.biometric_feat = 0.85 * self.biometric_feat + 0.15 * bio
            self.gait.update(kpts)

    def get_mean_appearance(self, use_clean=False):
        """
        use_clean=True: dùng clean snapshot khi đang lost (FIX-2).
        """
        pool = self.clean_appearances if (use_clean and self._profile_frozen) else list(self.appearances)
        if len(pool) == 0:
            return None
        feat = np.mean(pool, axis=0)
        feat_norm = np.linalg.norm(feat)
        if feat_norm > 1e-6:
            feat = feat / feat_norm
        return feat

    def get_last_box(self):
        return self.boxes[-1] if self.boxes else None

    def get_last_frame(self):
        return self.frame_idxs[-1] if self.frame_idxs else -999

    def get_median_box(self, max_samples=30):
        if not self.boxes:
            return None
        samples = list(self.boxes)[-max_samples:]
        return np.median(np.asarray(samples, dtype=np.float32), axis=0).tolist()

    def get_median_area(self, max_samples=30):
        if not self.boxes:
            return None
        areas = []
        for box in list(self.boxes)[-max_samples:]:
            w = max(box[2] - box[0], 1.0)
            h = max(box[3] - box[1], 1.0)
            areas.append(w * h)
        return float(np.median(areas)) if areas else None

    def get_predicted_box(self, frame_idx):
        last_box = self.get_last_box()
        if last_box is None or len(self.boxes) < 2 or len(self.frame_idxs) < 2:
            return last_box
        lookback = min(6, len(self.boxes))
        old_box   = np.asarray(self.boxes[-lookback], dtype=np.float32)
        new_box   = np.asarray(self.boxes[-1], dtype=np.float32)
        old_frame = self.frame_idxs[-lookback]
        new_frame = self.frame_idxs[-1]
        dt = max(new_frame - old_frame, 1)
        delta_frames = max(frame_idx - new_frame, 0)
        velocity = (new_box - old_box) / dt
        pred = np.asarray(last_box, dtype=np.float32) + velocity * min(delta_frames, 12)
        return pred.tolist()

    def get_velocity_vector(self):
        """
        [FIX-4] Tính velocity 2D (cx, cy) dựa trên lịch sử center.
        Trả về None nếu không đủ data.
        """
        n = CFG["min_vel_frames"]
        if len(self.boxes) < n or len(self.frame_idxs) < n:
            return None
        boxes_arr = np.asarray(list(self.boxes), dtype=np.float32)
        frames_arr = np.asarray(list(self.frame_idxs), dtype=np.float32)
        centers = np.stack([
            (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2,
            (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2,
        ], axis=1)
        # Dùng lookback để tính velocity ổn định hơn
        lookback = min(n, len(centers))
        dt = frames_arr[-1] - frames_arr[-lookback]
        if dt < 1:
            return None
        vel = (centers[-1] - centers[-lookback]) / dt
        return vel  # (vx, vy) pixels/frame

    def mark_lost(self):
        if self.lost_frames == 0:
            self.freeze_profile()       # [FIX-2] đóng băng profile ngay khi bắt đầu lost
        self.lost_frames += 1
        self.hit_streak = 0
        if self.lost_frames > CFG["tracklet_max_lost"]:
            self.is_active = False

    # ── [FIX-1] Tentative helpers ─────────────────────────────────────────
    def add_tentative_vote(self, candidate_gid: int, score: float, yolo_id: int):
        if candidate_gid not in self.tentative_candidates:
            self.tentative_candidates[candidate_gid] = {
                "scores":   collections.deque(maxlen=CFG["tentative_frames"] + 2),
                "yolo_ids": collections.deque(maxlen=CFG["tentative_frames"] + 2),
            }
        buf = self.tentative_candidates[candidate_gid]
        buf["scores"].append(score)
        buf["yolo_ids"].append(yolo_id)

    def check_tentative_confirmed(self, candidate_gid: int):
        """
        Trả về True nếu candidate_gid đã đủ N vote liên tiếp với score trung bình đủ cao.
        """
        buf = self.tentative_candidates.get(candidate_gid)
        if buf is None:
            return False
        if len(buf["scores"]) < CFG["tentative_frames"]:
            return False
        recent = list(buf["scores"])[-CFG["tentative_frames"]:]
        return float(np.mean(recent)) >= CFG["recover_confirm_score"]

    def clear_tentative(self, except_gid=None):
        """Xóa tất cả tentative ngoại trừ gid được chỉ định."""
        if except_gid is None:
            self.tentative_candidates.clear()
        else:
            self.tentative_candidates = {
                k: v for k, v in self.tentative_candidates.items()
                if k == except_gid
            }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: TRACKLET MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class TrackletManager:
    def __init__(self):
        self.tracklets: dict[int, Tracklet]      = {}
        self.tracker_to_global: dict[int, int]   = {}
        self.next_global_id = 1

        # [FIX-1] Tentative pool: yolo_id → (target_global_id, score_buffer)
        # Theo dõi ở manager level để cross-check dễ hơn.
        # Format: { yolo_id: {'target_gid': int, 'scores': deque} }
        self._tentative_pool: dict[int, dict] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Main update loop
    # ─────────────────────────────────────────────────────────────────────
    def update(self, detections, frame_idx):
        for det in detections:
            det["global_id"] = None
            det.pop("_tentative_target", None)
            det.pop("_recovery_failed", None)
            det.pop("_crowded", None)

        if detections:
            crowded_flags = self._crowded_detection_flags(detections)
            for det, crowded in zip(detections, crowded_flags):
                det["_crowded"] = crowded

        updated = set()
        used_det_idxs = set()

        # First pass: keep detections whose tracker ID is already mapped.
        # This preserves normal BoT-SORT continuity and lets low-confidence
        # detections rescue an existing global ID.
        for det_idx, det in enumerate(detections):
            yolo_id = det["yolo_id"]
            gid = self.tracker_to_global.get(yolo_id)
            if gid is None:
                continue
            trk = self.tracklets.get(gid)
            if trk is None or not trk.is_active:
                self.tracker_to_global.pop(yolo_id, None)
                continue
            self._assign_detection(trk, det, frame_idx)
            det["global_id"] = gid
            updated.add(gid)
            used_det_idxs.add(det_idx)

        # Second pass: a new tracker ID may still be an old person after an
        # occlusion, scale jump, or short detector dropout. Match those
        # detections back to lost global tracklets before creating new IDs.
        unmatched_idxs = [i for i in range(len(detections)) if i not in used_det_idxs]
        lost_gids = [
            gid for gid, trk in self.tracklets.items()
            if (gid not in updated and trk.is_active and trk.lost_frames > 0
                    and trk.lost_frames <= CFG["recover_lost_window"])
        ]
        if unmatched_idxs and lost_gids:
            score_matrix = np.zeros((len(lost_gids), len(unmatched_idxs)), dtype=np.float32)
            for row, gid in enumerate(lost_gids):
                trk = self.tracklets[gid]
                for col, det_idx in enumerate(unmatched_idxs):
                    det = detections[det_idx]
                    if not self._passes_motion_gate(det, trk, frame_idx):
                        continue
                    app_score = self._appearance_similarity(det, trk)
                    if (app_score is not None
                            and app_score < CFG["min_appearance_for_recover"]):
                        continue
                    score = self.compute_fusion_score(det, trk, frame_idx)
                    min_score = self._compute_min_score(det, trk, gid, score, None, row)
                    if score >= min_score:
                        score_matrix[row, col] = score

            if np.any(score_matrix > 0):
                rows, cols = linear_sum_assignment(-score_matrix)
                for row, col in zip(rows, cols):
                    score = float(score_matrix[row, col])
                    if score <= 0:
                        continue
                    det_idx = unmatched_idxs[col]
                    if det_idx in used_det_idxs:
                        continue
                    det = detections[det_idx]
                    gid = lost_gids[row]
                    trk = self.tracklets[gid]
                    margin = (CFG["crowded_min_margin"] if det.get("_crowded", False)
                              else CFG["min_assignment_margin"])
                    ambiguous = self._is_ambiguous_assignment(
                        score_matrix, row, col, margin=margin
                    )
                    if ambiguous and score < CFG["lost_in_crowd_min_score"]:
                        det["_tentative_target"] = gid
                        used_det_idxs.add(det_idx)
                        continue

                    confirmed = self._process_tentative(trk, det, gid, score, frame_idx)
                    if confirmed:
                        self._assign_detection(trk, det, frame_idx)
                        det["global_id"] = gid
                        updated.add(gid)
                        used_det_idxs.add(det_idx)
                    else:
                        pool_entry = self._tentative_pool.get(det["yolo_id"])
                        scores = pool_entry["scores"] if pool_entry is not None else []
                        if len(scores) < CFG["tentative_frames"]:
                            det["_tentative_target"] = gid
                            used_det_idxs.add(det_idx)
                        else:
                            det["_recovery_failed"] = True
                            trk.clear_tentative(except_gid=None)
                            self._tentative_pool.pop(det["yolo_id"], None)

        # Third pass: only create a new global ID when confidence is high and
        # there is no plausible lost tracklet waiting for this detection.
        recoverable_gids = [
            gid for gid, trk in self.tracklets.items()
            if (gid not in updated and trk.is_active and trk.lost_frames > 0
                    and trk.lost_frames <= CFG["recover_lost_window"])
        ]
        for det_idx, det in enumerate(detections):
            if det_idx in used_det_idxs or det.get("global_id") is not None:
                continue
            if det["conf"] < CFG["new_track_min_conf"]:
                continue
            if (not det.get("_recovery_failed", False)
                    and self._has_recoverable_lost_candidate(det, recoverable_gids, frame_idx)):
                det["_tentative_target"] = recoverable_gids[0] if recoverable_gids else None
                continue
            yolo_id = det["yolo_id"]
            gid = self.next_global_id
            self.next_global_id += 1
            self.tracker_to_global[yolo_id] = gid
            self.tracklets[gid] = Tracklet(
                gid, yolo_id, det["box"], frame_idx,
                det.get("appearance"), det.get("kpts"),
            )
            det["global_id"] = gid
            updated.add(gid)

        for gid, trk in self.tracklets.items():
            if gid not in updated:
                trk.mark_lost()
        return detections

    # ─────────────────────────────────────────────────────────────────────
    # [FIX-1] Tentative processing
    # ─────────────────────────────────────────────────────────────────────
    def _process_tentative(self, tracklet: Tracklet, det: dict, gid: int,
                            score: float, frame_idx: int) -> bool:
        """
        Thêm vote vào buffer tentative.
        Trả về True nếu đã đủ vote để confirm (có thể commit ngay).
        Trả về False nếu chưa đủ.
        """
        # Score quá thấp → không đủ điều kiện bắt đầu tentative
        if score < CFG["tentative_entry_score"]:
            return False

        yolo_id = det["yolo_id"]
        tracklet.add_tentative_vote(gid, score, yolo_id)

        # Cũng track ở manager level
        pool_entry = self._tentative_pool.get(yolo_id)
        if pool_entry is None or pool_entry.get("target_gid") != gid:
            self._tentative_pool[yolo_id] = {
                "target_gid": gid,
                "scores":     collections.deque(maxlen=CFG["tentative_frames"] + 2),
                "frame_start": frame_idx,
            }
        self._tentative_pool[yolo_id]["scores"].append(score)

        return tracklet.check_tentative_confirmed(gid)

    # ─────────────────────────────────────────────────────────────────────
    # Min score calculation (FIX-3 size-aware)
    # ─────────────────────────────────────────────────────────────────────
    def _compute_min_score(self, det, tracklet: Tracklet, gid, score, app_matrix, row):
        min_score = (
            CFG["lost_min_score"] if tracklet.lost_frames > 0
            else CFG["active_min_score"]
        )
        mapped_gid = self.tracker_to_global.get(det["yolo_id"])
        if mapped_gid == gid and tracklet.lost_frames > 0:
            min_score = min(min_score, CFG["same_tracker_lost_min_score"])
        if det.get("_crowded", False):
            min_score = max(min_score, CFG["crowded_min_score"])
        if tracklet.lost_frames > 0 and mapped_gid != gid:
            if det.get("_crowded", False):
                min_score = max(min_score, CFG["lost_in_crowd_min_score"])
            else:
                min_score = max(min_score, CFG["recover_min_score"])

        # ── [FIX-3] Size-aware penalty ─────────────────────────────────
        if tracklet.lost_frames > 0:
            median_area = tracklet.get_median_area()
            if median_area is not None and median_area > 0:
                det_area = self._box_area(det["box"])
                size_ratio = det_area / median_area
                if size_ratio > CFG["size_expand_thresh"]:
                    # Box detection to hơn hẳn → tăng ngưỡng
                    min_score = min(1.0, min_score + CFG["size_expand_penalty"])

        return min_score

    # ─────────────────────────────────────────────────────────────────────
    # Assignment helpers
    # ─────────────────────────────────────────────────────────────────────
    def _assign_detection(self, tracklet: Tracklet, det: dict, frame_idx: int):
        old_tracker_id      = tracklet.tracker_id
        incoming_tracker_id = det["yolo_id"]
        previous_gid = self.tracker_to_global.get(incoming_tracker_id)
        if previous_gid is not None and previous_gid != tracklet.id:
            self.tracker_to_global.pop(incoming_tracker_id, None)

        # [FIX-2] Profile freeze check
        median_area = tracklet.get_median_area()
        det_area    = self._box_area(det["box"])
        is_partial  = (
            median_area is not None
            and det_area < CFG["size_shrink_thresh"] * median_area
        )
        update_profile = not det.get("_crowded", False) and not is_partial

        # Nếu vừa recovered, thaw profile
        if tracklet._profile_frozen:
            tracklet.thaw_profile()

        tracklet.update(
            incoming_tracker_id, det["box"], frame_idx,
            det.get("appearance"), det.get("kpts"),
            update_profile=update_profile,
        )
        if (old_tracker_id != incoming_tracker_id
                and self.tracker_to_global.get(old_tracker_id) == tracklet.id):
            self.tracker_to_global.pop(old_tracker_id, None)
        self.tracker_to_global[incoming_tracker_id] = tracklet.id

    # ─────────────────────────────────────────────────────────────────────
    # Spatial helpers
    # ─────────────────────────────────────────────────────────────────────
    def _box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _box_center_dist(self, box_a, box_b):
        ca = self._box_center(box_a)
        cb = self._box_center(box_b)
        return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))

    def _box_diag(self, box):
        return float(np.hypot(max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)))

    def _box_area(self, box):
        return float(max(box[2] - box[0], 1.0) * max(box[3] - box[1], 1.0))

    def _compute_iou(self, box_a, box_b):
        xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
        area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _crowded_detection_flags(self, detections):
        flags = [False] * len(detections)
        for i, det_a in enumerate(detections):
            box_a = det_a["box"]; diag_a = self._box_diag(box_a)
            for j in range(i + 1, len(detections)):
                box_b = detections[j]["box"]
                iou   = self._compute_iou(box_a, box_b)
                cd    = self._box_center_dist(box_a, box_b)
                near  = CFG["crowded_center_dist_ratio"] * min(diag_a, self._box_diag(box_b))
                if iou >= CFG["crowded_iou_thresh"] or cd <= near:
                    flags[i] = flags[j] = True
        return flags

    def _has_nearby_track_candidate(self, det, candidate_ids, frame_idx):
        for gid in candidate_ids:
            trk = self.tracklets.get(gid)
            if trk is None or not trk.is_active:
                continue
            if self._passes_motion_gate(det, trk, frame_idx):
                return True
        return False

    def _has_recoverable_lost_candidate(self, det, candidate_ids, frame_idx):
        for gid in candidate_ids:
            trk = self.tracklets.get(gid)
            if (trk is None or not trk.is_active
                    or trk.lost_frames <= 0
                    or trk.lost_frames > CFG["recover_lost_window"]):
                continue
            if not self._passes_motion_gate(det, trk, frame_idx):
                continue
            score = self.compute_fusion_score(det, trk, frame_idx)
            if score >= CFG["same_tracker_lost_min_score"]:
                return True
        return False

    def _is_ambiguous_assignment(self, score_matrix, row, col, margin=None):
        if margin is None:
            margin = CFG["min_assignment_margin"]
        if margin <= 0:
            return False
        score = float(score_matrix[row, col])
        row_alt = np.delete(score_matrix[row], col)
        col_alt = np.delete(score_matrix[:, col], row)
        row_best = float(np.max(row_alt)) if row_alt.size else 0.0
        col_best = float(np.max(col_alt)) if col_alt.size else 0.0
        return (score - row_best < margin) or (score - col_best < margin)

    def _reference_box_for_match(self, tracklet: Tracklet, frame_idx: int):
        ref_box = tracklet.get_predicted_box(frame_idx)
        if ref_box is None:
            return None
        if ref_box[2] <= ref_box[0] or ref_box[3] <= ref_box[1]:
            return tracklet.get_median_box() or tracklet.get_last_box()
        if tracklet.lost_frames > 0:
            median_box  = tracklet.get_median_box()
            median_area = tracklet.get_median_area()
            last_box    = tracklet.get_last_box()
            if (median_box is not None and median_area is not None
                    and last_box is not None
                    and self._box_area(last_box) < 0.65 * median_area):
                return median_box
        return ref_box

    def _passes_motion_gate(self, det, tracklet: Tracklet, frame_idx: int):
        ref_box = self._reference_box_for_match(tracklet, frame_idx)
        if ref_box is None:
            return True
        delta_frames = max(frame_idx - tracklet.get_last_frame(), 1)
        dist = self._box_center_dist(ref_box, det["box"])
        speed = dist / delta_frames
        max_speed = (CFG["max_lost_pixel_speed"] if tracklet.lost_frames > 0
                     else CFG["max_pixel_speed"])
        if speed > max_speed:
            return False
        dist_ratio  = (CFG["max_lost_center_dist_ratio"] if tracklet.lost_frames > 0
                       else CFG["max_center_dist_ratio"])
        lost_growth = min(np.sqrt(delta_frames), 3.0) if tracklet.lost_frames > 0 else 1.0
        last_box    = tracklet.get_last_box()
        ref_diag    = max(
            self._box_diag(last_box),
            self._box_diag(tracklet.get_median_box() or last_box),
        )
        return dist <= dist_ratio * ref_diag * lost_growth

    # ─────────────────────────────────────────────────────────────────────
    # Similarity helpers
    # ─────────────────────────────────────────────────────────────────────
    def _appearance_similarity(self, det, tracklet: Tracklet):
        """[FIX-2] Dùng clean profile khi tracklet đang lost."""
        det_app = det.get("appearance")
        trk_app = tracklet.get_mean_appearance(use_clean=tracklet._profile_frozen)
        if det_app is None or trk_app is None:
            return None
        det_app = np.asarray(det_app, dtype=np.float32)
        trk_app = np.asarray(trk_app, dtype=np.float32)
        if det_app.shape != trk_app.shape:
            return None
        nd = np.linalg.norm(det_app); nt = np.linalg.norm(trk_app)
        if nd < 1e-6 or nt < 1e-6:
            return None
        s = float(np.dot(det_app, trk_app) / (nd * nt))
        if np.min(det_app) >= 0.0 and np.min(trk_app) >= 0.0:
            return float(np.clip(s, 0.0, 1.0))
        return float(np.clip((s + 1.0) / 2.0, 0.0, 1.0))

    def _shape_similarity(self, box_a, box_b):
        wa = max(box_a[2] - box_a[0], 1.0); ha = max(box_a[3] - box_a[1], 1.0)
        wb = max(box_b[2] - box_b[0], 1.0); hb = max(box_b[3] - box_b[1], 1.0)
        hs = np.exp(-abs(np.log(ha / hb)))
        ws = np.exp(-abs(np.log(wa / wb)))
        ap = np.exp(-abs(np.log((ha/wa) / (hb/wb))))
        return float(np.clip(0.45*hs + 0.25*ws + 0.30*ap, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # Fusion score (FIX-4, FIX-5, FIX-6)
    # ─────────────────────────────────────────────────────────────────────
    def compute_fusion_score(self, det, tracklet: Tracklet, frame_idx: int) -> float:
        last_box = tracklet.get_last_box()
        if last_box is None:
            return 0.0
        ref_box = self._reference_box_for_match(tracklet, frame_idx) or last_box

        # 1. IoU
        iou_score = self._compute_iou(det["box"], last_box)

        # 2. Center distance
        center_dist  = self._box_center_dist(det["box"], ref_box)
        dist_ratio   = (CFG["max_lost_center_dist_ratio"] if tracklet.lost_frames > 0
                        else CFG["max_center_dist_ratio"])
        delta_frames = max(frame_idx - tracklet.get_last_frame(), 1)
        lost_growth  = min(np.sqrt(delta_frames), 3.0) if tracklet.lost_frames > 0 else 1.0
        ref_diag     = max(
            self._box_diag(last_box),
            self._box_diag(tracklet.get_median_box() or last_box),
        )
        max_dist     = max(dist_ratio * ref_diag * lost_growth, 1.0)
        center_score = max(0.0, 1.0 - (center_dist / max_dist))
        if iou_score >= CFG["min_iou_for_direct"]:
            center_score = max(center_score, 0.9)

        # 3. Appearance (với staleness decay, FIX-5)
        app_score = self._appearance_similarity(det, tracklet)
        if app_score is not None and tracklet.lost_frames > CFG["appearance_stale_frames"]:
            stale_ratio = min(
                tracklet.lost_frames / CFG["appearance_max_stale"], 1.0
            )
            # Kéo app_score về 0.5 (neutral) theo mức độ stale
            app_score = app_score * (1 - stale_ratio) + 0.5 * stale_ratio

        # 4. Biometric
        det_bio  = BodyBiometrics.from_keypoints(det.get("kpts"))
        if det_bio is None:
            det_bio = BodyBiometrics.from_box(det["box"])
        # [FIX-2] Dùng clean biometric khi lost
        trk_bio = (tracklet.clean_biometric
                   if tracklet._profile_frozen and tracklet.clean_biometric is not None
                   else tracklet.biometric_feat)
        bio_score = (BodyBiometrics.similarity(det_bio, trk_bio) + 1) / 2

        # 5. Shape — dùng median box để tránh ảnh hưởng bởi box nhỏ khi bị che
        shape_ref  = tracklet.get_median_box() if tracklet.lost_frames > 0 else last_box
        shape_score = self._shape_similarity(det["box"], shape_ref or last_box)

        # [FIX-3] Size-aware penalty trong score (không chỉ trong min_score)
        if tracklet.lost_frames > 0:
            median_area = tracklet.get_median_area()
            if median_area is not None and median_area > 0:
                size_ratio = self._box_area(det["box"]) / median_area
                if size_ratio > CFG["size_expand_thresh"]:
                    shape_score *= 0.7   # penalty shape score nếu size bất thường

        # 6. Gait
        gait_score = None
        if det.get("kpts") is not None:
            mean_angles = tracklet.gait.get_mean_angles()
            if mean_angles is not None:
                det_angles = tracklet.gait._compute_joint_angles(det["kpts"])
                if det_angles is not None:
                    gait_score = (GaitSignature.similarity(det_angles, mean_angles) + 1) / 2

        # ── [FIX-4] Velocity direction penalty ────────────────────────────
        velocity_penalty = 0.0
        if CFG["velocity_direction_penalty"] > 0:
            vel = tracklet.get_velocity_vector()
            if vel is not None:
                vel_mag = float(np.linalg.norm(vel))
                if vel_mag > 1.0:   # chỉ tính khi có chuyển động rõ
                    pred_cx, pred_cy = self._box_center(ref_box)
                    det_cx, det_cy   = self._box_center(det["box"])
                    det_dir = np.array([det_cx - pred_cx, det_cy - pred_cy], dtype=np.float32)
                    det_mag = float(np.linalg.norm(det_dir))
                    if det_mag > 1.0:
                        cos_angle = float(np.dot(vel, det_dir) / (vel_mag * det_mag + 1e-6))
                        angle     = float(np.arccos(np.clip(cos_angle, -1, 1)))
                        if angle > CFG["velocity_angle_thresh"]:
                            velocity_penalty = CFG["velocity_direction_penalty"]

        # ── [FIX-6] Adaptive weight khi đồng phục ─────────────────────────
        # Nếu appearance score thấp/mờ (vì đồng phục giống nhau) → tăng spatial
        w_iou  = CFG["w_iou"]
        w_ctr  = CFG["w_center"]
        w_app  = CFG["w_appearance"]
        w_bio  = CFG["w_biometric"]
        w_shp  = CFG["w_shape"]
        w_gait = CFG["w_gait"]

        if app_score is not None and app_score < 0.55 and tracklet.lost_frames > 0:
            # Appearance không phân biệt được → giảm xuống, boost spatial
            boost = min(0.10, (0.55 - app_score) * 0.5)
            w_app  = max(0.05, w_app - boost)
            w_ctr  = w_ctr + boost * 0.6
            w_shp  = w_shp + boost * 0.4

        # Weighted fusion
        terms = [
            (w_iou,  iou_score),
            (w_ctr,  center_score),
            (w_bio,  bio_score),
            (w_shp,  shape_score),
        ]
        if app_score is not None:
            terms.append((w_app, app_score))
        if gait_score is not None:
            terms.append((w_gait, gait_score))

        total_w = sum(wt for wt, _ in terms)
        score   = sum(wt * val for wt, val in terms) / max(total_w, 1e-6)

        # Áp dụng velocity penalty
        score = max(0.0, score - velocity_penalty)

        return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    (255, 80,  80),  (80, 200,  80),  (80,  80, 255),
    (255, 200,  0),  (200,  0, 255),  (0,  200, 255),
    (255, 128,  0),  (128, 255,  0),  (0,  128, 255),
    (255,  0, 128),  (0,  255, 128),  (128,  0, 255),
]
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

def get_color(track_id):
    return PALETTE[int(track_id) % len(PALETTE)]

def draw_detection(frame, box, track_id, conf, lost=False, tentative=False,
                   kpts=None, debug=False):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = get_color(track_id)

    thickness = 1 if (lost or tentative) else 2
    line_type  = cv2.LINE_4 if tentative else cv2.LINE_AA
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, lineType=line_type)

    if tentative:
        label = f"ID:{track_id}?"
    else:
        label = f"ID:{track_id}" + (f" (c:{conf:.2f})" if debug else "")
    if lost:
        label += " [L]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

    if kpts is not None and CFG["show_skeleton"]:
        h_frame, w_frame = frame.shape[:2]
        kp      = kpts[:, :2].astype(int)
        conf_kp = kpts[:, 2] if kpts.shape[1] > 2 else np.ones(len(kpts))
        for i, (x, y) in enumerate(kp):
            if conf_kp[i] > 0.3:
                cv2.circle(frame, (x, y), 3, color, -1)
        for a, b in COCO_SKELETON:
            if conf_kp[a] > 0.3 and conf_kp[b] > 0.3:
                cv2.line(frame, tuple(kp[a]), tuple(kp[b]), color, 1, cv2.LINE_AA)

def draw_hud(frame, frame_idx, fps_val, n_active, n_lost, n_tentative):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, 108), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    lines = [
        f"Frame:     {frame_idx}",
        f"FPS:       {fps_val:.1f}",
        f"Active:    {n_active}",
        f"Lost:      {n_lost}",
        f"Tentative: {n_tentative}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 240, 200), 1, cv2.LINE_AA)


def assign_pose_to_detections(det_boxes, pose_boxes, pose_keypoints, iou_threshold=0.15):
    pose_map = {}
    if len(det_boxes) == 0 or len(pose_boxes) == 0:
        return pose_map
    used_pose = set()
    for det_idx, det_box in enumerate(det_boxes):
        best_pose_idx = -1; best_iou = 0.0
        for pose_idx, pose_box in enumerate(pose_boxes):
            if pose_idx in used_pose:
                continue
            xa = max(det_box[0], pose_box[0]); ya = max(det_box[1], pose_box[1])
            xb = min(det_box[2], pose_box[2]); yb = min(det_box[3], pose_box[3])
            inter = max(0, xb-xa) * max(0, yb-ya)
            da = max(det_box[2]-det_box[0],1) * max(det_box[3]-det_box[1],1)
            pa = max(pose_box[2]-pose_box[0],1) * max(pose_box[3]-pose_box[1],1)
            iou = inter / (da + pa - inter + 1e-6)
            if iou > best_iou:
                best_iou = iou; best_pose_idx = pose_idx
        if best_pose_idx >= 0 and best_iou >= iou_threshold:
            pose_map[det_idx] = pose_keypoints[best_pose_idx]
            used_pose.add(best_pose_idx)
    return pose_map


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def _get_runtime_config():
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
    requested  = str(CFG.get("device", "auto")).strip().lower()
    torch_device = "cpu"; yolo_device = "cpu"; use_half = False
    diagnostics = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": False,
        "gpu_name": None,
        "failure_reason": None,
    }
    def can_use_cuda():
        try:
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda:0")
            return True
        except Exception as exc:
            diagnostics["failure_reason"] = str(exc)
            return False

    if requested in {"auto", "cuda", "cuda:0", "0"} and can_use_cuda():
        torch_device = "cuda:0"; yolo_device = 0
        use_half = bool(CFG["use_half"])
        diagnostics["cuda_available"] = True
        diagnostics["gpu_name"] = torch.cuda.get_device_name(0)
        torch.backends.cudnn.benchmark = True
    elif requested in {"cuda", "cuda:0", "0"} and not CFG.get("allow_cpu_fallback", False):
        raise RuntimeError(f"CUDA yeu cau nhung khong san sang. reason={diagnostics['failure_reason']}")
    elif CFG.get("require_gpu", False) and not CFG.get("allow_cpu_fallback", False) and requested != "cpu":
        raise RuntimeError(f"GPU khong san sang. reason={diagnostics['failure_reason']}")

    return {"torch_device": torch_device, "yolo_device": yolo_device,
            "use_half": use_half, "diagnostics": diagnostics}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  UNIFORM PERSON TRACKING — v2 (Improved Re-ID Stability)")
    print("=" * 65)

    runtime      = _get_runtime_config()
    torch_device = runtime["torch_device"]
    yolo_device  = runtime["yolo_device"]
    use_half     = runtime["use_half"]
    d            = runtime["diagnostics"]

    print(f"[OK] Torch: {d['torch_version']}  CUDA: {d['torch_cuda_build']}")
    print(f"[OK] Device: {torch_device}  FP16: {use_half}")
    if d["gpu_name"]:
        print(f"[OK] GPU: {d['gpu_name']}")

    detect_model = YOLO(CFG["yolo_model"])
    detect_model.to(torch_device)
    pose_model = None
    if CFG["pose_model"]:
        pose_model = YOLO(CFG["pose_model"])
        pose_model.to(torch_device)

    cap = cv2.VideoCapture(CFG["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được: {CFG['video_path']}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = None
    if CFG["save_video"]:
        out = cv2.VideoWriter(
            CFG["output_path"], cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W, H)
        )
    print(f"[OK] Video: {W}x{H} @ {fps_src}fps → {CFG['output_path']}")

    show_window = CFG["show_window"]
    if show_window and not os.environ.get("DISPLAY", "").strip():
        show_window = False
        print("[WARN] DISPLAY trống — tắt show_window.")
    elif show_window:
        try:
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        except cv2.error:
            show_window = False
    print("-" * 65)

    manager     = TrackletManager()
    frame_idx   = 0
    fps_counter = collections.deque(maxlen=30)

    try:
        while cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # ── Detect + Track ────────────────────────────────────────────
            det_results = detect_model.track(
                source=frame, persist=True, tracker=CFG["tracker_cfg"],
                classes=[0], conf=CFG["conf_thresh"], iou=CFG["iou_thresh"],
                imgsz=CFG["imgsz"], half=use_half, device=yolo_device, verbose=False,
            )

            # ── Pose (optional) ───────────────────────────────────────────
            pose_map = {}
            if pose_model and frame_idx % CFG["reid_interval"] == 0:
                pose_results = pose_model(
                    frame, classes=[0], conf=CFG["conf_thresh"],
                    imgsz=CFG["imgsz"], half=use_half, device=yolo_device, verbose=False,
                )
            else:
                pose_results = None

            # ── Build detections ──────────────────────────────────────────
            detections = []
            if (det_results[0].boxes is not None and
                    det_results[0].boxes.id is not None):
                boxes     = det_results[0].boxes.xyxy.cpu().numpy()
                track_ids = det_results[0].boxes.id.int().cpu().numpy()
                confs     = det_results[0].boxes.conf.cpu().numpy()
                has_embed = hasattr(det_results[0].boxes, "feat")
                feats     = det_results[0].boxes.feat.cpu().numpy() if has_embed else None

                if (pose_results and pose_results[0].keypoints is not None
                        and pose_results[0].boxes is not None):
                    kp_data  = pose_results[0].keypoints.data.cpu().numpy()
                    kp_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                    pose_map = assign_pose_to_detections(boxes, kp_boxes, kp_data)

                for i, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                    app_feat = feats[i] if feats is not None else None
                    if app_feat is None and CFG["use_crop_appearance"]:
                        app_feat = extract_crop_appearance(frame, box)
                    detections.append({
                        "box":        box.tolist(),
                        "yolo_id":    int(tid),
                        "conf":       float(conf),
                        "appearance": app_feat,
                        "kpts":       pose_map.get(i),
                    })

            # ── Tracklet update ───────────────────────────────────────────
            detections = manager.update(detections, frame_idx)

            # ── Draw confirmed detections ─────────────────────────────────
            active_gids = set()
            for det in detections:
                gid = det.get("global_id")
                if gid is None:
                    continue
                active_gids.add(gid)
                draw_detection(
                    frame, det["box"], gid, det["conf"],
                    kpts=det.get("kpts"), debug=CFG["show_debug_info"],
                )

            # ── Draw tentative detections (dashed box) ────────────────────
            if CFG["show_debug_info"]:
                for det in detections:
                    tgt = det.get("_tentative_target")
                    if tgt is not None and det.get("global_id") is None:
                        draw_detection(
                            frame, det["box"], tgt, det["conf"],
                            tentative=True, debug=False,
                        )

            # ── Draw lost tracklets ───────────────────────────────────────
            if CFG["show_debug_info"]:
                for gid, trk in manager.tracklets.items():
                    if gid not in active_gids and trk.is_active and trk.get_last_box():
                        draw_detection(frame, trk.get_last_box(), trk.id, 0.0, lost=True)

            # ── HUD ───────────────────────────────────────────────────────
            t1 = time.perf_counter()
            fps_counter.append(1.0 / max(t1 - t0, 1e-6))
            cur_fps   = float(np.mean(fps_counter))
            n_active  = sum(1 for t in manager.tracklets.values() if t.is_active)
            n_lost    = sum(1 for t in manager.tracklets.values()
                           if t.is_active and t.lost_frames > 0)
            n_tent    = sum(1 for d in detections if d.get("_tentative_target") is not None)
            draw_hud(frame, frame_idx, cur_fps, n_active, n_lost, n_tent)

            if out is not None:
                out.write(frame)
            if show_window:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    print("\n[STOP] Dừng bởi người dùng.")
                    break

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx:5d} | FPS {cur_fps:5.1f} | "
                      f"Active {n_active} | Lost {n_lost} | Tentative {n_tent}")

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if show_window:
            cv2.destroyAllWindows()
        status = f"đã lưu: {CFG['output_path']}" if CFG["save_video"] else "không lưu (save_video=False)"
        print(f"\n[DONE] Video {status}")
        print(f"       Tổng frames: {frame_idx}")


if __name__ == "__main__":
    main()
