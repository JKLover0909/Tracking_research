"""
Person tracking with a fixed ROI filter.

This script reuses the improved TrackletManager/Re-ID logic from tracking.py,
but only sends person detections inside the configured ROI to the global
tracklet manager.
"""

import collections
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from tracking import (
    CFG,
    TrackletManager,
    _get_runtime_config,
    assign_pose_to_detections,
    draw_detection,
    draw_hud,
    extract_crop_appearance,
)


ROI = {
    "x1": 956,
    "y1": 73,
    "x2": 2472,
    "y2": 1276,
}

CFG.update({
    "output_path": r"/mnt/nvme/opt/Tracking_research/assets/video_results/result_v2_roi_1_2026-04-23_142309.mp4",
    "show_roi": True,
    # center: track person when bbox center is inside ROI.
    # overlap: track person when bbox has any overlap with ROI.
    "roi_filter_mode": "center",
})


def clip_roi_to_frame(roi, frame_width, frame_height):
    return {
        "x1": int(np.clip(roi["x1"], 0, frame_width - 1)),
        "y1": int(np.clip(roi["y1"], 0, frame_height - 1)),
        "x2": int(np.clip(roi["x2"], 1, frame_width)),
        "y2": int(np.clip(roi["y2"], 1, frame_height)),
    }


def box_center(box):
    x1, y1, x2, y2 = box
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def box_overlaps_roi(box, roi):
    x1, y1, x2, y2 = box
    return (
        max(float(x1), roi["x1"]) < min(float(x2), roi["x2"])
        and max(float(y1), roi["y1"]) < min(float(y2), roi["y2"])
    )


def is_person_in_roi(box, roi, mode="center"):
    if mode == "overlap":
        return box_overlaps_roi(box, roi)

    cx, cy = box_center(box)
    return roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]


def draw_roi(frame, roi, roi_count):
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    color = (0, 255, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"ROI det: {roi_count}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_label = max(y1 - 8, th + 8)
    cv2.rectangle(frame, (x1, y_label - th - 8), (x1 + tw + 8, y_label), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_label - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def main():
    print("=" * 65)
    print("  UNIFORM PERSON TRACKING - ROI FILTER")
    print("=" * 65)
    print(
        "[OK] ROI original coords: "
        f"({ROI['x1']}, {ROI['y1']}) -> ({ROI['x2']}, {ROI['y2']})"
    )

    runtime = _get_runtime_config()
    torch_device = runtime["torch_device"]
    yolo_device = runtime["yolo_device"]
    use_half = runtime["use_half"]
    d = runtime["diagnostics"]

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
        raise RuntimeError(f"Khong mo duoc: {CFG['video_path']}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    roi = clip_roi_to_frame(ROI, width, height)

    if roi["x2"] <= roi["x1"] or roi["y2"] <= roi["y1"]:
        raise ValueError(f"ROI khong hop le sau khi clip theo frame {width}x{height}: {roi}")

    out = None
    if CFG["save_video"]:
        os.makedirs(os.path.dirname(CFG["output_path"]), exist_ok=True)
        out = cv2.VideoWriter(
            CFG["output_path"],
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (width, height),
        )

    print(f"[OK] Video: {width}x{height} @ {fps_src}fps -> {CFG['output_path']}")
    print(f"[OK] ROI clipped: ({roi['x1']}, {roi['y1']}) -> ({roi['x2']}, {roi['y2']})")
    print(f"[OK] ROI mode: {CFG['roi_filter_mode']}")

    show_window = CFG["show_window"]
    if show_window and not os.environ.get("DISPLAY", "").strip():
        show_window = False
        print("[WARN] DISPLAY trong - tat show_window.")
    elif show_window:
        try:
            cv2.namedWindow("Tracking ROI", cv2.WINDOW_NORMAL)
        except cv2.error:
            show_window = False
    print("-" * 65)

    manager = TrackletManager()
    frame_idx = 0
    fps_counter = collections.deque(maxlen=30)

    try:
        while cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            det_results = detect_model.track(
                source=frame,
                persist=True,
                tracker=CFG["tracker_cfg"],
                classes=[0],
                conf=CFG["conf_thresh"],
                iou=CFG["iou_thresh"],
                imgsz=CFG["imgsz"],
                half=use_half,
                device=yolo_device,
                verbose=False,
            )

            if pose_model and frame_idx % CFG["reid_interval"] == 0:
                pose_results = pose_model(
                    frame,
                    classes=[0],
                    conf=CFG["conf_thresh"],
                    imgsz=CFG["imgsz"],
                    half=use_half,
                    device=yolo_device,
                    verbose=False,
                )
            else:
                pose_results = None

            detections = []
            total_persons = 0
            skipped_outside_roi = 0
            if det_results[0].boxes is not None and det_results[0].boxes.id is not None:
                boxes = det_results[0].boxes.xyxy.cpu().numpy()
                track_ids = det_results[0].boxes.id.int().cpu().numpy()
                confs = det_results[0].boxes.conf.cpu().numpy()
                total_persons = len(boxes)
                has_embed = hasattr(det_results[0].boxes, "feat")
                feats = det_results[0].boxes.feat.cpu().numpy() if has_embed else None

                pose_map = {}
                if (
                    pose_results
                    and pose_results[0].keypoints is not None
                    and pose_results[0].boxes is not None
                ):
                    kp_data = pose_results[0].keypoints.data.cpu().numpy()
                    kp_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                    pose_map = assign_pose_to_detections(boxes, kp_boxes, kp_data)

                for i, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                    if not is_person_in_roi(box, roi, CFG["roi_filter_mode"]):
                        skipped_outside_roi += 1
                        continue

                    app_feat = feats[i] if feats is not None else None
                    if app_feat is None and CFG["use_crop_appearance"]:
                        app_feat = extract_crop_appearance(frame, box)
                    detections.append({
                        "box": box.tolist(),
                        "yolo_id": int(tid),
                        "conf": float(conf),
                        "appearance": app_feat,
                        "kpts": pose_map.get(i),
                    })

            detections = manager.update(detections, frame_idx)

            active_gids = set()
            for det in detections:
                gid = det.get("global_id")
                if gid is None:
                    continue
                active_gids.add(gid)
                draw_detection(
                    frame,
                    det["box"],
                    gid,
                    det["conf"],
                    kpts=det.get("kpts"),
                    debug=CFG["show_debug_info"],
                )

            if CFG["show_debug_info"]:
                for det in detections:
                    tgt = det.get("_tentative_target")
                    if tgt is not None and det.get("global_id") is None:
                        draw_detection(
                            frame,
                            det["box"],
                            tgt,
                            det["conf"],
                            tentative=True,
                            debug=False,
                        )

                for gid, trk in manager.tracklets.items():
                    if gid in active_gids or not trk.is_active or not trk.get_last_box():
                        continue
                    if is_person_in_roi(trk.get_last_box(), roi, CFG["roi_filter_mode"]):
                        draw_detection(frame, trk.get_last_box(), trk.id, 0.0, lost=True)

            t1 = time.perf_counter()
            fps_counter.append(1.0 / max(t1 - t0, 1e-6))
            cur_fps = float(np.mean(fps_counter))
            n_active = sum(1 for t in manager.tracklets.values() if t.is_active)
            n_lost = sum(
                1 for t in manager.tracklets.values() if t.is_active and t.lost_frames > 0
            )
            n_tent = sum(1 for dct in detections if dct.get("_tentative_target") is not None)

            if CFG["show_roi"]:
                draw_roi(frame, roi, len(detections))
            draw_hud(frame, frame_idx, cur_fps, n_active, n_lost, n_tent)

            if out is not None:
                out.write(frame)
            if show_window:
                cv2.imshow("Tracking ROI", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("\n[STOP] Dung boi nguoi dung.")
                    break

            if frame_idx % 100 == 0:
                print(
                    f"  Frame {frame_idx:5d} | FPS {cur_fps:5.1f} | "
                    f"ROI {len(detections)}/{total_persons} | "
                    f"Skipped {skipped_outside_roi} | Active {n_active} | "
                    f"Lost {n_lost} | Tentative {n_tent}"
                )

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if show_window:
            cv2.destroyAllWindows()
        status = f"da luu: {CFG['output_path']}" if CFG["save_video"] else "khong luu"
        print(f"\n[DONE] Video {status}")
        print(f"       Tong frames: {frame_idx}")


if __name__ == "__main__":
    main()
