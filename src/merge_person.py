"""
Create a video where every detected person is duplicated once.

Default input priority:
  1. ./output.mp4
  2. ./assets/output.mp4
  3. ./assets/output_2.mp4

Example:
  python src/merge_person.py
  python src/merge_person.py --input assets/output_2.mp4 --output assets/output_doubled_people.mp4
  python src/merge_person.py --model yolo26s-seg.engine
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CANDIDATES = (
    ROOT / "output.mp4",
    ROOT / "assets" / "output.mp4",
    ROOT / "assets" / "output_2.mp4",
)
DEFAULT_MODEL_CANDIDATES = (
    ROOT / "yolo26s-seg.engine",
    ROOT / "yolo26s-seg.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate each tracked person in a video using YOLO segmentation masks."
    )
    parser.add_argument("--input", type=Path, default=None, help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "output_doubled_people.mp4",
        help="Output video path.",
    )
    parser.add_argument("--model", type=Path, default=None, help="YOLO segmentation model path.")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=ROOT / "bytetrack.yaml",
        help="Ultralytics tracker config path.",
    )
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.55, help="Detection IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument(
        "--offset-ratio",
        type=float,
        default=0.75,
        help="Horizontal clone offset as a ratio of each person bbox width.",
    )
    parser.add_argument(
        "--min-offset",
        type=int,
        default=45,
        help="Minimum horizontal clone offset in pixels.",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=9,
        help="Odd blur kernel for mask feathering. Use 0 to disable.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip cloned persons before compositing.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Copy audio from the source video with ffmpeg if available.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="YOLO device: cuda:0, 0, cpu, or auto. Default uses Jetson GPU.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Run on CPU if CUDA is unavailable. Off by default to avoid slow accidental CPU runs.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 on CUDA. FP16 is enabled by default for Jetson speed.",
    )
    parser.add_argument(
        "--export-engine",
        action="store_true",
        help="Export the .pt model to TensorRT .engine first, then run with that engine.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=2.0,
        help="TensorRT export workspace size in GiB when --export-engine is used.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames. 0 means full video.",
    )
    return parser.parse_args()


def resolve_input(path: Path | None) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Input video not found: {resolved}")
        return resolved

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No default input video found. Searched:\n  {searched}")


def resolve_model(path: Path | None) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Model not found: {resolved}")
        return resolved

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(f"No default YOLO segmentation model found. Searched:\n  {searched}")


def resolve_device(
    device_arg: str,
    allow_cpu_fallback: bool,
    disable_half: bool,
) -> tuple[str, str | int, bool]:
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")
    requested = str(device_arg).strip().lower()
    cuda_error = None
    has_cuda = False

    try:
        torch.cuda.init()
        _ = torch.empty(1, device="cuda:0")
        has_cuda = True
    except Exception as exc:
        cuda_error = exc

    if requested in {"auto", "cuda", "cuda:0", "0"} and has_cuda:
        torch.backends.cudnn.benchmark = True
        return "cuda:0", 0, not disable_half
    if requested in {"cuda", "cuda:0", "0"} and not has_cuda and not allow_cpu_fallback:
        reason = f"{type(cuda_error).__name__}: {cuda_error}" if cuda_error else "unknown reason"
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot initialize GPU.\n"
            f"torch={torch.__version__}, torch_cuda_build={torch.version.cuda}, reason={reason}\n"
            "On Jetson, run outside restricted sandboxes and use a PyTorch build matching your JetPack/CUDA."
        )
    if requested == "auto" and not has_cuda and not allow_cpu_fallback:
        reason = f"{type(cuda_error).__name__}: {cuda_error}" if cuda_error else "unknown reason"
        raise RuntimeError(
            "No CUDA GPU available and CPU fallback is disabled.\n"
            f"torch={torch.__version__}, torch_cuda_build={torch.version.cuda}, reason={reason}"
        )
    return "cpu", "cpu", False


def maybe_export_engine(model_path: Path, args: argparse.Namespace) -> Path:
    if not args.export_engine:
        return model_path
    if model_path.suffix == ".engine":
        print("[INFO] Model is already a TensorRT engine.")
        return model_path
    if model_path.suffix != ".pt":
        raise ValueError("--export-engine requires a .pt model as input.")

    print("[INFO] Exporting TensorRT engine. This can take several minutes on Jetson.")
    exporter = YOLO(str(model_path))
    engine_path = exporter.export(
        format="engine",
        imgsz=args.imgsz,
        half=not args.no_half,
        device=0,
        workspace=args.workspace,
        dynamic=False,
        verbose=False,
    )
    return Path(engine_path)


def ensure_odd_kernel(value: int) -> int:
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def clip_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def choose_clone_box(
    box: tuple[int, int, int, int],
    track_id: int,
    frame_width: int,
    frame_height: int,
    offset_ratio: float,
    min_offset: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    base_offset = max(min_offset, int(round(bw * offset_ratio)))
    direction = -1 if int(track_id) % 2 else 1
    vertical_shift = int(((int(track_id) * 37) % 21) - 10)

    candidates = [
        (direction * base_offset, vertical_shift),
        (-direction * base_offset, -vertical_shift),
        (direction * int(base_offset * 1.45), vertical_shift),
        (-direction * int(base_offset * 1.45), -vertical_shift),
    ]

    best_box = None
    best_score = -1e9
    for dx, dy in candidates:
        nx1 = x1 + dx
        ny1 = y1 + dy
        nx1 = max(0, min(frame_width - bw, nx1))
        ny1 = max(0, min(frame_height - bh, ny1))
        candidate = (nx1, ny1, nx1 + bw, ny1 + bh)

        separation = abs(((candidate[0] + candidate[2]) / 2) - ((x1 + x2) / 2))
        overlap_penalty = box_iou(box, candidate) * bw
        score = separation - overlap_penalty
        if score > best_score:
            best_score = score
            best_box = candidate

    return best_box if best_box is not None else box


def get_instance_masks(result, frame_shape: tuple[int, int]) -> list[np.ndarray] | None:
    if result.masks is None or result.masks.data is None:
        return None

    frame_h, frame_w = frame_shape
    masks = result.masks.data.detach().cpu().numpy()
    full_size_masks = []
    for mask in masks:
        if mask.shape[:2] != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        full_size_masks.append(mask.astype(np.float32))
    return full_size_masks


def composite_clone(
    source_frame: np.ndarray,
    output_frame: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    full_mask: np.ndarray | None,
    feather_kernel: int,
    mirror: bool,
) -> None:
    sx1, sy1, sx2, sy2 = source_box
    tx1, ty1, tx2, ty2 = target_box

    if sx2 <= sx1 or sy2 <= sy1 or tx2 <= tx1 or ty2 <= ty1:
        return

    crop = source_frame[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return

    if full_mask is None:
        alpha = np.ones(crop.shape[:2], dtype=np.float32)
    else:
        alpha = full_mask[sy1:sy2, sx1:sx2]
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    if mirror:
        crop = cv2.flip(crop, 1)
        alpha = cv2.flip(alpha, 1)

    target_w = tx2 - tx1
    target_h = ty2 - ty1
    if crop.shape[1] != target_w or crop.shape[0] != target_h:
        crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if feather_kernel > 0:
        alpha = cv2.GaussianBlur(alpha, (feather_kernel, feather_kernel), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    roi = output_frame[ty1:ty2, tx1:tx2].astype(np.float32)
    blended = crop.astype(np.float32) * alpha + roi * (1.0 - alpha)
    output_frame[ty1:ty2, tx1:tx2] = blended.astype(np.uint8)


def mux_audio(source_video: Path, silent_video: Path, final_video: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        print("[WARN] ffmpeg not found, output will not contain audio.")
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("[WARN] ffmpeg audio mux failed. Keeping silent video.")
        print(proc.stderr[-1200:])
        return False
    return True


def main() -> None:
    args = parse_args()
    input_path = resolve_input(args.input)
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    model_path = resolve_model(args.model)
    tracker_path = args.tracker if args.tracker.is_absolute() else ROOT / args.tracker
    feather_kernel = ensure_odd_kernel(args.feather)

    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_path}")

    torch_device, yolo_device, use_half = resolve_device(
        args.device,
        args.allow_cpu_fallback,
        args.no_half,
    )
    model_path = maybe_export_engine(model_path, args)
    is_engine = model_path.suffix == ".engine"

    print("[INFO] Input: ", input_path)
    print("[INFO] Output:", output_path)
    print("[INFO] Model: ", model_path)
    print("[INFO] Device:", "TensorRT" if is_engine else torch_device, "half=", use_half)

    model = YOLO(str(model_path))
    if not is_engine:
        model.to(torch_device)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_out_path = output_path
    tmp_dir = None
    if args.keep_audio:
        tmp_dir = tempfile.TemporaryDirectory(prefix="merge_person_")
        video_out_path = Path(tmp_dir.name) / "video_no_audio.mp4"

    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {video_out_path}")

    print(f"[INFO] Video: {frame_width}x{frame_height} @ {fps:.2f}fps, frames={total_frames}")
    print("[INFO] Processing...")

    frame_idx = 0
    start_time = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            source_frame = frame.copy()
            output_frame = frame.copy()

            track_kwargs = {
                "source": source_frame,
                "persist": True,
                "tracker": str(tracker_path),
                "classes": [0],
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if not is_engine:
                track_kwargs["half"] = use_half
                track_kwargs["device"] = yolo_device

            results = model.track(**track_kwargs)
            result = results[0]
            masks = get_instance_masks(result, (frame_height, frame_width))

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                track_ids = result.boxes.id.int().detach().cpu().numpy()

                for det_idx, (raw_box, track_id) in enumerate(zip(boxes, track_ids)):
                    source_box = clip_box(raw_box, frame_width, frame_height)
                    x1, y1, x2, y2 = source_box
                    if (x2 - x1) < 8 or (y2 - y1) < 16:
                        continue

                    target_box = choose_clone_box(
                        source_box,
                        int(track_id),
                        frame_width,
                        frame_height,
                        args.offset_ratio,
                        args.min_offset,
                    )
                    mask = masks[det_idx] if masks is not None and det_idx < len(masks) else None
                    composite_clone(
                        source_frame,
                        output_frame,
                        source_box,
                        target_box,
                        mask,
                        feather_kernel,
                        args.mirror,
                    )

            writer.write(output_frame)

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - start_time
                proc_fps = frame_idx / max(elapsed, 1e-6)
                print(f"[INFO] Frame {frame_idx}/{total_frames or '?'} | {proc_fps:.2f} FPS")

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()

    if args.keep_audio:
        muxed = mux_audio(input_path, video_out_path, output_path)
        if not muxed:
            shutil.copyfile(video_out_path, output_path)
        if tmp_dir is not None:
            tmp_dir.cleanup()

    elapsed = time.perf_counter() - start_time
    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Processed {frame_idx} frames in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
