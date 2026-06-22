"""
Create one video that contains two copies of the source video.

This doubles the visible person count without cutting people out of frames.
Default output uses a side-by-side canvas:

    [ original video ][ original video ]

Examples:
  python src/double_video.py
  python src/double_video.py --input assets/output_2.mp4 --output assets/output_double_canvas.mp4
  python src/double_video.py --layout vertical --scale 0.5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CANDIDATES = (
    ROOT / "output.mp4",
    ROOT / "assets" / "output.mp4",
    ROOT / "assets" / "output_2.mp4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Put two source videos into one canvas to double visible people."
    )
    parser.add_argument("--input", type=Path, default=None, help="First/source video path.")
    parser.add_argument(
        "--second-input",
        type=Path,
        default=None,
        help="Optional second video path. If omitted, the first video is duplicated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "output_double_canvas.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="horizontal = side-by-side, vertical = top-bottom.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale each input video before placing it on the canvas.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="Pixel gap between the two videos.",
    )
    parser.add_argument(
        "--bg-color",
        default="0,0,0",
        help="Canvas background color as B,G,R, for example 0,0,0.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Copy audio from the first input video with ffmpeg if available.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames. 0 means full video.",
    )
    return parser.parse_args()


def resolve_path(path: Path | None, candidates: tuple[Path, ...]) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(f"Video not found: {resolved}")
        return resolved

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No input video found. Searched:\n  {searched}")


def parse_bgr(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--bg-color must use B,G,R format, for example 0,0,0")
    color = tuple(max(0, min(255, int(p))) for p in parts)
    return color


def read_video_info(cap: cv2.VideoCapture, path: Path) -> tuple[int, int, float, int]:
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video size for: {path}")
    return width, height, fps, frames


def resize_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target_size
    if frame.shape[1] == target_w and frame.shape[0] == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def make_canvas(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    layout: str,
    gap: int,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = frame_a.shape[:2]
    if layout == "horizontal":
        canvas = np.full((h, w * 2 + gap, 3), bg_color, dtype=np.uint8)
        canvas[:, :w] = frame_a
        canvas[:, w + gap : w * 2 + gap] = frame_b
        return canvas

    canvas = np.full((h * 2 + gap, w, 3), bg_color, dtype=np.uint8)
    canvas[:h, :] = frame_a
    canvas[h + gap : h * 2 + gap, :] = frame_b
    return canvas


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
    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0")
    if args.gap < 0:
        raise ValueError("--gap must be greater than or equal to 0")

    input_a = resolve_path(args.input, DEFAULT_INPUT_CANDIDATES)
    input_b = resolve_path(args.second_input, (input_a,))
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    bg_color = parse_bgr(args.bg_color)

    cap_a = cv2.VideoCapture(str(input_a))
    cap_b = cv2.VideoCapture(str(input_b))
    width_a, height_a, fps_a, frames_a = read_video_info(cap_a, input_a)
    width_b, height_b, fps_b, frames_b = read_video_info(cap_b, input_b)

    target_w = int(round(width_a * args.scale))
    target_h = int(round(height_a * args.scale))
    if target_w <= 0 or target_h <= 0:
        raise ValueError("--scale produces an invalid output size")

    if (width_a, height_a) != (width_b, height_b):
        print(
            "[WARN] Input videos have different sizes. "
            f"Second video will be resized from {width_b}x{height_b} to {width_a}x{height_a}."
        )
    if abs(fps_a - fps_b) > 0.01:
        print(f"[WARN] FPS differs: first={fps_a:.3f}, second={fps_b:.3f}. Output uses first FPS.")

    if args.layout == "horizontal":
        out_size = (target_w * 2 + args.gap, target_h)
    else:
        out_size = (target_w, target_h * 2 + args.gap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = None
    video_out_path = output_path
    if args.keep_audio:
        tmp_dir = tempfile.TemporaryDirectory(prefix="double_video_")
        video_out_path = Path(tmp_dir.name) / "video_no_audio.mp4"

    writer = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_a,
        out_size,
    )
    if not writer.isOpened():
        cap_a.release()
        cap_b.release()
        raise RuntimeError(f"Cannot create output video: {video_out_path}")

    total_frames = min(frames_a, frames_b) if frames_a and frames_b else 0
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    print(f"[INFO] First input:  {input_a} ({width_a}x{height_a}, {fps_a:.2f}fps)")
    print(f"[INFO] Second input: {input_b} ({width_b}x{height_b}, {fps_b:.2f}fps)")
    print(f"[INFO] Output:       {output_path} ({out_size[0]}x{out_size[1]})")
    print(f"[INFO] Layout:       {args.layout}")

    frame_idx = 0
    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break

            frame_idx += 1
            frame_a = resize_frame(frame_a, (target_w, target_h))
            frame_b = resize_frame(frame_b, (target_w, target_h))
            canvas = make_canvas(frame_a, frame_b, args.layout, args.gap, bg_color)
            writer.write(canvas)

            if frame_idx % 100 == 0:
                suffix = f"/{total_frames}" if total_frames else ""
                print(f"[INFO] Frame {frame_idx}{suffix}")

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap_a.release()
        cap_b.release()
        writer.release()

    if args.keep_audio:
        muxed = mux_audio(input_a, video_out_path, output_path)
        if not muxed:
            shutil.copyfile(video_out_path, output_path)
        if tmp_dir is not None:
            tmp_dir.cleanup()

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Frames: {frame_idx}")


if __name__ == "__main__":
    main()
