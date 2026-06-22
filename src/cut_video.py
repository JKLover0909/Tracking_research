"""
Cut a video by start/end time.

Edit the CONFIG section below, then run:
  python src/cut_video.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG - chỉnh trực tiếp ở đây
# ─────────────────────────────────────────────────────────────────────────────
# Có thể dùng đường dẫn tương đối từ project root hoặc đường dẫn tuyệt đối.
INPUT_PATH = ROOT / "assets" / "video_results" / "difference_bytetrack_vs_botsort_19.mp4"
OUTPUT_PATH = ROOT / "assets" / "video_results" / "difference_bytetrack_vs_botsort_19_output.mp4"

# Có thể nhập số giây: 5, 12.5
# Hoặc chuỗi thời gian: "00:00:05", "01:20", "00:01:20.5"
START_TIME = 46
END_TIME = 86

# False: cắt nhanh bằng ffmpeg stream-copy, giữ audio nếu ffmpeg chạy được.
# True: re-encode bằng ffmpeg, cắt chính xác hơn nhưng chậm hơn.
REENCODE = False

# False: ưu tiên ffmpeg, nếu lỗi sẽ tự fallback OpenCV.
# True: dùng OpenCV luôn, video output sẽ không có audio.
FORCE_OPENCV = False


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_time_to_seconds(value: str) -> float:
    value = value.strip()
    if ":" not in value:
        seconds = float(value)
        if seconds < 0:
            raise ValueError("Time must be greater than or equal to 0.")
        return seconds

    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("Time must use SS, MM:SS, or HH:MM:SS format.")

    nums = [float(part) for part in parts]
    if len(nums) == 2:
        minutes, seconds = nums
        total = minutes * 60 + seconds
    else:
        hours, minutes, seconds = nums
        total = hours * 3600 + minutes * 60 + seconds

    if total < 0:
        raise ValueError("Time must be greater than or equal to 0.")
    return total


def seconds_to_ffmpeg_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def cut_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    reencode: bool,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or use --opencv.")

    duration = end_seconds - start_seconds
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        seconds_to_ffmpeg_time(start_seconds),
        "-i",
        str(input_path),
        "-t",
        seconds_to_ffmpeg_time(duration),
    ]

    if reencode:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
        ]
    else:
        cmd += ["-c", "copy"]

    cmd.append(str(output_path))

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def cut_with_opencv(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0
    try:
        while frame_idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_idx += 1
            written += 1
    finally:
        cap.release()
        writer.release()

    print(f"[INFO] OpenCV wrote {written} frames. Audio is not included.")


def main() -> None:
    input_path = resolve_path(Path(INPUT_PATH))
    output_path = resolve_path(Path(OUTPUT_PATH))
    start_seconds = parse_time_to_seconds(str(START_TIME))
    end_seconds = parse_time_to_seconds(str(END_TIME))

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if end_seconds <= start_seconds:
        raise ValueError("END_TIME must be greater than START_TIME")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input:  {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Range:  {start_seconds:.3f}s -> {end_seconds:.3f}s")

    if FORCE_OPENCV:
        cut_with_opencv(input_path, output_path, start_seconds, end_seconds)
    else:
        try:
            cut_with_ffmpeg(input_path, output_path, start_seconds, end_seconds, REENCODE)
        except RuntimeError as exc:
            print(f"[WARN] ffmpeg cut failed, falling back to OpenCV video-only cut.\n{exc}")
            cut_with_opencv(input_path, output_path, start_seconds, end_seconds)

    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Range: {start_seconds:.3f}s -> {end_seconds:.3f}s")


if __name__ == "__main__":
    main()
