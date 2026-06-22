"""
Merge 3 camera videos into one synchronized canvas for multi-camera tracking.
Tự động chuẩn hóa kích thước: tất cả camera dùng cùng cell size, hiển thị full canvas.

Default inputs:
  assets/ps_Long/Cam_01.mp4
  assets/ps_Long/Cam_02.mp4
  assets/ps_Long/Cam_03.mp4

Examples:
  python src/merge_video.py
  python src/merge_video.py --layout row --max-frames 300
  python src/merge_video.py --inputs cam1.mp4 cam2.mp4 cam3.mp4 --output assets/merged_3cams.mp4
  python src/merge_video.py --output-size 1920x1080 --fit contain
  python src/merge_video.py --output-size 1080p --normalize-cell

Layout modes:
  row         : 3 cameras side by side, giữ đúng aspect ratio từng camera [DEFAULT with auto]
  horizontal  : 3 cameras side by side, cùng cell size
  vertical    : 3 cameras stacked        (canvas ratio = cell_w : 3 * cell_h)
  grid        : 2x2 grid, bottom-right cell empty nếu chỉ có 3 cameras
  auto        : uses row for 3 cameras to avoid empty cells  [DEFAULT]

Fit modes:
  cover   : fill full cell, center-crop (default)
  contain : show full frame with black padding  <-- tốt khi cameras khác aspect ratio
  stretch : fill full cell, kéo giãn

Normalize cell (--normalize-cell):
  Tự động tính cell_w/cell_h lớn nhất sao cho tất cả camera vừa màn hình.
  Mặc định dùng camera có DIỆN TÍCH NHỎ NHẤT làm chuẩn (tránh quá to).

Output:
  - merged video
  - JSON metadata beside the video with each camera region/scale
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT / "assets" / "ps_Long" / "Cam_01.mp4",
    ROOT / "assets" / "ps_Long" / "Cam_02.mp4",
    ROOT / "assets" / "ps_Long" / "Cam_03.mp4",
)
DEFAULT_OUTPUT = ROOT / "assets" / "ps_Long" / "merged_3cams.mp4"

OUTPUT_PRESETS: dict[str, tuple[int, int]] = {
    "4k":    (3840, 2160),
    "1080p": (1920, 1080),
    "720p":  (1280,  720),
    "480p":  ( 854,  480),
}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class Placement:
    camera_id: int
    source_path: str
    cell_x: int
    cell_y: int
    cell_w: int
    cell_h: int
    image_x: int
    image_y: int
    image_w: int
    image_h: int
    scale_x: float
    scale_y: float
    source_visible_x: float
    source_visible_y: float
    source_visible_w: float
    source_visible_h: float


@dataclass(frozen=True)
class LayoutSlot:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class FitResult:
    cell: np.ndarray
    image_x: int
    image_y: int
    image_w: int
    image_h: int
    scale_x: float
    scale_y: float
    source_visible_x: float
    source_visible_y: float
    source_visible_w: float
    source_visible_h: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge 3 camera videos into one tracking-ready canvas (chuẩn hóa kích thước)."
    )
    parser.add_argument(
        "--inputs", type=Path, nargs=3, default=None,
        metavar=("CAM1", "CAM2", "CAM3"),
        help="Three input video paths.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layout", choices=("row", "horizontal", "vertical", "grid", "auto"), default="auto",
    )
    parser.add_argument(
        "--output-size", default=None, metavar="WxH_or_PRESET",
        help="Force canvas size, e.g. '1920x1080' or '1080p'.",
    )
    parser.add_argument("--cell-width",  type=int, default=0)
    parser.add_argument("--cell-height", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument(
        "--fit", choices=("cover", "contain", "stretch"), default="cover",
        help=(
            "cover=fill+crop | contain=full frame+padding (khuyến nghị khi cameras khác size) | stretch=kéo giãn"
        ),
    )
    parser.add_argument(
        "--normalize-cell", action="store_true",
        help=(
            "Tự động chuẩn hóa cell size từ tất cả cameras. "
            "Dùng khi các video có kích thước khác nhau để hiển thị đồng đều."
        ),
    )
    parser.add_argument(
        "--normalize-strategy",
        choices=("min_area", "max_area", "median_width", "first"),
        default="min_area",
        help=(
            "Chiến lược chọn cell size khi --normalize-cell: "
            "min_area=camera nhỏ nhất (tránh crop quá nhiều), "
            "max_area=camera lớn nhất, "
            "median_width=trung bình, "
            "first=camera đầu tiên."
        ),
    )
    parser.add_argument("--bg-color", default="20,20,20")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--end-mode", choices=("shortest", "longest"), default="shortest")
    parser.add_argument("--fps-mode", choices=("first", "min", "max"), default="first")
    parser.add_argument("--sync-mode", choices=("time", "frame"), default="time")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--metadata", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_bgr(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--bg-color must use B,G,R format, e.g. 20,20,20")
    return tuple(max(0, min(255, int(p))) for p in parts)


def parse_output_size(value: str) -> tuple[int, int]:
    if value.lower() in OUTPUT_PRESETS:
        return OUTPUT_PRESETS[value.lower()]
    parts = value.lower().split("x")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    raise ValueError(
        f"Cannot parse --output-size '{value}'. "
        f"Use WxH (e.g. 1920x1080) or a preset: {', '.join(OUTPUT_PRESETS)}."
    )


def read_video_info(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video size for: {path}")
    return cap, VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=frame_count)


def choose_fps(infos: list[VideoInfo], mode: str) -> float:
    values = [info.fps for info in infos if info.fps > 0]
    if mode == "min":  return min(values)
    if mode == "max":  return max(values)
    return values[0]


def video_duration(info: VideoInfo) -> float:
    if info.frame_count <= 0 or info.fps <= 0:
        return 0.0
    return info.frame_count / info.fps


# ─────────────────────────────────────────────────────────────────────────────
# Cell size normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_cell_size(
    infos: list[VideoInfo],
    strategy: str,
) -> tuple[int, int]:
    """
    Tính cell_w và cell_h thống nhất cho tất cả cameras.

    Vấn đề: nếu cameras có kích thước khác nhau và dùng cell_w/cell_h của camera đầu tiên,
    các camera nhỏ hơn sẽ bị kéo căng (stretch/cover) quá nhiều,
    còn camera lớn hơn sẽ bị crop nhiều.

    Giải pháp: chọn một cell size "đại diện" dựa trên strategy,
    sau đó tất cả camera dùng chung cell đó + fit mode để hiển thị đúng.

    Strategies:
    - min_area   : dùng camera có diện tích nhỏ nhất → ít crop nhất
    - max_area   : dùng camera lớn nhất → đủ chỗ cho tất cả
    - median_width: dùng camera có width trung bình
    - first      : dùng camera đầu tiên (hành vi cũ)
    """
    if strategy == "first":
        return infos[0].width, infos[0].height

    if strategy == "min_area":
        ref = min(infos, key=lambda i: i.width * i.height)
    elif strategy == "max_area":
        ref = max(infos, key=lambda i: i.width * i.height)
    elif strategy == "median_width":
        sorted_by_w = sorted(infos, key=lambda i: i.width)
        ref = sorted_by_w[len(sorted_by_w) // 2]
    else:
        ref = infos[0]

    print(
        f"[INFO] Normalized cell size from camera '{ref.path.name}': "
        f"{ref.width}x{ref.height} (strategy={strategy})"
    )
    return ref.width, ref.height


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _canvas_for_layout(layout: str, count: int, cell_w: int, cell_h: int, gap: int) -> tuple[int, int]:
    if layout in ("row", "horizontal"):
        return count * cell_w + (count - 1) * gap, cell_h
    if layout == "vertical":
        return cell_w, count * cell_h + (count - 1) * gap
    cols = 2
    rows = math.ceil(count / cols)
    return cols * cell_w + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap


def choose_layout(infos: list[VideoInfo], gap: int) -> str:
    if len(infos) == 3:
        return "row"

    target = 16 / 9
    best_layout, best_diff = "horizontal", float("inf")
    cell_w, cell_h = infos[0].width, infos[0].height
    for layout in ("horizontal", "vertical"):
        cw, ch = _canvas_for_layout(layout, len(infos), cell_w, cell_h, gap)
        diff = abs(cw / ch - target)
        if diff < best_diff:
            best_diff = diff
            best_layout = layout
    return best_layout


def get_layout_positions(
    layout: str, count: int, cell_w: int, cell_h: int, gap: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    canvas = _canvas_for_layout(layout, count, cell_w, cell_h, gap)
    if layout == "horizontal":
        positions = [(i * (cell_w + gap), 0) for i in range(count)]
    elif layout == "vertical":
        positions = [(0, i * (cell_h + gap)) for i in range(count)]
    else:
        cols = 2
        positions = [
            ((i % cols) * (cell_w + gap), (i // cols) * (cell_h + gap))
            for i in range(count)
        ]
    return positions, canvas


def _row_widths(infos: list[VideoInfo], row_h: int) -> list[int]:
    widths = [max(1, int(round(info.width * row_h / info.height))) for info in infos]
    # mp4 encoders commonly round odd frame dimensions down; keep metadata exact.
    return [max(2, width - (width % 2)) for width in widths]


def _proportional_row_widths(infos: list[VideoInfo], total_w: int) -> list[int]:
    aspects = [info.width / info.height for info in infos]
    aspect_sum = sum(aspects)
    widths = [max(1, int(round(total_w * aspect / aspect_sum))) for aspect in aspects]
    widths[-1] = max(1, widths[-1] + total_w - sum(widths))
    return widths


def get_layout_slots(
    layout: str,
    infos: list[VideoInfo],
    cell_w: int,
    cell_h: int,
    gap: int,
    forced_canvas_size: tuple[int, int] | None = None,
) -> tuple[list[LayoutSlot], tuple[int, int]]:
    """
    Trả về vùng vẽ cho từng camera.

    Layout "row" khác "horizontal": row giữ đúng aspect ratio của từng video,
    nhờ đó 3 camera lấp đầy một hàng liên tục mà không cần ô grid trống.
    """
    count = len(infos)

    if layout != "row":
        positions, canvas_size = get_layout_positions(layout, count, cell_w, cell_h, gap)
        slots = [LayoutSlot(x=x, y=y, w=cell_w, h=cell_h) for x, y in positions]
        return slots, canvas_size

    if forced_canvas_size:
        out_w, out_h = forced_canvas_size
        available_w = max(1, out_w - (count - 1) * gap)
        row_h = out_h
        row_widths = _proportional_row_widths(infos, available_w)
        row_w = out_w
        start_x = 0
        start_y = 0
        canvas_size = forced_canvas_size
    else:
        row_h = max(2, cell_h - (cell_h % 2))
        row_widths = _row_widths(infos, row_h)
        row_w = sum(row_widths) + (count - 1) * gap
        start_x = 0
        start_y = 0
        canvas_size = (row_w, row_h)

    slots: list[LayoutSlot] = []
    x = start_x
    for width in row_widths:
        slots.append(LayoutSlot(x=x, y=start_y, w=width, h=row_h))
        x += width + gap
    return slots, canvas_size


def cell_size_from_output(
    output_w: int, output_h: int,
    layout: str, count: int, gap: int,
) -> tuple[int, int]:
    if layout == "row":
        return output_w, output_h
    if layout == "horizontal":
        cols, rows = count, 1
    elif layout == "vertical":
        cols, rows = 1, count
    else:
        cols, rows = 2, math.ceil(count / 2)
    cell_w = (output_w - (cols - 1) * gap) // cols
    cell_h = (output_h - (rows - 1) * gap) // rows
    return max(1, cell_w), max(1, cell_h)


# ─────────────────────────────────────────────────────────────────────────────
# Frame fitting — cốt lõi của việc hiển thị đúng khi cameras khác size
# ─────────────────────────────────────────────────────────────────────────────

def fit_frame(
    frame: np.ndarray,
    cell_w: int,
    cell_h: int,
    bg_color: tuple[int, int, int],
    fit: str,
) -> FitResult:
    """
    Fit một frame vào cell.

    - cover  : scale để fill hoàn toàn cell, crop phần thừa ở center.
               Tốt khi cameras có cùng aspect ratio.
    - contain: scale để toàn bộ frame vừa trong cell, padding bằng bg_color.
               Tốt nhất khi cameras có kích thước / aspect ratio khác nhau
               vì không mất thông tin nào.
    - stretch: kéo giãn vừa cell, có thể méo.
    """
    src_h, src_w = frame.shape[:2]

    if fit == "stretch":
        cell = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        return FitResult(
            cell=cell, image_x=0, image_y=0, image_w=cell_w, image_h=cell_h,
            scale_x=cell_w / src_w, scale_y=cell_h / src_h,
            source_visible_x=0.0, source_visible_y=0.0,
            source_visible_w=float(src_w), source_visible_h=float(src_h),
        )

    if fit == "cover":
        scale = max(cell_w / src_w, cell_h / src_h)
        resized_w = max(cell_w, int(round(src_w * scale)))
        resized_h = max(cell_h, int(round(src_h * scale)))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        crop_x = max(0, (resized_w - cell_w) // 2)
        crop_y = max(0, (resized_h - cell_h) // 2)
        cell = resized[crop_y: crop_y + cell_h, crop_x: crop_x + cell_w].copy()
        return FitResult(
            cell=cell, image_x=0, image_y=0, image_w=cell_w, image_h=cell_h,
            scale_x=scale, scale_y=scale,
            source_visible_x=crop_x / scale, source_visible_y=crop_y / scale,
            source_visible_w=cell_w / scale, source_visible_h=cell_h / scale,
        )

    # contain — giữ toàn bộ frame, pad phần còn lại
    scale = min(cell_w / src_w, cell_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    cell = np.full((cell_h, cell_w, 3), bg_color, dtype=np.uint8)
    x = (cell_w - resized_w) // 2
    y = (cell_h - resized_h) // 2
    cell[y: y + resized_h, x: x + resized_w] = resized
    return FitResult(
        cell=cell, image_x=x, image_y=y, image_w=resized_w, image_h=resized_h,
        scale_x=resized_w / src_w, scale_y=resized_h / src_h,
        source_visible_x=0.0, source_visible_y=0.0,
        source_visible_w=float(src_w), source_visible_h=float(src_h),
    )


def draw_label(cell: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, margin = 0.8, 2, 10
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x1, y1 = margin, margin
    x2, y2 = x1 + text_w + 16, y1 + text_h + baseline + 12
    cv2.rectangle(cell, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(cell, text, (x1 + 8, y2 - baseline - 6),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_resolution_info(cell: np.ndarray, src_w: int, src_h: int) -> None:
    """Hiển thị resolution gốc của camera để dễ phân biệt."""
    text = f"{src_w}x{src_h}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.5, 1
    cell_h, cell_w = cell.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = cell_w - text_w - 14
    y = cell_h - baseline - 8
    cv2.rectangle(cell, (x - 6, y - text_h - 4), (x + text_w + 6, y + baseline + 4), (40, 40, 40), -1)
    cv2.putText(cell, text, (x, y), font, font_scale, (180, 180, 180), thickness, cv2.LINE_AA)


def read_until_frame(
    cap: cv2.VideoCapture,
    target_idx: int,
    current_idx: int,
    current_frame: np.ndarray | None,
) -> tuple[bool, int, np.ndarray | None]:
    if target_idx < current_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        current_idx = target_idx - 1
        current_frame = None
    while current_idx < target_idx:
        ok, frame = cap.read()
        if not ok:
            return False, current_idx, current_frame
        current_idx += 1
        current_frame = frame
    return current_frame is not None, current_idx, current_frame


def make_metadata(
    output_path, infos, placements, canvas_size, fps, sync_mode, layout, gap, fit
) -> dict:
    return {
        "output_video": str(output_path),
        "canvas_width": canvas_size[0],
        "canvas_height": canvas_size[1],
        "fps": fps,
        "sync_mode": sync_mode,
        "layout": layout,
        "gap": gap,
        "fit": fit,
        "cameras": [
            {
                "camera_id": p.camera_id,
                "source_path": p.source_path,
                "source_width": infos[p.camera_id - 1].width,
                "source_height": infos[p.camera_id - 1].height,
                "source_fps": infos[p.camera_id - 1].fps,
                "source_frame_count": infos[p.camera_id - 1].frame_count,
                "cell": {"x": p.cell_x, "y": p.cell_y, "width": p.cell_w, "height": p.cell_h},
                "image_region": {
                    "x": p.image_x, "y": p.image_y,
                    "width": p.image_w, "height": p.image_h,
                },
                "source_visible_region": {
                    "x": p.source_visible_x, "y": p.source_visible_y,
                    "width": p.source_visible_w, "height": p.source_visible_h,
                },
                "scale_x_source_to_canvas": p.scale_x,
                "scale_y_source_to_canvas": p.scale_y,
            }
            for p in placements
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.scale <= 0:
        raise ValueError("--scale must be > 0")
    if args.gap < 0:
        raise ValueError("--gap must be >= 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    input_paths = [resolve_path(p) for p in (args.inputs or DEFAULT_INPUTS)]
    missing = [p for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input video(s):\n  " + "\n  ".join(str(p) for p in missing))

    output_path   = resolve_path(args.output)
    metadata_path = resolve_path(args.metadata) if args.metadata else output_path.with_suffix(".json")
    bg_color      = parse_bgr(args.bg_color)

    caps:  list[cv2.VideoCapture] = []
    infos: list[VideoInfo]        = []
    for path in input_paths:
        cap, info = read_video_info(path)
        caps.append(cap)
        infos.append(info)

    # ── log source sizes ──────────────────────────────────────────────────────
    print("[INFO] Input video sizes:")
    for i, info in enumerate(infos):
        print(f"       Cam {i+1}: {info.width}x{info.height} @ {info.fps:.2f} fps  ({info.path.name})")

    sizes_differ = len({(i.width, i.height) for i in infos}) > 1
    if sizes_differ:
        print(
            "[WARN] Cameras have different resolutions. "
            "Using --normalize-cell or --fit contain is recommended."
        )

    fps = choose_fps(infos, args.fps_mode)

    # ── layout ────────────────────────────────────────────────────────────────
    layout = args.layout
    if layout == "auto":
        layout = choose_layout(infos, args.gap)
        print(f"[INFO] Auto-selected layout: {layout}")

    # ── cell size ─────────────────────────────────────────────────────────────
    forced_canvas_size: tuple[int, int] | None = None
    if args.output_size:
        # --output-size dominates everything: canvas is fixed, cell derived
        out_w, out_h = parse_output_size(args.output_size)
        forced_canvas_size = (out_w, out_h)
        cell_w, cell_h = cell_size_from_output(out_w, out_h, layout, len(infos), args.gap)
        if layout == "row":
            print(f"[INFO] Output size {out_w}x{out_h} → adaptive row slots ({layout})")
        else:
            print(f"[INFO] Output size {out_w}x{out_h} → cell {cell_w}x{cell_h} ({layout})")

    elif args.normalize_cell or (sizes_differ and not args.cell_width and not args.cell_height):
        # Auto-normalize: all cameras share the same cell size
        if sizes_differ and not args.normalize_cell:
            print(
                "[INFO] Cameras differ in size → auto-applying --normalize-cell "
                "(strategy=min_area). Pass --normalize-cell explicitly to suppress this notice."
            )
        cell_w, cell_h = normalize_cell_size(infos, args.normalize_strategy)
        cell_w = max(1, int(round(cell_w * args.scale)))
        cell_h = max(1, int(round(cell_h * args.scale)))

    else:
        # Legacy behaviour: explicit or first-camera dimensions
        first  = infos[0]
        cell_w = args.cell_width  if args.cell_width  > 0 else first.width
        cell_h = args.cell_height if args.cell_height > 0 else first.height
        cell_w = max(1, int(round(cell_w * args.scale)))
        cell_h = max(1, int(round(cell_h * args.scale)))

    slots, canvas_size = get_layout_slots(
        layout, infos, cell_w, cell_h, args.gap, forced_canvas_size=forced_canvas_size
    )
    if layout == "row":
        slot_text = ", ".join(f"Cam {i + 1}: {slot.w}x{slot.h}" for i, slot in enumerate(slots))
        print(f"[INFO] Row slots: {slot_text}")
        print(f"[INFO] Canvas: {canvas_size[0]}x{canvas_size[1]}  |  Fit: {args.fit}")
    else:
        print(f"[INFO] Cell size: {cell_w}x{cell_h}  |  Canvas: {canvas_size[0]}x{canvas_size[1]}  |  Fit: {args.fit}")

    # ── aspect ratio warning ──────────────────────────────────────────────────
    ratio = canvas_size[0] / canvas_size[1]
    if layout == "row" and forced_canvas_size is None and ratio >= 2.5:
        print(
            f"[INFO] Row layout keeps every camera fully visible, so the canvas is wide "
            f"({canvas_size[0]}x{canvas_size[1]}). Use --scale 0.5 if the output is too large."
        )
    elif not (1.2 < ratio < 2.5):
        print(
            f"[WARN] Canvas aspect ratio is {ratio:.2f} ({canvas_size[0]}x{canvas_size[1]}). "
            "Use --output-size 1920x1080 (or 1080p/720p) to force a standard size."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, canvas_size)
    if not writer.isOpened():
        for cap in caps:
            cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # ── total frames ──────────────────────────────────────────────────────────
    frame_counts = [info.frame_count for info in infos if info.frame_count > 0]
    if args.end_mode == "shortest":
        if args.sync_mode == "time":
            durations   = [video_duration(info) for info in infos if video_duration(info) > 0]
            total_frames = int(math.floor(min(durations) * fps)) if durations else 0
        else:
            total_frames = min(frame_counts) if frame_counts else 0
    else:
        if args.sync_mode == "time":
            durations   = [video_duration(info) for info in infos if video_duration(info) > 0]
            total_frames = int(math.ceil(max(durations) * fps)) if durations else 0
        else:
            total_frames = max(frame_counts) if frame_counts else 0
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    # ── write frames ──────────────────────────────────────────────────────────
    placements:     list[Placement]          = []
    last_frames:    list[np.ndarray | None]  = [None] * len(caps)
    current_indices = [-1] * len(caps)
    written = 0

    try:
        while total_frames == 0 or written < total_frames:
            canvas   = np.full((canvas_size[1], canvas_size[0], 3), bg_color, dtype=np.uint8)
            any_ok   = False

            for idx, (cap, info, slot) in enumerate(zip(caps, infos, slots)):
                if args.sync_mode == "time":
                    target_idx = int(round((written / fps) * info.fps))
                else:
                    target_idx = written

                ok, current_indices[idx], frame = read_until_frame(
                    cap, target_idx, current_indices[idx], last_frames[idx]
                )
                if ok:
                    any_ok = True
                    last_frames[idx] = frame
                elif args.end_mode == "longest" and last_frames[idx] is not None:
                    frame = np.zeros_like(last_frames[idx])
                else:
                    frame = None

                if frame is None:
                    continue

                fit_result = fit_frame(frame, slot.w, slot.h, bg_color, args.fit)
                cell = fit_result.cell.copy()

                if not args.no_labels:
                    draw_label(cell, f"CAM {idx + 1:02d}")
                    # Hiển thị resolution gốc ở góc dưới phải mỗi cell
                    draw_resolution_info(cell, info.width, info.height)

                canvas[slot.y: slot.y + slot.h, slot.x: slot.x + slot.w] = cell

                if written == 0:
                    placements.append(
                        Placement(
                            camera_id=idx + 1,
                            source_path=str(info.path),
                            cell_x=slot.x, cell_y=slot.y, cell_w=slot.w, cell_h=slot.h,
                            image_x=slot.x + fit_result.image_x,
                            image_y=slot.y + fit_result.image_y,
                            image_w=fit_result.image_w, image_h=fit_result.image_h,
                            scale_x=fit_result.scale_x, scale_y=fit_result.scale_y,
                            source_visible_x=fit_result.source_visible_x,
                            source_visible_y=fit_result.source_visible_y,
                            source_visible_w=fit_result.source_visible_w,
                            source_visible_h=fit_result.source_visible_h,
                        )
                    )

            if not any_ok and args.end_mode == "shortest":
                break

            writer.write(canvas)
            written += 1
            if written % 300 == 0:
                print(f"[INFO] Wrote {written} frames...")

            if total_frames == 0 and not any_ok:
                break

    finally:
        for cap in caps:
            cap.release()
        writer.release()

    metadata = make_metadata(
        output_path=output_path, infos=infos, placements=placements,
        canvas_size=canvas_size, fps=fps, sync_mode=args.sync_mode,
        layout=layout, gap=args.gap, fit=args.fit,
    )
    metadata["written_frames"] = written
    metadata["normalized_cell"] = args.normalize_cell or sizes_differ
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fps_values = ", ".join(f"{info.fps:.3f}" for info in infos)
    if any(abs(info.fps - infos[0].fps) > 0.01 for info in infos[1:]):
        print(f"[WARN] Input FPS differs ({fps_values}). Output FPS={fps:.3f}.")

    print(f"\n[DONE] Saved video:    {output_path}")
    print(f"[DONE] Saved metadata: {metadata_path}")
    print(f"[DONE] Frames: {written}  |  Canvas: {canvas_size[0]}x{canvas_size[1]}  |  FPS: {fps:.3f}")


if __name__ == "__main__":
    main()
