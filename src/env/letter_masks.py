"""Load, cache, and generate binary PNG masks for LightPaint letters."""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Font resolution order for portable letter rasterization.
_FONT_CANDIDATES = [
    os.environ.get("LIGHTPAINT_FONT", ""),
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

# Module-level cache: letter -> (mask_array, waypoints)
_MASK_CACHE: Dict[str, np.ndarray] = {}
_WAYPOINT_CACHE: Dict[str, List[Tuple[float, float, float]]] = {}

# World bounds for waypoint projection (matches env world_to_pixel constants)
X_MIN, X_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.5, 2.5
Y_CANVAS = 0.0  # canvas is in XZ plane at Y=0


def _resolve_font_at(size: int) -> ImageFont.FreeTypeFont:
    """Resolve a bold font at the requested pixel size; fall back gracefully."""
    font_size = max(1, int(size))
    for candidate in _FONT_CANDIDATES:
        if candidate and os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _resolve_font(size: int) -> ImageFont.FreeTypeFont:
    """Resolve the default single-letter font."""
    return _resolve_font_at(max(1, int(size * 0.85)))


def auto_fit_font_size(
    text: str,
    canvas_size: int = 64,
    target_fill: float = 0.85,
) -> ImageFont.FreeTypeFont:
    """Pick the largest font size whose bbox(text) fits within canvas × target_fill."""
    max_size = max(4, int(canvas_size * 0.95))
    probe_img = Image.new("L", (canvas_size, canvas_size), color=0)
    probe_draw = ImageDraw.Draw(probe_img)
    fit_w = canvas_size * target_fill
    fit_h = canvas_size * target_fill
    for fs in range(max_size, 3, -1):
        font = _resolve_font_at(fs)
        bbox = probe_draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= fit_w and h <= fit_h:
            return font
    return _resolve_font_at(4)


def render_letter(text: str, size: int = 64) -> np.ndarray:
    """
    Render an ASCII string as a float32 binary mask (0.0 or 1.0).

    Supports multi-character text (e.g. "RL", "Pig") with automatic font-size
    fitting. Case is preserved, so "Pig" and "PIG" produce different masks.
    Uses Pillow textbbox-centered rasterization with portable font resolution.
    Returns shape (size, size) float32 array.
    """
    font = auto_fit_font_size(text, canvas_size=size, target_fill=0.85)
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    # Center the text using textbbox (handles left-side bearing correctly)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x_offset = (size - text_w) // 2 - bbox[0]
    y_offset = (size - text_h) // 2 - bbox[1]
    draw.text((x_offset, y_offset), text, fill=255, font=font)

    arr = np.array(img, dtype=np.uint8)
    return (arr > 127).astype(np.float32)


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Save a float32 binary mask as 8-bit grayscale PNG (0 or 255)."""
    img_arr = (mask * 255).astype(np.uint8)
    Image.fromarray(img_arr, mode="L").save(str(out_path))


def load_mask(label: str, data_dir: Path) -> np.ndarray:
    """
    Load a 64x64 float32 binary mask from data_dir/{label}.png, with cache.

    Case-sensitive: "Pig" and "PIG" are distinct cache keys and filenames.
    Returns shape (64, 64) float32 with values 0.0 or 1.0.
    Raises FileNotFoundError if the PNG does not exist.
    """
    if label in _MASK_CACHE:
        return _MASK_CACHE[label]

    png_path = Path(data_dir) / f"{label}.png"
    if not png_path.is_file():
        raise FileNotFoundError(f"Label mask not found: {png_path}. Run gen_masks.py first.")

    arr = np.array(Image.open(str(png_path)).convert("L"), dtype=np.uint8)
    mask = (arr > 127).astype(np.float32)
    _MASK_CACHE[label] = mask
    return mask


def get_mask(label: str) -> np.ndarray:
    """
    Get mask from cache (must have been loaded/generated already).
    Renders on-the-fly if not in cache (no PNG required).
    Case-sensitive: "Pig" and "PIG" are distinct.
    """
    if label not in _MASK_CACHE:
        _MASK_CACHE[label] = render_letter(label, size=64)
    return _MASK_CACHE[label]


def _pixel_to_world(px: int, pz: int, size: int = 64) -> Tuple[float, float]:
    """Convert pixel coordinates (col, row) to world (x, z) coordinates."""
    x = X_MIN + (px / (size - 1)) * (X_MAX - X_MIN)
    # Row 0 = top = Z_MAX; row (size-1) = bottom = Z_MIN
    z = Z_MAX - (pz / (size - 1)) * (Z_MAX - Z_MIN)
    return x, z


def get_waypoints(label: str) -> List[Tuple[float, float, float]]:
    """
    Generate a list of (x, y, z) world-frame waypoints from the mask.

    Traces the label from top-left to bottom-right using a simple raster scan
    of nonzero pixels. Returns tuples (x, Y_CANVAS, z). Case-sensitive cache.
    """
    if label in _WAYPOINT_CACHE:
        return _WAYPOINT_CACHE[label]

    mask = get_mask(label)
    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        # Degenerate: return single center point
        return [(0.0, Y_CANVAS, 1.5)]

    # Sort by row then column to trace top-to-bottom, left-to-right
    order = np.argsort(ys * 64 + xs)
    ys_sorted = ys[order]
    xs_sorted = xs[order]

    waypoints = []
    for pz, px in zip(ys_sorted, xs_sorted):
        wx, wz = _pixel_to_world(int(px), int(pz), size=64)
        waypoints.append((wx, Y_CANVAS, wz))

    # Subsample to ~100 waypoints for trajectory parameterization
    if len(waypoints) > 100:
        indices = np.linspace(0, len(waypoints) - 1, 100, dtype=int)
        waypoints = [waypoints[i] for i in indices]

    _WAYPOINT_CACHE[label] = waypoints
    return waypoints


def generate_all_masks(out_dir: Path, labels: List[str], size: int = 64) -> None:
    """
    Render and save masks for all specified labels to out_dir.

    Case-sensitive: "Pig" and "PIG" produce distinct files.
    Prints one line per label: 'letter=X px=N' matching AC-1 grep regex.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        mask = render_letter(label, size=size)
        _MASK_CACHE[label] = mask
        out_path = out_dir / f"{label}.png"
        save_mask(mask, out_path)
        px = int(mask.sum())
        print(f"letter={label} px={px}", flush=True)


def main() -> None:
    """CLI entry point: generate letter masks and save as PNGs."""
    parser = argparse.ArgumentParser(
        description="Generate 64x64 binary PNG masks for letters."
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for PNG masks.")
    parser.add_argument("--letters", nargs="+", default=["R", "L", "T", "I"],
                        help="Letters to render (default: R L T I).")
    parser.add_argument("--size", type=int, default=64,
                        help="Canvas size in pixels (default: 64).")
    args = parser.parse_args()

    generate_all_masks(args.out, args.letters, args.size)


if __name__ == "__main__":
    main()
