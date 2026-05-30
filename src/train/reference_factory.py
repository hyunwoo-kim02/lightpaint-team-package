"""Shared reference construction for train/eval scripts.

The training stack should consume one compiled ``LightPaintRef`` regardless of
whether the source is a square, a rendered letter, or a JSON file exported from
``tools/draw_path.html``.  This module keeps those CLI options and metadata in
one place so team experiments do not drift between scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.env.lightpaint_ref import (
    LightPaintRef,
    load_drawn_path_ref,
    make_letter_ref,
    make_square_ref,
)


_PKG_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ReferenceBuild:
    """Compiled reference plus the metadata needed for artifacts."""

    label: str
    token: str
    reference: LightPaintRef
    metadata: dict[str, Any]


def _positive_scale(raw: float | None) -> float:
    if raw is None:
        return 1.0
    scale = float(raw)
    if scale <= 0.0:
        raise ValueError("--path-scale은 0보다 커야 합니다")
    return scale


def _safe_token(value: str) -> str:
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep) or "run"


def _resolve_package_relative_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return _PKG_ROOT / path


def _package_relative(path: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(_PKG_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distance_along_ref_for_point(ref: LightPaintRef, point: Any) -> float:
    pts = ref.waypoints
    point_arr = np.asarray(point, dtype=np.float32).reshape(-1)
    if point_arr.size == 2:
        if ref.plane == "xz":
            point_arr = np.asarray([point_arr[0], 0.0, point_arr[1]], dtype=np.float32)
        else:
            point_arr = np.asarray([point_arr[0], point_arr[1], 0.0], dtype=np.float32)
    if point_arr.size != 3:
        raise ValueError("corner point must contain 2 or 3 coordinates")
    seg = pts[1:] - pts[:-1]
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-9)
    rel = point_arr[None, :] - pts[:-1]
    alpha = np.clip(np.sum(rel * seg, axis=1) / seg_len2, 0.0, 1.0)
    closest = pts[:-1] + alpha[:, None] * seg
    dist = np.linalg.norm(closest - point_arr[None, :], axis=1)
    idx = int(np.argmin(dist))
    return float(ref.cumlen[idx] + alpha[idx] * ref.segment_lengths[idx])


def _load_corner_distance_hints(path: Path, ref: LightPaintRef) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw = data.get("corner_distance_hints_m", data.get("corner_distances_m", data.get("corner_s_m")))
    values: list[float] = []
    if raw is not None:
        values.extend(float(v) for v in raw)
    for corner in data.get("corners", []):
        if isinstance(corner, (int, float)):
            values.append(float(corner))
        elif isinstance(corner, dict):
            if "s_m" in corner:
                values.append(float(corner["s_m"]))
            elif "distance_m" in corner:
                values.append(float(corner["distance_m"]))
            elif "t_s" in corner:
                values.append(float(corner["t_s"]) * float(ref.speed))
            elif "time_s" in corner:
                values.append(float(corner["time_s"]) * float(ref.speed))
            elif "point" in corner:
                values.append(_distance_along_ref_for_point(ref, corner["point"]))
    if not values:
        raise ValueError(f"corner hint file has no corner distances: {path}")
    hints = np.asarray(sorted(set(round(float(v), 6) for v in values)), dtype=np.float32)
    return np.clip(hints, 0.0, float(ref.length)).astype(np.float32)


def _with_corner_distance_hints(ref: LightPaintRef, hints: np.ndarray) -> LightPaintRef:
    return LightPaintRef(
        waypoints=ref.waypoints.copy(),
        speed=float(ref.speed),
        yaw_value=float(ref.yaw_value),
        name=ref.name,
        plane=ref.plane,
        segment_led=ref.segment_led.copy(),
        corner_distance_hints=hints,
    )


def _apply_optional_corner_hints(
    args: argparse.Namespace,
    ref: LightPaintRef,
    metadata: dict[str, Any],
) -> tuple[LightPaintRef, dict[str, Any]]:
    raw_path = getattr(args, "corner_hints_path", None)
    if raw_path in (None, ""):
        return ref, metadata
    path = _resolve_package_relative_path(raw_path)
    hints = _load_corner_distance_hints(path, ref)
    updated = dict(metadata)
    updated.update({
        "corner_hints_path": str(raw_path),
        "resolved_corner_hints_path": str(path),
        "corner_hints_relative_path": _package_relative(path),
        "corner_hints_sha256": _file_sha256(path),
        "corner_distance_hints_m": [float(v) for v in hints],
    })
    return _with_corner_distance_hints(ref, hints), updated


def _smooth_arg(args: argparse.Namespace) -> bool | None:
    if hasattr(args, "smooth_ref"):
        value = getattr(args, "smooth_ref")
        return None if value is None else bool(value)
    if hasattr(args, "no_smooth_ref"):
        return not bool(getattr(args, "no_smooth_ref"))
    return None


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    """Add common reference-source arguments to a script parser."""
    parser.add_argument(
        "--trajectory",
        choices=("square", "letter", "drawn"),
        default="square",
        help="Reference source: generated square, rendered text, or drawn JSON.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Text to render when --trajectory letter is selected. Defaults to L.",
    )
    parser.add_argument(
        "--square-side",
        type=float,
        default=0.8,
        help="Unscaled square side length in meters.",
    )
    parser.add_argument(
        "--letter-plane",
        choices=("xz", "xy"),
        default="xz",
        help="Plane used when --trajectory letter is selected.",
    )
    parser.add_argument(
        "--drawn-path",
        type=str,
        default=None,
        help="JSON path exported by tools/draw_path.html or another stroke editor.",
    )
    parser.add_argument(
        "--drawn-plane",
        choices=("xz", "xy"),
        default=None,
        help="Override plane from drawn JSON. If omitted, the JSON plane is used.",
    )
    parser.add_argument(
        "--drawn-space",
        choices=("normalized", "pixel", "world"),
        default=None,
        help="Override coordinate_space from drawn JSON. If omitted, the JSON value is used.",
    )
    parser.add_argument(
        "--width-m",
        type=float,
        default=None,
        help="Unscaled reference width in meters for letter/drawn paths. Drawn JSON width_m is used when omitted.",
    )
    parser.add_argument(
        "--height-m",
        type=float,
        default=None,
        help="Unscaled reference height in meters for letter/drawn paths. Drawn JSON height_m is used when omitted.",
    )
    parser.add_argument(
        "--path-scale",
        type=float,
        default=None,
        help="Positive distance multiplier applied when generating the reference path.",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=None,
        help="Maximum smoothed waypoints for letter trajectories and per drawn stroke.",
    )
    parser.add_argument(
        "--smooth-window-m",
        type=float,
        default=None,
        help="Smoothing window in meters for generated references.",
    )
    parser.add_argument(
        "--smooth-ref",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable reference smoothing. If omitted, drawn JSON settings are used when present.",
    )
    parser.add_argument(
        "--corner-hints-path",
        type=str,
        default=None,
        help="Optional JSON file with manual corner_distance_hints_m values exported by tools/corner_hint_editor.html.",
    )


def build_reference_from_args(args: argparse.Namespace) -> ReferenceBuild:
    """Build one LightPaintRef from common argparse fields."""
    scale = _positive_scale(getattr(args, "path_scale", None))
    speed = float(getattr(args, "speed", 0.35))
    trajectory = str(getattr(args, "trajectory", "square"))

    if trajectory == "square":
        side = float(args.square_side) * scale
        ref = make_square_ref(side_m=side, center_x=0.0, center_z=1.5, speed=speed)
        label = "square"
        metadata = {
            "trajectory": "square",
            "label": label,
            "path_scale": scale,
            "square_side_m": side,
            "unscaled_square_side_m": float(args.square_side),
        }
        ref, metadata = _apply_optional_corner_hints(args, ref, metadata)
        return ReferenceBuild(label=label, token=_safe_token(label), reference=ref, metadata=metadata)

    if trajectory == "drawn":
        if getattr(args, "drawn_path", None) is None:
            raise ValueError("--trajectory drawn은 --drawn-path가 필요합니다")
        raw_json_path = Path(args.drawn_path)
        json_path = _resolve_package_relative_path(raw_json_path)
        raw_path_scale = getattr(args, "path_scale", None)
        ref = load_drawn_path_ref(
            path=json_path,
            speed=speed,
            plane=getattr(args, "drawn_plane", None),
            coordinate_space=getattr(args, "drawn_space", None),
            width_m=getattr(args, "width_m", None),
            height_m=getattr(args, "height_m", None),
            path_scale=raw_path_scale,
            max_waypoints_per_stroke=(
                None if getattr(args, "max_waypoints", None) is None else int(args.max_waypoints)
            ),
            smooth=_smooth_arg(args),
            smooth_window_m=(
                None if getattr(args, "smooth_window_m", None) is None else float(args.smooth_window_m)
            ),
        )
        if raw_path_scale is None:
            with open(str(json_path), "r", encoding="utf-8") as f:
                data = json.load(f)
            effective_scale = _positive_scale(data.get("path_scale", data.get("distance_scale", data.get("scale", 1.0))))
        else:
            effective_scale = scale
        label = raw_json_path.stem
        metadata = {
            "trajectory": "drawn",
            "label": label,
            "drawn_path": str(raw_json_path),
            "resolved_drawn_path": str(json_path),
            "drawn_relative_path": _package_relative(json_path),
            "drawn_sha256": _file_sha256(json_path),
            "drawn_size_bytes": json_path.stat().st_size,
            "path_scale": effective_scale,
            "plane": ref.plane,
        }
        ref, metadata = _apply_optional_corner_hints(args, ref, metadata)
        return ReferenceBuild(label=label, token=_safe_token(label), reference=ref, metadata=metadata)

    label = str(args.label if getattr(args, "label", None) is not None else "L")
    width = float(args.width_m if getattr(args, "width_m", None) is not None else 0.9) * scale
    height = float(args.height_m if getattr(args, "height_m", None) is not None else 0.9) * scale
    ref = make_letter_ref(
        letter=label,
        plane=str(args.letter_plane),
        speed=speed,
        width_m=width,
        height_m=height,
        center_x=0.0,
        center_z=1.5,
        fixed_y=0.0,
        fixed_z=1.5,
        max_waypoints=int(args.max_waypoints if getattr(args, "max_waypoints", None) is not None else 240),
        smooth=True if _smooth_arg(args) is None else bool(_smooth_arg(args)),
        smooth_window_m=float(
            args.smooth_window_m if getattr(args, "smooth_window_m", None) is not None else 0.08
        ),
    )
    metadata = {
        "trajectory": "letter",
        "label": label,
        "path_scale": scale,
        "width_m": width,
        "height_m": height,
        "plane": ref.plane,
    }
    ref, metadata = _apply_optional_corner_hints(args, ref, metadata)
    return ReferenceBuild(label=label, token=_safe_token(label), reference=ref, metadata=metadata)
