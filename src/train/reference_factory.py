"""Shared reference construction for train/eval scripts.

The training stack should consume one compiled ``LightPaintRef`` regardless of
whether the source is a square, a rendered letter, or a JSON file exported from
``tools/draw_path.html``.  This module keeps those CLI options and metadata in
one place so team experiments do not drift between scripts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
            "path_scale": effective_scale,
            "plane": ref.plane,
        }
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
    return ReferenceBuild(label=label, token=_safe_token(label), reference=ref, metadata=metadata)
