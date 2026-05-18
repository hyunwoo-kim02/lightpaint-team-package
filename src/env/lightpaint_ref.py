"""DATT-style time-indexed references for light-painting trajectories.

The contract intentionally mirrors the small surface used from DATT refs:
``pos(t)``, ``vel(t)``, ``acc(t)``, and ``yaw(t)``.  Trajectory sources such as
waypoints, stroke JSON, or image-derived paths should compile into this class
before entering the PyBullet/PID stack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class LightPaintRef:
    """Piecewise-linear constant-speed reference with DATT-style methods."""

    waypoints: np.ndarray
    speed: float = 0.35
    yaw_value: float = 0.0
    name: str = "waypoint_ref"
    plane: str = "xz"
    segment_led: np.ndarray | None = None
    cumlen: np.ndarray = field(init=False)
    segment_lengths: np.ndarray = field(init=False)
    duration: float = field(init=False)
    length: float = field(init=False)
    corner_times: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        pts_raw = np.asarray(self.waypoints, dtype=np.float32)
        pts = pts_raw
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")
        if len(pts) < 2:
            raise ValueError("at least two waypoints are required")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive")
        if self.segment_led is None:
            raw_led = np.ones(len(pts) - 1, dtype=np.float32)
        else:
            raw_led = np.asarray(self.segment_led, dtype=np.float32).reshape(-1)
            if raw_led.shape != (len(pts) - 1,):
                raise ValueError("segment_led must have shape (N-1,)")
            raw_led = np.clip(raw_led, 0.0, 1.0)

        keep = [0]
        led_keep: List[float] = []
        last_kept_raw = 0
        for i in range(1, len(pts)):
            if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > 1e-6:
                led_keep.append(float(np.max(raw_led[last_kept_raw:i])))
                keep.append(i)
                last_kept_raw = i
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("waypoints collapse to a single point")
        seg_led = np.asarray(led_keep, dtype=np.float32)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        length = float(cum[-1])
        duration = length / float(self.speed)

        object.__setattr__(self, "waypoints", pts)
        object.__setattr__(self, "segment_led", seg_led)
        object.__setattr__(self, "segment_lengths", seg)
        object.__setattr__(self, "cumlen", cum)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "corner_times", (cum / float(self.speed)).astype(np.float32))

    def _flat_times(self, t) -> Tuple[np.ndarray, bool, Tuple[int, ...]]:
        arr = np.asarray(t, dtype=np.float32)
        scalar = arr.ndim == 0
        shape = () if scalar else arr.shape
        return arr.reshape(-1), scalar, shape

    def _segment_indices(self, flat_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.clip(flat_t * float(self.speed), 0.0, self.length)
        idx = np.searchsorted(self.cumlen, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.waypoints) - 2)
        return s.astype(np.float32), idx.astype(np.int64)

    @staticmethod
    def _restore(values: np.ndarray, scalar: bool, shape: Tuple[int, ...]):
        if scalar:
            return values[0]
        return values.reshape(shape + values.shape[1:])

    def pos(self, t):
        flat_t, scalar, shape = self._flat_times(t)
        s, idx = self._segment_indices(flat_t)
        seg_len = self.segment_lengths[idx]
        denom = np.maximum(seg_len, 1e-9)
        alpha = np.clip((s - self.cumlen[idx]) / denom, 0.0, 1.0).astype(np.float32)
        p0 = self.waypoints[idx]
        p1 = self.waypoints[idx + 1]
        out = p0 * (1.0 - alpha[:, None]) + p1 * alpha[:, None]
        return self._restore(out.astype(np.float32), scalar, shape)

    def vel(self, t):
        flat_t, scalar, shape = self._flat_times(t)
        _, idx = self._segment_indices(flat_t)
        tangent = self.waypoints[idx + 1] - self.waypoints[idx]
        denom = np.maximum(self.segment_lengths[idx], 1e-9)
        out = tangent / denom[:, None] * float(self.speed)
        moving = (flat_t >= 0.0) & (flat_t < self.duration)
        out[~moving] = 0.0
        return self._restore(out.astype(np.float32), scalar, shape)

    def acc(self, t):
        flat_t, scalar, shape = self._flat_times(t)
        out = np.zeros((len(flat_t), 3), dtype=np.float32)
        return self._restore(out, scalar, shape)

    def yaw(self, t):
        flat_t, scalar, shape = self._flat_times(t)
        out = np.full((len(flat_t),), float(self.yaw_value), dtype=np.float32)
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def led(self, t):
        """Return scheduled LED brightness for time t."""
        flat_t, scalar, shape = self._flat_times(t)
        _, idx = self._segment_indices(flat_t)
        out = self.segment_led[idx].astype(np.float32)
        moving = (flat_t >= 0.0) & (flat_t < self.duration)
        out[~moving] = 0.0
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def future_positions(self, t0: float, count: int = 15, dt: float = 0.1) -> np.ndarray:
        times = float(t0) + np.arange(int(count), dtype=np.float32) * float(dt)
        return np.asarray(self.pos(times), dtype=np.float32)


def make_waypoint_ref(
    waypoints: Iterable[Iterable[float]],
    speed: float = 0.35,
    yaw_value: float = 0.0,
    name: str = "waypoint_ref",
    plane: str = "xz",
    segment_led: Iterable[float] | None = None,
) -> LightPaintRef:
    pts = np.asarray(list(waypoints), dtype=np.float32)
    led = None if segment_led is None else np.asarray(list(segment_led), dtype=np.float32)
    return LightPaintRef(
        waypoints=pts,
        speed=float(speed),
        yaw_value=float(yaw_value),
        name=name,
        plane=plane,
        segment_led=led,
    )


def make_square_ref(
    side_m: float = 0.8,
    center_x: float = 0.0,
    center_z: float = 1.5,
    y: float = 0.0,
    speed: float = 0.35,
) -> LightPaintRef:
    """Create the active Square-PID-M0 gate reference in the x-z canvas plane."""
    half = float(side_m) / 2.0
    z0 = float(center_z) - half
    z1 = float(center_z) + half
    x0 = float(center_x) - half
    x1 = float(center_x) + half
    pts = np.array(
        [
            [x0, y, z0],
            [x1, y, z0],
            [x1, y, z1],
            [x0, y, z1],
            [x0, y, z0],
        ],
        dtype=np.float32,
    )
    return LightPaintRef(waypoints=pts, speed=float(speed), yaw_value=0.0, name="square", plane="xz")


def _dedupe_waypoints(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(pts) <= 1:
        return pts.copy()
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > eps:
            keep.append(i)
    return pts[keep].astype(np.float32)


def _resample_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    pts = _dedupe_waypoints(points)
    count = int(max(2, count))
    if len(pts) <= 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
    total = float(cum[-1])
    if total < 1e-6:
        return pts[:1].copy()
    samples = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.empty((count, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(samples, cum, pts[:, axis]).astype(np.float32)
    return out


def smooth_waypoint_path(
    waypoints: np.ndarray,
    max_waypoints: int = 240,
    samples_per_meter: float = 120.0,
    smooth_window_m: float = 0.08,
    polyorder: int = 3,
) -> np.ndarray:
    """Return a PID-friendly, arc-length-resampled version of a waypoint path.

    Raster skeleton paths contain pixel-grid stair steps. A PID controller sees
    those as rapid heading changes, so this function first resamples by arc
    length and then smooths the coordinate sequence while preserving endpoints.
    """
    pts = _dedupe_waypoints(waypoints)
    if len(pts) <= 3:
        return pts

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
    total = float(seg.sum())
    if total < 1e-6:
        return pts[:1].copy()

    dense_count = int(np.ceil(total * float(samples_per_meter))) + 1
    dense_count = int(np.clip(dense_count, min(len(pts), int(max_waypoints)), max(int(max_waypoints), 2)))
    dense = _resample_by_arclength(pts, dense_count)
    if len(dense) <= 5:
        return dense

    spacing = total / float(max(len(dense) - 1, 1))
    window = int(round(float(smooth_window_m) / max(spacing, 1e-6)))
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    if window >= len(dense):
        window = len(dense) - 1 if len(dense) % 2 == 0 else len(dense)
    if window < 5:
        return dense.astype(np.float32)

    smoothed = dense.copy()
    try:
        from scipy.signal import savgol_filter

        order = min(int(polyorder), window - 2)
        for axis in range(3):
            smoothed[:, axis] = savgol_filter(
                dense[:, axis],
                window_length=window,
                polyorder=max(1, order),
                mode="interp",
            ).astype(np.float32)
    except Exception:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        for axis in range(3):
            padded = np.pad(dense[:, axis], pad_width=pad, mode="edge")
            smoothed[:, axis] = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]
    return _dedupe_waypoints(smoothed)


def _drawn_point_to_world(
    point: Iterable[float],
    coordinate_space: str,
    plane: str,
    width_m: float,
    height_m: float,
    center_x: float,
    center_y: float,
    center_z: float,
    fixed_y: float,
    fixed_z: float,
    size: int,
) -> np.ndarray:
    vals = [float(v) for v in point]
    if coordinate_space == "world":
        if len(vals) == 3:
            return np.asarray(vals, dtype=np.float32)
        if len(vals) != 2:
            raise ValueError("world points must be [x,z], [x,y], or [x,y,z]")
        if plane == "xz":
            return np.asarray([vals[0], fixed_y, vals[1]], dtype=np.float32)
        return np.asarray([vals[0], vals[1], fixed_z], dtype=np.float32)

    if len(vals) != 2:
        raise ValueError(f"{coordinate_space} points must be 2-D")
    if coordinate_space == "pixel":
        u = vals[0] / float(size - 1)
        v = vals[1] / float(size - 1)
    elif coordinate_space == "normalized":
        u, v = vals
    else:
        raise ValueError("coordinate_space must be 'normalized', 'pixel', or 'world'")

    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    x = float(center_x) - float(width_m) / 2.0 + u * float(width_m)
    vertical = float(height_m) / 2.0 - v * float(height_m)
    if plane == "xz":
        return np.asarray([x, float(fixed_y), float(center_z) + vertical], dtype=np.float32)
    return np.asarray([x, float(center_y) + vertical, float(fixed_z)], dtype=np.float32)


def make_drawn_path_ref(
    strokes: Iterable[Any],
    plane: str = "xz",
    coordinate_space: str = "normalized",
    speed: float = 0.35,
    width_m: float = 0.9,
    height_m: float = 0.9,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 1.5,
    fixed_y: float = 0.0,
    fixed_z: float = 1.5,
    size: int = 64,
    max_waypoints_per_stroke: int = 120,
    smooth: bool = True,
    smooth_window_m: float = 0.06,
    connect_strokes: bool = True,
    name: str = "drawn_path",
) -> LightPaintRef:
    """Build a continuous flight reference from user-drawn strokes.

    Each stroke is an LED-on painting segment by default. Connector segments
    inserted between strokes are LED-off, so the drone can move continuously
    without painting unwanted lines.
    """
    if plane not in ("xz", "xy"):
        raise ValueError("plane must be 'xz' or 'xy'")

    all_points: List[np.ndarray] = []
    segment_led: List[float] = []

    for stroke_idx, stroke in enumerate(strokes):
        if isinstance(stroke, dict):
            raw_points = stroke.get("points", [])
            led_value = float(stroke.get("led", stroke.get("led_on", True)))
        else:
            raw_points = stroke
            led_value = 1.0
        raw_points = list(raw_points)
        if len(raw_points) < 2:
            continue

        stroke_pts = np.asarray(
            [
                _drawn_point_to_world(
                    pt,
                    coordinate_space=coordinate_space,
                    plane=plane,
                    width_m=width_m,
                    height_m=height_m,
                    center_x=center_x,
                    center_y=center_y,
                    center_z=center_z,
                    fixed_y=fixed_y,
                    fixed_z=fixed_z,
                    size=size,
                )
                for pt in raw_points
            ],
            dtype=np.float32,
        )
        stroke_pts = _dedupe_waypoints(stroke_pts)
        if len(stroke_pts) < 2:
            continue
        if smooth:
            stroke_pts = smooth_waypoint_path(
                stroke_pts,
                max_waypoints=int(max_waypoints_per_stroke),
                smooth_window_m=float(smooth_window_m),
                samples_per_meter=120.0,
            )
        elif len(stroke_pts) > int(max_waypoints_per_stroke):
            idx = np.linspace(0, len(stroke_pts) - 1, int(max_waypoints_per_stroke), dtype=int)
            stroke_pts = stroke_pts[idx]

        if not all_points:
            all_points.extend([p.copy() for p in stroke_pts])
            segment_led.extend([led_value] * (len(stroke_pts) - 1))
            continue

        prev = all_points[-1]
        first = stroke_pts[0]
        dist = float(np.linalg.norm(first - prev))
        if connect_strokes and dist > 1e-6:
            all_points.append(first.copy())
            segment_led.append(0.0)

        start_idx = 1 if connect_strokes and dist > 1e-6 else 0
        for p in stroke_pts[start_idx:]:
            all_points.append(p.copy())
            segment_led.append(led_value)

    if len(all_points) < 2:
        raise ValueError("drawn path must contain at least one stroke with two distinct points")

    return LightPaintRef(
        waypoints=np.asarray(all_points, dtype=np.float32),
        speed=float(speed),
        yaw_value=0.0,
        name=str(name),
        plane=plane,
        segment_led=np.asarray(segment_led, dtype=np.float32),
    )


def load_drawn_path_ref(
    path: str | Path,
    speed: float = 0.35,
    plane: str | None = None,
    coordinate_space: str | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    max_waypoints_per_stroke: int = 120,
    smooth: bool = True,
    smooth_window_m: float = 0.06,
) -> LightPaintRef:
    """Load a user-drawn stroke JSON file and compile it into LightPaintRef."""
    json_path = Path(path)
    with open(str(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    strokes = data.get("strokes")
    if strokes is None:
        raise ValueError("drawn path JSON must contain a 'strokes' list")
    return make_drawn_path_ref(
        strokes=strokes,
        plane=str(plane or data.get("plane", "xz")),
        coordinate_space=str(coordinate_space or data.get("coordinate_space", "normalized")),
        speed=float(speed if speed is not None else data.get("speed", 0.35)),
        width_m=float(width_m if width_m is not None else data.get("width_m", 0.9)),
        height_m=float(height_m if height_m is not None else data.get("height_m", 0.9)),
        center_x=float(data.get("center_x", 0.0)),
        center_y=float(data.get("center_y", 0.0)),
        center_z=float(data.get("center_z", 1.5)),
        fixed_y=float(data.get("fixed_y", 0.0)),
        fixed_z=float(data.get("fixed_z", 1.5)),
        size=int(data.get("size", 64)),
        max_waypoints_per_stroke=int(data.get("max_waypoints_per_stroke", max_waypoints_per_stroke)),
        smooth=bool(data.get("smooth", smooth)),
        smooth_window_m=float(data.get("smooth_window_m", smooth_window_m)),
        connect_strokes=bool(data.get("connect_strokes", True)),
        name=str(data.get("name", json_path.stem)),
    )


def _walk_mask_pixels(binary: np.ndarray, max_waypoints: int = 160) -> List[Tuple[int, int]]:
    """Return ordered skeleton pixels as (col, row) pairs."""
    binary = (np.asarray(binary) > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return []
    try:
        from skimage.morphology import skeletonize
        from skimage.measure import label as cc_label

        skel = skeletonize(binary, method="lee").astype(np.uint8)
        if skel.sum() == 0:
            skel = binary
        labels = cc_label(skel, connectivity=2)
        num_components = int(labels.max())
    except Exception:
        skel = binary
        labels = (binary > 0).astype(np.int32)
        num_components = 1

    if num_components <= 0:
        labels = (skel > 0).astype(np.int32)
        num_components = 1

    if num_components > 1:
        ordered_ids = []
        for cid in range(1, num_components + 1):
            rows, cols = np.where(labels == cid)
            if len(cols):
                ordered_ids.append((float(cols.mean()), cid))
        ordered_ids = [cid for _, cid in sorted(ordered_ids)]
    else:
        ordered_ids = [1]

    out: List[Tuple[int, int]] = []
    for cid in ordered_ids:
        if num_components > 1:
            rows, cols = np.where(labels == cid)
        else:
            rows, cols = np.where(skel > 0)
        out.extend(_walk_component_pixels(cols, rows))

    if len(out) > max_waypoints:
        idx = np.linspace(0, len(out) - 1, int(max_waypoints), dtype=int)
        out = [out[i] for i in idx]
    return out


def _walk_component_pixels(cols, rows) -> List[Tuple[int, int]]:
    pixels = set(zip([int(v) for v in cols], [int(v) for v in rows]))
    if not pixels:
        return []

    def degree(col: int, row: int) -> int:
        return sum(
            1
            for dc in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if (dc, dr) != (0, 0) and (col + dc, row + dr) in pixels
        )

    endpoints = [pt for pt in pixels if degree(*pt) == 1]
    if endpoints:
        endpoints.sort(key=lambda pt: (pt[1], pt[0]))
        start = endpoints[0]
    else:
        start = min(pixels, key=lambda pt: (pt[1], pt[0]))

    ordered: List[Tuple[int, int]] = [start]
    visited = {start}
    current = start
    while True:
        best, best_d = None, 10.0
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if dc == 0 and dr == 0:
                    continue
                cand = (current[0] + dc, current[1] + dr)
                if cand in pixels and cand not in visited:
                    d = float((dc * dc + dr * dr) ** 0.5)
                    if d < best_d:
                        best_d = d
                        best = cand
        if best is None:
            remaining = [p for p in pixels if p not in visited]
            if not remaining:
                break
            col, row = current
            best = min(remaining, key=lambda q: (q[0] - col) ** 2 + (q[1] - row) ** 2)
        ordered.append(best)
        visited.add(best)
        current = best
    return ordered


def make_letter_ref(
    letter: str,
    plane: str,
    speed: float = 0.35,
    width_m: float = 0.9,
    height_m: float = 0.9,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 1.5,
    fixed_y: float = 0.0,
    fixed_z: float = 1.5,
    size: int = 64,
    max_waypoints: int = 240,
    smooth: bool = True,
    smooth_window_m: float = 0.08,
    samples_per_meter: float = 120.0,
) -> LightPaintRef:
    """Create a text-mask reference in either x-z or fixed-z x-y plane."""
    if len(str(letter)) < 1 or not any(ch.isalpha() for ch in str(letter)):
        raise ValueError("letter must contain at least one alphabetic character")
    if plane not in ("xz", "xy"):
        raise ValueError("plane must be 'xz' or 'xy'")

    from src.env.letter_masks import render_letter

    label = str(letter)
    mask = render_letter(label.upper(), size=size)
    raw_max_waypoints = max(int(max_waypoints) * 3, int(max_waypoints))
    pixels = _walk_mask_pixels(mask, max_waypoints=raw_max_waypoints)
    if len(pixels) < 2:
        pixels = [(size // 2, size // 2), (size // 2 + 1, size // 2)]

    half_w = float(width_m) / 2.0
    half_h = float(height_m) / 2.0
    pts = []
    for col, row in pixels:
        u = float(col) / float(size - 1)
        v = float(row) / float(size - 1)
        x = float(center_x) - half_w + u * float(width_m)
        vertical = half_h - v * float(height_m)
        if plane == "xz":
            pts.append([x, float(fixed_y), float(center_z) + vertical])
        else:
            pts.append([x, float(center_y) + vertical, float(fixed_z)])

    pts_arr = np.asarray(pts, dtype=np.float32)
    if smooth:
        pts_arr = smooth_waypoint_path(
            pts_arr,
            max_waypoints=int(max_waypoints),
            smooth_window_m=float(smooth_window_m),
            samples_per_meter=float(samples_per_meter),
        )
    elif len(pts_arr) > int(max_waypoints):
        idx = np.linspace(0, len(pts_arr) - 1, int(max_waypoints), dtype=int)
        pts_arr = pts_arr[idx]

    return LightPaintRef(
        waypoints=pts_arr,
        speed=float(speed),
        yaw_value=0.0,
        name=f"letter_{label.upper()}_{plane}_{'smooth' if smooth else 'raw'}",
        plane=plane,
    )
