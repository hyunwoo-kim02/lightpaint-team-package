"""Time-indexed references for light-painting trajectories.

The reference contract exposes ``pos(t)``, ``vel(t)``, ``acc(t)``, and
``yaw(t)``. Trajectory sources such as
waypoints, stroke JSON, or image-derived paths should compile into this class
before entering the PyBullet/PID stack.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

DEFAULT_CORNER_MIN_SHARPNESS = 0.15
DEFAULT_CORNER_HINT_MIN_SHARPNESS = 0.12
DEFAULT_CORNER_MIN_SPACING_M = 0.36
DEFAULT_EFFECTIVE_CORNER_MAX_COUNT = 8
DEFAULT_CORNER_MASK_MAX_FRACTION = 0.18


@dataclass(frozen=True)
class LightPaintRef:
    """Piecewise-linear constant-speed reference."""

    waypoints: np.ndarray
    speed: float = 0.35
    yaw_value: float = 0.0
    name: str = "waypoint_ref"
    plane: str = "xz"
    segment_led: np.ndarray | None = None
    corner_distance_hints: np.ndarray | None = None
    cumlen: np.ndarray = field(init=False)
    segment_lengths: np.ndarray = field(init=False)
    duration: float = field(init=False)
    length: float = field(init=False)
    waypoint_times: np.ndarray = field(init=False)
    corner_times: np.ndarray = field(init=False)
    corner_indices: np.ndarray = field(init=False)
    corner_sharpness: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        pts_raw = np.asarray(self.waypoints, dtype=np.float32)
        pts = pts_raw
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("waypoints는 shape (N, 3)이어야 합니다")
        if len(pts) < 2:
            raise ValueError("waypoints는 최소 2개 이상 필요합니다")
        if self.speed <= 0.0:
            raise ValueError("speed는 0보다 커야 합니다")
        if self.segment_led is None:
            raw_led = np.ones(len(pts) - 1, dtype=np.float32)
        else:
            raw_led = np.asarray(self.segment_led, dtype=np.float32).reshape(-1)
            if raw_led.shape != (len(pts) - 1,):
                raise ValueError("segment_led는 shape (N-1,)이어야 합니다")
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
            raise ValueError("waypoints가 하나의 점으로 축약되었습니다")
        seg_led = np.asarray(led_keep, dtype=np.float32)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        length = float(cum[-1])
        duration = length / float(self.speed)
        waypoint_times = (cum / float(self.speed)).astype(np.float32)
        corner_indices, corner_sharpness = _detect_corner_indices(pts, cum)
        if self.corner_distance_hints is not None:
            hinted_indices = _corner_indices_from_distance_hints(pts, cum, self.corner_distance_hints)
            if hinted_indices.size:
                corner_indices = hinted_indices
                corner_sharpness = np.asarray(
                    [
                        max(
                            _corner_sharpness_at(pts, int(idx)),
                            DEFAULT_CORNER_HINT_MIN_SHARPNESS,
                        )
                        for idx in corner_indices
                    ],
                    dtype=np.float32,
                )

        object.__setattr__(self, "waypoints", pts)
        object.__setattr__(self, "segment_led", seg_led)
        object.__setattr__(self, "segment_lengths", seg)
        object.__setattr__(self, "cumlen", cum)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "waypoint_times", waypoint_times)
        object.__setattr__(self, "corner_times", waypoint_times[corner_indices].astype(np.float32))
        object.__setattr__(self, "corner_indices", corner_indices)
        object.__setattr__(self, "corner_sharpness", corner_sharpness)

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


def _corner_sharpness_at(pts: np.ndarray, idx: int) -> float:
    if len(pts) < 3 or idx <= 0 or idx >= len(pts) - 1:
        return 0.0
    a = pts[idx] - pts[idx - 1]
    b = pts[idx + 1] - pts[idx]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.clip((1.0 - cosang) * 0.5, 0.0, 1.0))


def _detect_corner_indices(
    pts: np.ndarray,
    cumlen: np.ndarray,
    *,
    min_sharpness: float = DEFAULT_CORNER_MIN_SHARPNESS,
    min_spacing_m: float = DEFAULT_CORNER_MIN_SPACING_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return waypoint indices whose heading change is a real corner."""
    pts = np.asarray(pts, dtype=np.float32)
    cumlen = np.asarray(cumlen, dtype=np.float32)
    if len(pts) < 3:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    candidates: list[tuple[float, int]] = []
    for idx in range(1, len(pts) - 1):
        sharpness = _corner_sharpness_at(pts, idx)
        if sharpness >= float(min_sharpness):
            candidates.append((sharpness, idx))
    kept: list[tuple[int, float]] = []
    for sharpness, idx in sorted(candidates, reverse=True):
        s = float(cumlen[idx])
        if all(abs(s - float(cumlen[other_idx])) >= float(min_spacing_m) for other_idx, _ in kept):
            kept.append((idx, sharpness))
    kept.sort(key=lambda item: item[0])
    return (
        np.asarray([idx for idx, _ in kept], dtype=np.int32),
        np.asarray([sharpness for _, sharpness in kept], dtype=np.float32),
    )


def _corner_indices_from_distance_hints(
    pts: np.ndarray,
    cumlen: np.ndarray,
    corner_distances: np.ndarray,
) -> np.ndarray:
    """Map raw-source corner arc lengths onto this reference's waypoint indices."""
    pts = np.asarray(pts, dtype=np.float32)
    cumlen = np.asarray(cumlen, dtype=np.float32)
    hints = np.asarray(corner_distances, dtype=np.float32).reshape(-1)
    if len(pts) < 3 or cumlen.size < 3 or hints.size == 0:
        return np.zeros(0, dtype=np.int32)
    length = float(cumlen[-1])
    kept: list[int] = []
    for raw_s in hints:
        s = float(np.clip(float(raw_s), 0.0, length))
        if s <= 1e-6 or s >= length - 1e-6:
            continue
        right = int(np.searchsorted(cumlen, s, side="left"))
        candidates = [
            int(np.clip(right - 1, 1, len(pts) - 2)),
            int(np.clip(right, 1, len(pts) - 2)),
        ]
        idx = min(candidates, key=lambda cand: abs(float(cumlen[cand]) - s))
        if idx not in kept:
            kept.append(idx)
    kept.sort()
    return np.asarray(kept, dtype=np.int32)


def effective_corner_indices(
    pts: np.ndarray,
    cumlen: np.ndarray,
    *,
    corner_indices: np.ndarray | None = None,
    corner_sharpness: np.ndarray | None = None,
    window_m: float = 0.18,
    max_corners: int = DEFAULT_EFFECTIVE_CORNER_MAX_COUNT,
) -> np.ndarray:
    """Return the corner index set shared by reward, teacher, and metrics."""
    pts = np.asarray(pts, dtype=np.float32)
    cumlen = np.asarray(cumlen, dtype=np.float32)
    if len(pts) < 3:
        return np.zeros(0, dtype=np.int32)
    if corner_indices is None:
        indices, sharpness = _detect_corner_indices(pts, cumlen)
    else:
        indices = np.asarray(corner_indices, dtype=np.int32).reshape(-1)
        if corner_sharpness is None:
            sharpness = np.asarray([_corner_sharpness_at(pts, int(i)) for i in indices], dtype=np.float32)
        else:
            sharpness = np.asarray(corner_sharpness, dtype=np.float32).reshape(-1)
            if sharpness.size != indices.size:
                sharpness = np.asarray([_corner_sharpness_at(pts, int(i)) for i in indices], dtype=np.float32)
    if indices.size <= 1:
        return indices.astype(np.int32)
    min_spacing = max(2.0 * float(window_m), DEFAULT_CORNER_MIN_SPACING_M)
    order = sorted(range(indices.size), key=lambda i: float(sharpness[i]), reverse=True)
    kept: list[int] = []
    for rel_idx in order:
        idx = int(indices[rel_idx])
        s = float(cumlen[idx])
        if all(abs(s - float(cumlen[int(indices[other_rel])])) >= min_spacing for other_rel in kept):
            kept.append(rel_idx)
    if int(max_corners) > 0 and len(kept) > int(max_corners):
        kept = sorted(kept, key=lambda rel_idx: float(sharpness[rel_idx]), reverse=True)[: int(max_corners)]
    kept.sort(key=lambda rel_idx: int(indices[rel_idx]))
    return np.asarray([int(indices[rel_idx]) for rel_idx in kept], dtype=np.int32)


def corner_window_radius_m(
    cumlen: np.ndarray,
    corner_indices: np.ndarray,
    window_m: float,
    *,
    max_fraction: float = DEFAULT_CORNER_MASK_MAX_FRACTION,
) -> float:
    """Cap each corner window so dense letter paths cannot become all-corner."""
    cumlen = np.asarray(cumlen, dtype=np.float32)
    indices = np.asarray(corner_indices, dtype=np.int32).reshape(-1)
    if indices.size == 0 or cumlen.size == 0:
        return 0.0
    length = float(cumlen[-1])
    radius_m = min(
        float(window_m),
        length * float(max_fraction) / max(2.0 * float(indices.size), 1.0),
    )
    if indices.size > 1:
        corner_s = cumlen[indices]
        gaps = np.diff(corner_s)
        if gaps.size:
            radius_m = min(radius_m, max(0.45 * float(np.min(gaps)), 1e-6))
    return float(max(radius_m, 0.0))


def corner_distances_for_ref(ref: "LightPaintRef", window_m: float) -> np.ndarray:
    indices = effective_corner_indices(
        ref.waypoints,
        ref.cumlen,
        corner_indices=getattr(ref, "corner_indices", None),
        corner_sharpness=getattr(ref, "corner_sharpness", None),
        window_m=window_m,
    )
    if indices.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(ref.cumlen[indices], dtype=np.float32)


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
            raise ValueError("world points는 [x,z], [x,y], [x,y,z] 중 하나여야 합니다")
        if plane == "xz":
            return np.asarray([vals[0], fixed_y, vals[1]], dtype=np.float32)
        return np.asarray([vals[0], vals[1], fixed_z], dtype=np.float32)

    if len(vals) != 2:
        raise ValueError(f"{coordinate_space} points는 2-D여야 합니다")
    if coordinate_space == "pixel":
        u = vals[0] / float(size - 1)
        v = vals[1] / float(size - 1)
    elif coordinate_space == "normalized":
        u, v = vals
    else:
        raise ValueError("coordinate_space는 'normalized', 'pixel', 'world' 중 하나여야 합니다")

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
    corner_distance_hints: Iterable[float] | None = None,
) -> LightPaintRef:
    """Build a continuous flight reference from user-drawn strokes.

    Each stroke is an LED-on painting segment by default. Connector segments
    inserted between strokes are LED-off, so the drone can move continuously
    without painting unwanted lines.
    """
    if plane not in ("xz", "xy"):
        raise ValueError("plane은 'xz' 또는 'xy'여야 합니다")

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
        corner_distance_hints=(
            None if corner_distance_hints is None else np.asarray(list(corner_distance_hints), dtype=np.float32)
        ),
    )


def load_drawn_path_ref(
    path: str | Path,
    speed: float = 0.35,
    plane: str | None = None,
    coordinate_space: str | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    path_scale: float | None = None,
    max_waypoints_per_stroke: int | None = None,
    smooth: bool | None = None,
    smooth_window_m: float | None = None,
) -> LightPaintRef:
    """Load a user-drawn stroke JSON file and compile it into LightPaintRef."""
    json_path = Path(path)
    with open(str(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    strokes = data.get("strokes")
    if strokes is None:
        raise ValueError("drawn path JSON에는 'strokes' list가 필요합니다")
    json_scale = data.get("path_scale", data.get("distance_scale", data.get("scale", 1.0)))
    effective_scale = float(path_scale if path_scale is not None else json_scale)
    if effective_scale <= 0.0:
        raise ValueError("drawn path scale은 0보다 커야 합니다")
    base_width_m = float(width_m if width_m is not None else data.get("width_m", 0.9))
    base_height_m = float(height_m if height_m is not None else data.get("height_m", 0.9))
    corner_distance_hints = data.get("corner_distance_hints_m", data.get("corner_distances_m"))
    return make_drawn_path_ref(
        strokes=strokes,
        plane=str(plane or data.get("plane", "xz")),
        coordinate_space=str(coordinate_space or data.get("coordinate_space", "normalized")),
        speed=float(speed if speed is not None else data.get("speed", 0.35)),
        width_m=base_width_m * effective_scale,
        height_m=base_height_m * effective_scale,
        center_x=float(data.get("center_x", 0.0)),
        center_y=float(data.get("center_y", 0.0)),
        center_z=float(data.get("center_z", 1.5)),
        fixed_y=float(data.get("fixed_y", 0.0)),
        fixed_z=float(data.get("fixed_z", 1.5)),
        size=int(data.get("size", 64)),
        max_waypoints_per_stroke=int(
            max_waypoints_per_stroke
            if max_waypoints_per_stroke is not None
            else data.get("max_waypoints_per_stroke", 120)
        ),
        smooth=bool(smooth if smooth is not None else data.get("smooth", True)),
        smooth_window_m=float(
            smooth_window_m
            if smooth_window_m is not None
            else data.get("smooth_window_m", 0.06)
        ),
        connect_strokes=bool(data.get("connect_strokes", True)),
        name=str(data.get("name", json_path.stem)),
        corner_distance_hints=corner_distance_hints,
    )


def _thin_binary_zhang_suen(binary: np.ndarray) -> np.ndarray:
    """Return a one-pixel skeleton without requiring scikit-image."""
    img = np.pad((np.asarray(binary) > 0).astype(np.uint8), 1)
    height, width = img.shape
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            delete: list[tuple[int, int]] = []
            for row in range(1, height - 1):
                for col in range(1, width - 1):
                    if img[row, col] == 0:
                        continue
                    p2 = img[row - 1, col]
                    p3 = img[row - 1, col + 1]
                    p4 = img[row, col + 1]
                    p5 = img[row + 1, col + 1]
                    p6 = img[row + 1, col]
                    p7 = img[row + 1, col - 1]
                    p8 = img[row, col - 1]
                    p9 = img[row - 1, col - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    neighbor_count = sum(int(v) for v in neighbors)
                    transitions = sum(
                        1
                        for left, right in zip(neighbors, neighbors[1:] + neighbors[:1])
                        if left == 0 and right == 1
                    )
                    if step == 0:
                        keep_shape = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        keep_shape = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if 2 <= neighbor_count <= 6 and transitions == 1 and keep_shape:
                        delete.append((row, col))
            if delete:
                changed = True
                for row, col in delete:
                    img[row, col] = 0
    return img[1:-1, 1:-1].astype(np.uint8)


def _label_binary_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    binary = (np.asarray(binary) > 0).astype(np.uint8)
    labels = np.zeros(binary.shape, dtype=np.int32)
    component_id = 0
    height, width = binary.shape
    rows, cols = np.where(binary > 0)
    for row, col in zip(rows, cols):
        row_i = int(row)
        col_i = int(col)
        if labels[row_i, col_i] != 0:
            continue
        component_id += 1
        labels[row_i, col_i] = component_id
        stack = [(row_i, col_i)]
        while stack:
            cur_row, cur_col = stack.pop()
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    if drow == 0 and dcol == 0:
                        continue
                    next_row = cur_row + drow
                    next_col = cur_col + dcol
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if binary[next_row, next_col] == 0 or labels[next_row, next_col] != 0:
                        continue
                    labels[next_row, next_col] = component_id
                    stack.append((next_row, next_col))
    return labels, component_id


def _walk_mask_components(binary: np.ndarray, max_waypoints: int = 160) -> List[List[Tuple[int, int]]]:
    """Return ordered skeleton components as (col, row) pixel paths."""
    binary = (np.asarray(binary) > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return []
    try:
        from skimage.morphology import skeletonize

        skel = skeletonize(binary, method="lee").astype(np.uint8)
    except Exception:
        skel = _thin_binary_zhang_suen(binary)
    if skel.sum() == 0:
        skel = binary
    try:
        from skimage.measure import label as cc_label

        labels = cc_label(skel, connectivity=2)
        num_components = int(labels.max())
    except Exception:
        labels, num_components = _label_binary_components(skel)

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

    components: List[List[Tuple[int, int]]] = []
    for cid in ordered_ids:
        if num_components > 1:
            rows, cols = np.where(labels == cid)
        else:
            rows, cols = np.where(skel > 0)
        walked = _walk_component_pixels(cols, rows)
        if walked:
            components.append(walked)

    total = sum(len(component) for component in components)
    if total > int(max_waypoints) and components:
        budget = max(int(max_waypoints), len(components))
        lengths = np.asarray([len(component) for component in components], dtype=np.float32)
        raw_counts = lengths / max(float(lengths.sum()), 1.0) * float(budget)
        counts = np.maximum(1, np.floor(raw_counts).astype(np.int32))
        while int(counts.sum()) > budget:
            candidates = np.where(counts > 1)[0]
            if candidates.size == 0:
                break
            idx = int(candidates[np.argmax(counts[candidates])])
            counts[idx] -= 1
        while int(counts.sum()) < budget:
            idx = int(np.argmax(raw_counts - counts))
            counts[idx] += 1
        limited: List[List[Tuple[int, int]]] = []
        for component, count in zip(components, counts):
            if len(component) > int(count):
                idx = np.linspace(0, len(component) - 1, int(count), dtype=int)
                limited.append([component[i] for i in idx])
            else:
                limited.append(component)
        components = limited
    return components


def _walk_mask_pixels(binary: np.ndarray, max_waypoints: int = 160) -> List[Tuple[int, int]]:
    """Return ordered skeleton pixels as (col, row) pairs."""
    out: List[Tuple[int, int]] = []
    for component in _walk_mask_components(binary, max_waypoints=max_waypoints):
        out.extend(component)
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
        raise ValueError("letter에는 알파벳 문자가 최소 1개 필요합니다")
    if plane not in ("xz", "xy"):
        raise ValueError("plane은 'xz' 또는 'xy'여야 합니다")

    from src.env.letter_masks import render_letter

    label = str(letter)
    mask = render_letter(label, size=size)
    raw_max_waypoints = max(int(max_waypoints) * 3, int(max_waypoints))
    components = _walk_mask_components(mask, max_waypoints=raw_max_waypoints)
    if not components:
        components = [[(size // 2, size // 2), (size // 2 + 1, size // 2)]]

    half_w = float(width_m) / 2.0
    half_h = float(height_m) / 2.0
    total_pixels = max(sum(len(component) for component in components), 1)
    component_budgets = [
        max(2, int(round(float(max_waypoints) * len(component) / float(total_pixels))))
        for component in components
    ]
    while sum(component_budgets) > int(max_waypoints) and any(count > 2 for count in component_budgets):
        idx = max(range(len(component_budgets)), key=lambda i: component_budgets[i])
        component_budgets[idx] -= 1

    all_points: List[np.ndarray] = []
    segment_led: List[float] = []
    corner_hint_values: List[float] = []
    raw_length_offset = 0.0
    last_raw_point: np.ndarray | None = None

    def pixel_to_point(col: int, row: int) -> List[float]:
        u = float(col) / float(size - 1)
        v = float(row) / float(size - 1)
        x = float(center_x) - half_w + u * float(width_m)
        vertical = half_h - v * float(height_m)
        if plane == "xz":
            return [x, float(fixed_y), float(center_z) + vertical]
        return [x, float(center_y) + vertical, float(fixed_z)]

    for component, budget in zip(components, component_budgets):
        pts_arr = np.asarray([pixel_to_point(col, row) for col, row in component], dtype=np.float32)
        pts_arr = _dedupe_waypoints(pts_arr)
        if len(pts_arr) < 2:
            continue
        if last_raw_point is not None:
            connector_dist = float(np.linalg.norm(pts_arr[0] - last_raw_point))
            if connector_dist > 1e-6:
                raw_length_offset += connector_dist
        raw_seg = np.linalg.norm(np.diff(pts_arr, axis=0), axis=1).astype(np.float32)
        raw_cum = np.concatenate([[0.0], np.cumsum(raw_seg)]).astype(np.float32)
        raw_corner_indices, _ = _detect_corner_indices(
            pts_arr,
            raw_cum,
            min_sharpness=DEFAULT_CORNER_HINT_MIN_SHARPNESS,
        )
        if raw_corner_indices.size:
            corner_hint_values.extend((raw_length_offset + raw_cum[raw_corner_indices]).astype(np.float32).tolist())
        raw_length_offset += float(raw_cum[-1])
        last_raw_point = pts_arr[-1].copy()

        if smooth:
            pts_arr = smooth_waypoint_path(
                pts_arr,
                max_waypoints=int(budget),
                smooth_window_m=float(smooth_window_m),
                samples_per_meter=float(samples_per_meter),
            )
        elif len(pts_arr) > int(budget):
            idx = np.linspace(0, len(pts_arr) - 1, int(budget), dtype=int)
            pts_arr = pts_arr[idx]
        if len(pts_arr) < 2:
            continue

        if not all_points:
            all_points.extend([p.copy() for p in pts_arr])
            segment_led.extend([1.0] * (len(pts_arr) - 1))
            continue

        prev = all_points[-1]
        first = pts_arr[0]
        dist = float(np.linalg.norm(first - prev))
        if dist > 1e-6:
            all_points.append(first.copy())
            segment_led.append(0.0)
            start_idx = 1
        else:
            start_idx = 1
        for p in pts_arr[start_idx:]:
            all_points.append(p.copy())
            segment_led.append(1.0)

    if len(all_points) < 2:
        all_points = [
            np.asarray(pixel_to_point(size // 2, size // 2), dtype=np.float32),
            np.asarray(pixel_to_point(size // 2 + 1, size // 2), dtype=np.float32),
        ]
        segment_led = [1.0]

    pts_arr = np.asarray(all_points, dtype=np.float32)
    corner_distance_hints = (
        np.asarray(corner_hint_values, dtype=np.float32) if corner_hint_values else None
    )
    return LightPaintRef(
        waypoints=pts_arr,
        speed=float(speed),
        yaw_value=0.0,
        name=f"letter_{label}_{plane}_{'smooth' if smooth else 'raw'}",
        plane=plane,
        segment_led=np.asarray(segment_led, dtype=np.float32),
        corner_distance_hints=corner_distance_hints,
    )
