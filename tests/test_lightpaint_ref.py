from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.lightpaint_ref import (
    LightPaintRef,
    load_drawn_path_ref,
    make_drawn_path_ref,
    make_letter_ref,
    make_square_ref,
)


def test_square_ref_contract_and_geometry():
    ref = make_square_ref(side_m=0.8, center_z=1.5, speed=0.4)

    assert isinstance(ref, LightPaintRef)
    assert ref.waypoints.shape == (5, 3)
    assert np.isclose(ref.length, 3.2, atol=1e-6)
    assert np.isclose(ref.duration, 8.0, atol=1e-6)
    assert ref.waypoint_times.shape == (5,)
    assert ref.corner_times.shape == (3,)
    np.testing.assert_array_equal(ref.corner_indices, np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(ref.corner_times, ref.waypoint_times[ref.corner_indices], atol=1e-6)

    np.testing.assert_allclose(ref.pos(0.0), [-0.4, 0.0, 1.1], atol=1e-6)
    np.testing.assert_allclose(ref.pos(ref.duration), [-0.4, 0.0, 1.1], atol=1e-6)
    np.testing.assert_allclose(ref.vel(ref.duration + 0.1), [0.0, 0.0, 0.0], atol=1e-6)
    assert float(ref.yaw(0.0)) == 0.0


def test_straight_ref_has_no_sharp_corners():
    ref = LightPaintRef(
        waypoints=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
                [0.4, 0.0, 1.0],
                [0.6, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        speed=0.3,
    )

    assert ref.corner_indices.size == 0


def test_square_ref_vectorized_future_positions():
    ref = make_square_ref(speed=0.2)
    times = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pos = ref.pos(times)
    vel = ref.vel(times)
    future = ref.future_positions(0.0, count=15, dt=0.1)

    assert pos.shape == (3, 3)
    assert vel.shape == (3, 3)
    assert future.shape == (15, 3)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(vel))
    assert np.all(np.isfinite(future))


def test_letter_ref_supports_xz_and_xy_planes():
    ref_xz = make_letter_ref("A", plane="xz", speed=0.3, max_waypoints=40)
    ref_xy = make_letter_ref("A", plane="xy", speed=0.3, max_waypoints=40)

    assert ref_xz.plane == "xz"
    assert ref_xy.plane == "xy"
    assert ref_xz.waypoints.shape[1] == 3
    assert ref_xy.waypoints.shape[1] == 3
    assert len(ref_xz.waypoints) <= 40
    assert len(ref_xy.waypoints) <= 40
    assert np.ptp(ref_xz.waypoints[:, 2]) > 0.1
    assert np.ptp(ref_xy.waypoints[:, 1]) > 0.1
    np.testing.assert_allclose(ref_xy.waypoints[:, 2], np.full(len(ref_xy.waypoints), 1.5), atol=1e-6)
    np.testing.assert_allclose(ref_xz.waypoints[:, 1], np.zeros(len(ref_xz.waypoints)), atol=1e-6)


def test_drawn_path_ref_adds_led_off_connectors():
    ref = make_drawn_path_ref(
        strokes=[
            {"points": [[0.1, 0.2], [0.4, 0.2]], "led": True},
            {"points": [[0.8, 0.7], [0.8, 0.4]], "led": True},
        ],
        plane="xz",
        coordinate_space="normalized",
        speed=0.3,
        smooth=False,
    )

    assert isinstance(ref, LightPaintRef)
    assert ref.waypoints.shape[0] == 4
    assert ref.segment_led.shape == (3,)
    assert np.any(ref.segment_led == 0.0)
    assert np.any(ref.segment_led == 1.0)

    off_idx = int(np.where(ref.segment_led == 0.0)[0][0])
    t_mid_connector = float((ref.cumlen[off_idx] + ref.cumlen[off_idx + 1]) / (2.0 * ref.speed))
    assert float(ref.led(t_mid_connector)) == 0.0


def test_drawn_path_ref_can_leave_strokes_disconnected():
    ref = make_drawn_path_ref(
        strokes=[
            {"points": [[0.1, 0.2], [0.4, 0.2]], "led": True},
            {"points": [[0.8, 0.7], [0.8, 0.4]], "led": True},
        ],
        plane="xz",
        coordinate_space="normalized",
        speed=0.3,
        smooth=False,
        connect_strokes=False,
    )

    assert ref.waypoints.shape[0] == 4
    assert ref.segment_led.shape == (3,)
    assert np.all(ref.segment_led == 1.0)


def test_load_drawn_path_ref_from_json(tmp_path):
    path = tmp_path / "drawn.json"
    path.write_text(
        json.dumps({
            "name": "unit_drawn",
            "coordinate_space": "normalized",
            "plane": "xz",
            "strokes": [
                {"points": [[0.2, 0.2], [0.5, 0.5], [0.8, 0.2]], "led": True}
            ],
        }),
        encoding="utf-8",
    )

    ref = load_drawn_path_ref(path, speed=0.25, smooth=False)
    assert ref.name == "unit_drawn"
    assert ref.plane == "xz"
    assert ref.duration > 0.0
    assert np.all(ref.segment_led == 1.0)


def test_load_drawn_path_ref_applies_json_path_scale(tmp_path):
    path = tmp_path / "drawn_scaled.json"
    path.write_text(
        json.dumps({
            "name": "scaled_drawn",
            "coordinate_space": "normalized",
            "plane": "xz",
            "width_m": 0.5,
            "height_m": 0.5,
            "path_scale": 2.0,
            "strokes": [
                {"points": [[0.0, 0.0], [1.0, 0.0]], "led": True}
            ],
        }),
        encoding="utf-8",
    )

    ref = load_drawn_path_ref(path, speed=0.25, smooth=False)
    assert ref.name == "scaled_drawn"
    assert np.isclose(ref.length, 1.0, atol=1e-6)


def test_load_drawn_path_ref_cli_path_scale_overrides_json_scale(tmp_path):
    path = tmp_path / "drawn_scaled_override.json"
    path.write_text(
        json.dumps({
            "name": "scaled_override_drawn",
            "coordinate_space": "normalized",
            "plane": "xz",
            "width_m": 0.5,
            "height_m": 0.5,
            "path_scale": 2.0,
            "strokes": [
                {"points": [[0.0, 0.0], [1.0, 0.0]], "led": True}
            ],
        }),
        encoding="utf-8",
    )

    ref = load_drawn_path_ref(path, speed=0.25, path_scale=0.5, smooth=False)
    assert np.isclose(ref.length, 0.25, atol=1e-6)


def test_load_drawn_path_ref_explicit_args_override_json_smoothing(tmp_path):
    path = tmp_path / "drawn_json_smoothing.json"
    points = [[float(i) / 9.0, 0.0] for i in range(10)]
    path.write_text(
        json.dumps({
            "name": "json_smoothing",
            "coordinate_space": "normalized",
            "plane": "xz",
            "max_waypoints_per_stroke": 2,
            "smooth": True,
            "smooth_window_m": 0.2,
            "strokes": [{"points": points, "led": True}],
        }),
        encoding="utf-8",
    )

    ref = load_drawn_path_ref(
        path,
        speed=0.25,
        max_waypoints_per_stroke=5,
        smooth=False,
    )

    assert ref.waypoints.shape[0] == 5
