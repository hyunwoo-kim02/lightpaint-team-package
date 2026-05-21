from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.train.reference_factory import build_reference_from_args
from src.train.train_phase_b_m0_corner import _corner_mask, _effective_corner_distances


def _args(**overrides) -> Namespace:
    base = {
        "trajectory": "square",
        "label": None,
        "square_side": 0.8,
        "letter_plane": "xz",
        "drawn_path": None,
        "drawn_plane": None,
        "drawn_space": None,
        "width_m": None,
        "height_m": None,
        "path_scale": None,
        "max_waypoints": 240,
        "smooth_window_m": 0.08,
        "no_smooth_ref": False,
        "speed": 0.35,
    }
    base.update(overrides)
    return Namespace(**base)


def test_reference_factory_scales_square():
    built = build_reference_from_args(_args(trajectory="square", square_side=0.8, path_scale=0.5))

    assert built.label == "square"
    assert built.token == "square"
    assert built.metadata["square_side_m"] == 0.4
    assert np.isclose(built.reference.length, 1.6, atol=1e-6)


def test_reference_factory_loads_draw_path_html_json(tmp_path):
    drawn = tmp_path / "team_path.json"
    drawn.write_text(
        json.dumps({
            "name": "team_path",
            "coordinate_space": "normalized",
            "plane": "xz",
            "width_m": 0.9,
            "height_m": 0.9,
            "path_scale": 1.0,
            "connect_strokes": True,
            "strokes": [
                {"led": True, "points": [[0.0, 0.0], [1.0, 0.0]]}
            ],
        }),
        encoding="utf-8",
    )

    built = build_reference_from_args(
        _args(trajectory="drawn", drawn_path=str(drawn), path_scale=0.5, no_smooth_ref=True)
    )

    assert built.label == "team_path"
    assert built.metadata["trajectory"] == "drawn"
    assert built.metadata["drawn_path"] == str(drawn)
    assert np.isclose(built.reference.length, 0.45, atol=1e-6)


def test_reference_factory_uses_drawn_json_scale_when_cli_scale_omitted(tmp_path):
    drawn = tmp_path / "team_path_scaled.json"
    drawn.write_text(
        json.dumps({
            "name": "team_path_scaled",
            "coordinate_space": "normalized",
            "plane": "xz",
            "width_m": 0.9,
            "height_m": 0.9,
            "path_scale": 0.5,
            "connect_strokes": True,
            "strokes": [
                {"led": True, "points": [[0.0, 0.0], [1.0, 0.0]]}
            ],
        }),
        encoding="utf-8",
    )

    built = build_reference_from_args(
        _args(trajectory="drawn", drawn_path=str(drawn), path_scale=None, no_smooth_ref=True)
    )

    assert built.metadata["path_scale"] == 0.5
    assert np.isclose(built.reference.length, 0.45, atol=1e-6)


def test_reference_factory_resolves_relative_drawn_path_from_package_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    built = build_reference_from_args(
        _args(
            trajectory="drawn",
            drawn_path="data/drawn_paths/examples/example_two_strokes.json",
            path_scale=0.5,
            no_smooth_ref=True,
        )
    )

    assert built.label == "example_two_strokes"
    assert built.metadata["drawn_path"].replace("\\", "/") == "data/drawn_paths/examples/example_two_strokes.json"
    assert Path(built.metadata["resolved_drawn_path"]).is_absolute()
    assert built.reference.length > 0.0


def test_reference_factory_scales_letter():
    built = build_reference_from_args(
        _args(trajectory="letter", label="L", width_m=0.6, height_m=0.8, path_scale=0.5, max_waypoints=40)
    )

    assert built.label == "L"
    assert built.metadata["width_m"] == 0.3
    assert built.metadata["height_m"] == 0.4
    assert built.reference.length > 0.0


def test_phase_b_corner_mask_uses_sharp_corner_indices_only():
    built = build_reference_from_args(_args(trajectory="square", square_side=0.8))
    ref = built.reference
    times = np.linspace(0.0, ref.duration, 120, dtype=np.float32)
    mask = _corner_mask(ref, times, window_m=0.18)

    assert mask.any()
    assert not mask.all()


def test_phase_b_corner_mask_empty_for_straight_drawn_path(tmp_path):
    drawn = tmp_path / "straight.json"
    drawn.write_text(
        json.dumps({
            "coordinate_space": "normalized",
            "plane": "xz",
            "strokes": [{"led": True, "points": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]}],
        }),
        encoding="utf-8",
    )
    built = build_reference_from_args(
        _args(trajectory="drawn", drawn_path=str(drawn), smooth_ref=False, no_smooth_ref=False)
    )
    times = np.linspace(0.0, built.reference.duration, 20, dtype=np.float32)

    assert not _corner_mask(built.reference, times, window_m=0.18).any()


def test_letter_corner_mask_does_not_cover_whole_path():
    for label, max_fraction in (
        ("L", 0.20),
        ("DG", 0.40),
        ("CAT", 0.40),
        ("Pig", 0.40),
        ("RL", 0.40),
    ):
        built = build_reference_from_args(_args(trajectory="letter", label=label))
        times = np.linspace(0.0, built.reference.duration, 500, dtype=np.float32)
        mask = _corner_mask(built.reference, times, window_m=0.18)
        corners = _effective_corner_distances(built.reference, window_m=0.18)

        assert len(corners) <= 8
        assert float(np.mean(mask)) <= max_fraction
