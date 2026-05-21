from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from src.env.lightpaint_ref import make_square_ref
from src.train import train_phase_b_m0_corner
from src.train.train_phase_b_m0_corner import (
    PAINTING_METRIC_KEYS,
    PAINTING_THRESHOLD,
    RolloutResult,
    _metrics,
    _painting_quality_metrics,
    _validate_config,
)


def _cfg(**overrides) -> Namespace:
    base = {
        "settle_time": 2.0,
        "init_box_size": 0.0,
        "ctrl_freq": 30,
        "pyb_freq": 240,
        "total_timesteps": 0,
        "eval_only": False,
        "load_model": None,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "bc_epochs": 0,
        "bc_episodes": 4,
        "bc_batch_size": 64,
        "speed": 0.35,
        "corner_window_m": 0.18,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.0,
        "clip_range": 0.2,
        "log_std_init": -2.0,
        "teacher_gain": 0.1,
        "teacher_window_m": 0.15,
        "bc_learning_rate": 1e-3,
        "bc_nonzero_weight": 1.0,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"learning_rate": -1.0}, "--learning-rate는 0보다 커야 합니다"),
        ({"gamma": 1.5}, "--gamma는 \\[0, 1\\] 범위여야 합니다"),
        ({"gae_lambda": -0.1}, "--gae-lambda는 \\[0, 1\\] 범위여야 합니다"),
        ({"clip_range": -0.2}, "--clip-range는 0보다 커야 합니다"),
        ({"teacher_gain": -0.1}, "--teacher-gain은 0 이상이어야 합니다"),
        ({"bc_learning_rate": -1.0}, "--bc-learning-rate는 0보다 커야 합니다"),
        ({"bc_nonzero_weight": -1.0}, "--bc-nonzero-weight는 0 이상이어야 합니다"),
        ({"init_box_size": -0.1}, "--init-box-size는 0 이상이어야 합니다"),
        ({"settle_time": -0.1}, "--settle-time은 0 이상이어야 합니다"),
    ],
)
def test_invalid_phase_b_config_fails_early(override, message):
    with pytest.raises(ValueError, match=message):
        _validate_config(_cfg(**override), max_steps=10)


def test_runtime_dependency_check_reports_missing_pybullet(monkeypatch):
    def fake_find_spec(name):
        if name == "pybullet":
            return None
        return object()

    monkeypatch.setattr(train_phase_b_m0_corner.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="pip install -r requirements.txt"):
        train_phase_b_m0_corner._check_runtime_dependencies()


def test_painting_quality_metrics_penalize_off_target_paint():
    target = [[1, 1, 0], [0, 1, 0]]
    cumulative = [[1, 0, 1], [0, 1, 0]]

    metrics = _painting_quality_metrics(cumulative, target)

    assert metrics["painted_pixel_coverage"] == pytest.approx(2 / 3)
    assert metrics["painted_pixel_recall"] == pytest.approx(2 / 3)
    assert metrics["painted_pixel_precision"] == pytest.approx(2 / 3)
    assert metrics["painted_pixel_iou"] == pytest.approx(0.5)
    assert metrics["painted_pixel_dice"] == pytest.approx(4 / 6)
    assert metrics["off_target_pixel_ratio"] == pytest.approx(1 / 3)
    assert metrics["painted_off_target_px"] == pytest.approx(1.0)


def test_painting_metric_schema_is_explicit():
    assert PAINTING_THRESHOLD == pytest.approx(0.3)
    assert {
        "painted_pixel_coverage",
        "painted_pixel_recall",
        "painted_pixel_precision",
        "painted_pixel_iou",
        "painted_pixel_dice",
        "off_target_pixel_ratio",
    } <= set(PAINTING_METRIC_KEYS)


def test_rollout_metrics_export_painting_quality_fields():
    ref = make_square_ref(side_m=0.2, speed=0.1)
    pos = np.asarray(ref.waypoints[:3], dtype=np.float32)
    result = RolloutResult(
        tag="phaseB_trained",
        phase="B",
        wind_mode="M0",
        time_arr=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        ref_arr=pos.copy(),
        v_ref_arr=np.zeros((3, 3), dtype=np.float32),
        pos_arr=pos.copy(),
        brightness_arr=np.ones(3, dtype=np.float32),
        rpm_arr=np.zeros((3, 4), dtype=np.float32),
        action_arr=np.zeros((3, 4), dtype=np.float32),
        delta_v_arr=np.zeros((3, 3), dtype=np.float32),
        reward_arr=np.asarray([0.1, 0.2], dtype=np.float32),
        info_rows=[],
        cumulative=np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.float32),
        target_mask=np.asarray([[1, 1, 0], [0, 1, 0]], dtype=np.float32),
        crash=False,
        walltime_s=0.01,
    )

    metrics = _metrics(result, ref, corner_window_m=0.1)

    assert metrics["painted_pixel_precision"] == "0.666667"
    assert metrics["painted_pixel_iou"] == "0.500000"
    assert metrics["painted_pixel_dice"] == "0.666667"
    assert metrics["off_target_pixel_ratio"] == "0.333333"
