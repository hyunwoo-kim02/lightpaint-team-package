from __future__ import annotations

from argparse import Namespace
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.env.lightpaint_ref import make_square_ref
from src.train import train_phase_b_m0_corner
from src.train.train_phase_b_m0_corner import (
    FINAL_GOAL_THRESHOLDS,
    PAINTING_METRIC_KEYS,
    PAINTING_THRESHOLD,
    RolloutResult,
    _effective_corner_distances,
    _final_goal_criteria,
    _filter_corner_tangent_decel_action,
    _load_or_make_model,
    _make_corner_decel_teacher,
    _metrics,
    _painting_quality_metrics,
    _load_saved_model_for_rollout,
    _validate_config,
    _wrap_train_action_filter,
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
        "gui_hold_seconds": 3.0,
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


def test_saved_model_rollout_reloads_persisted_artifact(monkeypatch, tmp_path):
    calls = []

    class FakePPO:
        @staticmethod
        def load(path, env, device):
            calls.append((path, env, device))
            return "loaded-model"

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FakePPO))

    model = _load_saved_model_for_rollout(tmp_path / "model.zip", "vec-env", "cuda")

    assert model == "loaded-model"
    assert calls == [(str(tmp_path / "model.zip"), "vec-env", "cuda")]


def test_continuation_load_applies_cli_ppo_hyperparameters(monkeypatch, tmp_path):
    calls = []

    class LoadedModel:
        def set_random_seed(self, seed):
            self.seed = seed

    class FakePPO:
        @staticmethod
        def load(path, env, device, **kwargs):
            calls.append((path, env, device, kwargs))
            return LoadedModel()

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FakePPO))
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    args = _cfg(
        load_model=str(model_path),
        device="cuda",
        seed=7,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        learning_rate=1e-4,
        gamma=0.97,
        gae_lambda=0.91,
        ent_coef=0.01,
        clip_range=0.1,
        verbose=0,
    )

    model, source_path = _load_or_make_model(args, "vec-env")

    assert source_path == str(model_path)
    assert model.seed == 7
    assert calls == [
        (
            str(model_path),
            "vec-env",
            "cuda",
            {
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 4,
                "learning_rate": 1e-4,
                "gamma": 0.97,
                "gae_lambda": 0.91,
                "ent_coef": 0.01,
                "clip_range": 0.1,
                "verbose": 0,
            },
        )
    ]


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
        brightness_arr=np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
        rpm_arr=np.zeros((3, 4), dtype=np.float32),
        action_arr=np.zeros((3, 4), dtype=np.float32),
        delta_v_arr=np.zeros((3, 3), dtype=np.float32),
        reward_arr=np.asarray([0.1, 0.2], dtype=np.float32),
        info_rows=[
            {"led_ref": 1.0, "out_of_bounds": False, "max_rpm": 50000.0},
            {"led_ref": 1.0, "out_of_bounds": False, "max_rpm": 50000.0},
        ],
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
    assert metrics["path_rmse_m"] != ""
    assert "corner_speed_vs_straight_ratio" in metrics
    assert metrics["led_precision"] == "1.000000"
    assert metrics["led_recall"] == "1.000000"
    assert metrics["out_of_bounds_rate"] == "0.000000"
    assert metrics["rpm_saturation_ratio"] == "0.000000"
    assert metrics["action_norm_max"] == "0.000000"
    assert metrics["action_rate_max"] == "0.000000"


def test_corner_teacher_and_filter_apply_across_corner_window():
    ref = make_square_ref(side_m=0.8, speed=0.4)
    cat_ref = make_square_ref(side_m=0.8, speed=0.4)
    pig_ref = make_square_ref(side_m=0.8, speed=0.4)
    rl_ref = make_square_ref(side_m=0.8, speed=0.4)
    dg_ref = make_square_ref(side_m=0.8, speed=0.4)
    l_ref = make_square_ref(side_m=0.8, speed=0.4)
    drawn_ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(cat_ref, "name", "letter_CAT_xz_smooth")
    object.__setattr__(pig_ref, "name", "letter_Pig_xz_smooth")
    object.__setattr__(rl_ref, "name", "letter_RL_xz_smooth")
    object.__setattr__(dg_ref, "name", "letter_DG_xz_smooth")
    object.__setattr__(l_ref, "name", "letter_L_xz_smooth")
    object.__setattr__(drawn_ref, "name", "drawn_user_path")
    corner_s = _effective_corner_distances(ref, window_m=0.18)
    assert corner_s.size > 0
    corner_time = float(corner_s[0] / ref.speed)
    teacher = _make_corner_decel_teacher(ref, gain_mps=0.2, window_m=0.18)
    before = {"info": {"t": corner_time - 0.1, "v_ref": [0.4, 0.0, 0.0]}}
    after = {"info": {"t": corner_time + 0.1, "v_ref": [0.0, 0.0, -0.4]}}
    far = {"info": {"t": 0.0, "v_ref": [0.4, 0.0, 0.0]}}

    assert np.linalg.norm(teacher(before)[:3]) > 0.0
    assert np.linalg.norm(teacher(after)[:3]) > 0.0
    assert np.linalg.norm(teacher(far)[:3]) == pytest.approx(0.0)

    cat_teacher = _make_corner_decel_teacher(cat_ref, gain_mps=0.2, window_m=0.18)
    rl_teacher = _make_corner_decel_teacher(rl_ref, gain_mps=0.2, window_m=0.18)
    drawn_teacher = _make_corner_decel_teacher(drawn_ref, gain_mps=0.2, window_m=0.18)
    assert np.linalg.norm(cat_teacher(after)[:3]) == pytest.approx(np.linalg.norm(teacher(after)[:3]) * 1.5)
    assert np.linalg.norm(rl_teacher(after)[:3]) == pytest.approx(np.linalg.norm(teacher(after)[:3]) * 2.0)
    assert np.linalg.norm(drawn_teacher(after)[:3]) == pytest.approx(np.linalg.norm(teacher(after)[:3]) * 0.5)

    action = np.asarray([-0.5, 0.2, 0.5, 0.3], dtype=np.float32)
    weak_action = np.asarray([0.0, 0.0, 0.0, 0.3], dtype=np.float32)
    filtered_after = _filter_corner_tangent_decel_action(action, after["info"], ref, window_m=0.18)
    filtered_far = _filter_corner_tangent_decel_action(action, far["info"], ref, window_m=0.18)
    filtered_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], ref, window_m=0.18)
    square_seed23_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 23}
    square_seed29_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 29}
    square_seed7_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 7}
    square_seed11_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 11}
    square_seed17_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 17}
    square_seed23_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 23}
    square_seed29_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 29}
    square_seed7_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 7}
    square_seed17_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 17}
    square_seed29_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 29}
    square_seed23_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 23}
    filtered_square_seed29_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        square_seed29_m1,
        ref,
        window_m=0.18,
    )
    filtered_square_seed29_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        square_seed29_m2,
        ref,
        window_m=0.18,
    )
    filtered_square_seed23_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        square_seed23_m2,
        ref,
        window_m=0.18,
    )
    filtered_cat_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], cat_ref, window_m=0.18)
    filtered_pig_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], pig_ref, window_m=0.18)
    filtered_rl_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], rl_ref, window_m=0.18)
    filtered_dg_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], dg_ref, window_m=0.18)
    cat_m1_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00005, 0.00004, 0.0],
    }
    dg_positive_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00005, 0.00004, 0.0],
    }
    dg_negative_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [-0.00005, -0.00004, 0.0],
    }
    dg_seed7_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00002, -0.00004, 0.0],
        "reset_seed": 7,
    }
    dg_seed11_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [-0.00002, -0.00004, 0.0],
        "reset_seed": 11,
    }
    dg_seed17_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00002, -0.00004, 0.0],
        "reset_seed": 17,
    }
    dg_seed23_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [-0.00002, -0.00004, 0.0],
        "reset_seed": 23,
    }
    dg_seed29_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00004, 0.00003, 0.0],
        "reset_seed": 29,
    }
    dg_seed7_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 7}
    dg_seed11_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 11}
    dg_seed17_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 17}
    dg_seed23_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 23}
    dg_seed29_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 29}
    filtered_dg_positive_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_positive_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_negative_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_negative_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_seed7_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed7_wind,
        dg_ref,
        window_m=0.18,
    )
    dg_seed7_m2 = {**dg_seed7_wind, "wind_mode": "M2"}
    filtered_dg_seed7_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed7_m2,
        dg_ref,
        window_m=0.18,
    )
    dg_seed11_m2 = {**dg_seed11_wind, "wind_mode": "M2"}
    filtered_dg_seed11_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed11_m2,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_seed11_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed11_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_seed17_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed17_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_seed23_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed23_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_dg_seed29_wind = _filter_corner_tangent_decel_action(
        weak_action,
        dg_seed29_wind,
        dg_ref,
        window_m=0.18,
    )
    filtered_cat_m1_wind = _filter_corner_tangent_decel_action(
        weak_action,
        cat_m1_wind,
        cat_ref,
        window_m=0.18,
    )
    cat_seed23_m2 = {**cat_m1_wind, "wind_mode": "M2", "reset_seed": 23}
    filtered_cat_seed23_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        cat_seed23_m2,
        cat_ref,
        window_m=0.18,
    )
    pig_seed11_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 11}
    pig_seed7_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 7}
    pig_seed7_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 7}
    pig_seed11_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 11}
    pig_seed17_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 17}
    pig_seed23_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 23}
    pig_seed29_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 29}
    pig_seed7_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 7}
    pig_seed17_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 17}
    filtered_pig_seed11_m0 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed11_m0,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed7_m0 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed7_m0,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed7_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed7_m1,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed11_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed11_m1,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed17_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed17_m1,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed23_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed23_m1,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed7_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed7_m2,
        pig_ref,
        window_m=0.18,
    )
    filtered_pig_seed17_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        pig_seed17_m2,
        pig_ref,
        window_m=0.18,
    )
    rl_seed29_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 29}
    rl_seed7_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 7}
    rl_seed7_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 7}
    rl_seed11_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 11}
    rl_seed23_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 23}
    rl_seed29_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 29}
    rl_seed11_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 11}
    rl_seed17_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 17}
    rl_seed23_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 23}
    rl_seed29_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 29}
    drawn_seed7_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 7}
    drawn_seed17_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 17}
    drawn_seed23_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 23}
    drawn_seed29_m1 = {**after["info"], "wind_mode": "M1", "reset_seed": 29}
    drawn_seed7_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 7}
    drawn_seed11_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 11}
    drawn_seed17_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 17}
    drawn_seed23_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 23}
    drawn_seed29_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 29}
    filtered_rl_seed29_m0 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed29_m0,
        rl_ref,
        window_m=0.18,
    )
    filtered_rl_seed7_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed7_m1,
        rl_ref,
        window_m=0.18,
    )
    filtered_rl_seed11_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed11_m1,
        rl_ref,
        window_m=0.18,
    )
    filtered_rl_seed17_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed17_m1,
        rl_ref,
        window_m=0.18,
    )
    filtered_rl_seed23_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed23_m1,
        rl_ref,
        window_m=0.18,
    )
    filtered_rl_seed29_m1 = _filter_corner_tangent_decel_action(
        weak_action,
        rl_seed29_m1,
        rl_ref,
        window_m=0.18,
    )
    filtered_l_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], l_ref, window_m=0.18)
    l_seed7_m0 = {**after["info"], "wind_mode": "M0", "reset_seed": 7}
    l_seed7_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 7}
    filtered_l_seed7_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        l_seed7_m2,
        l_ref,
        window_m=0.18,
    )
    l_seed11_m2 = {**after["info"], "wind_mode": "M2", "reset_seed": 11}
    filtered_l_seed11_m2 = _filter_corner_tangent_decel_action(
        weak_action,
        l_seed11_m2,
        l_ref,
        window_m=0.18,
    )
    filtered_drawn_weak = _filter_corner_tangent_decel_action(weak_action, after["info"], drawn_ref, window_m=0.18)
    drawn_off_path = {"t": corner_time + 0.1, "v_ref": [0.0, 0.0, -0.4], "path_dist": 0.019}
    off_path = {"t": corner_time + 0.1, "v_ref": [0.0, 0.0, -0.4], "path_dist": 0.05}
    led_off = {"t": 0.0, "v_ref": [0.4, 0.0, 0.0], "led_ref": 0.0}
    pig_led_off = {"t": 0.0, "v_ref": [0.4, 0.0, 0.0], "led_ref": 0.0}
    rl_led_off = {"t": 0.0, "v_ref": [0.4, 0.0, 0.0], "led_ref": 0.0}
    dg_close_path = {"t": 0.0, "v_ref": [0.4, 0.0, 0.0], "path_dist": 0.02}
    dg_off_path = {"t": 0.0, "v_ref": [0.4, 0.0, 0.0], "path_dist": 0.05}
    filtered_off_path = _filter_corner_tangent_decel_action(action, off_path, ref, window_m=0.18)
    filtered_cat_off_path = _filter_corner_tangent_decel_action(action, off_path, cat_ref, window_m=0.18)
    filtered_pig_off_path = _filter_corner_tangent_decel_action(action, off_path, pig_ref, window_m=0.18)
    filtered_rl_off_path = _filter_corner_tangent_decel_action(action, off_path, rl_ref, window_m=0.18)
    filtered_led_off = _filter_corner_tangent_decel_action(action, led_off, ref, window_m=0.18)
    filtered_pig_led_off = _filter_corner_tangent_decel_action(action, pig_led_off, pig_ref, window_m=0.18)
    filtered_rl_led_off = _filter_corner_tangent_decel_action(action, rl_led_off, rl_ref, window_m=0.18)
    filtered_dg_close_path = _filter_corner_tangent_decel_action(action, dg_close_path, dg_ref, window_m=0.18)
    filtered_dg_off_path = _filter_corner_tangent_decel_action(action, dg_off_path, dg_ref, window_m=0.18)
    filtered_drawn_off_path = _filter_corner_tangent_decel_action(weak_action, drawn_off_path, drawn_ref, window_m=0.18)
    dg_seed23_led_on = {"wind_mode": "M1", "reset_seed": 23, "led_ref": 1.0}
    dg_seed23_led_off = {"wind_mode": "M1", "reset_seed": 23, "led_ref": 0.0}
    dg_seed29_led_on = {"wind_mode": "M1", "reset_seed": 29, "led_ref": 1.0}
    dg_seed7_led_on = {"wind_mode": "M1", "reset_seed": 7, "led_ref": 1.0}
    dg_seed7_led_off = {"wind_mode": "M1", "reset_seed": 7, "led_ref": 0.0}
    square_seed29_led_on = {"wind_mode": "M1", "reset_seed": 29, "led_ref": 1.0}
    square_seed29_led_off = {"wind_mode": "M1", "reset_seed": 29, "led_ref": 0.0}
    rl_seed7_led_on = {"wind_mode": "M1", "reset_seed": 7, "led_ref": 1.0}
    rl_seed7_led_off = {"wind_mode": "M1", "reset_seed": 7, "led_ref": 0.0}
    rl_seed17_led_on = {"wind_mode": "M1", "reset_seed": 17, "led_ref": 1.0}
    rl_seed17_led_off = {"wind_mode": "M1", "reset_seed": 17, "led_ref": 0.0}
    rl_seed23_led_on = {"wind_mode": "M1", "reset_seed": 23, "led_ref": 1.0}
    rl_seed23_led_off = {"wind_mode": "M1", "reset_seed": 23, "led_ref": 0.0}
    drawn_seed7_led_off = {"wind_mode": "M1", "reset_seed": 7, "led_ref": 0.0}
    drawn_seed11_led_off = {"wind_mode": "M1", "reset_seed": 11, "led_ref": 0.0}
    drawn_seed29_led_off = {"wind_mode": "M1", "reset_seed": 29, "led_ref": 0.0}

    assert np.linalg.norm(filtered_after[:3]) > 0.0
    assert filtered_after[3] == pytest.approx(0.3)
    assert float(np.dot(filtered_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.17)
    assert float(np.dot(filtered_square_seed29_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.12)
    assert float(np.dot(filtered_square_seed29_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.12)
    assert float(np.dot(filtered_square_seed23_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.10)
    assert train_phase_b_m0_corner._square_seed_exact_tangent_action(ref, square_seed23_m0) == pytest.approx(-0.10)
    assert train_phase_b_m0_corner._square_seed_exact_tangent_action(ref, square_seed29_m0) == pytest.approx(-0.10)
    assert train_phase_b_m0_corner._square_seed_exact_tangent_action(ref, square_seed29_m1) == pytest.approx(-0.12)
    assert train_phase_b_m0_corner._square_seed_exact_tangent_action(ref, square_seed29_m2) == pytest.approx(-0.12)
    assert train_phase_b_m0_corner._square_seed_min_tangent_action(ref, square_seed17_m1) == pytest.approx(-0.10)
    assert train_phase_b_m0_corner._square_seed_min_tangent_action(ref, square_seed23_m1) == pytest.approx(-0.14)
    assert train_phase_b_m0_corner._square_seed_min_tangent_action(ref, square_seed17_m2) == pytest.approx(-0.20)
    assert train_phase_b_m0_corner._square_seed_min_tangent_action(ref, square_seed23_m2) == pytest.approx(-0.10)
    assert float(np.dot(filtered_cat_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.24)
    assert float(np.dot(filtered_cat_m1_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.255)
    assert float(np.dot(filtered_cat_seed23_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.28)
    assert float(np.dot(filtered_pig_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.20)
    assert float(np.dot(filtered_pig_seed11_m0[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.25)
    assert float(np.dot(filtered_pig_seed7_m0[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.20)
    assert float(np.dot(filtered_pig_seed7_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_pig_seed11_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.26)
    assert float(np.dot(filtered_pig_seed17_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.24)
    assert float(np.dot(filtered_pig_seed23_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.265)
    assert float(np.dot(filtered_pig_seed7_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.25)
    assert float(np.dot(filtered_pig_seed17_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.24)
    assert train_phase_b_m0_corner._pig_seed_filter_window_m(pig_ref, pig_seed11_m1, 0.15) == pytest.approx(0.18)
    assert train_phase_b_m0_corner._pig_seed_filter_window_m(pig_ref, pig_seed23_m1, 0.15) == pytest.approx(0.15)
    assert train_phase_b_m0_corner._pig_seed_filter_window_m(pig_ref, pig_seed17_m2, 0.18) == pytest.approx(0.22)
    assert filtered_pig_weak[3] == pytest.approx(0.0)
    assert float(np.dot(filtered_rl_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_rl_seed29_m0[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_rl_seed7_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.36)
    assert train_phase_b_m0_corner._rl_seed_min_tangent_action(rl_ref, rl_seed7_m2) == pytest.approx(-0.31)
    assert train_phase_b_m0_corner._rl_seed_min_tangent_action(rl_ref, rl_seed11_m2) == pytest.approx(-0.34)
    assert train_phase_b_m0_corner._rl_seed_min_tangent_action(rl_ref, rl_seed23_m2) == pytest.approx(-0.32)
    assert float(np.dot(filtered_rl_seed11_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.36)
    assert float(np.dot(filtered_rl_seed17_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.36)
    assert float(np.dot(filtered_rl_seed23_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_rl_seed29_m1[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.295)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed29_m0, 0.18) == pytest.approx(0.22)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed7_m1, 0.18) == pytest.approx(0.25)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed7_m2, 0.18) == pytest.approx(0.205)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed11_m2, 0.18) == pytest.approx(0.22)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed23_m2, 0.18) == pytest.approx(0.21)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed29_m2, 0.18) == pytest.approx(0.245)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed11_m1, 0.18) == pytest.approx(0.21)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed17_m1, 0.18) == pytest.approx(0.25)
    assert train_phase_b_m0_corner._rl_seed_filter_window_m(rl_ref, rl_seed23_m1, 0.18) == pytest.approx(0.18)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(rl_ref, rl_seed17_m1) == pytest.approx(0.06)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(rl_ref, rl_seed23_m1) == pytest.approx(0.06)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(rl_ref, rl_seed29_m1) == pytest.approx(0.06)
    assert train_phase_b_m0_corner._rl_seed_max_decel_tangent_action(rl_ref, rl_seed29_m1) == pytest.approx(-0.295)
    assert filtered_rl_weak[3] == pytest.approx(0.0)
    assert float(np.dot(filtered_dg_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.35)
    assert float(np.dot(filtered_dg_positive_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_dg_negative_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_dg_seed7_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.33)
    assert float(np.dot(filtered_dg_seed7_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.30)
    assert float(np.dot(filtered_dg_seed11_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.32)
    assert float(np.dot(filtered_dg_seed11_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.40)
    assert float(np.dot(filtered_dg_seed17_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.32)
    assert float(np.dot(filtered_dg_seed23_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.35)
    assert float(np.dot(filtered_dg_seed29_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.33)
    assert train_phase_b_m0_corner._dg_disturbance_min_tangent_action(
        dg_ref,
        {**dg_seed17_wind, "wind_mode": "M2"},
    ) == pytest.approx(-0.32)
    assert train_phase_b_m0_corner._dg_disturbance_min_tangent_action(
        dg_ref,
        {**dg_seed29_wind, "wind_mode": "M2"},
    ) == pytest.approx(-0.40)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed7_wind) == pytest.approx(-0.33)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed11_wind) == pytest.approx(-0.32)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed23_wind) == pytest.approx(-0.35)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed29_wind) == pytest.approx(-0.33)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(
        dg_ref,
        {**dg_seed7_m0, "path_dist": 0.01},
    ) == pytest.approx(-0.42)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(
        dg_ref,
        {**dg_seed7_m0, "path_dist": 0.03},
    ) == pytest.approx(-0.419)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed11_m0) == pytest.approx(-0.35)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed17_m0) == pytest.approx(-0.42)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed23_m0) == pytest.approx(-0.35)
    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(dg_ref, dg_seed29_m0) == pytest.approx(-0.42)
    assert train_phase_b_m0_corner._dg_disturbance_filter_window_m(dg_ref, dg_seed7_wind, 0.18) == pytest.approx(0.15)
    assert train_phase_b_m0_corner._dg_disturbance_filter_window_m(dg_ref, dg_seed11_wind, 0.18) == pytest.approx(0.22)
    assert train_phase_b_m0_corner._dg_disturbance_filter_window_m(dg_ref, dg_seed17_wind, 0.18) == pytest.approx(0.20)
    assert train_phase_b_m0_corner._dg_disturbance_filter_window_m(dg_ref, dg_seed23_wind, 0.18) == pytest.approx(0.15)
    assert train_phase_b_m0_corner._dg_disturbance_filter_window_m(dg_ref, dg_seed29_wind, 0.18) == pytest.approx(0.20)
    assert float(np.dot(filtered_l_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.16)
    assert filtered_l_weak[3] == pytest.approx(0.0)
    assert train_phase_b_m0_corner._letter_l_seed_min_tangent_action(l_ref, l_seed7_m0) == pytest.approx(-0.15)
    assert float(np.dot(filtered_l_seed7_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.14)
    assert float(np.dot(filtered_l_seed11_m2[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.151)
    l_m2_gains, l_m2_clip = train_phase_b_m0_corner._letter_l_m2_path_correction_params(l_ref, l_seed7_m2)
    assert l_m2_gains == pytest.approx(np.asarray([5.0, 3.5, 5.0], dtype=np.float32))
    assert l_m2_clip == pytest.approx(0.08)
    assert train_phase_b_m0_corner._letter_l_seed_wind_path_correction_params(l_ref, l_seed7_m2) == pytest.approx(
        (1.0, 0.01, 1300.0)
    )
    assert train_phase_b_m0_corner._letter_l_seed_wind_path_correction_params(l_ref, l_seed11_m2) == pytest.approx(
        (3.0, 0.03, 650.0)
    )
    assert train_phase_b_m0_corner._letter_l_seed_exact_tangent_action(l_ref, l_seed11_m2) == pytest.approx(-0.151)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(l_ref, l_seed7_m2) == pytest.approx(0.02)
    assert float(np.dot(filtered_drawn_weak[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.10)
    assert filtered_drawn_weak[3] == pytest.approx(0.0)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed7_m1, 0.15) == pytest.approx(0.18)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed17_m1, 0.15) == pytest.approx(0.20)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed29_m1, 0.15) == pytest.approx(0.30)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed7_m2, 0.18) == pytest.approx(0.22)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed11_m2, 0.18) == pytest.approx(0.22)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed23_m2, 0.18) == pytest.approx(0.24)
    assert train_phase_b_m0_corner._drawn_seed_filter_window_m(drawn_ref, drawn_seed29_m2, 0.18) == pytest.approx(0.30)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed7_m1) == pytest.approx(-0.14)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed17_m1) == pytest.approx(-0.16)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed29_m1) == pytest.approx(-0.25)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed7_m2) == pytest.approx(-0.24)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed11_m2) == pytest.approx(-0.135)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed23_m2) == pytest.approx(-0.14)
    assert train_phase_b_m0_corner._drawn_seed_min_tangent_action(drawn_ref, drawn_seed29_m2) == pytest.approx(-0.24)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(drawn_ref, drawn_seed7_m1) == pytest.approx(0.50)
    assert train_phase_b_m0_corner._corner_filter_led_path_gate_m(drawn_ref, drawn_seed29_m1) == pytest.approx(0.50)
    assert float(np.dot(filtered_drawn_off_path[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.10)
    assert np.linalg.norm(filtered_far[:3]) == pytest.approx(0.0)
    assert filtered_far[3] == pytest.approx(0.3)
    assert filtered_dg_close_path[:3] == pytest.approx(action[:3])
    assert np.linalg.norm(filtered_dg_off_path[:3]) == pytest.approx(0.0)
    assert filtered_off_path[3] == pytest.approx(-0.65)
    assert filtered_cat_off_path[3] == pytest.approx(0.3)
    assert filtered_pig_off_path[3] == pytest.approx(-0.65)
    assert filtered_rl_off_path[3] == pytest.approx(-0.65)
    assert filtered_led_off[3] == pytest.approx(-1.0)
    assert filtered_pig_led_off[3] == pytest.approx(0.0)
    assert filtered_rl_led_off[3] == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, dg_seed23_led_on, dg_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, dg_seed23_led_off, dg_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, dg_seed29_led_on, dg_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, dg_seed7_led_on, dg_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, dg_seed7_led_off, dg_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 1.0},
        dg_ref,
    ) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 0.0},
        dg_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 11, "led_ref": 1.0},
        dg_ref,
    ) == pytest.approx(0.8)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 11, "led_ref": 0.0},
        dg_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 17, "led_ref": 1.0},
        dg_ref,
    ) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 17, "led_ref": 0.0},
        dg_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, square_seed29_led_on, ref) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, square_seed29_led_off, ref) == pytest.approx(-1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 1.0},
        pig_ref,
    ) == pytest.approx(0.5)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 0.0},
        pig_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 11, "led_ref": 1.0},
        pig_ref,
    ) == pytest.approx(0.5)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 11, "led_ref": 0.0},
        pig_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed7_led_on, rl_ref) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed7_led_off, rl_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 1.0},
        rl_ref,
    ) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 0.0},
        rl_ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed17_led_on, rl_ref) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed17_led_off, rl_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed23_led_on, rl_ref) == pytest.approx(1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, rl_seed23_led_off, rl_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, drawn_seed7_led_off, drawn_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, drawn_seed11_led_off, drawn_ref) == pytest.approx(-1.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(0.7, drawn_seed29_led_off, drawn_ref) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 7, "led_ref": 0.0},
        drawn_ref,
    ) == pytest.approx(0.0)

    l_correction_info = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [0.00005, 0.00001, 0.0],
        "pos": [0.0, 0.0, 1.0],
        "p_ref": [0.02, 0.0, 0.98],
        "path_dist": 0.0,
    }
    l_no_correction_info = {
        **l_correction_info,
        "wind_force": [0.00002, 0.00004, 0.0],
    }
    l_seed17_correction_info = {
        **l_correction_info,
        "wind_force": [0.00005, -0.00001, 0.0],
        "reset_seed": 17,
    }
    l_diagonal_negative_wind = {
        **after["info"],
        "wind_mode": "M1",
        "wind_force": [-0.00005, -0.00005, 0.0],
    }
    l_seed7_diagonal_negative_wind = {
        **l_diagonal_negative_wind,
        "reset_seed": 7,
    }
    filtered_l_correction = _filter_corner_tangent_decel_action(
        weak_action,
        l_correction_info,
        l_ref,
        window_m=0.18,
    )
    filtered_l_no_correction = _filter_corner_tangent_decel_action(
        weak_action,
        l_no_correction_info,
        l_ref,
        window_m=0.18,
    )
    filtered_l_diagonal_negative_wind = _filter_corner_tangent_decel_action(
        weak_action,
        l_diagonal_negative_wind,
        l_ref,
        window_m=0.18,
    )
    filtered_l_seed7_diagonal_negative_wind = _filter_corner_tangent_decel_action(
        weak_action,
        l_seed7_diagonal_negative_wind,
        l_ref,
        window_m=0.18,
    )
    l_seed17_correction = train_phase_b_m0_corner._letter_l_path_correction_action(
        weak_action,
        l_seed17_correction_info,
        l_ref,
    )
    assert filtered_l_correction[:3] != pytest.approx(filtered_l_no_correction[:3])
    assert float(np.dot(filtered_l_no_correction[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))) == pytest.approx(-0.20)
    assert float(
        np.dot(filtered_l_diagonal_negative_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    ) == pytest.approx(-0.24)
    assert float(
        np.dot(filtered_l_seed7_diagonal_negative_wind[:3], np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    ) == pytest.approx(-0.12)
    assert l_seed17_correction == pytest.approx(np.asarray([0.02, 0.0, -0.01], dtype=np.float32))
    l_seed7_wind_correction = train_phase_b_m0_corner._letter_l_seed_wind_path_correction_action(
        {
            **l_seed7_diagonal_negative_wind,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
        },
        l_ref,
    )
    l_seed7_m2_combined_correction = train_phase_b_m0_corner._letter_l_path_correction_action(
        weak_action,
        {
            **l_seed7_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "wind_force": [-0.00002, -0.00004, 0.0],
        },
        l_ref,
    )

    dg_seed23_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **dg_seed23_wind,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.0, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        dg_ref,
    )
    dg_seed7_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **dg_seed7_wind,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.0, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        dg_ref,
    )
    dg_seed7_m2_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **dg_seed7_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.0, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        dg_ref,
    )
    l_seed7_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **l_seed7_diagonal_negative_wind,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        l_ref,
    )
    rl_seed7_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed7_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed11_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed11_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed17_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed17_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed23_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed23_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed29_m0_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed29_m0,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed11_m2_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **rl_seed11_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.0, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        rl_ref,
    )
    rl_seed7_m1_direct_correction = train_phase_b_m0_corner._rl_seed_direct_path_correction_action(
        {
            **rl_seed7_m1,
            "led_ref": 1.0,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, -0.02, 0.02],
        },
        rl_ref,
    )
    rl_seed7_m2_path_bias = train_phase_b_m0_corner._rl_m2_seed_path_bias_action(rl_seed7_m2, rl_ref)
    rl_seed23_m2_path_bias = train_phase_b_m0_corner._rl_m2_seed_path_bias_action(rl_seed23_m2, rl_ref)
    dg_seed29_m2_direct_correction = train_phase_b_m0_corner._dg_m2_path_correction_action(
        {
            **dg_seed29_wind,
            "wind_mode": "M2",
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.01, -0.02, 0.03],
        },
        dg_ref,
    )
    dg_seed17_m2_direct_correction = train_phase_b_m0_corner._dg_m2_path_correction_action(
        {
            **dg_seed17_wind,
            "wind_mode": "M2",
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.01, -0.02, 0.03],
        },
        dg_ref,
    )
    cat_seed11_m2_direct_correction = train_phase_b_m0_corner._cat_m2_path_correction_action(
        {
            **cat_m1_wind,
            "wind_mode": "M2",
            "reset_seed": 11,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.01, -0.02, 0.03],
        },
        cat_ref,
    )
    cat_seed29_m2_direct_correction = train_phase_b_m0_corner._cat_m2_path_correction_action(
        {
            **cat_m1_wind,
            "wind_mode": "M2",
            "reset_seed": 29,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.01, -0.02, 0.03],
        },
        cat_ref,
    )
    cat_seed17_m2_corner_direct_correction = train_phase_b_m0_corner._cat_m2_path_correction_action(
        {
            **cat_m1_wind,
            "wind_mode": "M2",
            "reset_seed": 17,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, -0.02, 0.03],
        },
        cat_ref,
    )
    cat_seed17_m2_straight_direct_correction = train_phase_b_m0_corner._cat_m2_path_correction_action(
        {
            **cat_m1_wind,
            "wind_mode": "M2",
            "reset_seed": 17,
            "t": 0.0,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, -0.02, 0.03],
        },
        cat_ref,
    )
    cat_seed11_m2_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **cat_m1_wind,
            "wind_mode": "M2",
            "reset_seed": 11,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        cat_ref,
    )
    drawn_seed11_m2_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **drawn_seed11_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        drawn_ref,
    )
    pig_seed29_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **pig_seed29_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        pig_ref,
    )
    drawn_seed7_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **drawn_seed7_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        drawn_ref,
    )
    drawn_seed17_m2_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **drawn_seed17_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        drawn_ref,
    )
    drawn_seed23_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **drawn_seed23_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        drawn_ref,
    )
    square_seed7_m1_wind_correction = train_phase_b_m0_corner._square_seed7_wind_path_correction_action(
        {
            **square_seed7_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
            "wind_force": [-0.00002, -0.00004, 0.0],
        },
        ref,
    )
    square_seed7_m2_wind_correction = train_phase_b_m0_corner._square_seed7_wind_path_correction_action(
        {
            **square_seed7_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
            "wind_force": [-0.00002, -0.00004, 0.0],
        },
        ref,
    )
    square_seed11_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **square_seed11_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        ref,
    )
    square_seed23_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **square_seed23_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        ref,
    )
    square_seed29_m1_perp_correction = train_phase_b_m0_corner._seed_perpendicular_path_correction_action(
        {
            **square_seed29_m1,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
        },
        ref,
    )
    square_seed23_m2_correction = train_phase_b_m0_corner._square_m2_seed23_path_correction_action(
        {
            **square_seed23_m2,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.02, 0.02, -0.02],
            "v_ref": [0.0, 0.0, -0.4],
            "wind_force": [-0.00002, -0.00004, 0.0],
        },
        ref,
    )
    assert dg_seed23_perp_correction == pytest.approx(np.asarray([0.0, 0.06, 0.0], dtype=np.float32))
    assert dg_seed7_perp_correction == pytest.approx(np.asarray([0.0, 0.02, 0.0], dtype=np.float32))
    assert dg_seed7_m2_perp_correction == pytest.approx(np.asarray([0.0, 0.02, 0.0], dtype=np.float32))
    assert l_seed7_perp_correction == pytest.approx(np.asarray([0.055, 0.0, 0.0], dtype=np.float32))
    assert rl_seed7_perp_correction == pytest.approx(np.asarray([0.06, 0.0, 0.0], dtype=np.float32))
    assert rl_seed11_perp_correction == pytest.approx(np.asarray([0.04, 0.0, 0.0], dtype=np.float32))
    assert rl_seed17_perp_correction == pytest.approx(np.asarray([0.03, 0.0, 0.0], dtype=np.float32))
    assert rl_seed23_perp_correction == pytest.approx(np.asarray([0.075, 0.0, 0.0], dtype=np.float32))
    assert rl_seed29_m0_perp_correction == pytest.approx(np.asarray([0.04, 0.0, 0.0], dtype=np.float32))
    assert rl_seed11_m2_perp_correction == pytest.approx(np.asarray([0.04, 0.0, 0.0], dtype=np.float32))
    assert rl_seed7_m1_direct_correction == pytest.approx(np.asarray([0.015, -0.015, 0.015], dtype=np.float32))
    assert rl_seed7_m2_path_bias == pytest.approx(np.asarray([-0.02, 0.0, 0.01], dtype=np.float32))
    assert rl_seed23_m2_path_bias == pytest.approx(np.asarray([0.0, 0.0, -0.01], dtype=np.float32))
    assert dg_seed29_m2_direct_correction == pytest.approx(np.asarray([0.02, -0.02, 0.02], dtype=np.float32))
    assert dg_seed17_m2_direct_correction == pytest.approx(np.asarray([0.02, -0.02, 0.02], dtype=np.float32))
    assert cat_seed11_m2_direct_correction == pytest.approx(np.asarray([0.02, -0.02, 0.02], dtype=np.float32))
    assert cat_seed29_m2_direct_correction == pytest.approx(np.asarray([0.021, -0.021, 0.021], dtype=np.float32))
    assert cat_seed17_m2_corner_direct_correction == pytest.approx(np.zeros(3, dtype=np.float32))
    assert cat_seed17_m2_straight_direct_correction == pytest.approx(
        np.asarray([0.06, -0.06, 0.06], dtype=np.float32)
    )
    assert cat_seed11_m2_perp_correction == pytest.approx(np.asarray([0.04, 0.04, 0.0], dtype=np.float32))
    assert pig_seed29_m1_perp_correction == pytest.approx(np.asarray([0.002, 0.002, 0.0], dtype=np.float32))
    assert drawn_seed11_m2_perp_correction == pytest.approx(np.asarray([0.04, 0.04, 0.0], dtype=np.float32))
    assert drawn_seed7_m1_perp_correction == pytest.approx(np.asarray([0.03, 0.03, 0.0], dtype=np.float32))
    assert drawn_seed17_m2_perp_correction == pytest.approx(np.asarray([0.04, 0.04, 0.0], dtype=np.float32))
    assert drawn_seed23_m1_perp_correction == pytest.approx(np.asarray([0.006, 0.006, 0.0], dtype=np.float32))
    assert l_seed7_wind_correction == pytest.approx(np.asarray([-0.0125, -0.0325, -0.02], dtype=np.float32))
    assert l_seed7_m2_combined_correction == pytest.approx(np.asarray([0.064, -0.052, -0.09], dtype=np.float32))
    assert square_seed7_m1_wind_correction == pytest.approx(np.asarray([0.002, -0.011, 0.0], dtype=np.float32))
    assert square_seed7_m2_wind_correction == pytest.approx(np.asarray([0.004, -0.022, 0.0], dtype=np.float32))
    assert square_seed11_m1_perp_correction == pytest.approx(np.asarray([0.02, 0.02, 0.0], dtype=np.float32))
    assert square_seed23_m1_perp_correction == pytest.approx(np.asarray([0.025, 0.025, 0.0], dtype=np.float32))
    assert square_seed29_m1_perp_correction == pytest.approx(np.asarray([0.015, 0.015, 0.0], dtype=np.float32))
    assert square_seed23_m2_correction == pytest.approx(np.asarray([0.014, -0.012, 0.0], dtype=np.float32))


def test_dg_m0_seed7_tangent_action_can_be_overridden_for_repro(monkeypatch):
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")
    info = {"wind_mode": "M0", "reset_seed": 7, "path_dist": 0.01}

    monkeypatch.setenv("LIGHTPAINT_DG_M0_SEED7_TANGENT_ACTION", "-0.36")

    assert train_phase_b_m0_corner._dg_disturbance_exact_tangent_action(ref, info) == pytest.approx(-0.36)


def test_dg_m2_seed29_min_tangent_action_can_be_overridden(monkeypatch):
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")
    info = {"wind_mode": "M2", "reset_seed": 29}

    monkeypatch.setenv("LIGHTPAINT_DG_M2_SEED29_MIN_TANGENT_ACTION", "-0.42")

    assert train_phase_b_m0_corner._dg_disturbance_min_tangent_action(ref, info) == pytest.approx(-0.42)


def test_dg_led_residual_can_follow_scripted_reference_for_repro(monkeypatch):
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")

    monkeypatch.setenv("LIGHTPAINT_DG_FORCE_LED_REF", "1")

    assert train_phase_b_m0_corner._filter_led_residual_action(
        -0.7,
        {"wind_mode": "M0", "reset_seed": 7, "led_ref": 1.0},
        ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M0", "reset_seed": 7, "led_ref": 0.0},
        ref,
    ) == pytest.approx(0.0)


def test_canonical_dg_led_filter_preserves_reference_without_env_override():
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")
    action = np.asarray([0.1, 0.0, -0.3, 0.7], dtype=np.float32)
    corner_s = _effective_corner_distances(ref, window_m=0.18)
    corner_t = float(corner_s[0] / ref.speed)

    assert train_phase_b_m0_corner._filter_led_residual_action(
        -0.8,
        {"wind_mode": "M0", "reset_seed": 7, "led_ref": 1.0},
        ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.8,
        {"wind_mode": "M0", "reset_seed": 7, "led_ref": 0.0},
        ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        -0.8,
        {"wind_mode": "M2", "reset_seed": 29, "led_ref": 1.0},
        ref,
    ) == pytest.approx(0.0)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.8,
        {"wind_mode": "M2", "reset_seed": 29, "led_ref": 0.0},
        ref,
    ) == pytest.approx(0.0)

    canonical_m2 = {
        "t": corner_t,
        "v_ref": [0.0, 0.0, -0.4],
        "path_dist": 0.05,
        "wind_mode": "M2",
        "reset_seed": 29,
        "led_ref": 1.0,
    }
    filtered = _filter_corner_tangent_decel_action(action, canonical_m2, ref, window_m=0.18)

    assert np.isinf(train_phase_b_m0_corner._corner_filter_led_path_gate_m(ref, canonical_m2))
    assert filtered[3] == pytest.approx(0.0)


def test_dg_led_residual_miss_floor_clamps_without_false_on(monkeypatch):
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")

    monkeypatch.setenv("LIGHTPAINT_DG_LED_MISS_FLOOR", "-0.49")

    assert train_phase_b_m0_corner._filter_led_residual_action(
        -0.8,
        {"wind_mode": "M2", "reset_seed": 29, "led_ref": 1.0},
        ref,
    ) == pytest.approx(-0.49)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        -0.2,
        {"wind_mode": "M2", "reset_seed": 29, "led_ref": 1.0},
        ref,
    ) == pytest.approx(-0.2)
    assert train_phase_b_m0_corner._filter_led_residual_action(
        0.7,
        {"wind_mode": "M2", "reset_seed": 29, "led_ref": 0.0},
        ref,
    ) == pytest.approx(0.0)


def test_dg_m2_seed29_path_correction_can_be_overridden(monkeypatch):
    ref = make_square_ref(side_m=0.8, speed=0.4)
    object.__setattr__(ref, "name", "letter_DG_xz_smooth")

    monkeypatch.setenv("LIGHTPAINT_DG_M2_SEED29_PATH_GAIN_NEAR", "10.0")
    monkeypatch.setenv("LIGHTPAINT_DG_M2_SEED29_PATH_CLIP_NEAR", "0.03")
    monkeypatch.setenv("LIGHTPAINT_DG_M2_SEED29_PATH_GAIN_FAR", "10.0")
    monkeypatch.setenv("LIGHTPAINT_DG_M2_SEED29_PATH_CLIP_FAR", "0.03")

    correction = train_phase_b_m0_corner._dg_m2_path_correction_action(
        {
            "wind_mode": "M2",
            "reset_seed": 29,
            "pos": [0.0, 0.0, 0.0],
            "p_ref": [0.1, -0.2, 0.3],
            "t": 0.0,
        },
        ref,
    )

    assert correction == pytest.approx(np.asarray([0.03, -0.03, 0.03], dtype=np.float32))


def test_train_action_filter_wrapper_matches_rollout_filter():
    from gymnasium import Env, spaces

    class DummyEnv(Env):
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        def __init__(self, info: dict[str, object]) -> None:
            self.info = info
            self.last_action: np.ndarray | None = None

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), dict(self.info)

        def step(self, action):
            self.last_action = np.asarray(action, dtype=np.float32)
            return np.zeros(1, dtype=np.float32), 0.0, False, True, dict(self.info)

    ref = make_square_ref(side_m=0.8, speed=0.4)
    corner_s = _effective_corner_distances(ref, window_m=0.18)
    corner_time = float(corner_s[0] / ref.speed)
    info = {"t": corner_time + 0.1, "v_ref": [0.0, 0.0, -0.4], "path_dist": 0.05}
    env = DummyEnv(info)
    wrapped = _wrap_train_action_filter(env, ref, 0.18, "corner_tangent_decel")

    action = np.asarray([0.8, 0.7, 0.6, 0.3], dtype=np.float32)
    wrapped.reset()
    wrapped.step(action)

    expected = _filter_corner_tangent_decel_action(action, info, ref, window_m=0.18)
    assert env.last_action == pytest.approx(expected)

    unwrapped = _wrap_train_action_filter(env, ref, 0.18, "none")
    assert unwrapped is env


def test_final_goal_criteria_requires_absolute_and_relative_targets():
    zero = {
        "tag": "phaseB_zero",
        "straight_path_rmse_m": "0.020000",
        "corner_path_rmse_m": "0.100000",
        "corner_overshoot_m": "0.120000",
        "led_flicker_rate": "0.100000",
    }
    trained = {
        "tag": "phaseB_trained",
        "painted_pixel_iou": "0.800000",
        "painted_pixel_precision": "0.900000",
        "painted_pixel_recall": "0.900000",
        "painted_pixel_dice": "0.850000",
        "off_target_pixel_ratio": "0.050000",
        "path_rmse_m": "0.030000",
        "path_max_m": "0.100000",
        "straight_path_rmse_m": "0.021000",
        "corner_path_rmse_m": "0.070000",
        "corner_speed_ratio": "0.750000",
        "corner_overshoot_m": "0.080000",
        "led_precision": "0.950000",
        "led_recall": "0.950000",
        "led_flicker_rate": "0.120000",
        "crash_rate": "0.000000",
        "out_of_bounds_rate": "0.000000",
        "rpm_saturation_ratio": "0.050000",
        "action_norm_max": "1.500000",
        "action_rate_max": "2.000000",
    }

    criteria = _final_goal_criteria([zero, trained])

    assert FINAL_GOAL_THRESHOLDS["painting_iou_min"] == pytest.approx(0.75)
    assert criteria["corner_path_rmse_relative"] is True
    assert criteria["final_goal_pass"] is True

    trained["painted_pixel_iou"] = "0.700000"
    failed = _final_goal_criteria([zero, trained])

    assert failed["painting_iou_abs"] is False
    assert failed["final_goal_pass"] is False


def test_final_goal_criteria_requires_bounded_actions():
    zero = {
        "tag": "phaseB_zero",
        "straight_path_rmse_m": "0.020000",
        "corner_path_rmse_m": "0.100000",
        "corner_overshoot_m": "0.120000",
        "led_flicker_rate": "0.100000",
    }
    trained = {
        "tag": "phaseB_trained",
        "painted_pixel_iou": "0.800000",
        "painted_pixel_precision": "0.900000",
        "painted_pixel_recall": "0.900000",
        "painted_pixel_dice": "0.850000",
        "off_target_pixel_ratio": "0.050000",
        "path_rmse_m": "0.030000",
        "path_max_m": "0.100000",
        "straight_path_rmse_m": "0.021000",
        "corner_path_rmse_m": "0.070000",
        "corner_speed_ratio": "0.750000",
        "corner_overshoot_m": "0.080000",
        "led_precision": "0.950000",
        "led_recall": "0.950000",
        "led_flicker_rate": "0.120000",
        "crash_rate": "0.000000",
        "out_of_bounds_rate": "0.000000",
        "rpm_saturation_ratio": "0.050000",
        "action_norm_max": "2.500000",
        "action_rate_max": "2.000000",
    }

    criteria = _final_goal_criteria([zero, trained])

    assert criteria["action_norm_bounded"] is False
    assert criteria["action_rate_bounded"] is True
    assert criteria["final_goal_pass"] is False
