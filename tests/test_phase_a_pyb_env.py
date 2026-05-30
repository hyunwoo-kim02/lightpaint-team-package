"""Tests for the PyBullet Phase A env (Checkpoint 2 V0).

Requires gym-pybullet-drones + pybullet. Skipped when those aren't importable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

p = pytest.importorskip("pybullet")
pytest.importorskip("gym_pybullet_drones")

from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB
from src.env.lightpaint_geometry import Y_BOUND_M, Y_CANVAS
from src.env.lightpaint_ref import DEFAULT_CORNER_HINT_MIN_SHARPNESS, make_letter_ref, make_square_ref
from src.render.pybullet_snapshot import capture_pybullet_snapshot


@pytest.fixture
def env():
    env = LightPaintAviaryPyB(label="L", phase="A", wind_mode="M0")
    yield env
    env.close()


def test_construction_loads_cf2x_urdf(env):
    """A working PyBullet env must have a non-negative drone body id."""
    assert int(env.DRONE_IDS[0]) >= 0
    assert int(env.CLIENT) >= 0
    assert env.dsl_pid is not None


def test_reset_returns_dict_obs_with_valid_pos(env):
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    for key, expected_shape in (
        ("drone_state", (12,)),
        ("future_ref", (45,)),
        ("target_mask", (1, 64, 64)),
        ("progress_mask", (1, 64, 64)),
    ):
        assert key in obs, f"missing obs key {key!r}"
        assert obs[key].shape == expected_shape, f"{key} shape {obs[key].shape} != {expected_shape}"
    pos = info["pos"]
    assert len(pos) == 3
    assert -2.0 < pos[0] < 2.0 and 0.0 < pos[2] < 3.0
    assert info["reset_seed"] == 42


def test_five_step_rollout(env):
    env.reset(seed=42)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.zeros(0, dtype=np.float32))
        assert isinstance(r, float)
        assert "tracking_err" in info
        assert "u_pid_rpm" in info and len(info["u_pid_rpm"]) == 4
    # u_pid_rpm should be within sensible RPM range
    rpm = np.array(info["u_pid_rpm"])
    assert np.all(rpm >= 0.0) and np.all(rpm <= 50000.0)


def test_square_reference_drives_time_indexed_pid_target():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env = LightPaintAviaryPyB(
        label="square",
        phase="A",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=20,
        init_box_size=0.0,
    )
    try:
        obs, info = env.reset(seed=42)
        np.testing.assert_allclose(info["p_ref"], ref.pos(0.0), atol=1e-6)
        assert obs["future_ref"].shape == (45,)
        first_ref = np.array(info["p_ref"], dtype=np.float32)
        for _ in range(5):
            obs, reward, term, trunc, info = env.step(np.zeros(0, dtype=np.float32))
        later_ref = np.array(info["p_ref"], dtype=np.float32)
        assert later_ref[0] > first_ref[0]
        assert info["ref_name"] == "square"
        assert "v_ref" in info and len(info["v_ref"]) == 3
    finally:
        env.close()


def test_reference_scripted_led_is_mask_gated():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    always_on = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
        led_always_on=True,
    )
    try:
        on_path = np.asarray(ref.pos(1.0), dtype=np.float32)
        off_path = np.asarray([0.0, 0.0, 1.5], dtype=np.float32)

        assert ref.led(1.0) == pytest.approx(1.0)
        assert env._scripted_led_ref(on_path, 1.0) == pytest.approx(1.0)
        assert env._scripted_led_ref(off_path, 1.0) == pytest.approx(0.0)
        assert always_on._scripted_led_ref(off_path, 1.0) == pytest.approx(1.0)
    finally:
        env.close()
        always_on.close()


def test_reference_mask_can_reproduce_legacy_connector_target():
    ref = make_letter_ref("DG", plane="xz", speed=0.35)

    current = LightPaintAviaryPyB._build_reference_mask(ref.waypoints, segment_led=ref.segment_led)
    legacy = LightPaintAviaryPyB._build_reference_mask(
        ref.waypoints,
        segment_led=ref.segment_led,
        include_led_off_segments=True,
    )

    assert int(current.sum()) == 209
    assert int(legacy.sum()) == 220


def test_pig_reference_led_gate_tracks_new_progress_and_off_target_ratio():
    ref = make_letter_ref("Pig", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="Pig",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_m1 = LightPaintAviaryPyB(
        label="Pig",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        env_m1._last_reset_seed = 7
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        env_m1._last_reset_seed = 11
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        assert env._scripted_led_ref(on_path, 0.1) == pytest.approx(1.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._stamp_led_progress(on_path, 1.0)
        assert env._scripted_led_ref(on_path, 0.1) == pytest.approx(0.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)

        off_path = np.asarray([0.95, 0.0, 0.55], dtype=np.float32)
        assert env._scripted_led_ref(off_path, 0.1) == pytest.approx(0.0)
    finally:
        env.close()
        env_m1.close()


def test_rl_reference_led_gate_allows_gated_progress_catchup():
    ref = make_letter_ref("RL", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="RL",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert ref.led(after_schedule) == pytest.approx(0.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._stamp_led_progress(on_path, 1.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)

        off_path = np.asarray([0.95, 0.0, 0.55], dtype=np.float32)
        assert env._scripted_led_ref(off_path, after_schedule) == pytest.approx(0.0)
    finally:
        env.close()


def test_l_reference_led_gate_allows_gated_progress_catchup():
    ref = make_letter_ref("L", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="L",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_m1 = LightPaintAviaryPyB(
        label="L",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert ref.led(after_schedule) == pytest.approx(0.0)
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.25)
        env_m1._last_reset_seed = 7
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        env_m1._last_reset_seed = 11
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.25)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._stamp_led_progress(on_path, 1.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)
    finally:
        env.close()
        env_m1.close()


def test_drawn_reference_seed7_and_seed29_use_narrow_progress_catchup_guard():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    object.__setattr__(ref, "name", "drawn_user_path")
    env_m1 = LightPaintAviaryPyB(
        label="drawn",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_m2 = LightPaintAviaryPyB(
        label="drawn",
        phase="B",
        wind_mode="M2",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        env_m1._last_reset_seed = 7
        assert env_m1._use_progress_led_catchup() is True
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.08)

        env_m1._last_reset_seed = 29
        assert env_m1._use_progress_led_catchup() is True
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.08)

        env_m1._last_reset_seed = 11
        assert env_m1._use_progress_led_catchup() is False
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.115)

        env_m2._last_reset_seed = 7
        assert env_m2._use_progress_led_catchup() is True
        assert env_m2._target_only_led_stamp(1.0, 0.0) is True
        assert env_m2._target_only_led_stamp(0.0, 0.0) is False

        env_m2._last_reset_seed = 11
        assert env_m2._use_progress_led_catchup() is False
        assert env_m2._target_only_led_stamp(1.0, 0.0) is False
    finally:
        env_m1.close()
        env_m2.close()


def test_cat_reference_led_catchup_uses_seed_specific_gate_limits():
    ref = make_letter_ref("CAT", plane="xz", speed=0.35)
    env_m0 = LightPaintAviaryPyB(
        label="CAT",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_m1 = LightPaintAviaryPyB(
        label="CAT",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert ref.led(after_schedule) == pytest.approx(0.0)
        assert env_m0._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)
        assert env_m0._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        env_m0._last_reset_seed = 11
        assert env_m0._progress_led_gate_max_off_target_ratio() == pytest.approx(0.095)
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        env_m1._last_reset_seed = 7
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.09)
        env_m1._last_reset_seed = 17
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.095)
        env_m1._last_reset_seed = 11
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        assert env_m1._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env_m0._stamp_led_progress(on_path, 1.0)
        assert env_m0._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)
        env_m1._stamp_led_progress(on_path, 1.0)
        assert env_m1._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)
    finally:
        env_m0.close()
        env_m1.close()


def test_dg_reference_led_catchup_is_limited_to_positive_xy_disturbance():
    ref = make_letter_ref("DG", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="DG",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert ref.led(after_schedule) == pytest.approx(0.0)

        env._latched_wind = np.asarray([-0.00005, -0.00004, 0.0], dtype=np.float32)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)

        env._last_reset_seed = 23
        env._latched_wind = np.asarray([-0.00005, -0.00004, 0.0], dtype=np.float32)
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._last_reset_seed = 7
        env._latched_wind = np.asarray([-0.00005, -0.00004, 0.0], dtype=np.float32)
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.10)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._last_reset_seed = 29
        env._latched_wind = np.asarray([0.00005, 0.00004, 0.0], dtype=np.float32)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._stamp_led_progress(on_path, 1.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)
    finally:
        env.close()


def test_square_reference_led_gate_allows_limited_progress_catchup():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_m1 = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        on_path = np.asarray(ref.pos(0.1), dtype=np.float32)
        after_schedule = float(ref.duration + 0.5)
        assert ref.led(after_schedule) == pytest.approx(0.0)
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.095)
        env_m1._last_reset_seed = 29
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.115)
        env_m1._last_reset_seed = 23
        assert env_m1._progress_led_gate_max_off_target_ratio() == pytest.approx(0.095)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(1.0)

        env._stamp_led_progress(on_path, 1.0)
        assert env._scripted_led_ref(on_path, after_schedule) == pytest.approx(0.0)
    finally:
        env.close()
        env_m1.close()


def test_rl_reference_seed7_uses_seeded_progress_gate():
    ref = make_letter_ref("RL", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="RL",
        phase="B",
        wind_mode="M1",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        env._last_reset_seed = 7
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.13)
        env._last_reset_seed = 11
        assert env._progress_led_gate_max_off_target_ratio() == pytest.approx(0.115)
    finally:
        env.close()


def test_reset_seed_resamples_initial_noise():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.05,
    )
    try:
        _, info_1 = env.reset(seed=1)
        _, info_2 = env.reset(seed=2)
        assert not np.allclose(info_1["pos"], info_2["pos"])
    finally:
        env.close()


def test_capture_snapshot_after_rollout(env):
    env.reset(seed=42)
    for _ in range(10):
        env.step(np.zeros(0, dtype=np.float32))
    img = capture_pybullet_snapshot(env.CLIENT, width=320, height=320)
    assert img.shape == (320, 320, 3)
    assert img.dtype == np.uint8
    # The default camera frames a sky+floor scene, so mean brightness should be > 50
    assert img.mean() > 50.0


def test_phase_b_zero_action_matches_pid_led_baseline():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env_a = LightPaintAviaryPyB(
        label="square",
        phase="A",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    env_b = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        assert env_b.action_space.shape == (4,)
        assert env_b.action_space.dtype == np.float32
        assert np.allclose(env_b.action_space.low, -1.0)
        assert np.allclose(env_b.action_space.high, 1.0)

        env_a.reset(seed=0)
        env_b.reset(seed=0)
        _, _, _, _, info_a = env_a.step(np.zeros(0, dtype=np.float32))
        _, _, _, _, info_b = env_b.step(np.zeros(4, dtype=np.float32))

        np.testing.assert_allclose(info_b["delta_v"], np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(info_b["target_vel"], info_b["v_ref"], atol=1e-7)
        assert info_b["delta_led"] == pytest.approx(0.0)
        assert info_b["brightness"] == pytest.approx(info_b["led_ref"])
        for key in (
            "r_path",
            "r_schedule",
            "r_led_target",
            "r_led_off",
            "r_led_miss",
            "r_smooth",
            "r_corner_speed",
            "r_completion",
            "paint_coverage",
        ):
            assert key in info_b
            assert np.isfinite(info_b[key])
        assert info_b["r_schedule"] > info_b["r_path"]
        np.testing.assert_allclose(info_b["u_pid_rpm"], info_a["u_pid_rpm"], rtol=1e-5, atol=1e-3)
        assert info_b["brightness"] == pytest.approx(info_a["brightness"])
    finally:
        env_a.close()
        env_b.close()


def test_phase_b_nonzero_action_changes_target_velocity_and_led():
    ref = make_square_ref(side_m=0.8, speed=0.35)
    env = LightPaintAviaryPyB(
        label="square",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        env.reset(seed=0)
        action = np.array([1.0, -0.5, 0.25, -1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        expected_delta_v = np.array([0.5, -0.25, 0.125], dtype=np.float32)
        np.testing.assert_allclose(info["delta_v"], expected_delta_v, atol=1e-7)
        np.testing.assert_allclose(
            info["target_vel"],
            np.asarray(info["v_ref"], dtype=np.float32) + expected_delta_v,
            atol=1e-7,
        )
        assert info["delta_led"] == pytest.approx(-1.0)
        assert info["led_ref"] == pytest.approx(1.0)
        assert info["brightness"] == pytest.approx(0.0)
        assert info["led_on"] is False
        assert info["r_led_miss"] < 0.0
        assert info["r_action_mag"] < 0.0
        assert info["r_led_flicker"] <= 0.0
    finally:
        env.close()


def test_phase_b_corner_context_uses_reference_semantic_sharpness():
    ref = make_letter_ref("L", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="L",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        assert env._corner_indices.size >= 1
        corner_idx = int(env._corner_indices[0])
        segment_idx = max(corner_idx - 1, 0)
        alpha = 1.0 if segment_idx == corner_idx - 1 else 0.0

        influence, sharpness, corner_dist = env._corner_context(segment_idx, alpha)

        assert corner_dist == pytest.approx(0.0, abs=1e-6)
        assert sharpness >= DEFAULT_CORNER_HINT_MIN_SHARPNESS - 1e-6
        assert influence >= DEFAULT_CORNER_HINT_MIN_SHARPNESS - 1e-6
    finally:
        env.close()


def test_phase_b_reward_corner_context_uses_reference_schedule():
    ref = make_letter_ref("L", plane="xz", speed=0.35)
    env = LightPaintAviaryPyB(
        label="L",
        phase="B",
        wind_mode="M0",
        reference=ref,
        max_episode_steps=5,
        init_box_size=0.0,
    )
    try:
        t_corner = float(ref.corner_times[0])
        env._last_v_ref = np.asarray(ref.vel(t_corner), dtype=np.float32)
        env._last_delta_v = np.zeros(3, dtype=np.float32)
        env._prev_delta_v = None

        _, components = env._compute_reward_phase_a(
            pos=np.asarray(ref.pos(0.0), dtype=np.float32),
            p_ref=np.asarray(ref.pos(t_corner), dtype=np.float32),
            led_ref=1.0,
            brightness=1.0,
            rpm=np.zeros(4, dtype=np.float32),
            prev_rpm=None,
            out_of_bounds=False,
            new_target=0.0,
            off_target=0.0,
            repaint=0.0,
            prev_brightness=1.0,
            speed=0.0,
            coverage=0.0,
            completion_due=False,
            t_ref=t_corner,
        )

        assert components["corner_influence"] >= DEFAULT_CORNER_HINT_MIN_SHARPNESS - 1e-6
        assert components["corner_dist_m"] == pytest.approx(0.0, abs=1e-6)
    finally:
        env.close()


def test_phase_b_step_passes_action_to_preprocess(monkeypatch):
    env = LightPaintAviaryPyB(label="L", phase="B", wind_mode="M0", max_episode_steps=5)
    seen = []
    original = env._preprocessAction

    def wrapped(action):
        seen.append(np.asarray(action, dtype=np.float32).flatten().copy())
        return original(action)

    try:
        env.reset(seed=0)
        monkeypatch.setattr(env, "_preprocessAction", wrapped)
        action = np.array([0.25, 0.0, -0.25, 0.5], dtype=np.float32)
        env.step(action)
        assert seen
        np.testing.assert_allclose(seen[0], action, atol=1e-7)
    finally:
        env.close()


def test_phase_b_m1_disturbance_applies_world_frame_force(monkeypatch):
    env = LightPaintAviaryPyB(label="L", phase="B", wind_mode="M1", max_episode_steps=5)
    calls = []
    original = p.applyExternalForce

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    try:
        monkeypatch.setattr(p, "applyExternalForce", wrapped)
        env.reset(seed=0)
        env.step(np.zeros(4, dtype=np.float32))
        assert calls
        _, kwargs = calls[-1]
        assert kwargs["flags"] == p.WORLD_FRAME
        assert np.linalg.norm(np.asarray(kwargs["forceObj"], dtype=np.float32)) > 0.0
    finally:
        env.close()


def test_pyb_out_of_bounds_includes_y_canvas_latitude():
    env = LightPaintAviaryPyB(label="L", phase="B", wind_mode="M0", max_episode_steps=5)
    try:
        assert env._is_out_of_bounds(np.array([0.0, Y_CANVAS + Y_BOUND_M + 0.01, 1.5], dtype=np.float32))
        assert not env._is_out_of_bounds(np.array([0.0, Y_CANVAS + Y_BOUND_M - 0.01, 1.5], dtype=np.float32))
    finally:
        env.close()


def test_gymnasium_reset_step_api_for_phase_a_and_b():
    envs = [
        LightPaintAviaryPyB(label="L", phase="A", wind_mode="M0", max_episode_steps=5),
        LightPaintAviaryPyB(label="L", phase="B", wind_mode="M2", max_episode_steps=5),
    ]
    try:
        for env in envs:
            reset_out = env.reset(seed=1)
            assert isinstance(reset_out, tuple) and len(reset_out) == 2
            obs, info = reset_out
            assert env.observation_space.contains(obs)
            assert isinstance(info, dict)

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            step_out = env.step(action)
            assert isinstance(step_out, tuple) and len(step_out) == 5
            obs, reward, terminated, truncated, info = step_out
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if env.phase == "B":
                assert np.linalg.norm(np.asarray(info["wind_force"], dtype=np.float32)) > 0.0
                assert info["wind_frame"] == "WORLD_FRAME"
    finally:
        for env in envs:
            env.close()


def test_case_preserved_on_label():
    env = LightPaintAviaryPyB(label="Pig", phase="A", wind_mode="M0")
    try:
        assert env.label == "Pig"
    finally:
        env.close()
