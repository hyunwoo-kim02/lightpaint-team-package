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
from src.env.lightpaint_ref import make_square_ref
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
