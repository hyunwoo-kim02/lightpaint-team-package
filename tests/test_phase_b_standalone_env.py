"""Fallback Phase B contract tests for the numpy-only env."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.light_paint_aviary_standalone import LightPaintAviaryW1


def test_phase_b_action_space_and_zero_action_smoke():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        max_episode_steps=5,
    )
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (4,)
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert np.isfinite(reward)
    assert info["r"] == pytest.approx(reward)
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
        assert key in info
        assert np.isfinite(info[key])
    assert info["r_schedule"] > info["r_path"]
    assert terminated is False
    assert truncated is False
    np.testing.assert_allclose(info["delta_v"], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(info["target_vel"], info["v_ref"], atol=1e-7)
    assert info["delta_led"] == pytest.approx(0.0)
    assert info["brightness"] == pytest.approx(info["led_ref"])


def test_phase_b_nonzero_action_affects_velocity_and_led():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=5,
    )
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
    np.testing.assert_allclose(info["u_pid"], info["u_final"], atol=1e-7)
    assert info["led_ref"] == pytest.approx(1.0)
    assert info["delta_led"] == pytest.approx(-1.0)
    assert info["brightness"] == pytest.approx(0.0)
    assert info["led_on"] is False
    assert info["r_led_miss"] < 0.0
    assert info["r_action_mag"] < 0.0
    assert info["r_led_flicker"] <= 0.0


def test_phase_b_m2_disturbance_smoke():
    env = LightPaintAviaryW1(label="L", phase="B", wind_mode="M2", max_episode_steps=5)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert np.linalg.norm(np.asarray(info["wind_force"], dtype=np.float32)) > 0.0
    assert info["wind_frame"] == "WORLD_FRAME"
    assert np.isfinite(info["r"])
    assert "r_path" in info and "r_led_off" in info


def test_phase_b_half_brightness_does_not_paint_when_led_off():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=5,
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -0.5], dtype=np.float32))
    assert info["brightness"] == pytest.approx(0.5)
    assert info["led_on"] is False
    assert info["painted_px"] == 0
    assert info["paint_new_target"] == pytest.approx(0.0)


def test_led_off_is_worse_than_scripted_on_when_paint_is_expected():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=5,
    )
    env.reset(seed=0)
    _, reward_on, _, _, info_on = env.step(np.zeros(4, dtype=np.float32))

    env.reset(seed=0)
    _, reward_off, _, _, info_off = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))

    assert info_on["led_ref"] == pytest.approx(1.0)
    assert info_on["brightness"] == pytest.approx(1.0)
    assert info_off["brightness"] == pytest.approx(0.0)
    assert info_off["r_led_miss"] < 0.0
    assert reward_on > reward_off


def test_reward_state_is_reset_between_episodes():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=5,
    )
    env.reset(seed=0)
    env.step(np.zeros(4, dtype=np.float32))

    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32))

    assert info["brightness"] == pytest.approx(0.0)
    assert info["r_action_rate"] == pytest.approx(0.0)
    assert info["r_led_flicker"] == pytest.approx(0.0)


def test_completion_reward_is_one_shot_and_coverage_based():
    env = LightPaintAviaryW1(
        label="L",
        phase="B",
        wind_mode="M0",
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=5,
    )
    env.reset(seed=0)
    env.cumulative = env.G_letter.copy()

    reward, components = env._compute_reward_phase_a(
        env._pos,
        env._pos,
        1.0,
        1.0,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        False,
        np.zeros(3, dtype=np.float32),
        None,
        0.0,
        0.0,
        0.0,
        1.0,
        env._paint_coverage(),
        True,
    )

    assert np.isfinite(reward)
    assert components["paint_coverage"] == pytest.approx(1.0)
    assert components["r_completion"] == pytest.approx(3.0)
