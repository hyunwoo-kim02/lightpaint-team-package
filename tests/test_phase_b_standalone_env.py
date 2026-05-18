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
    env = LightPaintAviaryW1(label="L", phase="B", wind_mode="M0", max_episode_steps=5)
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (4,)
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
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
    assert info["led_ref"] == pytest.approx(1.0)
    assert info["delta_led"] == pytest.approx(-1.0)
    assert info["brightness"] == pytest.approx(0.0)
    assert info["led_on"] is False


def test_phase_b_m2_disturbance_smoke():
    env = LightPaintAviaryW1(label="L", phase="B", wind_mode="M2", max_episode_steps=5)
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert np.linalg.norm(np.asarray(info["wind_force"], dtype=np.float32)) > 0.0
    assert info["wind_frame"] == "WORLD_FRAME"
