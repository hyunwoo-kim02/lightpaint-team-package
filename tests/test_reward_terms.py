from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env import reward_config as cfg
from src.env.reward_terms import compute_lightpaint_reward, is_led_on, led_brightness_from_ref


def _reward_kwargs(**overrides):
    base = {
        "path_dist": 0.02,
        "schedule_err": 0.03,
        "led_ref": 1.0,
        "brightness": 1.0,
        "new_target": 0.2,
        "off_target": 0.1,
        "repaint": 0.0,
        "command": np.array([0.1, 0.0, 0.0], dtype=np.float32),
        "prev_command": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "delta_v": np.array([-0.1, 0.0, 0.0], dtype=np.float32),
        "prev_delta_v": np.zeros(3, dtype=np.float32),
        "prev_brightness": 0.5,
        "corner_influence": 0.7,
        "corner_sharpness": 1.0,
        "corner_dist": 0.04,
        "speed": 0.3,
        "ref_speed": 0.35,
        "v_ref": np.array([0.35, 0.0, 0.0], dtype=np.float32),
        "coverage": 0.8,
        "completion_due": True,
        "out_of_bounds": False,
    }
    base.update(overrides)
    return base


def test_led_brightness_contract():
    assert led_brightness_from_ref(1.0, -0.5) == pytest.approx(0.5)
    assert is_led_on(0.5) is False
    assert is_led_on(0.5001) is True


def test_reward_config_env_override_file(monkeypatch, tmp_path):
    override_path = tmp_path / "reward_override.json"
    override_path.write_text(
        json.dumps({"W_PATH": 1.75, "RESIDUAL_DELTA_MAX": 0.25, "W_LED_FLICKER": 0.12}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LIGHTPAINT_REWARD_CONFIG", str(override_path))

    overridden = importlib.reload(cfg)
    assert overridden.W_PATH == pytest.approx(1.75)
    assert overridden.RESIDUAL_DELTA_MAX == pytest.approx(0.25)
    assert overridden.W_FLICKER == pytest.approx(0.12)
    assert overridden.ACTIVE_REWARD_CONFIG_PATH == str(override_path)
    assert overridden.ACTIVE_REWARD_OVERRIDES["W_PATH"] == pytest.approx(1.75)

    monkeypatch.delenv("LIGHTPAINT_REWARD_CONFIG")
    restored = importlib.reload(cfg)
    assert restored.W_PATH == pytest.approx(0.6)
    assert restored.RESIDUAL_DELTA_MAX == pytest.approx(0.5)


def test_reward_config_env_override_accepts_windows_utf8_bom(monkeypatch, tmp_path):
    override_path = tmp_path / "reward_override_bom.json"
    override_path.write_text(json.dumps({"W_PATH": 1.25}), encoding="utf-8-sig")
    monkeypatch.setenv("LIGHTPAINT_REWARD_CONFIG", str(override_path))

    overridden = importlib.reload(cfg)
    assert overridden.W_PATH == pytest.approx(1.25)

    monkeypatch.delenv("LIGHTPAINT_REWARD_CONFIG")
    importlib.reload(cfg)


def test_reward_components_include_contract_keys():
    total, components = compute_lightpaint_reward(**_reward_kwargs())

    assert np.isfinite(total)
    assert set(cfg.REWARD_COMPONENT_KEYS).issubset(components)
    assert components["r_track"] == pytest.approx(components["r_path"] + components["r_schedule"])
    assert components["r_paint_outcome"] == pytest.approx(
        components["r_led_target"] + components["r_led_off"] + components["r_repaint"]
    )
    assert components["r_led_prior"] == pytest.approx(components["r_led_miss"])
    assert components["r_led_scripted"] == pytest.approx(
        components["r_paint_outcome"] + components["r_led_prior"]
    )
    assert components["r_led_total"] == pytest.approx(
        components["r_paint_outcome"] + components["r_led_prior"] + components["r_led_flicker"]
    )
    assert components["corner_delta_v_along_ref"] < 0.0


def test_reward_terminal_and_led_miss_terms():
    _, components = compute_lightpaint_reward(
        **_reward_kwargs(brightness=0.0, out_of_bounds=True, completion_due=False)
    )

    assert components["r_led_miss"] < 0.0
    assert components["r_led_prior"] == pytest.approx(components["r_led_miss"])
    assert components["r_terminal"] == pytest.approx(-cfg.W_TERMINAL_BOUNDS)
    assert components["r_completion"] == pytest.approx(0.0)


def test_led_prior_is_one_sided_miss_prior():
    _, components = compute_lightpaint_reward(
        **_reward_kwargs(
            led_ref=0.0,
            brightness=1.0,
            new_target=0.0,
            off_target=0.0,
            repaint=0.0,
        )
    )

    assert components["r_led_miss"] == pytest.approx(0.0)
    assert components["r_led_prior"] == pytest.approx(0.0)
    assert components["r_paint_outcome"] == pytest.approx(0.0)


def test_standalone_reward_config_parity():
    from src.env import light_paint_aviary_standalone as standalone

    assert standalone.RESIDUAL_DELTA_MAX == cfg.RESIDUAL_DELTA_MAX
    assert standalone.LED_RESIDUAL_SCALE == cfg.LED_RESIDUAL_SCALE
    assert standalone.CORNER_WINDOW_M == cfg.CORNER_WINDOW_M
    assert standalone.W_FLICKER == cfg.W_LED_FLICKER
    assert standalone.W_ACTION_MAG == cfg.W_ACTION_MAG


def test_pybullet_reward_config_parity_when_available():
    pytest.importorskip("pybullet")
    pytest.importorskip("pybullet_data")
    pytest.importorskip("gym_pybullet_drones")
    from src.env import light_paint_aviary_pyb as pyb

    assert pyb.RESIDUAL_DELTA_MAX == cfg.RESIDUAL_DELTA_MAX
    assert pyb.LED_RESIDUAL_SCALE == cfg.LED_RESIDUAL_SCALE
    assert pyb.CORNER_WINDOW_M == cfg.CORNER_WINDOW_M
    assert pyb.W_LED_FLICKER == cfg.W_LED_FLICKER
    assert pyb.W_ACTION_MAG == cfg.W_ACTION_MAG
