"""Shared reward and action-scale constants for light-painting RL.

PyBullet remains the production physics backend and the standalone env remains
the dependency-free fallback.  Both should read reward/action contract values
from this module so reward experiments do not drift by backend.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

RESIDUAL_DELTA_MAX = 0.5
LED_RESIDUAL_SCALE = 1.0
LED_ON_THRESHOLD = 0.5

LAMBDA_SMOOTH = 0.4
PATH_SIGMA_M = 0.08
SCHEDULE_SIGMA_M = 0.15
W_PATH = 0.6
W_SCHEDULE = 1.0
W_NEW_TARGET = 1.2
W_OFF_TARGET = 1.6
W_REPAINT = 0.2
W_ACTION_MAG = 0.8
W_ACTION_RATE = 0.3
W_LED_MISS = 1.2
W_LED_FLICKER = 0.05

CORNER_WINDOW_M = 0.28
CORNER_SPEED_FRACTION = 0.65
CORNER_SIGMA_M = 0.08
W_CORNER_SPEED = 16.0
W_CORNER_TRACK = 0.25
W_CORNER_PATH = 0.7
W_CORNER_DECEL = 2.0
W_CORNER_ACCEL = 2.5
W_CORNER_LATERAL = 1.0

W_COMPLETION = 3.0
W_INCOMPLETE = 1.0
W_TERMINAL_BOUNDS = 100.0

REWARD_COMPONENT_KEYS = (
    "r_path",
    "r_schedule",
    "r_track",
    "r_led_target",
    "r_led_off",
    "r_repaint",
    "r_led_miss",
    "r_paint_outcome",
    "r_led_prior",
    "r_led_scripted",
    "r_led_total",
    "r_smooth",
    "r_action_mag",
    "r_action_rate",
    "r_led_flicker",
    "r_corner_track",
    "r_corner_path",
    "r_corner_speed",
    "r_corner_decel",
    "r_corner_accel",
    "r_corner_lateral",
    "r_completion",
    "r_terminal",
    "path_dist",
    "schedule_err",
    "paint_coverage",
    "corner_sharpness",
    "corner_influence",
    "corner_dist_m",
    "corner_delta_v_along_ref",
    "paint_new_target",
    "paint_off_target",
    "paint_repaint",
)

OVERRIDABLE_REWARD_KEYS = (
    "RESIDUAL_DELTA_MAX",
    "LED_RESIDUAL_SCALE",
    "LED_ON_THRESHOLD",
    "LAMBDA_SMOOTH",
    "PATH_SIGMA_M",
    "SCHEDULE_SIGMA_M",
    "W_PATH",
    "W_SCHEDULE",
    "W_NEW_TARGET",
    "W_OFF_TARGET",
    "W_REPAINT",
    "W_ACTION_MAG",
    "W_ACTION_RATE",
    "W_LED_MISS",
    "W_LED_FLICKER",
    "CORNER_WINDOW_M",
    "CORNER_SPEED_FRACTION",
    "CORNER_SIGMA_M",
    "W_CORNER_SPEED",
    "W_CORNER_TRACK",
    "W_CORNER_PATH",
    "W_CORNER_DECEL",
    "W_CORNER_ACCEL",
    "W_CORNER_LATERAL",
    "W_COMPLETION",
    "W_INCOMPLETE",
    "W_TERMINAL_BOUNDS",
)

ACTIVE_REWARD_CONFIG_PATH: str | None = None
ACTIVE_REWARD_OVERRIDES: dict[str, float] = {}


def _apply_env_reward_overrides() -> None:
    """Apply per-run reward constants from LIGHTPAINT_REWARD_CONFIG JSON."""
    global ACTIVE_REWARD_CONFIG_PATH, ACTIVE_REWARD_OVERRIDES

    raw_path = os.environ.get("LIGHTPAINT_REWARD_CONFIG")
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"LIGHTPAINT_REWARD_CONFIG 경로가 존재하지 않습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("LIGHTPAINT_REWARD_CONFIG는 JSON object여야 합니다")

    allowed = set(OVERRIDABLE_REWARD_KEYS)
    overrides: dict[str, float] = {}
    for key, value in data.items():
        if key not in allowed:
            raise RuntimeError(f"지원하지 않는 reward override key입니다: {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Reward override {key}는 유한한 숫자여야 합니다")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"Reward override {key}는 유한한 숫자여야 합니다")
        globals()[key] = numeric
        overrides[key] = numeric

    ACTIVE_REWARD_CONFIG_PATH = str(path)
    ACTIVE_REWARD_OVERRIDES = overrides


_apply_env_reward_overrides()

# Backward-compatible alias kept in sync with the canonical key.
W_FLICKER = W_LED_FLICKER
