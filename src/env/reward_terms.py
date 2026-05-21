"""Pure reward helpers shared by PyBullet and standalone light-paint envs."""
from __future__ import annotations

from typing import Any

import numpy as np

from src.env.reward_config import (
    CORNER_SIGMA_M,
    CORNER_SPEED_FRACTION,
    LAMBDA_SMOOTH,
    LED_ON_THRESHOLD,
    PATH_SIGMA_M,
    SCHEDULE_SIGMA_M,
    W_ACTION_MAG,
    W_ACTION_RATE,
    W_COMPLETION,
    W_CORNER_ACCEL,
    W_CORNER_DECEL,
    W_CORNER_LATERAL,
    W_CORNER_PATH,
    W_CORNER_SPEED,
    W_CORNER_TRACK,
    W_INCOMPLETE,
    W_LED_FLICKER,
    W_LED_MISS,
    W_NEW_TARGET,
    W_OFF_TARGET,
    W_PATH,
    W_REPAINT,
    W_SCHEDULE,
    W_TERMINAL_BOUNDS,
)


def led_brightness_from_ref(led_ref: float, delta_led: float) -> float:
    return float(np.clip(float(led_ref) + float(delta_led), 0.0, 1.0))


def is_led_on(brightness: float) -> bool:
    return bool(float(brightness) > LED_ON_THRESHOLD)


def compute_lightpaint_reward(
    *,
    path_dist: float,
    schedule_err: float,
    led_ref: float,
    brightness: float,
    new_target: float,
    off_target: float,
    repaint: float,
    command: np.ndarray,
    prev_command: np.ndarray | None,
    delta_v: np.ndarray,
    prev_delta_v: np.ndarray | None,
    prev_brightness: float,
    corner_influence: float,
    corner_sharpness: float,
    corner_dist: float,
    speed: float,
    ref_speed: float,
    v_ref: np.ndarray,
    coverage: float,
    completion_due: bool,
    out_of_bounds: bool,
    command_smooth_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Compute reward components without depending on an env instance."""
    path_dist = float(path_dist)
    schedule_err = float(schedule_err)
    corner_influence = float(corner_influence)
    r_path = W_PATH * float(np.exp(-((path_dist / PATH_SIGMA_M) ** 2)))
    r_schedule = W_SCHEDULE * float(np.exp(-((schedule_err / SCHEDULE_SIGMA_M) ** 2)))
    r_corner_track = W_CORNER_TRACK * corner_influence * float(
        np.exp(-((schedule_err / CORNER_SIGMA_M) ** 2))
    )
    r_corner_path = W_CORNER_PATH * corner_influence * float(
        np.exp(-((path_dist / CORNER_SIGMA_M) ** 2))
    )
    r_led_target = W_NEW_TARGET * float(new_target)
    r_led_off = -W_OFF_TARGET * float(off_target)
    r_repaint = -W_REPAINT * float(repaint)
    # One-sided LED miss prior: penalize being dimmer than the scripted LED
    # reference, while false-on behavior is judged by painting outcome terms.
    r_led_miss = -W_LED_MISS * max(0.0, float(led_ref) - float(brightness))
    r_paint_outcome = r_led_target + r_led_off + r_repaint
    r_led_prior = r_led_miss

    command_arr = np.asarray(command, dtype=np.float32).reshape(-1)
    if prev_command is None:
        r_smooth = LAMBDA_SMOOTH
    else:
        prev_command_arr = np.asarray(prev_command, dtype=np.float32).reshape(-1)
        scale = max(float(command_smooth_scale), 1e-9)
        du = (command_arr - prev_command_arr) / scale
        r_smooth = float(np.exp(-float(np.linalg.norm(du)))) * LAMBDA_SMOOTH

    delta_v_arr = np.asarray(delta_v, dtype=np.float32).reshape(3)
    r_action_mag = -W_ACTION_MAG * float(np.dot(delta_v_arr, delta_v_arr))
    if prev_delta_v is not None:
        prev_delta_arr = np.asarray(prev_delta_v, dtype=np.float32).reshape(3)
        delta_rate = delta_v_arr - prev_delta_arr
        r_action_rate = -W_ACTION_RATE * float(np.dot(delta_rate, delta_rate))
    else:
        r_action_rate = 0.0
    r_led_flicker = -W_LED_FLICKER * abs(float(brightness) - float(prev_brightness))
    r_led_total = r_paint_outcome + r_led_prior + r_led_flicker

    ref_speed = float(ref_speed)
    if ref_speed < 1e-6:
        ref_speed = 0.0
    v_corner_limit = ref_speed * CORNER_SPEED_FRACTION
    r_corner_speed = -W_CORNER_SPEED * corner_influence * max(0.0, float(speed) - v_corner_limit) ** 2

    v_ref_arr = np.asarray(v_ref, dtype=np.float32).reshape(3)
    v_ref_norm = float(np.linalg.norm(v_ref_arr))
    if v_ref_norm > 1e-6:
        v_ref_unit = v_ref_arr / v_ref_norm
        delta_along_ref = float(np.dot(delta_v_arr, v_ref_unit))
        delta_lateral = delta_v_arr - delta_along_ref * v_ref_unit
    else:
        delta_along_ref = 0.0
        delta_lateral = delta_v_arr
    r_corner_decel = W_CORNER_DECEL * corner_influence * max(0.0, -delta_along_ref)
    r_corner_accel = -W_CORNER_ACCEL * corner_influence * max(0.0, delta_along_ref)
    r_corner_lateral = -W_CORNER_LATERAL * corner_influence * float(np.dot(delta_lateral, delta_lateral))

    coverage = float(np.clip(coverage, 0.0, 1.0))
    r_completion = (W_COMPLETION * coverage - W_INCOMPLETE * (1.0 - coverage)) if completion_due else 0.0
    r_terminal = -W_TERMINAL_BOUNDS if out_of_bounds else 0.0
    total = (
        r_path
        + r_schedule
        + r_led_target
        + r_led_off
        + r_repaint
        + r_led_miss
        + r_smooth
        + r_action_mag
        + r_action_rate
        + r_led_flicker
        + r_corner_track
        + r_corner_path
        + r_corner_speed
        + r_corner_decel
        + r_corner_accel
        + r_corner_lateral
        + r_completion
        + r_terminal
    )
    components: dict[str, float] = {
        "r_path": r_path,
        "r_schedule": r_schedule,
        "r_track": r_path + r_schedule,
        "r_led_target": r_led_target,
        "r_led_off": r_led_off,
        "r_repaint": r_repaint,
        "r_led_miss": r_led_miss,
        "r_paint_outcome": r_paint_outcome,
        "r_led_prior": r_led_prior,
        "r_led_scripted": r_paint_outcome + r_led_prior,
        "r_led_total": r_led_total,
        "r_smooth": r_smooth,
        "r_action_mag": r_action_mag,
        "r_action_rate": r_action_rate,
        "r_led_flicker": r_led_flicker,
        "r_corner_track": r_corner_track,
        "r_corner_path": r_corner_path,
        "r_corner_speed": r_corner_speed,
        "r_corner_decel": r_corner_decel,
        "r_corner_accel": r_corner_accel,
        "r_corner_lateral": r_corner_lateral,
        "r_completion": r_completion,
        "r_terminal": r_terminal,
        "path_dist": path_dist,
        "schedule_err": schedule_err,
        "paint_coverage": coverage,
        "corner_sharpness": float(corner_sharpness),
        "corner_influence": corner_influence,
        "corner_dist_m": float(corner_dist),
        "corner_delta_v_along_ref": delta_along_ref,
        "paint_new_target": float(new_target),
        "paint_off_target": float(off_target),
        "paint_repaint": float(repaint),
    }
    return float(total), components
