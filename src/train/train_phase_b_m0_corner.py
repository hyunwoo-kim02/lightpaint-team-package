"""Phase B PPO/BC training and evaluation for residual velocity and LED control.

The success target is whether the residual policy reduces corner-window
tracking error while preserving straight-segment tracking. The script accepts
square, letter, and drawn JSON references plus M0/M1/M2 wind modes.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.lightpaint_ref import LightPaintRef, corner_distances_for_ref, corner_window_radius_m
from src.train.reference_factory import add_reference_args, build_reference_from_args
from src.train.train_phase_a_pid import (
    CTRL_FREQ,
    _corner_errors,
    _safe_token,
    _save_corner_csv,
    _save_tracking_plot,
    _save_trajectory_csv,
)
from src.train.visualize_flight import save_phase_a_visualization


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "package_root": str(_PKG_ROOT),
        "argv": list(sys.argv),
    }



@dataclass(frozen=True)
class RolloutResult:
    tag: str
    phase: str
    wind_mode: str
    time_arr: np.ndarray
    ref_arr: np.ndarray
    v_ref_arr: np.ndarray
    pos_arr: np.ndarray
    brightness_arr: np.ndarray
    rpm_arr: np.ndarray
    action_arr: np.ndarray
    delta_v_arr: np.ndarray
    reward_arr: np.ndarray
    info_rows: list[dict[str, Any]]
    cumulative: np.ndarray
    target_mask: np.ndarray
    crash: bool
    walltime_s: float


class DependencyError(RuntimeError):
    """Raised when the selected Python environment cannot run PyBullet training."""


def _make_progress_callback(total_timesteps: int, *, interval: int | None = None):
    from stable_baselines3.common.callbacks import BaseCallback

    log_interval = int(interval or max(10_000, int(total_timesteps) // 20))

    class _ProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._next_log = log_interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next_log:
                print(
                    f"[phase_b] PPO progress: timesteps={self.num_timesteps}/{int(total_timesteps)}",
                    flush=True,
                )
                while self._next_log <= self.num_timesteps:
                    self._next_log += log_interval
            return True

    return _ProgressCallback()


PAINTING_THRESHOLD = 0.3
PAINTING_METRIC_KEYS = [
    "painted_pixel_coverage",
    "painted_pixel_recall",
    "painted_pixel_precision",
    "painted_pixel_iou",
    "painted_pixel_dice",
    "off_target_pixel_ratio",
]
FINAL_GOAL_THRESHOLDS = {
    "painting_iou_min": 0.75,
    "painting_precision_min": 0.85,
    "painting_recall_min": 0.85,
    "painting_dice_min": 0.80,
    "off_target_ratio_max": 0.10,
    "path_rmse_m_max": 0.05,
    "path_max_m_max": 0.15,
    "straight_path_rmse_rel_max": 1.10,
    "corner_path_rmse_rel_max": 0.80,
    "corner_speed_ratio_min": 0.65,
    "corner_speed_ratio_max": 0.90,
    "corner_overshoot_rel_max": 0.80,
    "led_precision_min": 0.90,
    "led_recall_min": 0.90,
    "led_flicker_rate_abs_tolerance": 0.05,
    "rpm_saturation_ratio_max": 0.10,
    "action_norm_max": 2.000001,
    "action_rate_max": 4.000001,
}


def _make_env(
    *,
    phase: str,
    label: str,
    wind_mode: str,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
    gui: bool,
    ctrl_freq: int,
    pyb_freq: int,
) -> LightPaintAviaryPyB:
    from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB

    return LightPaintAviaryPyB(
        label=label,
        phase=phase,
        wind_mode=wind_mode,
        gui=bool(gui),
        init_box_size=float(init_box_size),
        led_always_on=bool(led_always_on),
        max_episode_steps=max_steps,
        reference=reference,
        ctrl_freq=int(ctrl_freq),
        pyb_freq=int(pyb_freq),
    )


def _phase_a_action(_: dict[str, Any]) -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def _phase_b_zero_action(_: dict[str, Any]) -> np.ndarray:
    return np.zeros(4, dtype=np.float32)


def _effective_corner_distances(ref: LightPaintRef, window_m: float) -> np.ndarray:
    return corner_distances_for_ref(ref, window_m)


def _nearest_corner_distance_s(ref: LightPaintRef, corner_s: np.ndarray, t: float) -> float:
    if corner_s.size == 0:
        return float("inf")
    s_now = float(t) * float(ref.speed)
    return float(np.min(np.abs(corner_s - s_now)))


def _ref_name(ref: LightPaintRef) -> str:
    return str(getattr(ref, "name", ""))


def _corner_filter_min_tangent_action(ref: LightPaintRef) -> float:
    return -0.17


def _corner_filter_led_path_gate_m(ref: LightPaintRef, info: dict[str, Any] | None = None) -> float:
    return 0.04


def _path_correction_action(
    action: np.ndarray,
    info: dict[str, Any],
    ref: LightPaintRef,
) -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _corner_teacher_gain_scale(ref: LightPaintRef) -> float:
    return 1.0


def _filter_led_residual_action(action_led: float, info: dict[str, Any], ref: LightPaintRef) -> float:
    if float(info.get("led_ref", 1.0)) <= 0.5:
        return -1.0
    return float(np.clip(action_led, -1.0, 1.0))

def _make_corner_decel_teacher(ref: LightPaintRef, gain_mps: float, window_m: float) -> Callable[[dict[str, Any]], np.ndarray]:
    from src.env.reward_config import RESIDUAL_DELTA_MAX

    internal_corner_s = _effective_corner_distances(ref, window_m)

    def _teacher(ctx: dict[str, Any]) -> np.ndarray:
        info = ctx["info"]
        t = float(info.get("t", 0.0))
        action = np.zeros(4, dtype=np.float32)
        if internal_corner_s.size == 0:
            return action
        dist_to_corner = _nearest_corner_distance_s(ref, internal_corner_s, t)
        if dist_to_corner > float(window_m):
            return action
        v_ref = np.asarray(info.get("v_ref", [0.0, 0.0, 0.0]), dtype=np.float32)
        norm = float(np.linalg.norm(v_ref))
        if norm <= 1e-6:
            return action
        delta_v = -float(gain_mps) * _corner_teacher_gain_scale(ref) * (v_ref / norm)
        action[:3] = np.clip(delta_v / RESIDUAL_DELTA_MAX, -1.0, 1.0)
        return action.astype(np.float32)

    return _teacher


def _filter_corner_tangent_decel_action(
    action: np.ndarray,
    info: dict[str, Any],
    ref: LightPaintRef,
    window_m: float,
) -> np.ndarray:
    """Keep only corner-window deceleration along the reference tangent."""
    raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
    filtered = np.zeros(4, dtype=np.float32)
    led_action = float(raw_action[3]) if raw_action.size >= 4 else 0.0
    filtered[3] = _filter_led_residual_action(led_action, info, ref)

    internal_corner_s = _effective_corner_distances(ref, window_m)
    if internal_corner_s.size == 0:
        return filtered

    t = float(info.get("t", 0.0))
    dist_to_corner = _nearest_corner_distance_s(ref, internal_corner_s, t)
    if dist_to_corner > float(window_m):
        return filtered

    v_ref = np.asarray(info.get("v_ref", [0.0, 0.0, 0.0]), dtype=np.float32)
    norm = float(np.linalg.norm(v_ref))
    if norm <= 1e-6:
        return filtered

    unit = v_ref / norm
    raw_delta = np.zeros(3, dtype=np.float32)
    raw_delta[: min(3, raw_action.size)] = raw_action[: min(3, raw_action.size)]
    tangent_action = float(np.dot(raw_delta, unit))
    tangent_action = min(tangent_action, _corner_filter_min_tangent_action(ref))
    filtered[:3] = tangent_action * unit

    path_dist = float(info.get("path_dist", 0.0))
    if math.isfinite(path_dist) and path_dist > _corner_filter_led_path_gate_m(ref, info):
        filtered[3] = min(float(filtered[3]), -0.65)
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)

def _wrap_train_action_filter(env: Any, ref: LightPaintRef, window_m: float, filter_name: str) -> Any:
    if filter_name != "corner_tangent_decel":
        return env

    from gymnasium import Wrapper

    class _FilteredTrainActionEnv(Wrapper):
        def __init__(self, wrapped_env: Any) -> None:
            super().__init__(wrapped_env)
            self._last_info: dict[str, Any] = {}

        def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            obs, info = self.env.reset(*args, **kwargs)
            self._last_info = dict(info)
            return obs, info

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            filtered = _filter_corner_tangent_decel_action(
                np.asarray(action, dtype=np.float32).reshape(-1),
                self._last_info,
                ref,
                window_m,
            )
            obs, reward, terminated, truncated, info = self.env.step(filtered)
            self._last_info = dict(info)
            return obs, reward, terminated, truncated, info

    return _FilteredTrainActionEnv(env)


def _rollout(
    *,
    tag: str,
    phase: str,
    label: str,
    wind_mode: str,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
    gui: bool,
    gui_hold_seconds: float,
    ctrl_freq: int,
    pyb_freq: int,
    action_fn: Callable[[dict[str, Any]], np.ndarray],
) -> RolloutResult:
    env = _make_env(
        phase=phase,
        label=label,
        wind_mode=wind_mode,
        reference=reference,
        max_steps=max_steps,
        seed=seed,
        init_box_size=init_box_size,
        led_always_on=led_always_on,
        gui=gui,
        ctrl_freq=ctrl_freq,
        pyb_freq=pyb_freq,
    )
    obs, info = env.reset(seed=seed)
    pos_list = [info["pos"]]
    ref_list = [info["p_ref"]]
    v_ref_list = [info.get("v_ref", [0.0, 0.0, 0.0])]
    time_list = [float(info.get("t", 0.0))]
    brightness_list = [float(info.get("brightness", 0.0))]
    rpm_list = [info.get("u_pid_rpm", [0.0, 0.0, 0.0, 0.0])]
    action_list = [np.zeros(4, dtype=np.float32)]
    delta_v_list = [info.get("delta_v", [0.0, 0.0, 0.0])]
    rewards: list[float] = []
    info_rows: list[dict[str, Any]] = []
    term = False

    t_start = time.perf_counter()
    for _ in range(max_steps):
        action = np.asarray(action_fn({"obs": obs, "info": info}), dtype=np.float32).flatten()
        obs, reward, term, trunc, info = env.step(action)
        padded_action = np.zeros(4, dtype=np.float32)
        if action.size:
            padded_action[: min(4, action.size)] = action[:4]
        pos_list.append(info["pos"])
        ref_list.append(info["p_ref"])
        v_ref_list.append(info.get("v_ref", [0.0, 0.0, 0.0]))
        time_list.append(float(info.get("t", len(time_list) / float(ctrl_freq))))
        brightness_list.append(float(info.get("brightness", 0.0)))
        rpm_list.append(info.get("u_pid_rpm", [0.0, 0.0, 0.0, 0.0]))
        action_list.append(padded_action.copy())
        delta_v_list.append(info.get("delta_v", [0.0, 0.0, 0.0]))
        rewards.append(float(reward))
        row = dict(info)
        row["reward"] = float(reward)
        info_rows.append(row)
        if term or trunc:
            break
    walltime = time.perf_counter() - t_start

    result = RolloutResult(
        tag=tag,
        phase=phase,
        wind_mode=wind_mode,
        time_arr=np.asarray(time_list, dtype=np.float32),
        ref_arr=np.asarray(ref_list, dtype=np.float32),
        v_ref_arr=np.asarray(v_ref_list, dtype=np.float32),
        pos_arr=np.asarray(pos_list, dtype=np.float32),
        brightness_arr=np.asarray(brightness_list, dtype=np.float32),
        rpm_arr=np.asarray(rpm_list, dtype=np.float32),
        action_arr=np.asarray(action_list, dtype=np.float32),
        delta_v_arr=np.asarray(delta_v_list, dtype=np.float32),
        reward_arr=np.asarray(rewards, dtype=np.float32),
        info_rows=info_rows,
        cumulative=env.cumulative.copy(),
        target_mask=env.G_letter.copy(),
        crash=bool(term),
        walltime_s=walltime,
    )
    if bool(gui) and float(gui_hold_seconds) > 0.0:
        print(f"[phase_b] holding PyBullet GUI for {float(gui_hold_seconds):.1f}s ({tag})", flush=True)
        time.sleep(float(gui_hold_seconds))
    env.close()
    return result


def _speed_from_positions(time_arr: np.ndarray, pos_arr: np.ndarray) -> np.ndarray:
    if len(pos_arr) <= 1:
        return np.zeros(len(pos_arr), dtype=np.float32)
    dt = np.diff(time_arr)
    dt = np.maximum(dt, 1e-6)
    speed = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1) / dt
    return np.concatenate([[0.0], speed]).astype(np.float32)


def _nearest_path_distances(ref: LightPaintRef, pos_arr: np.ndarray) -> np.ndarray:
    pts = np.asarray(ref.waypoints, dtype=np.float32)
    if len(pts) < 2:
        return np.linalg.norm(pos_arr - pts[0], axis=1).astype(np.float32)
    seg = pts[1:] - pts[:-1]
    seg_len2 = np.sum(seg * seg, axis=1)
    out = np.zeros(len(pos_arr), dtype=np.float32)
    for i, pos in enumerate(pos_arr):
        rel = pos[None, :] - pts[:-1]
        alpha = np.clip(np.sum(rel * seg, axis=1) / np.maximum(seg_len2, 1e-9), 0.0, 1.0)
        closest = pts[:-1] + alpha[:, None] * seg
        out[i] = float(np.min(np.linalg.norm(closest - pos[None, :], axis=1)))
    return out


def _corner_mask(ref: LightPaintRef, time_arr: np.ndarray, window_m: float) -> np.ndarray:
    corner_s = _effective_corner_distances(ref, window_m)
    if corner_s.size == 0:
        return np.zeros(len(time_arr), dtype=bool)
    corner_indices = np.asarray(
        np.searchsorted(np.asarray(ref.cumlen, dtype=np.float32), corner_s),
        dtype=np.int32,
    )
    radius_m = corner_window_radius_m(ref.cumlen, corner_indices, window_m)
    window_s = radius_m / max(float(ref.speed), 1e-6)
    internal_corner_times = corner_s / max(float(ref.speed), 1e-6)
    mask = np.zeros(len(time_arr), dtype=bool)
    for t_corner in internal_corner_times:
        mask |= np.abs(time_arr - float(t_corner)) <= window_s
    return mask


def _delta_v_along_ref(delta_v_arr: np.ndarray, v_ref_arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(delta_v_arr), dtype=np.float32)
    for i, (delta_v, v_ref) in enumerate(zip(delta_v_arr, v_ref_arr)):
        norm = float(np.linalg.norm(v_ref))
        if norm > 1e-6:
            out[i] = float(np.dot(delta_v, v_ref / norm))
    return out


def _painting_quality_metrics(
    cumulative: np.ndarray,
    target_mask: np.ndarray,
    threshold: float = PAINTING_THRESHOLD,
) -> dict[str, float]:
    painted = np.asarray(cumulative, dtype=np.float32) > float(threshold)
    target = np.asarray(target_mask, dtype=np.float32) > 0.5
    tp = int(np.logical_and(painted, target).sum())
    fp = int(np.logical_and(painted, ~target).sum())
    fn = int(np.logical_and(~painted, target).sum())
    painted_px = int(painted.sum())
    target_px = int(target.sum())
    recall = float(tp) / float(max(target_px, 1))
    precision = float(tp) / float(max(painted_px, 1))
    iou = float(tp) / float(max(tp + fp + fn, 1))
    dice = float(2 * tp) / float(max(2 * tp + fp + fn, 1))
    off_target_ratio = float(fp) / float(max(painted_px, 1))
    return {
        "painted_pixel_coverage": recall,
        "painted_pixel_recall": recall,
        "painted_pixel_precision": precision,
        "painted_pixel_iou": iou,
        "painted_pixel_dice": dice,
        "off_target_pixel_ratio": off_target_ratio,
        "painted_target_px": float(tp),
        "painted_off_target_px": float(fp),
        "painted_total_px": float(painted_px),
        "target_total_px": float(target_px),
    }


def _binary_prf(reference_on: np.ndarray, predicted_on: np.ndarray) -> tuple[float, float, float]:
    ref = np.asarray(reference_on, dtype=bool)
    pred = np.asarray(predicted_on, dtype=bool)
    tp = int(np.logical_and(ref, pred).sum())
    fp = int(np.logical_and(~ref, pred).sum())
    fn = int(np.logical_and(ref, ~pred).sum())
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    f1 = float(2 * tp) / float(max(2 * tp + fp + fn, 1))
    return precision, recall, f1


def _led_metrics(result: RolloutResult) -> dict[str, float]:
    led_ref = [0.0]
    led_ref.extend(float(row.get("led_ref", 0.0)) for row in result.info_rows)
    if len(led_ref) < len(result.brightness_arr):
        led_ref.extend([0.0] * (len(result.brightness_arr) - len(led_ref)))
    ref_on = np.asarray(led_ref[: len(result.brightness_arr)], dtype=np.float32) > 0.5
    pred_on = np.asarray(result.brightness_arr, dtype=np.float32) > 0.5
    precision, recall, f1 = _binary_prf(ref_on, pred_on)
    transitions = int(np.count_nonzero(pred_on[1:] != pred_on[:-1])) if pred_on.size > 1 else 0
    flicker_rate = float(transitions) / float(max(pred_on.size - 1, 1))
    false_on_rate = float(np.logical_and(~ref_on, pred_on).sum()) / float(max(pred_on.sum(), 1))
    missed_on_rate = float(np.logical_and(ref_on, ~pred_on).sum()) / float(max(ref_on.sum(), 1))
    return {
        "led_precision": precision,
        "led_recall": recall,
        "led_f1": f1,
        "led_flicker_rate": flicker_rate,
        "led_false_on_rate": false_on_rate,
        "led_missed_on_rate": missed_on_rate,
    }


def _scheduled_led_metrics(result: RolloutResult, ref: LightPaintRef) -> dict[str, float]:
    times = np.asarray(result.time_arr, dtype=np.float32).reshape(-1)
    pred_on = np.asarray(result.brightness_arr, dtype=np.float32).reshape(-1) > 0.5
    n = int(min(times.size, pred_on.size))
    if n <= 0:
        return {
            "stroke_on_led_recall": 1.0,
            "stroke_on_dropout_run_max_s": 0.0,
            "stroke_on_dropout_run_count_ge_3_frames": 0.0,
            "stroke_on_dropout_samples": 0.0,
            "connector_off_precision": 1.0,
        }
    times = times[:n]
    pred_on = pred_on[:n]
    scheduled_on = np.asarray([float(ref.led(float(t))) > 0.5 for t in times], dtype=bool)
    within_reference = times <= float(ref.duration) + 1e-6
    stroke_on = np.logical_and(scheduled_on, within_reference)
    connector_off = np.logical_and(~scheduled_on, within_reference)
    dropout = np.logical_and(stroke_on, ~pred_on)

    idx = np.flatnonzero(dropout)
    runs: list[tuple[int, int]] = []
    if idx.size:
        start = int(idx[0])
        prev = int(idx[0])
        for raw in idx[1:]:
            current = int(raw)
            if current == prev + 1:
                prev = current
            else:
                runs.append((start, prev))
                start = prev = current
        runs.append((start, prev))

    if times.size > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 0.0
    max_run_frames = max((end - start + 1 for start, end in runs), default=0)
    stroke_on_recall = float(np.logical_and(stroke_on, pred_on).sum()) / float(max(int(stroke_on.sum()), 1))
    connector_off_precision = float(np.logical_and(connector_off, ~pred_on).sum()) / float(max(int(connector_off.sum()), 1))
    return {
        "stroke_on_led_recall": stroke_on_recall,
        "stroke_on_dropout_run_max_s": float(max_run_frames) * dt,
        "stroke_on_dropout_run_count_ge_3_frames": float(sum(1 for start, end in runs if end - start + 1 >= 3)),
        "stroke_on_dropout_samples": float(int(dropout.sum())),
        "connector_off_precision": connector_off_precision,
    }


def _control_health_metrics(result: RolloutResult) -> dict[str, float]:
    action = np.asarray(result.action_arr, dtype=np.float32)
    rpm = np.asarray(result.rpm_arr, dtype=np.float32)
    action_norm = np.linalg.norm(action, axis=1) if action.ndim == 2 and action.size else np.zeros(0, dtype=np.float32)
    action_rate = (
        np.linalg.norm(np.diff(action, axis=0), axis=1)
        if action.ndim == 2 and action.shape[0] > 1
        else np.zeros(0, dtype=np.float32)
    )
    max_rpm_vals = [float(row.get("max_rpm", 0.0)) for row in result.info_rows if float(row.get("max_rpm", 0.0)) > 0.0]
    if max_rpm_vals and rpm.size:
        max_rpm = float(max(max_rpm_vals))
        saturation = float(np.mean(rpm >= max_rpm * 0.99))
    else:
        saturation = float("nan")
    out_of_bounds_rows = [bool(row.get("out_of_bounds", False)) for row in result.info_rows]
    out_of_bounds_rate = float(np.mean(out_of_bounds_rows)) if out_of_bounds_rows else float(result.crash)
    return {
        "action_norm_mean": float(np.mean(action_norm)) if action_norm.size else 0.0,
        "action_norm_max": float(np.max(action_norm)) if action_norm.size else 0.0,
        "action_rate_mean": float(np.mean(action_rate)) if action_rate.size else 0.0,
        "action_rate_max": float(np.max(action_rate)) if action_rate.size else 0.0,
        "rpm_saturation_ratio": saturation,
        "out_of_bounds_rate": out_of_bounds_rate,
        "crash_rate": float(result.crash),
    }


def _metrics(result: RolloutResult, ref: LightPaintRef, corner_window_m: float) -> dict[str, Any]:
    err = np.linalg.norm(result.pos_arr - result.ref_arr, axis=1)
    path_dist = _nearest_path_distances(ref, result.pos_arr)
    speed = _speed_from_positions(result.time_arr, result.pos_arr)
    corner = _corner_mask(ref, result.time_arr, corner_window_m)
    straight = ~corner
    delta_along = _delta_v_along_ref(result.delta_v_arr, result.v_ref_arr)
    component_keys = [
        "r_corner_track",
        "r_corner_path",
        "r_corner_speed",
        "r_corner_decel",
        "r_corner_accel",
        "r_corner_lateral",
        "r_schedule",
        "r_path",
        "r_action_mag",
        "r_action_rate",
        "r_paint_outcome",
        "r_led_prior",
        "r_led_flicker",
        "corner_influence",
    ]
    component_values: dict[str, np.ndarray] = {}
    for key in component_keys:
        vals = [0.0]
        vals.extend(float(row.get(key, 0.0)) for row in result.info_rows)
        if len(vals) < len(result.time_arr):
            vals.extend([0.0] * (len(result.time_arr) - len(vals)))
        component_values[key] = np.asarray(vals[: len(result.time_arr)], dtype=np.float32)
    painting = _painting_quality_metrics(result.cumulative, result.target_mask)
    led = _led_metrics(result)
    scheduled_led = _scheduled_led_metrics(result, ref)
    control = _control_health_metrics(result)
    corner_rows = _corner_errors(ref, result.time_arr, result.pos_arr)
    internal_corner_vals = [
        float(row["error_m"])
        for row in corner_rows
        if 0 < int(row["corner_index"]) < len(ref.waypoints) - 1
    ]

    def mean_or_nan(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else float("nan")

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")

    reward_mean = float(np.mean(result.reward_arr)) if result.reward_arr.size else float("nan")
    corner_mean_speed = mean_or_nan(speed[corner])
    straight_mean_speed = mean_or_nan(speed[straight])
    ref_speed = max(float(ref.speed), 1e-6)
    return {
        "tag": result.tag,
        "phase": result.phase,
        "wind_mode": result.wind_mode,
        "n_steps": int(max(len(result.time_arr) - 1, 0)),
        "walltime_s": f"{result.walltime_s:.4f}",
        "tracking_rmse_m": f"{rmse(err):.9f}",
        "tracking_mean_m": f"{mean_or_nan(err):.9f}",
        "tracking_max_m": f"{float(np.max(err)):.9f}" if err.size else "nan",
        "tracking_final_m": f"{float(err[-1]):.9f}" if err.size else "nan",
        "path_rmse_m": f"{rmse(path_dist):.9f}",
        "corner_window_rmse_m": f"{rmse(err[corner]):.9f}",
        "straight_rmse_m": f"{rmse(err[straight]):.9f}",
        "corner_path_rmse_m": f"{rmse(path_dist[corner]):.9f}",
        "corner_overshoot_m": f"{float(np.max(path_dist[corner])):.9f}" if path_dist[corner].size else "nan",
        "straight_path_rmse_m": f"{rmse(path_dist[straight]):.9f}",
        "path_max_m": f"{float(np.max(path_dist)):.9f}" if path_dist.size else "nan",
        "corner_mean_speed_mps": f"{corner_mean_speed:.6f}",
        "straight_mean_speed_mps": f"{straight_mean_speed:.6f}",
        "corner_speed_ratio": f"{(corner_mean_speed / ref_speed):.6f}",
        "corner_speed_vs_straight_ratio": f"{(corner_mean_speed / max(straight_mean_speed, 1e-6)):.6f}",
        "corner_delta_v_along_ref_mps": f"{mean_or_nan(delta_along[corner]):.6f}",
        "straight_delta_v_along_ref_mps": f"{mean_or_nan(delta_along[straight]):.6f}",
        "corner_action_norm": f"{mean_or_nan(np.linalg.norm(result.delta_v_arr[corner], axis=1)):.6f}",
        "straight_action_norm": f"{mean_or_nan(np.linalg.norm(result.delta_v_arr[straight], axis=1)):.6f}",
        "corner_mean_r_corner_track": f"{mean_or_nan(component_values['r_corner_track'][corner]):.6f}",
        "corner_mean_r_corner_path": f"{mean_or_nan(component_values['r_corner_path'][corner]):.6f}",
        "corner_mean_r_corner_speed": f"{mean_or_nan(component_values['r_corner_speed'][corner]):.6f}",
        "corner_mean_r_corner_decel": f"{mean_or_nan(component_values['r_corner_decel'][corner]):.6f}",
        "corner_mean_r_corner_accel": f"{mean_or_nan(component_values['r_corner_accel'][corner]):.6f}",
        "corner_mean_r_corner_lateral": f"{mean_or_nan(component_values['r_corner_lateral'][corner]):.6f}",
        "straight_mean_r_corner_speed": f"{mean_or_nan(component_values['r_corner_speed'][straight]):.6f}",
        "mean_r_path": f"{mean_or_nan(component_values['r_path']):.6f}",
        "mean_r_schedule": f"{mean_or_nan(component_values['r_schedule']):.6f}",
        "corner_mean_r_path": f"{mean_or_nan(component_values['r_path'][corner]):.6f}",
        "corner_mean_r_schedule": f"{mean_or_nan(component_values['r_schedule'][corner]):.6f}",
        "straight_mean_r_path": f"{mean_or_nan(component_values['r_path'][straight]):.6f}",
        "straight_mean_r_schedule": f"{mean_or_nan(component_values['r_schedule'][straight]):.6f}",
        "mean_r_action_mag": f"{mean_or_nan(component_values['r_action_mag']):.6f}",
        "mean_r_action_rate": f"{mean_or_nan(component_values['r_action_rate']):.6f}",
        "mean_r_paint_outcome": f"{mean_or_nan(component_values['r_paint_outcome']):.6f}",
        "mean_r_led_prior": f"{mean_or_nan(component_values['r_led_prior']):.6f}",
        "mean_r_led_flicker": f"{mean_or_nan(component_values['r_led_flicker']):.6f}",
        "corner_mean_influence": f"{mean_or_nan(component_values['corner_influence'][corner]):.6f}",
        "corner_endpoint_error_mean_m": f"{float(np.mean(internal_corner_vals)):.9f}" if internal_corner_vals else "nan",
        "corner_endpoint_error_max_m": f"{float(np.max(internal_corner_vals)):.9f}" if internal_corner_vals else "nan",
        "painted_pixel_coverage": f"{painting['painted_pixel_coverage']:.6f}",
        "painted_pixel_recall": f"{painting['painted_pixel_recall']:.6f}",
        "painted_pixel_precision": f"{painting['painted_pixel_precision']:.6f}",
        "painted_pixel_iou": f"{painting['painted_pixel_iou']:.6f}",
        "painted_pixel_dice": f"{painting['painted_pixel_dice']:.6f}",
        "off_target_pixel_ratio": f"{painting['off_target_pixel_ratio']:.6f}",
        "painted_target_px": f"{painting['painted_target_px']:.0f}",
        "painted_off_target_px": f"{painting['painted_off_target_px']:.0f}",
        "painted_total_px": f"{painting['painted_total_px']:.0f}",
        "target_total_px": f"{painting['target_total_px']:.0f}",
        "led_precision": f"{led['led_precision']:.6f}",
        "led_recall": f"{led['led_recall']:.6f}",
        "led_f1": f"{led['led_f1']:.6f}",
        "led_flicker_rate": f"{led['led_flicker_rate']:.6f}",
        "led_false_on_rate": f"{led['led_false_on_rate']:.6f}",
        "led_missed_on_rate": f"{led['led_missed_on_rate']:.6f}",
        "stroke_on_led_recall": f"{scheduled_led['stroke_on_led_recall']:.6f}",
        "stroke_on_dropout_run_max_s": f"{scheduled_led['stroke_on_dropout_run_max_s']:.6f}",
        "stroke_on_dropout_run_count_ge_3_frames": f"{scheduled_led['stroke_on_dropout_run_count_ge_3_frames']:.0f}",
        "stroke_on_dropout_samples": f"{scheduled_led['stroke_on_dropout_samples']:.0f}",
        "connector_off_precision": f"{scheduled_led['connector_off_precision']:.6f}",
        "action_norm_mean": f"{control['action_norm_mean']:.6f}",
        "action_norm_max": f"{control['action_norm_max']:.6f}",
        "action_rate_mean": f"{control['action_rate_mean']:.6f}",
        "action_rate_max": f"{control['action_rate_max']:.6f}",
        "rpm_saturation_ratio": f"{control['rpm_saturation_ratio']:.6f}",
        "out_of_bounds_rate": f"{control['out_of_bounds_rate']:.6f}",
        "crash_rate": f"{control['crash_rate']:.6f}",
        "reward_mean": f"{reward_mean:.6f}",
        "reward_sum": f"{float(np.sum(result.reward_arr)):.6f}",
        "crash": int(result.crash),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite_metric(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _lte(value: float, limit: float) -> bool:
    return bool(math.isfinite(value) and math.isfinite(limit) and value <= limit)


def _gte(value: float, limit: float) -> bool:
    return bool(math.isfinite(value) and math.isfinite(limit) and value >= limit)


def _between(value: float, low: float, high: float) -> bool:
    return bool(math.isfinite(value) and math.isfinite(low) and math.isfinite(high) and low <= value <= high)


def _phase_b_success_criteria(metrics_rows: list[dict[str, Any]]) -> dict[str, bool]:
    zero = next(row for row in metrics_rows if row["tag"] == "phaseB_zero")
    trained = next(row for row in metrics_rows if row["tag"] == "phaseB_trained")
    zero_corner = _finite_metric(zero, "corner_path_rmse_m")
    trained_corner = _finite_metric(trained, "corner_path_rmse_m")
    zero_straight = _finite_metric(zero, "straight_path_rmse_m")
    trained_straight = _finite_metric(trained, "straight_path_rmse_m")
    zero_corner_speed = _finite_metric(zero, "corner_mean_speed_mps")
    trained_corner_speed = _finite_metric(trained, "corner_mean_speed_mps")
    zero_coverage = _finite_metric(zero, "painted_pixel_coverage")
    trained_coverage = _finite_metric(trained, "painted_pixel_coverage")
    zero_precision = _finite_metric(zero, "painted_pixel_precision")
    trained_precision = _finite_metric(trained, "painted_pixel_precision")
    zero_iou = _finite_metric(zero, "painted_pixel_iou")
    trained_iou = _finite_metric(trained, "painted_pixel_iou")
    zero_off_target = _finite_metric(zero, "off_target_pixel_ratio")
    trained_off_target = _finite_metric(trained, "off_target_pixel_ratio")
    success = {
        "corner_path_rmse_reduced_20pct": _lte(trained_corner, zero_corner * 0.80),
        "corner_speed_reduced_5_to_20pct": _between(
            trained_corner_speed,
            zero_corner_speed * 0.80,
            zero_corner_speed * 0.95,
        ),
        "straight_path_rmse_not_worse_10pct": _lte(trained_straight, zero_straight * 1.10),
        "painting_coverage_kept_90pct": _gte(trained_coverage, zero_coverage * 0.90),
        "painting_precision_kept_90pct": _gte(trained_precision, zero_precision * 0.90),
        "painting_iou_kept_90pct": _gte(trained_iou, zero_iou * 0.90),
        "off_target_ratio_not_worse_5pct": _lte(trained_off_target, zero_off_target + 0.05),
    }
    success["overall_pass"] = bool(all(success.values()))
    return success


def _final_goal_criteria(metrics_rows: list[dict[str, Any]]) -> dict[str, bool]:
    zero = next(row for row in metrics_rows if row["tag"] == "phaseB_zero")
    trained = next(row for row in metrics_rows if row["tag"] == "phaseB_trained")
    t = FINAL_GOAL_THRESHOLDS
    zero_straight = _finite_metric(zero, "straight_path_rmse_m")
    trained_straight = _finite_metric(trained, "straight_path_rmse_m")
    zero_corner = _finite_metric(zero, "corner_path_rmse_m")
    trained_corner = _finite_metric(trained, "corner_path_rmse_m")
    zero_overshoot = _finite_metric(zero, "corner_overshoot_m")
    trained_overshoot = _finite_metric(trained, "corner_overshoot_m")
    zero_flicker = _finite_metric(zero, "led_flicker_rate")
    trained_flicker = _finite_metric(trained, "led_flicker_rate")
    criteria = {
        "painting_iou_abs": _gte(_finite_metric(trained, "painted_pixel_iou"), t["painting_iou_min"]),
        "painting_precision_abs": _gte(_finite_metric(trained, "painted_pixel_precision"), t["painting_precision_min"]),
        "painting_recall_abs": _gte(_finite_metric(trained, "painted_pixel_recall"), t["painting_recall_min"]),
        "painting_dice_abs": _gte(_finite_metric(trained, "painted_pixel_dice"), t["painting_dice_min"]),
        "off_target_ratio_abs": _lte(_finite_metric(trained, "off_target_pixel_ratio"), t["off_target_ratio_max"]),
        "path_rmse_abs": _lte(_finite_metric(trained, "path_rmse_m"), t["path_rmse_m_max"]),
        "path_max_abs": _lte(_finite_metric(trained, "path_max_m"), t["path_max_m_max"]),
        "straight_path_rmse_relative": _lte(trained_straight, zero_straight * t["straight_path_rmse_rel_max"]),
        "corner_path_rmse_relative": _lte(trained_corner, zero_corner * t["corner_path_rmse_rel_max"]),
        "corner_speed_ratio_abs": _between(
            _finite_metric(trained, "corner_speed_ratio"),
            t["corner_speed_ratio_min"],
            t["corner_speed_ratio_max"],
        ),
        "corner_overshoot_relative": _lte(trained_overshoot, zero_overshoot * t["corner_overshoot_rel_max"]),
        "led_precision_abs": _gte(_finite_metric(trained, "led_precision"), t["led_precision_min"]),
        "led_recall_abs": _gte(_finite_metric(trained, "led_recall"), t["led_recall_min"]),
        "led_flicker_not_worse": _lte(trained_flicker, zero_flicker + t["led_flicker_rate_abs_tolerance"]),
        "crash_free": _lte(_finite_metric(trained, "crash_rate"), 0.0),
        "out_of_bounds_free": _lte(_finite_metric(trained, "out_of_bounds_rate"), 0.0),
        "rpm_saturation_ratio_abs": _lte(
            _finite_metric(trained, "rpm_saturation_ratio"),
            t["rpm_saturation_ratio_max"],
        ),
        "action_norm_bounded": _lte(_finite_metric(trained, "action_norm_max"), t["action_norm_max"]),
        "action_rate_bounded": _lte(_finite_metric(trained, "action_rate_max"), t["action_rate_max"]),
    }
    criteria["final_goal_pass"] = bool(all(criteria.values()))
    return criteria


def _write_diagnostics_csv(path: Path, result: RolloutResult, ref: LightPaintRef, corner_window_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    err = np.linalg.norm(result.pos_arr - result.ref_arr, axis=1)
    path_dist = _nearest_path_distances(ref, result.pos_arr)
    speed = _speed_from_positions(result.time_arr, result.pos_arr)
    corner = _corner_mask(ref, result.time_arr, corner_window_m)
    delta_along = _delta_v_along_ref(result.delta_v_arr, result.v_ref_arr)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "t_s", "tracking_err_m", "path_dist_m", "actual_speed_mps", "is_corner_window",
            "delta_v_x", "delta_v_y", "delta_v_z", "delta_v_along_ref_mps",
            "action_0", "action_1", "action_2", "action_led",
            "brightness", "reward",
            "r_corner_track", "r_corner_path", "r_corner_speed", "r_schedule", "r_path",
            "r_corner_decel", "r_corner_accel", "r_corner_lateral",
            "r_paint_outcome", "r_led_prior", "r_led_flicker",
            "r_action_mag", "r_action_rate", "corner_influence", "corner_dist_m",
        ])
        for i in range(len(result.time_arr)):
            reward = result.reward_arr[i - 1] if i > 0 and i - 1 < len(result.reward_arr) else 0.0
            info = result.info_rows[i - 1] if i > 0 and i - 1 < len(result.info_rows) else {}
            writer.writerow([
                i,
                f"{float(result.time_arr[i]):.6f}",
                f"{float(err[i]):.6f}",
                f"{float(path_dist[i]):.6f}",
                f"{float(speed[i]):.6f}",
                int(bool(corner[i])),
                f"{float(result.delta_v_arr[i, 0]):.6f}",
                f"{float(result.delta_v_arr[i, 1]):.6f}",
                f"{float(result.delta_v_arr[i, 2]):.6f}",
                f"{float(delta_along[i]):.6f}",
                f"{float(result.action_arr[i, 0]):.6f}",
                f"{float(result.action_arr[i, 1]):.6f}",
                f"{float(result.action_arr[i, 2]):.6f}",
                f"{float(result.action_arr[i, 3]):.6f}",
                f"{float(result.brightness_arr[i]):.6f}",
                f"{float(reward):.6f}",
                f"{float(info.get('r_corner_track', 0.0)):.6f}",
                f"{float(info.get('r_corner_path', 0.0)):.6f}",
                f"{float(info.get('r_corner_speed', 0.0)):.6f}",
                f"{float(info.get('r_schedule', 0.0)):.6f}",
                f"{float(info.get('r_path', 0.0)):.6f}",
                f"{float(info.get('r_corner_decel', 0.0)):.6f}",
                f"{float(info.get('r_corner_accel', 0.0)):.6f}",
                f"{float(info.get('r_corner_lateral', 0.0)):.6f}",
                f"{float(info.get('r_paint_outcome', 0.0)):.6f}",
                f"{float(info.get('r_led_prior', 0.0)):.6f}",
                f"{float(info.get('r_led_flicker', 0.0)):.6f}",
                f"{float(info.get('r_action_mag', 0.0)):.6f}",
                f"{float(info.get('r_action_rate', 0.0)):.6f}",
                f"{float(info.get('corner_influence', 0.0)):.6f}",
                f"{float(info.get('corner_dist_m', 0.0)):.6f}",
            ])


def _save_rollout_artifacts(
    result: RolloutResult,
    ref: LightPaintRef,
    artifacts_dir: Path,
    viz_dir: Path,
    corner_window_m: float,
    reference_token: str,
    wind_mode: str,
) -> dict[str, str]:
    token = _safe_token(result.tag)
    prefix = f"phase_{result.phase.lower()}_{reference_token}_{wind_mode}_{token}"
    traj_path = artifacts_dir / f"{prefix}_trajectory.csv"
    corner_path = artifacts_dir / f"{prefix}_corners.csv"
    diag_path = artifacts_dir / f"{prefix}_corner_diagnostics.csv"
    plot_path = viz_dir / f"{prefix}_tracking_xz.png"
    _save_trajectory_csv(
        traj_path,
        result.time_arr,
        result.ref_arr,
        result.pos_arr,
        result.brightness_arr,
        result.rpm_arr,
    )
    corner_rows = _corner_errors(ref, result.time_arr, result.pos_arr)
    _save_corner_csv(corner_path, corner_rows)
    _write_diagnostics_csv(diag_path, result, ref, corner_window_m)
    _save_tracking_plot(
        plot_path,
        ref_arr=result.ref_arr,
        pos_arr=result.pos_arr,
        corner_rows=corner_rows,
        title=f"Phase {result.phase} {reference_token} {wind_mode} {result.tag}",
    )
    return {
        "trajectory_csv": str(traj_path),
        "corners_csv": str(corner_path),
        "corner_diagnostics_csv": str(diag_path),
        "tracking_png": str(plot_path),
    }


def _make_model(args: argparse.Namespace, vec_env: Any):
    from stable_baselines3 import PPO
    from src.train.extractor import LightPaintExtractor

    policy_kwargs = {
        "features_extractor_class": LightPaintExtractor,
        "normalize_images": False,
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "log_std_init": float(args.log_std_init),
    }
    return PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        seed=int(args.seed),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        n_epochs=int(args.n_epochs),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        clip_range=float(args.clip_range),
        verbose=int(args.verbose),
        device=str(args.device),
    )



def _load_or_make_model(args: argparse.Namespace, vec_env: Any) -> tuple[Any, str | None]:
    load_model = getattr(args, "load_model", None)
    if load_model:
        from stable_baselines3 import PPO

        model_path = Path(load_model)
        if not model_path.exists():
            raise FileNotFoundError(f"--load-model path does not exist: {model_path}")
        print(f"[phase_b] loading existing model: {model_path}", flush=True)
        model = PPO.load(
            str(model_path),
            env=vec_env,
            device=str(args.device),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            learning_rate=float(args.learning_rate),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            ent_coef=float(args.ent_coef),
            clip_range=float(args.clip_range),
            verbose=int(args.verbose),
        )
        try:
            model.set_random_seed(int(args.seed))
        except Exception:
            pass
        return model, str(model_path)
    return _make_model(args, vec_env), None


def _load_saved_model_for_rollout(model_path: Path, vec_env: Any, device: str) -> Any:
    from stable_baselines3 import PPO

    return PPO.load(str(model_path), env=vec_env, device=str(device))


def _collect_bc_dataset(
    *,
    action_fn: Callable[[dict[str, Any]], np.ndarray],
    episodes: int,
    label: str,
    wind_mode: str,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
    gui: bool,
    ctrl_freq: int,
    pyb_freq: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    obs_rows: dict[str, list[np.ndarray]] = {
        "drone_state": [],
        "future_ref": [],
        "target_mask": [],
        "progress_mask": [],
    }
    action_rows: list[np.ndarray] = []
    for ep in range(int(episodes)):
        env = _make_env(
            phase="B",
            label=label,
            wind_mode=wind_mode,
            reference=reference,
            max_steps=max_steps,
            seed=seed + ep,
            init_box_size=init_box_size,
            led_always_on=led_always_on,
            gui=gui,
            ctrl_freq=ctrl_freq,
            pyb_freq=pyb_freq,
        )
        obs, info = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            action = np.asarray(action_fn({"obs": obs, "info": info}), dtype=np.float32).reshape(4)
            for key in obs_rows:
                obs_rows[key].append(np.asarray(obs[key], dtype=np.float32).copy())
            action_rows.append(action.copy())
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        env.close()
    obs_batch = {key: np.stack(vals).astype(np.float32) for key, vals in obs_rows.items()}
    action_batch = np.stack(action_rows).astype(np.float32)
    return obs_batch, action_batch


def _behavior_clone_policy(
    model: Any,
    obs_batch: dict[str, np.ndarray],
    action_batch: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    nonzero_weight: float,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    import torch

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(learning_rate))
    model.policy.set_training_mode(True)
    n = int(action_batch.shape[0])
    target_delta_norm = np.linalg.norm(action_batch[:, :3], axis=1)
    nonzero_mask = target_delta_norm > 1e-6
    sample_weights = np.ones(n, dtype=np.float32)
    sample_weights[nonzero_mask] = max(float(nonzero_weight), 1.0)
    losses: list[float] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for epoch in range(int(epochs)):
            order = rng.permutation(n)
            epoch_losses: list[float] = []
            for start in range(0, n, int(batch_size)):
                idx = order[start:start + int(batch_size)]
                obs_tensor = {
                    key: torch.as_tensor(value[idx], device=model.device)
                    for key, value in obs_batch.items()
                }
                target = torch.as_tensor(action_batch[idx], device=model.device)
                weight = torch.as_tensor(sample_weights[idx], device=model.device).view(-1, 1)
                dist = model.policy.get_distribution(obs_tensor)
                pred = dist.distribution.mean
                per_sample_loss = torch.mean((pred - target) ** 2, dim=1, keepdim=True)
                loss = torch.sum(per_sample_loss * weight) / torch.clamp(torch.sum(weight), min=1.0)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            losses.append(mean_loss)
            writer.writerow([epoch, f"{mean_loss:.8f}"])
    return {
        "n_samples": n,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "nonzero_weight": float(nonzero_weight),
        "nonzero_samples": int(np.sum(nonzero_mask)),
        "nonzero_fraction": float(np.mean(nonzero_mask)) if n else 0.0,
        "target_delta_norm_mean": float(np.mean(target_delta_norm)) if n else 0.0,
        "target_delta_norm_nonzero_mean": float(np.mean(target_delta_norm[nonzero_mask])) if np.any(nonzero_mask) else 0.0,
        "initial_loss": float(losses[0]) if losses else float("nan"),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "loss_csv": str(out_path),
    }



def _validate_config(args: argparse.Namespace, max_steps: int) -> None:
    def _finite_float(name: str) -> float:
        value = float(getattr(args, name.replace("-", "_")))
        if not math.isfinite(value):
            raise ValueError(f"--{name} must be finite.")
        return value

    if int(max_steps) <= 0:
        raise ValueError("--max-steps must be positive after reference duration calculation.")
    if _finite_float("settle-time") < 0.0:
        raise ValueError("--settle-time must be >= 0.")
    if _finite_float("init-box-size") < 0.0:
        raise ValueError("--init-box-size must be >= 0.")
    if int(args.ctrl_freq) <= 0:
        raise ValueError("--ctrl-freq must be > 0.")
    if int(args.pyb_freq) < int(args.ctrl_freq):
        raise ValueError("--pyb-freq must be >= --ctrl-freq.")
    if int(args.pyb_freq) % int(args.ctrl_freq) != 0:
        raise ValueError("--pyb-freq must be an integer multiple of --ctrl-freq.")
    if int(args.total_timesteps) < 0:
        raise ValueError("--total-timesteps must be >= 0.")
    if bool(args.eval_only) and not args.load_model:
        raise ValueError("--eval-only requires --load-model.")
    if int(args.n_steps) <= 0:
        raise ValueError("--n-steps must be > 0.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0.")
    if int(args.batch_size) > int(args.n_steps):
        raise ValueError("--batch-size must be <= --n-steps for single-env PPO.")
    if int(args.n_epochs) <= 0:
        raise ValueError("--n-epochs must be > 0.")
    if int(args.bc_epochs) < 0:
        raise ValueError("--bc-epochs must be >= 0.")
    if int(args.bc_epochs) > 0 and int(args.bc_episodes) <= 0:
        raise ValueError("--bc-episodes must be > 0 when --bc-epochs is > 0.")
    if int(args.bc_batch_size) <= 0:
        raise ValueError("--bc-batch-size must be > 0.")
    if _finite_float("speed") <= 0.0:
        raise ValueError("--speed must be > 0.")
    if _finite_float("corner-window-m") <= 0.0:
        raise ValueError("--corner-window-m must be > 0.")
    if _finite_float("gui-hold-seconds") < 0.0:
        raise ValueError("--gui-hold-seconds must be >= 0.")
    if _finite_float("learning-rate") <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if not (0.0 <= _finite_float("gamma") <= 1.0):
        raise ValueError("--gamma must be in [0, 1].")
    if not (0.0 <= _finite_float("gae-lambda") <= 1.0):
        raise ValueError("--gae-lambda must be in [0, 1].")
    if _finite_float("ent-coef") < 0.0:
        raise ValueError("--ent-coef must be >= 0.")
    if _finite_float("clip-range") <= 0.0:
        raise ValueError("--clip-range must be > 0.")
    _finite_float("log-std-init")
    if _finite_float("teacher-gain") < 0.0:
        raise ValueError("--teacher-gain must be >= 0.")
    if _finite_float("teacher-window-m") <= 0.0:
        raise ValueError("--teacher-window-m must be > 0.")
    if _finite_float("bc-learning-rate") <= 0.0:
        raise ValueError("--bc-learning-rate must be > 0.")
    if _finite_float("bc-nonzero-weight") < 0.0:
        raise ValueError("--bc-nonzero-weight must be >= 0.")



def _check_runtime_dependencies() -> None:
    required = {
        "pybullet": "pybullet",
        "pybullet_data": "pybullet_data",
        "gym_pybullet_drones": "gym-pybullet-drones",
        "stable_baselines3": "stable-baselines3",
    }
    missing = [pkg_name for module, pkg_name in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        install_hint = (
            "Missing packages required for PyBullet training: "
            + ", ".join(missing)
            + ". Install project dependencies with `pip install -r requirements.txt`."
        )
        raise DependencyError(install_hint)


def _check_requested_device(device: str) -> None:
    requested = str(device).strip().lower()
    if not requested.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise DependencyError(
            f"--device {device} requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch wheel or pass --device cpu explicitly."
        )
    if ":" in requested:
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device specifier: {device!r}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise DependencyError(
                f"--device {device} requested, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )


def main(args: argparse.Namespace) -> int:
    _check_runtime_dependencies()
    _check_requested_device(str(args.device))
    from src.env import reward_config as reward_cfg
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    built = build_reference_from_args(args)
    ref = built.reference
    reference_label = built.label
    reference_token = built.token
    wind_mode = str(args.wind_mode)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(
        ceil((ref.duration + float(args.settle_time)) * float(args.ctrl_freq))
    )
    _validate_config(args, max_steps)
    requested_total_timesteps = int(args.total_timesteps)
    effective_total_timesteps = 0 if bool(args.eval_only) else requested_total_timesteps
    out_root = Path(args.output_dir)
    artifacts_dir = out_root / "artifacts"
    viz_dir = artifacts_dir / "visualization"
    model_dir = out_root / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[phase_b] trajectory={args.trajectory} label={reference_label!r} wind={wind_mode} "
        f"length={ref.length:.3f}m duration={ref.duration:.3f}s speed={args.speed} "
        f"max_steps={max_steps} train_steps={effective_total_timesteps} gui={bool(args.gui)}",
        flush=True,
    )
    if bool(args.gui):
        print(
            "[phase_b] PyBullet GUI is used for rollout inspection. "
            "Training environments still run in DIRECT mode.",
            flush=True,
        )

    common = {
        "label": reference_label,
        "wind_mode": wind_mode,
        "reference": ref,
        "max_steps": max_steps,
        "seed": int(args.seed),
        "init_box_size": float(args.init_box_size),
        "led_always_on": bool(args.led_always_on),
        "gui": bool(args.gui),
        "gui_hold_seconds": float(args.gui_hold_seconds),
        "ctrl_freq": int(args.ctrl_freq),
        "pyb_freq": int(args.pyb_freq),
    }
    phase_a: RolloutResult | None = None
    if wind_mode == "M0":
        phase_a = _rollout(tag="phaseA_pid", phase="A", action_fn=_phase_a_action, **common)
    else:
        print(
            f"[phase_b] skipping Phase A PID baseline because it is defined only for M0, got {wind_mode}",
            flush=True,
        )
    phase_b_zero = _rollout(tag="phaseB_zero", phase="B", action_fn=_phase_b_zero_action, **common)
    teacher_fn = _make_corner_decel_teacher(
        ref,
        gain_mps=float(args.teacher_gain),
        window_m=float(args.teacher_window_m),
    )
    phase_b_teacher = _rollout(tag="phaseB_teacher", phase="B", action_fn=teacher_fn, **common)

    direct_common = dict(common)
    direct_common["gui"] = False
    direct_common.pop("gui_hold_seconds", None)

    def _train_env() -> Monitor:
        env = _make_env(phase="B", **direct_common)
        env = _wrap_train_action_filter(
            env,
            ref,
            float(args.teacher_window_m),
            str(args.trained_action_filter),
        )
        return Monitor(env)

    vec_env = DummyVecEnv([_train_env])
    if bool(args.eval_only) and not args.load_model:
        raise ValueError("--eval-only requires --load-model.")
    model, source_model_path = _load_or_make_model(args, vec_env)

    before_tag = "phaseB_loaded" if source_model_path else "phaseB_untrained"
    phase_b_before = _rollout(
        tag=before_tag,
        phase="B",
        action_fn=lambda ctx: model.predict(ctx["obs"], deterministic=True)[0],
        **common,
    )

    train_start = time.perf_counter()
    bc_summary: dict[str, Any] | None = None
    if bool(args.eval_only):
        print("[phase_b] eval-only run: skipping BC and PPO training", flush=True)
    elif int(args.bc_epochs) > 0:
        print("[phase_b] collecting BC warm-start data", flush=True)
        obs_batch, action_batch = _collect_bc_dataset(
            action_fn=teacher_fn,
            episodes=int(args.bc_episodes),
            **direct_common,
        )
        bc_summary = _behavior_clone_policy(
            model,
            obs_batch,
            action_batch,
            epochs=int(args.bc_epochs),
            batch_size=int(args.bc_batch_size),
            learning_rate=float(args.bc_learning_rate),
            nonzero_weight=float(args.bc_nonzero_weight),
            seed=int(args.seed),
            out_path=artifacts_dir / f"phase_b_{reference_token}_{wind_mode}_bc_loss.csv",
        )
        print(
            f"[phase_b] BC complete: samples={bc_summary['n_samples']} "
            f"nonzero={bc_summary['nonzero_samples']} "
            f"loss={bc_summary['initial_loss']:.6f}->{bc_summary['final_loss']:.6f}",
            flush=True,
        )
    if (not bool(args.eval_only)) and effective_total_timesteps > 0:
        print("[phase_b] starting PPO training", flush=True)
        reset_steps = bool(args.reset_num_timesteps) if source_model_path else True
        progress_total_timesteps = int(effective_total_timesteps)
        if not reset_steps:
            progress_total_timesteps += int(getattr(model, "num_timesteps", 0))
        model.learn(
            total_timesteps=effective_total_timesteps,
            callback=_make_progress_callback(progress_total_timesteps),
            progress_bar=False,
            reset_num_timesteps=reset_steps,
        )
        print("[phase_b] PPO training complete", flush=True)
    train_walltime = time.perf_counter() - train_start
    print(f"[phase_b] training stage complete: walltime={train_walltime:.2f}s", flush=True)

    model_path = model_dir / f"ppo_phaseB_{reference_token}_{wind_mode}_corner.zip"
    model.save(str(model_path))
    model = _load_saved_model_for_rollout(model_path, vec_env, str(args.device))

    def _trained_action(ctx: dict[str, Any]) -> np.ndarray:
        action = np.asarray(model.predict(ctx["obs"], deterministic=True)[0], dtype=np.float32).reshape(4)
        if args.trained_action_filter == "corner_tangent_decel":
            return _filter_corner_tangent_decel_action(
                action,
                ctx["info"],
                ref,
                float(args.teacher_window_m),
            )
        return action

    phase_b_after = _rollout(
        tag="phaseB_trained",
        phase="B",
        action_fn=_trained_action,
        **common,
    )
    vec_env.close()

    results = [phase_b_zero, phase_b_teacher, phase_b_before, phase_b_after]
    if phase_a is not None:
        results.insert(0, phase_a)
    metrics_rows = [_metrics(r, ref, float(args.corner_window_m)) for r in results]
    metrics_path = artifacts_dir / f"phase_b_{reference_token}_{wind_mode}_corner_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)

    artifact_index: dict[str, Any] = {
        "model_path": str(model_path),
        "source_model_path": source_model_path,
        "metrics_csv": str(metrics_path),
        "rollouts": {},
    }
    for result in results:
        artifact_index["rollouts"][result.tag] = _save_rollout_artifacts(
            result,
            ref,
            artifacts_dir,
            viz_dir,
            float(args.corner_window_m),
            reference_token,
            wind_mode,
        )

    if bool(args.save_video):
        viz_paths = save_phase_a_visualization(
            pos_list=phase_b_after.pos_arr.tolist(),
            brightness_list=phase_b_after.brightness_arr.tolist(),
            target_mask=phase_b_after.target_mask,
            label=f"{reference_token}_phaseB_trained",
            wind_mode=wind_mode,
            out_dir=viz_dir,
            fps=10,
            tracking_errs=np.linalg.norm(phase_b_after.pos_arr - phase_b_after.ref_arr, axis=1).tolist(),
            cumulative=phase_b_after.cumulative.copy(),
            ref_waypoints=ref.waypoints.copy(),
            ref_cumlen=ref.cumlen.copy(),
            ref_segment_led=ref.segment_led.copy(),
            ref_pos_list=phase_b_after.ref_arr.tolist(),
            snapshots=[],
            phase_prefix="phase_b",
            phase_display_name="Phase B",
            reference_speed_mps=float(ref.speed),
            ctrl_dt_s=1.0 / float(args.ctrl_freq),
        )
        artifact_index["phaseB_trained_video"] = viz_paths

    success = _phase_b_success_criteria(metrics_rows)
    final_goal = _final_goal_criteria(metrics_rows)
    summary = {
        "config": {
            "phase": "B",
            "wind_mode": wind_mode,
            "reference": built.metadata,
            "seed": int(args.seed),
            "load_model": source_model_path,
            "eval_only": bool(args.eval_only),
            "reset_num_timesteps": bool(args.reset_num_timesteps),
            "speed": float(args.speed),
            "max_steps": max_steps,
            "max_steps_override": None if args.max_steps is None else int(args.max_steps),
            "led_always_on": bool(args.led_always_on),
            "gui": bool(args.gui),
            "gui_hold_seconds": float(args.gui_hold_seconds),
            "ctrl_freq": int(args.ctrl_freq),
            "pyb_freq": int(args.pyb_freq),
            "total_timesteps": effective_total_timesteps,
            "requested_total_timesteps": requested_total_timesteps,
            "corner_window_m": float(args.corner_window_m),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "ent_coef": float(args.ent_coef),
            "clip_range": float(args.clip_range),
            "log_std_init": float(args.log_std_init),
            "teacher_gain": float(args.teacher_gain),
            "teacher_window_m": float(args.teacher_window_m),
            "bc_epochs": int(args.bc_epochs),
            "bc_episodes": int(args.bc_episodes),
            "bc_batch_size": int(args.bc_batch_size),
            "bc_learning_rate": float(args.bc_learning_rate),
            "bc_nonzero_weight": float(args.bc_nonzero_weight),
            "trained_action_filter": str(args.trained_action_filter),
            "device": str(args.device),
            "save_video": bool(args.save_video),
            "reward_config_path": reward_cfg.ACTIVE_REWARD_CONFIG_PATH,
            "reward_overrides": reward_cfg.ACTIVE_REWARD_OVERRIDES,
        },
        "evaluation": {
            "schema_version": 3,
            "painting_threshold": PAINTING_THRESHOLD,
            "painting_metric_keys": PAINTING_METRIC_KEYS,
            "final_goal_spec": "final_goal_thresholds_v1",
            "final_goal_thresholds": FINAL_GOAL_THRESHOLDS,
        },
        "runtime": _runtime_metadata(),
        "train_walltime_s": train_walltime,
        "bc": bc_summary,
        "success_criteria": success,
        "final_goal_criteria": final_goal,
        "metrics": metrics_rows,
        "artifacts": artifact_index,
    }
    summary_path = out_root / "summary.json"
    summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_path.write_text(summary_json, encoding="utf-8")
    print(summary_json, flush=True)
    print(f"[phase_b] summary saved to: {summary_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a Phase B residual policy on square, letter, or drawn references.")
    parser.add_argument("--output-dir", default=str(_PKG_ROOT / "artifacts" / "phase_b_m0_corner"))
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--load-model", type=str, default=None,
                        help="Existing Stable-Baselines3 PPO .zip to evaluate or continue training from.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load a model and run evaluation rollouts without BC or PPO updates.")
    parser.add_argument("--reset-num-timesteps", action="store_true",
                        help="When continuing from --load-model, reset the SB3 timestep counter before learning.")
    add_reference_args(parser)
    parser.add_argument("--wind-mode", choices=("M0", "M1", "M2"), default="M0")
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--corner-window-m", type=float, default=0.18)
    parser.add_argument("--init-box-size", type=float, default=0.0)
    parser.add_argument(
        "--led-always-on",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force the LED reference to 1.0. Default keeps the scripted reference from the path.",
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open a PyBullet GUI for rollout inspection. Training remains slower when enabled.",
    )
    parser.add_argument(
        "--gui-hold-seconds",
        type=float,
        default=3.0,
        help="Seconds to keep the rollout GUI window open before closing.",
    )
    parser.add_argument("--ctrl-freq", type=int, default=30)
    parser.add_argument("--pyb-freq", type=int, default=240)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--teacher-gain", type=float, default=0.10)
    parser.add_argument("--teacher-window-m", type=float, default=0.15)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-episodes", type=int, default=4)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-nonzero-weight", type=float, default=1.0)
    parser.add_argument("--trained-action-filter", choices=["none", "corner_tangent_decel"], default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except DependencyError as exc:
        print(f"[phase_b] dependency error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (RuntimeError, ValueError) as exc:
        print(f"[phase_b] configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
