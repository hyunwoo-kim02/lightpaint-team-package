"""Phase B M0 PPO pilot focused on square-corner slowing.

This is a short, reproducible training/evaluation entrypoint.  The success
target is not raw reward increase; it is whether the residual policy reduces
corner-window tracking error while preserving straight-segment tracking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB, RESIDUAL_DELTA_MAX
from src.env.lightpaint_ref import LightPaintRef, make_square_ref
from src.train.train_phase_a_pid import (
    CTRL_FREQ,
    _corner_errors,
    _safe_token,
    _save_corner_csv,
    _save_tracking_plot,
    _save_trajectory_csv,
)
from src.train.visualize_flight import save_phase_a_visualization


@dataclass(frozen=True)
class RolloutResult:
    tag: str
    phase: str
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


def _make_env(
    *,
    phase: str,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
    ctrl_freq: int,
    pyb_freq: int,
) -> LightPaintAviaryPyB:
    return LightPaintAviaryPyB(
        label="square",
        phase=phase,
        wind_mode="M0",
        gui=False,
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


def _make_corner_decel_teacher(ref: LightPaintRef, gain_mps: float, window_m: float) -> Callable[[dict[str, Any]], np.ndarray]:
    internal_corner_s = np.asarray(ref.cumlen[1:-1], dtype=np.float32)

    def _teacher(ctx: dict[str, Any]) -> np.ndarray:
        info = ctx["info"]
        t = float(info.get("t", 0.0))
        s_now = t * float(ref.speed)
        action = np.zeros(4, dtype=np.float32)
        if internal_corner_s.size == 0:
            return action
        ahead = internal_corner_s[internal_corner_s >= s_now]
        if ahead.size == 0:
            return action
        dist_to_next = float(ahead[0] - s_now)
        if not (0.0 <= dist_to_next <= float(window_m)):
            return action
        v_ref = np.asarray(info.get("v_ref", [0.0, 0.0, 0.0]), dtype=np.float32)
        norm = float(np.linalg.norm(v_ref))
        if norm <= 1e-6:
            return action
        delta_v = -float(gain_mps) * (v_ref / norm)
        action[:3] = np.clip(delta_v / RESIDUAL_DELTA_MAX, -1.0, 1.0)
        return action.astype(np.float32)

    return _teacher


def _filter_corner_tangent_decel_action(
    action: np.ndarray,
    info: dict[str, Any],
    ref: LightPaintRef,
    window_m: float,
) -> np.ndarray:
    """Keep only learned pre-corner deceleration along the reference tangent."""
    filtered = np.zeros(4, dtype=np.float32)
    internal_corner_s = np.asarray(ref.cumlen[1:-1], dtype=np.float32)
    if internal_corner_s.size == 0:
        filtered[3] = float(action[3])
        return filtered
    t = float(info.get("t", 0.0))
    s_now = t * float(ref.speed)
    ahead = internal_corner_s[internal_corner_s >= s_now]
    if ahead.size == 0:
        filtered[3] = float(action[3])
        return filtered
    dist_to_next = float(ahead[0] - s_now)
    if not (0.0 <= dist_to_next <= float(window_m)):
        filtered[3] = float(action[3])
        return filtered
    v_ref = np.asarray(info.get("v_ref", [0.0, 0.0, 0.0]), dtype=np.float32)
    norm = float(np.linalg.norm(v_ref))
    if norm <= 1e-6:
        filtered[3] = float(action[3])
        return filtered
    unit = v_ref / norm
    tangent_action = float(np.dot(np.asarray(action[:3], dtype=np.float32), unit))
    tangent_action = min(tangent_action, 0.0)
    filtered[:3] = tangent_action * unit
    filtered[3] = float(action[3])
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)


def _rollout(
    *,
    tag: str,
    phase: str,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
    ctrl_freq: int,
    pyb_freq: int,
    action_fn: Callable[[dict[str, Any]], np.ndarray],
) -> RolloutResult:
    env = _make_env(
        phase=phase,
        reference=reference,
        max_steps=max_steps,
        seed=seed,
        init_box_size=init_box_size,
        led_always_on=led_always_on,
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
    if len(ref.corner_times) <= 2:
        return np.zeros(len(time_arr), dtype=bool)
    window_s = float(window_m) / max(float(ref.speed), 1e-6)
    internal_corner_times = np.asarray(ref.corner_times[1:-1], dtype=np.float32)
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
        "corner_influence",
    ]
    component_values: dict[str, np.ndarray] = {}
    for key in component_keys:
        vals = [0.0]
        vals.extend(float(row.get(key, 0.0)) for row in result.info_rows)
        component_values[key] = np.asarray(vals[: len(result.time_arr)], dtype=np.float32)
    target_px = max(int(result.target_mask.sum()), 1)
    painted_bin = (result.cumulative > 0.3).astype(np.float32)
    coverage = float((painted_bin * (result.target_mask > 0.5)).sum()) / float(target_px)
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
    return {
        "tag": result.tag,
        "phase": result.phase,
        "wind_mode": "M0",
        "n_steps": int(max(len(result.time_arr) - 1, 0)),
        "walltime_s": f"{result.walltime_s:.4f}",
        "tracking_rmse_m": f"{rmse(err):.6f}",
        "tracking_mean_m": f"{mean_or_nan(err):.6f}",
        "tracking_max_m": f"{float(np.max(err)):.6f}" if err.size else "nan",
        "tracking_final_m": f"{float(err[-1]):.6f}" if err.size else "nan",
        "corner_window_rmse_m": f"{rmse(err[corner]):.6f}",
        "straight_rmse_m": f"{rmse(err[straight]):.6f}",
        "corner_path_rmse_m": f"{rmse(path_dist[corner]):.6f}",
        "straight_path_rmse_m": f"{rmse(path_dist[straight]):.6f}",
        "path_max_m": f"{float(np.max(path_dist)):.6f}" if path_dist.size else "nan",
        "corner_mean_speed_mps": f"{mean_or_nan(speed[corner]):.6f}",
        "straight_mean_speed_mps": f"{mean_or_nan(speed[straight]):.6f}",
        "corner_speed_ratio": f"{(mean_or_nan(speed[corner]) / max(mean_or_nan(speed[straight]), 1e-6)):.6f}",
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
        "corner_mean_influence": f"{mean_or_nan(component_values['corner_influence'][corner]):.6f}",
        "corner_endpoint_error_mean_m": f"{float(np.mean(internal_corner_vals)):.6f}" if internal_corner_vals else "nan",
        "corner_endpoint_error_max_m": f"{float(np.max(internal_corner_vals)):.6f}" if internal_corner_vals else "nan",
        "painted_pixel_coverage": f"{coverage:.6f}",
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
) -> dict[str, str]:
    token = _safe_token(result.tag)
    prefix = f"phase_{result.phase.lower()}_square_M0_{token}"
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
        title=f"Phase {result.phase} square M0 {result.tag}",
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


def _collect_bc_dataset(
    *,
    action_fn: Callable[[dict[str, Any]], np.ndarray],
    episodes: int,
    reference: LightPaintRef,
    max_steps: int,
    seed: int,
    init_box_size: float,
    led_always_on: bool,
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
            reference=reference,
            max_steps=max_steps,
            seed=seed + ep,
            init_box_size=init_box_size,
            led_always_on=led_always_on,
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


def main(args: argparse.Namespace) -> int:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    ref = make_square_ref(side_m=float(args.square_side), speed=float(args.speed))
    max_steps = int(args.max_steps) if args.max_steps is not None else int(
        ceil((ref.duration + float(args.settle_time)) * float(args.ctrl_freq))
    )
    out_root = Path(args.output_dir)
    artifacts_dir = out_root / "artifacts"
    viz_dir = artifacts_dir / "visualization"
    model_dir = out_root / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[phase_b_m0_corner] square side={args.square_side} speed={args.speed} "
        f"duration={ref.duration:.3f}s max_steps={max_steps} total_timesteps={args.total_timesteps}",
        flush=True,
    )

    common = {
        "reference": ref,
        "max_steps": max_steps,
        "seed": int(args.seed),
        "init_box_size": float(args.init_box_size),
        "led_always_on": bool(args.led_always_on),
        "ctrl_freq": int(args.ctrl_freq),
        "pyb_freq": int(args.pyb_freq),
    }
    phase_a = _rollout(tag="phaseA_pid", phase="A", action_fn=_phase_a_action, **common)
    phase_b_zero = _rollout(tag="phaseB_zero", phase="B", action_fn=_phase_b_zero_action, **common)
    teacher_fn = _make_corner_decel_teacher(
        ref,
        gain_mps=float(args.teacher_gain),
        window_m=float(args.teacher_window_m),
    )
    phase_b_teacher = _rollout(tag="phaseB_teacher", phase="B", action_fn=teacher_fn, **common)

    def _train_env() -> Monitor:
        return Monitor(_make_env(phase="B", **common))

    vec_env = DummyVecEnv([_train_env])
    model = _make_model(args, vec_env)

    phase_b_before = _rollout(
        tag="phaseB_untrained",
        phase="B",
        action_fn=lambda ctx: model.predict(ctx["obs"], deterministic=True)[0],
        **common,
    )

    train_start = time.perf_counter()
    bc_summary: dict[str, Any] | None = None
    if int(args.bc_epochs) > 0:
        print("[phase_b_m0_corner] BC warm-start collect start", flush=True)
        obs_batch, action_batch = _collect_bc_dataset(
            action_fn=teacher_fn,
            episodes=int(args.bc_episodes),
            **common,
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
            out_path=artifacts_dir / "phase_b_square_M0_bc_loss.csv",
        )
        print(
            f"[phase_b_m0_corner] BC done samples={bc_summary['n_samples']} "
            f"nonzero={bc_summary['nonzero_samples']} "
            f"loss={bc_summary['initial_loss']:.6f}->{bc_summary['final_loss']:.6f}",
            flush=True,
        )
    if int(args.total_timesteps) > 0:
        print("[phase_b_m0_corner] PPO learn start", flush=True)
        model.learn(total_timesteps=int(args.total_timesteps), progress_bar=False)
        print("[phase_b_m0_corner] PPO learn done", flush=True)
    train_walltime = time.perf_counter() - train_start
    print(f"[phase_b_m0_corner] train stage done walltime={train_walltime:.2f}s", flush=True)

    model_path = model_dir / "ppo_phaseB_square_M0_corner.zip"
    model.save(str(model_path))

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

    results = [phase_a, phase_b_zero, phase_b_teacher, phase_b_before, phase_b_after]
    metrics_rows = [_metrics(r, ref, float(args.corner_window_m)) for r in results]
    metrics_path = artifacts_dir / "phase_b_square_M0_corner_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)

    artifact_index: dict[str, Any] = {
        "model_path": str(model_path),
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
        )

    if bool(args.save_video):
        viz_paths = save_phase_a_visualization(
            pos_list=phase_b_after.pos_arr.tolist(),
            brightness_list=phase_b_after.brightness_arr.tolist(),
            target_mask=phase_b_after.target_mask,
            label="square_phaseB_trained",
            wind_mode="M0",
            out_dir=viz_dir,
            fps=10,
            tracking_errs=np.linalg.norm(phase_b_after.pos_arr - phase_b_after.ref_arr, axis=1).tolist(),
            cumulative=phase_b_after.cumulative.copy(),
            ref_waypoints=ref.waypoints.copy(),
            ref_cumlen=ref.cumlen.copy(),
            ref_segment_led=ref.segment_led.copy(),
            ref_pos_list=phase_b_after.ref_arr.tolist(),
            snapshots=[],
        )
        artifact_index["phaseB_trained_video"] = viz_paths

    zero = next(row for row in metrics_rows if row["tag"] == "phaseB_zero")
    trained = next(row for row in metrics_rows if row["tag"] == "phaseB_trained")
    zero_corner = float(zero["corner_path_rmse_m"])
    trained_corner = float(trained["corner_path_rmse_m"])
    zero_straight = float(zero["straight_path_rmse_m"])
    trained_straight = float(trained["straight_path_rmse_m"])
    zero_corner_speed = float(zero["corner_mean_speed_mps"])
    trained_corner_speed = float(trained["corner_mean_speed_mps"])
    zero_coverage = float(zero["painted_pixel_coverage"])
    trained_coverage = float(trained["painted_pixel_coverage"])
    success = {
        "corner_path_rmse_reduced_20pct": bool(trained_corner <= zero_corner * 0.80),
        "corner_speed_reduced_5_to_20pct": bool(
            zero_corner_speed * 0.80 <= trained_corner_speed <= zero_corner_speed * 0.95
        ),
        "straight_path_rmse_not_worse_10pct": bool(trained_straight <= zero_straight * 1.10),
        "painting_coverage_kept_90pct": bool(trained_coverage >= zero_coverage * 0.90),
        "overall_pass": bool(
            trained_corner <= zero_corner * 0.80
            and zero_corner_speed * 0.80 <= trained_corner_speed <= zero_corner_speed * 0.95
            and trained_straight <= zero_straight * 1.10
            and trained_coverage >= zero_coverage * 0.90
        ),
    }
    summary = {
        "config": {
            "phase": "B",
            "wind_mode": "M0",
            "reference": "square",
            "square_side": float(args.square_side),
            "speed": float(args.speed),
            "max_steps": max_steps,
            "total_timesteps": int(args.total_timesteps),
            "corner_window_m": float(args.corner_window_m),
            "log_std_init": float(args.log_std_init),
            "teacher_gain": float(args.teacher_gain),
            "teacher_window_m": float(args.teacher_window_m),
            "bc_epochs": int(args.bc_epochs),
            "bc_episodes": int(args.bc_episodes),
            "trained_action_filter": str(args.trained_action_filter),
        },
        "train_walltime_s": train_walltime,
        "bc": bc_summary,
        "success_criteria": success,
        "metrics": metrics_rows,
        "artifacts": artifact_index,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[phase_b_m0_corner] summary: {summary_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase B M0 PPO for square corner slowing.")
    parser.add_argument("--output-dir", default=str(_PKG_ROOT / "artifacts" / "phase_b_m0_corner"))
    parser.add_argument("--total-timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--square-side", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--corner-window-m", type=float, default=0.18)
    parser.add_argument("--init-box-size", type=float, default=0.0)
    parser.add_argument("--led-always-on", action="store_true", default=True)
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
    raise SystemExit(main(parse_args()))
