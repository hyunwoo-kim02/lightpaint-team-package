"""Phase A PID-only PyBullet rollout for the active Square-PID-M0 gate.

Default run:
    python -m src.train.train_phase_a_pid

Outputs:
    artifacts/phase_a_<trajectory>_<wind>_trajectory.csv
    artifacts/phase_a_<trajectory>_<wind>_metrics.csv
    artifacts/phase_a_<trajectory>_<wind>_corners.csv
    artifacts/visualization/phase_a_<trajectory>_<wind>.mp4
    artifacts/visualization/phase_a_<trajectory>_<wind>_*.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from math import ceil
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB
from src.env.lightpaint_ref import (
    LightPaintRef,
    load_drawn_path_ref,
    make_letter_ref,
    make_square_ref,
)
from src.render.pybullet_snapshot import capture_pybullet_snapshot
from src.train.visualize_flight import save_phase_a_visualization


SNAPSHOT_STRIDE = 5
CTRL_FREQ = 30.0


def _safe_token(value: str) -> str:
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep) or "run"


def _build_reference(args: argparse.Namespace) -> tuple[str, LightPaintRef | None]:
    if args.trajectory == "square":
        ref = make_square_ref(
            side_m=float(args.square_side),
            center_x=0.0,
            center_z=1.5,
            speed=float(args.speed),
        )
        return "square", ref
    if args.trajectory == "drawn":
        if args.drawn_path is None:
            raise ValueError("--trajectory drawn requires --drawn-path")
        ref = load_drawn_path_ref(
            path=args.drawn_path,
            speed=float(args.speed),
            plane=str(args.drawn_plane),
            coordinate_space=str(args.drawn_space),
            width_m=float(args.width_m),
            height_m=float(args.height_m),
            max_waypoints_per_stroke=int(args.max_waypoints),
            smooth=not bool(args.no_smooth_ref),
            smooth_window_m=float(args.smooth_window_m),
        )
        return Path(args.drawn_path).stem, ref
    label = args.label if args.label is not None else "L"
    ref = make_letter_ref(
        letter=str(label),
        plane=str(args.letter_plane),
        speed=float(args.speed),
        width_m=float(args.width_m),
        height_m=float(args.height_m),
        center_x=0.0,
        center_z=1.5,
        fixed_y=0.0,
        fixed_z=1.5,
        max_waypoints=int(args.max_waypoints),
        smooth=not bool(args.no_smooth_ref),
        smooth_window_m=float(args.smooth_window_m),
    )
    return str(label), ref


def _corner_errors(ref: LightPaintRef, time_arr: np.ndarray, pos_arr: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for idx, (corner_time, corner_pos) in enumerate(zip(ref.corner_times, ref.waypoints)):
        nearest = int(np.argmin(np.abs(time_arr - float(corner_time))))
        actual = pos_arr[nearest]
        err = float(np.linalg.norm(actual - corner_pos))
        rows.append({
            "corner_index": idx,
            "target_time_s": float(corner_time),
            "sample_step": nearest,
            "sample_time_s": float(time_arr[nearest]),
            "ref_x": float(corner_pos[0]),
            "ref_y": float(corner_pos[1]),
            "ref_z": float(corner_pos[2]),
            "actual_x": float(actual[0]),
            "actual_y": float(actual[1]),
            "actual_z": float(actual[2]),
            "error_m": err,
        })
    return rows


def _save_trajectory_csv(
    path: Path,
    time_arr: np.ndarray,
    ref_arr: np.ndarray,
    pos_arr: np.ndarray,
    brightness_arr: np.ndarray,
    rpm_arr: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "t_s",
            "ref_x", "ref_y", "ref_z",
            "actual_x", "actual_y", "actual_z",
            "tracking_err_m",
            "brightness",
            "rpm_0", "rpm_1", "rpm_2", "rpm_3",
        ])
        for i in range(len(time_arr)):
            err = float(np.linalg.norm(pos_arr[i] - ref_arr[i]))
            rpm = rpm_arr[i] if i < len(rpm_arr) else np.zeros(4, dtype=np.float32)
            w.writerow([
                i, f"{float(time_arr[i]):.6f}",
                f"{float(ref_arr[i, 0]):.6f}",
                f"{float(ref_arr[i, 1]):.6f}",
                f"{float(ref_arr[i, 2]):.6f}",
                f"{float(pos_arr[i, 0]):.6f}",
                f"{float(pos_arr[i, 1]):.6f}",
                f"{float(pos_arr[i, 2]):.6f}",
                f"{err:.6f}",
                f"{float(brightness_arr[i]):.3f}",
                f"{float(rpm[0]):.3f}",
                f"{float(rpm[1]):.3f}",
                f"{float(rpm[2]):.3f}",
                f"{float(rpm[3]):.3f}",
            ])


def _save_corner_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _save_tracking_plot(
    path: Path,
    ref_arr: np.ndarray,
    pos_arr: np.ndarray,
    corner_rows: list[dict],
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.plot(ref_arr[:, 0], ref_arr[:, 2], color="cyan", linewidth=2.0, label="reference")
    ax.plot(pos_arr[:, 0], pos_arr[:, 2], color="orange", linewidth=1.2, alpha=0.9, label="actual")
    if corner_rows:
        ax.scatter(
            [r["ref_x"] for r in corner_rows],
            [r["ref_z"] for r in corner_rows],
            c="lime",
            s=45,
            marker="s",
            label="corners",
            zorder=5,
        )
    ax.scatter([pos_arr[0, 0]], [pos_arr[0, 2]], c="white", s=50, label="start", zorder=6)
    ax.scatter([pos_arr[-1, 0]], [pos_arr[-1, 2]], c="yellow", s=50, label="end", zorder=6)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("z (m)", color="white")
    ax.set_title(title, color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.grid(True, color="gray", alpha=0.25)
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(path), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main(args: argparse.Namespace) -> int:
    label, reference = _build_reference(args)
    run_token = _safe_token(label)
    wind_mode = args.wind_mode
    if args.max_steps is None:
        if reference is not None:
            max_steps = int(ceil((reference.duration + float(args.settle_time)) * CTRL_FREQ))
        else:
            max_steps = 600
    else:
        max_steps = int(args.max_steps)
    seed = int(args.seed)

    out_root = Path(args.output_dir) if args.output_dir else _PKG_ROOT
    artifacts_dir = out_root / "artifacts"
    viz_dir = artifacts_dir / "visualization"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[phase_a] trajectory={args.trajectory} label={label!r} wind_mode={wind_mode} "
        f"seed={seed} max_steps={max_steps}",
        flush=True,
    )
    if reference is not None:
        print(
            f"[phase_a] ref={reference.name} length={reference.length:.3f}m "
            f"duration={reference.duration:.3f}s speed={reference.speed:.3f}m/s",
            flush=True,
        )

    env = LightPaintAviaryPyB(
        label=label,
        phase="A",
        wind_mode=wind_mode,
        gui=False,
        init_box_size=float(args.init_box_size),
        led_always_on=bool(args.led_always_on),
        max_episode_steps=max_steps,
        reference=reference,
    )
    target_mask = env.G_letter.copy()
    target_px = max(int(target_mask.sum()), 1)
    print(f"[phase_a] target_px={target_px}", flush=True)

    obs, info = env.reset(seed=seed)
    pos_list: list = [info["pos"]]
    ref_list: list = [info["p_ref"]]
    time_list: list = [float(info.get("t", 0.0))]
    brightness_list: list = [float(info.get("brightness", 0.0))]
    rpm_list: list = [info.get("u_pid_rpm", [0.0, 0.0, 0.0, 0.0])]
    rewards: list = []
    snapshots: list = []
    term = False

    t_start = time.perf_counter()
    try:
        snapshots.append((0, capture_pybullet_snapshot(env.CLIENT)))
    except Exception as exc:
        print(f"[phase_a] snapshot capture skipped: {exc}", flush=True)

    stride = max(1, int(args.snapshot_stride))
    for k in range(max_steps):
        obs, reward, term, trunc, info = env.step(np.zeros(0, dtype=np.float32))
        pos_list.append(info["pos"])
        ref_list.append(info["p_ref"])
        time_list.append(float(info.get("t", (k + 1) / CTRL_FREQ)))
        brightness_list.append(float(info["brightness"]))
        rpm_list.append(info["u_pid_rpm"])
        rewards.append(float(reward))
        if (k + 1) % stride == 0:
            try:
                snapshots.append((k + 1, capture_pybullet_snapshot(env.CLIENT)))
            except Exception:
                pass
        if term or trunc:
            break
    walltime = time.perf_counter() - t_start

    time_arr = np.asarray(time_list, dtype=np.float32)
    pos_arr = np.asarray(pos_list, dtype=np.float32)
    ref_arr = np.asarray(ref_list, dtype=np.float32)
    brightness_arr = np.asarray(brightness_list, dtype=np.float32)
    rpm_arr = np.asarray(rpm_list, dtype=np.float32)
    err_arr = np.linalg.norm(pos_arr - ref_arr, axis=1)
    rmse = float(np.sqrt(np.mean(err_arr ** 2))) if len(err_arr) else float("nan")
    mean_err = float(np.mean(err_arr)) if len(err_arr) else float("nan")
    max_err = float(np.max(err_arr)) if len(err_arr) else float("nan")
    final_err = float(err_arr[-1]) if len(err_arr) else float("nan")
    corner_rows = _corner_errors(reference, time_arr, pos_arr) if reference is not None else []
    corner_vals = [r["error_m"] for r in corner_rows]
    corner_max = float(max(corner_vals)) if corner_vals else float("nan")
    corner_mean = float(np.mean(corner_vals)) if corner_vals else float("nan")

    painted_px_total = int(env.cumulative.sum())
    painted_bin = (env.cumulative > 0.3).astype(np.float32)
    on_target_paint = float((painted_bin * (target_mask > 0.5)).sum())
    coverage = on_target_paint / target_px
    crash = bool(term)

    print(
        f"[phase_a] DONE: steps={len(rewards)} walltime={walltime:.2f}s "
        f"tracking_rmse={rmse:.4f} max_err={max_err:.4f} "
        f"corner_max={corner_max:.4f} crash={crash}",
        flush=True,
    )

    traj_path = artifacts_dir / f"phase_a_{run_token}_{wind_mode}_trajectory.csv"
    corner_path = artifacts_dir / f"phase_a_{run_token}_{wind_mode}_corners.csv"
    metrics_path = artifacts_dir / f"phase_a_{run_token}_{wind_mode}_metrics.csv"
    tracking_plot_path = viz_dir / f"phase_a_{run_token}_{wind_mode}_tracking_xz.png"

    _save_trajectory_csv(traj_path, time_arr, ref_arr, pos_arr, brightness_arr, rpm_arr)
    _save_corner_csv(corner_path, corner_rows)
    _save_tracking_plot(
        tracking_plot_path,
        ref_arr=ref_arr,
        pos_arr=pos_arr,
        corner_rows=corner_rows,
        title=f"Phase A {label} {wind_mode}: reference vs actual",
    )

    with open(str(metrics_path), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "trajectory", "label", "wind_mode", "seed", "n_steps", "walltime_s",
            "ref_name", "ref_length_m", "ref_duration_s", "ref_speed_mps",
            "tracking_rmse_m", "tracking_mean_m", "tracking_max_m", "tracking_final_m",
            "corner_endpoint_error_mean_m", "corner_endpoint_error_max_m",
            "painted_pixel_coverage", "painted_px_total", "crash",
        ])
        w.writerow([
            args.trajectory, label, wind_mode, seed, len(rewards), f"{walltime:.4f}",
            reference.name if reference is not None else "letter_skeleton",
            f"{reference.length:.6f}" if reference is not None else f"{float(env._ref_cumlen[-1]):.6f}",
            f"{reference.duration:.6f}" if reference is not None else "",
            f"{reference.speed:.6f}" if reference is not None else "",
            f"{rmse:.6f}", f"{mean_err:.6f}", f"{max_err:.6f}", f"{final_err:.6f}",
            f"{corner_mean:.6f}", f"{corner_max:.6f}",
            f"{coverage:.6f}", painted_px_total, int(crash),
        ])

    print(f"[phase_a] trajectory CSV: {traj_path}", flush=True)
    if corner_rows:
        print(f"[phase_a] corner CSV: {corner_path}", flush=True)
    print(f"[phase_a] metrics CSV: {metrics_path}", flush=True)
    print(f"[phase_a] tracking plot: {tracking_plot_path}", flush=True)

    paths = save_phase_a_visualization(
        pos_list=pos_list,
        brightness_list=brightness_list,
        target_mask=target_mask,
        label=run_token,
        wind_mode=wind_mode,
        out_dir=viz_dir,
        fps=10,
        tracking_errs=err_arr.tolist(),
        cumulative=env.cumulative.copy(),
        ref_waypoints=env._ref_waypoints.copy(),
        ref_cumlen=env._ref_cumlen.copy(),
        ref_segment_led=reference.segment_led.copy() if reference is not None else None,
        snapshots=snapshots,
    )
    print(f"[phase_a] visualization paths: {paths}", flush=True)

    env.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A PID-only sanity run.")
    parser.add_argument("--trajectory", choices=("square", "letter", "drawn"), default="square",
                        help="Trajectory source for the PID gate.")
    parser.add_argument("--label", default=None,
                        help="Text label for --trajectory letter. Defaults to 'L'.")
    parser.add_argument("--wind-mode", default="M0",
                        help="Wind mode. Phase A enforces M0.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Episode horizon. Defaults to reference duration plus settle time.")
    parser.add_argument("--speed", type=float, default=0.35,
                        help="Reference speed in m/s for waypoint trajectories.")
    parser.add_argument("--square-side", type=float, default=0.8,
                        help="Square side length in meters.")
    parser.add_argument("--letter-plane", choices=("xz", "xy"), default="xz",
                        help="Plane used when --trajectory letter is selected.")
    parser.add_argument("--drawn-path", type=str, default=None,
                        help="JSON path exported by tools/draw_path.html or another stroke editor.")
    parser.add_argument("--drawn-plane", choices=("xz", "xy"), default="xz",
                        help="Plane used when --trajectory drawn is selected.")
    parser.add_argument("--drawn-space", choices=("normalized", "pixel", "world"), default="normalized",
                        help="Coordinate space for drawn JSON points.")
    parser.add_argument("--width-m", type=float, default=0.9,
                        help="Letter reference width in meters.")
    parser.add_argument("--height-m", type=float, default=0.9,
                        help="Letter reference height in meters.")
    parser.add_argument("--max-waypoints", type=int, default=240,
                        help="Maximum smooth reference waypoints for letter trajectories.")
    parser.add_argument("--smooth-window-m", type=float, default=0.08,
                        help="Smoothing window in meters for image-derived letter references.")
    parser.add_argument("--no-smooth-ref", action="store_true",
                        help="Use raw skeleton waypoints for letter trajectories.")
    parser.add_argument("--settle-time", type=float, default=2.0,
                        help="Extra seconds after the reference endpoint for settling.")
    parser.add_argument("--init-box-size", type=float, default=0.02,
                        help="Initial random position box half-width in meters.")
    parser.add_argument("--snapshot-stride", type=int, default=SNAPSHOT_STRIDE,
                        help="PyBullet camera snapshot stride in env steps.")
    parser.add_argument("--led-always-on", action="store_true",
                        help="Force LED on for visualization instead of scripted target-mask LED.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for env reset.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output root. Defaults to package root.")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
