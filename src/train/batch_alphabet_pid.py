"""Batch PID rollouts for A-Z letter references in two movement modes.

Modes:
- ``xy_fixed_z``: draw the letter on the x-y plane while z is fixed.
- ``xz_z_motion``: draw the letter on the x-z plane while y is fixed.

The script keeps outputs organized per mode/letter under
``artifacts/alphabet_pid`` and writes a single summary CSV for audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import string
import sys
import time
from math import ceil
from pathlib import Path
from typing import Any

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

import numpy as np

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.light_paint_aviary_pyb import LightPaintAviaryPyB
from src.env.lightpaint_ref import LightPaintRef, make_letter_ref
from src.render.pybullet_snapshot import capture_pybullet_snapshot
from src.train.visualize_flight import save_phase_a_visualization


CTRL_FREQ = 30.0
MODE_TO_PLANE = {
    "xy_fixed_z": "xy",
    "xz_z_motion": "xz",
}


def _project(points: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    if plane == "xy":
        return points[:, 0], points[:, 1], "x (m)", "y (m)"
    return points[:, 0], points[:, 2], "x (m)", "z (m)"


def _write_trajectory_csv(
    path: Path,
    time_arr: np.ndarray,
    ref_arr: np.ndarray,
    pos_arr: np.ndarray,
    err_arr: np.ndarray,
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
            "rpm_0", "rpm_1", "rpm_2", "rpm_3",
        ])
        for i in range(len(time_arr)):
            rpm = rpm_arr[i] if i < len(rpm_arr) else np.zeros(4, dtype=np.float32)
            w.writerow([
                i,
                f"{float(time_arr[i]):.6f}",
                f"{float(ref_arr[i, 0]):.6f}",
                f"{float(ref_arr[i, 1]):.6f}",
                f"{float(ref_arr[i, 2]):.6f}",
                f"{float(pos_arr[i, 0]):.6f}",
                f"{float(pos_arr[i, 1]):.6f}",
                f"{float(pos_arr[i, 2]):.6f}",
                f"{float(err_arr[i]):.6f}",
                f"{float(rpm[0]):.3f}",
                f"{float(rpm[1]):.3f}",
                f"{float(rpm[2]):.3f}",
                f"{float(rpm[3]):.3f}",
            ])


def _save_tracking_plot(
    path: Path,
    letter: str,
    mode: str,
    ref_arr: np.ndarray,
    pos_arr: np.ndarray,
    err_arr: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane = MODE_TO_PLANE[mode]
    rx, ry, x_label, y_label = _project(ref_arr, plane)
    axx, ayy, _, _ = _project(pos_arr, plane)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#111111")
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#111111")
    ax.plot(rx, ry, color="cyan", linewidth=1.6, label="reference")
    ax.plot(axx, ayy, color="orange", linewidth=1.0, alpha=0.9, label="actual")
    ax.scatter([axx[0]], [ayy[0]], c="lime", s=40, label="start", zorder=4)
    ax.scatter([axx[-1]], [ayy[-1]], c="yellow", s=40, label="end", zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label, color="white")
    ax.set_ylabel(y_label, color="white")
    ax.set_title(f"{letter} {mode} projection", color="white")
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color="gray", alpha=0.25)
    ax.legend(facecolor="#222222", labelcolor="white", fontsize=8)
    for sp in ax.spines.values():
        sp.set_color("gray")

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3.set_facecolor("#111111")
    ax3.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], color="cyan", linewidth=1.4)
    ax3.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], color="orange", linewidth=1.0)
    ax3.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], c="lime", s=40)
    ax3.scatter([pos_arr[-1, 0]], [pos_arr[-1, 1]], [pos_arr[-1, 2]], c="yellow", s=40)
    ax3.set_xlabel("x", color="white")
    ax3.set_ylabel("y", color="white")
    ax3.set_zlabel("z", color="white")
    ax3.tick_params(colors="white", labelsize=7)
    ax3.set_title("3D path", color="white")

    ax_err = fig.add_subplot(gs[1, 0])
    ax_err.set_facecolor("#111111")
    ax_err.plot(err_arr, color="orange", linewidth=1.0)
    ax_err.set_xlabel("step", color="white")
    ax_err.set_ylabel("tracking error (m)", color="white")
    ax_err.set_title(f"RMSE={metrics['tracking_rmse_m']:.4f} m", color="white")
    ax_err.tick_params(colors="white", labelsize=8)
    ax_err.grid(True, color="gray", alpha=0.25)
    for sp in ax_err.spines.values():
        sp.set_color("gray")

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.set_facecolor("#111111")
    ax_txt.axis("off")
    text = (
        f"letter: {letter}\n"
        f"mode: {mode}\n"
        f"reference length: {metrics['ref_length_m']:.3f} m\n"
        f"reference duration: {metrics['ref_duration_s']:.3f} s\n"
        f"steps: {metrics['n_steps']}\n"
        f"RMSE: {metrics['tracking_rmse_m']:.4f} m\n"
        f"max error: {metrics['tracking_max_m']:.4f} m\n"
        f"final error: {metrics['tracking_final_m']:.4f} m\n"
        f"crash: {int(metrics['crash'])}"
    )
    ax_txt.text(
        0.03,
        0.95,
        text,
        color="white",
        fontsize=11,
        family="monospace",
        va="top",
        transform=ax_txt.transAxes,
    )

    fig.suptitle(f"Alphabet PID rollout: {letter} / {mode}", color="white", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(str(path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _latest_snapshot(snapshots: list[tuple[int, np.ndarray]], step: int):
    if not snapshots:
        return None
    chosen = snapshots[0][1]
    for s, img in snapshots:
        if s <= step:
            chosen = img
        else:
            break
    return chosen


def _save_tracking_video(
    path: Path,
    letter: str,
    mode: str,
    ref_arr: np.ndarray,
    pos_arr: np.ndarray,
    err_arr: np.ndarray,
    snapshots: list[tuple[int, np.ndarray]],
    max_frames: int,
    fps: int,
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import imageio
        import imageio_ffmpeg  # noqa: F401
    except Exception:
        return False

    plane = MODE_TO_PLANE[mode]
    rx, ry, x_label, y_label = _project(ref_arr, plane)
    axx, ayy, _, _ = _project(pos_arr, plane)
    indices = np.linspace(0, len(pos_arr) - 1, max(2, int(max_frames)), dtype=int)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=6, macro_block_size=1)
    try:
        for idx in indices:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            fig.patch.set_facecolor("#111111")

            ax = axes[0]
            ax.set_facecolor("#111111")
            ax.plot(rx, ry, color="cyan", linewidth=1.2, alpha=0.8)
            ax.plot(axx[: idx + 1], ayy[: idx + 1], color="orange", linewidth=1.2)
            ax.scatter([axx[idx]], [ayy[idx]], c="lime", s=45, zorder=4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(x_label, color="white")
            ax.set_ylabel(y_label, color="white")
            ax.set_title(f"{letter} {mode} step={idx}", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=7)
            ax.grid(True, color="gray", alpha=0.25)
            for sp in ax.spines.values():
                sp.set_color("gray")

            ax2 = axes[1]
            ax2.set_facecolor("#111111")
            shot = _latest_snapshot(snapshots, int(idx))
            if shot is not None:
                ax2.imshow(shot, origin="upper")
                ax2.axis("off")
                ax2.set_title("PyBullet view", color="white", fontsize=9)
            else:
                ax2.plot(err_arr[: idx + 1], color="orange", linewidth=1.0)
                ax2.set_title("tracking error", color="white", fontsize=9)
                ax2.tick_params(colors="white", labelsize=7)
                ax2.grid(True, color="gray", alpha=0.25)

            plt.tight_layout(pad=0.5)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
            plt.close(fig)
    finally:
        writer.close()
    return True


def _run_one(
    letter: str,
    mode: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    plane = MODE_TO_PLANE[mode]
    ref = make_letter_ref(
        letter=letter,
        plane=plane,
        speed=float(args.speed),
        width_m=float(args.width_m),
        height_m=float(args.height_m),
        max_waypoints=int(args.max_waypoints),
        smooth=not bool(args.no_smooth_ref),
        smooth_window_m=float(args.smooth_window_m),
    )
    max_steps = int(ceil((ref.duration + float(args.settle_time)) * CTRL_FREQ))
    env = LightPaintAviaryPyB(
        label=f"{letter}_{mode}",
        phase="A",
        wind_mode="M0",
        gui=False,
        init_box_size=0.0,
        led_always_on=True,
        max_episode_steps=max_steps,
        reference=ref,
    )

    pos_list = []
    ref_list = []
    time_list = []
    rpm_list = []
    brightness_list = []
    snapshots: list[tuple[int, np.ndarray]] = []
    term = False
    trunc = False
    t_start = time.perf_counter()
    try:
        _, info = env.reset(seed=int(args.seed))
        pos_list.append(info["pos"])
        ref_list.append(info["p_ref"])
        time_list.append(float(info.get("t", 0.0)))
        rpm_list.append(info.get("u_pid_rpm", [0.0, 0.0, 0.0, 0.0]))
        brightness_list.append(float(info.get("brightness", 0.0)))
        if not args.no_video:
            try:
                snapshots.append((0, capture_pybullet_snapshot(env.CLIENT, width=320, height=240)))
            except Exception:
                pass

        snapshot_stride = max(1, max_steps // max(1, int(args.snapshot_count)))
        for k in range(max_steps):
            _, reward, term, trunc, info = env.step(np.zeros(0, dtype=np.float32))
            pos_list.append(info["pos"])
            ref_list.append(info["p_ref"])
            time_list.append(float(info.get("t", (k + 1) / CTRL_FREQ)))
            rpm_list.append(info["u_pid_rpm"])
            brightness_list.append(float(info.get("brightness", 0.0)))
            if not args.no_video and (k + 1) % snapshot_stride == 0:
                try:
                    snapshots.append((k + 1, capture_pybullet_snapshot(env.CLIENT, width=320, height=240)))
                except Exception:
                    pass
            if term or trunc:
                break
    finally:
        cumulative = env.cumulative.copy()
        env.close()

    walltime = time.perf_counter() - t_start
    time_arr = np.asarray(time_list, dtype=np.float32)
    pos_arr = np.asarray(pos_list, dtype=np.float32)
    ref_arr = np.asarray(ref_list, dtype=np.float32)
    rpm_arr = np.asarray(rpm_list, dtype=np.float32)
    err_arr = np.linalg.norm(pos_arr - ref_arr, axis=1)

    run_dir = out_dir / mode / letter
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory.csv"
    metrics_path = run_dir / "metrics.csv"
    plot_path = run_dir / "tracking.png"
    video_path = run_dir / "flight.mp4"

    metrics = {
        "letter": letter,
        "mode": mode,
        "plane": plane,
        "seed": int(args.seed),
        "n_steps": int(len(pos_arr) - 1),
        "walltime_s": float(walltime),
        "ref_name": ref.name,
        "ref_length_m": float(ref.length),
        "ref_duration_s": float(ref.duration),
        "ref_speed_mps": float(ref.speed),
        "tracking_rmse_m": float(np.sqrt(np.mean(err_arr ** 2))),
        "tracking_mean_m": float(np.mean(err_arr)),
        "tracking_max_m": float(np.max(err_arr)),
        "tracking_final_m": float(err_arr[-1]),
        "painted_px_total": int(cumulative.sum()),
        "crash": bool(term),
        "truncated": bool(trunc),
        "trajectory_csv": str(trajectory_path),
        "tracking_png": str(plot_path),
        "flight_mp4": "",
        "flight_gif": "",
        "summary_png": "",
        "refpath_png": "",
        "frames_dir": "",
    }

    _write_trajectory_csv(trajectory_path, time_arr, ref_arr, pos_arr, err_arr, rpm_arr)
    _save_tracking_plot(plot_path, letter, mode, ref_arr, pos_arr, err_arr, metrics)
    if not args.no_video:
        if args.quick_video:
            ok = _save_tracking_video(
                video_path,
                letter,
                mode,
                ref_arr,
                pos_arr,
                err_arr,
                snapshots,
                max_frames=int(args.video_frames),
                fps=int(args.video_fps),
            )
            if ok:
                metrics["flight_mp4"] = str(video_path)
        else:
            viz_paths = save_phase_a_visualization(
                pos_list=pos_list,
                brightness_list=brightness_list,
                target_mask=env.G_letter.copy() if "env" in locals() else np.zeros((64, 64), dtype=np.float32),
                label=f"{letter}_{mode}",
                wind_mode="M0",
                out_dir=run_dir,
                fps=int(args.video_fps),
                tracking_errs=err_arr.tolist(),
                cumulative=cumulative,
                ref_waypoints=ref.waypoints.copy(),
                ref_cumlen=ref.cumlen.copy(),
                ref_segment_led=ref.segment_led.copy(),
                snapshots=snapshots,
            )
            metrics["flight_mp4"] = viz_paths.get("mp4", "")
            metrics["flight_gif"] = viz_paths.get("gif", "")
            metrics["summary_png"] = viz_paths.get("summary", "")
            metrics["refpath_png"] = viz_paths.get("refpath", "")
            metrics["frames_dir"] = viz_paths.get("frames_dir", "")

    with open(str(metrics_path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    with open(str(run_dir / "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _save_summary_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    letters = sorted({r["letter"] for r in rows})
    modes = [m for m in MODE_TO_PLANE if any(r["mode"] == m for r in rows)]
    x = np.arange(len(letters))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    for i, mode in enumerate(modes):
        vals = []
        for letter in letters:
            row = next(r for r in rows if r["letter"] == letter and r["mode"] == mode)
            vals.append(float(row["tracking_rmse_m"]))
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.set_ylabel("RMSE (m)", color="white")
    ax.set_title("A-Z PID tracking RMSE by movement mode", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, axis="y", color="gray", alpha=0.25)
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.legend(facecolor="#222222", labelcolor="white")
    plt.tight_layout()
    plt.savefig(str(path), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A-Z PID letter rollouts in xy/xz modes.")
    parser.add_argument("--letters", default=string.ascii_uppercase,
                        help="Letters to run. Default: ABC...Z")
    parser.add_argument("--modes", nargs="+", default=["xy_fixed_z", "xz_z_motion"],
                        choices=tuple(MODE_TO_PLANE.keys()))
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to lightpaint-team-package/artifacts/alphabet_pid.")
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--width-m", type=float, default=0.9)
    parser.add_argument("--height-m", type=float, default=0.9)
    parser.add_argument("--max-waypoints", type=int, default=240)
    parser.add_argument("--smooth-window-m", type=float, default=0.08)
    parser.add_argument("--no-smooth-ref", action="store_true")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--snapshot-count", type=int, default=8)
    parser.add_argument("--video-frames", type=int, default=18)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--quick-video", action="store_true",
                        help="Use the old abbreviated flight.mp4 instead of the full square-style artifact family.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    letters = [ch.upper() for ch in args.letters if ch.isalpha()]
    out_dir = Path(args.output_dir) if args.output_dir else _PKG_ROOT / "artifacts" / "alphabet_pid"
    rows: list[dict[str, Any]] = []
    total = len(letters) * len(args.modes)
    idx = 0
    for mode in args.modes:
        for letter in letters:
            idx += 1
            print(f"[alphabet_pid] {idx}/{total} letter={letter} mode={mode}", flush=True)
            row = _run_one(letter, mode, args, out_dir)
            rows.append(row)
            print(
                f"[alphabet_pid] done letter={letter} mode={mode} "
                f"rmse={row['tracking_rmse_m']:.4f} max={row['tracking_max_m']:.4f} "
                f"crash={int(row['crash'])}",
                flush=True,
            )

    summary_csv = out_dir / "summary_metrics.csv"
    summary_plot = out_dir / "summary_rmse.png"
    manifest = out_dir / "manifest.json"
    _save_summary_csv(summary_csv, rows)
    _save_summary_plot(summary_plot, rows)
    with open(str(manifest), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary_csv": str(summary_csv), "summary_plot": str(summary_plot)}, f, indent=2)
    print(f"[alphabet_pid] summary CSV: {summary_csv}", flush=True)
    print(f"[alphabet_pid] summary plot: {summary_plot}", flush=True)
    print(f"[alphabet_pid] manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
