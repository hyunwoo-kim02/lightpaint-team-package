"""
visualize_flight.py - Drone-flight visualization utilities.
Purpose: Load trained policy, run 1 deterministic episode, save 3D flight PNG,
         per-frame snapshots, and mp4 to artifacts/visualization/.

CLI:
    python -m src.train.visualize_flight --model models/policy.zip --letter L
"""
import os
import sys
import argparse
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Resolve package root for both `python -m` and direct invocation.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
_REPO_ROOT = _PKG_ROOT.parent.parent
for _p in [str(_PKG_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# SB3 import deferred to main() so Phase A visualization avoids importing torch.
def _lazy_import_ppo():
    from stable_baselines3 import PPO
    return PPO


_WIND_ALIAS_TO_MODE = {"W0": "M0", "W1": "M1", "W2": "M2"}


def _normalize_wind_arg(wind: str) -> str:
    """Normalize W0-W2 CLI aliases to M0-M2 wind mode names."""
    wind_key = str(wind).upper()
    return _WIND_ALIAS_TO_MODE.get(wind_key, wind_key)


# ----------------------------------------------------------------------
# Phase A visualization without an SB3 dependency.
# ----------------------------------------------------------------------

def save_phase_a_visualization(
    pos_list: list,
    brightness_list: list,
    target_mask: np.ndarray,
    label: str,
    wind_mode: str,
    out_dir: Path,
    fps: int = 10,
    tracking_errs: Optional[list] = None,
    cumulative: Optional[np.ndarray] = None,
    ref_waypoints: Optional[np.ndarray] = None,
    ref_cumlen: Optional[np.ndarray] = None,
    ref_segment_led: Optional[np.ndarray] = None,
    ref_pos_list: Optional[list] = None,
    snapshots: Optional[list] = None,
    phase_prefix: str = "phase_a",
    phase_display_name: str = "Phase A",
    reference_speed_mps: Optional[float] = None,
    ctrl_dt_s: Optional[float] = None,
) -> dict:
    """
    Save the Phase A visualization family (consistent naming). The optional
    phase_prefix/phase_display_name keep old Phase A defaults while allowing
    verification runs to reuse the same format for later phases:
      - <phase_prefix>_<label>_<wind_mode>_3d.png
      - <phase_prefix>_<label>_<wind_mode>_frames/
      - <phase_prefix>_<label>_<wind_mode>.mp4
      - <phase_prefix>_<label>_<wind_mode>.gif
      - <phase_prefix>_<label>_<wind_mode>_summary.png

    The mp4 and gif are both produced from the same frame set and contain the
    full simulation start-to-end. Returns a dict with the actual artifact paths.
    """
    from src.env.lightpaint_geometry import (
        X_MIN, X_MAX, Z_MIN, Z_MAX,
        world_to_pixel, LED_STAMP_RADIUS_PX,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label  # case-preserved
    phase_prefix = str(phase_prefix).strip() or "phase_a"
    phase_display_name = str(phase_display_name).strip() or "Phase A"
    log_prefix = f"[{phase_prefix}_viz]"
    png_3d = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}_3d.png"
    frames_dir = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}_frames"
    mp4_path = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}.mp4"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.png"):
        old_frame.unlink()

    # ---------- 3D PNG ----------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#111111")
    if len(pos_list) >= 2:
        xs = [float(p[0]) for p in pos_list]
        ys = [float(p[1]) for p in pos_list]
        zs = [float(p[2]) for p in pos_list]
        # Color by brightness (red = on, cyan = off)
        colors = ["red" if b > 0.5 else "cyan" for b in brightness_list]
        for i in range(len(xs) - 1):
            ax.plot3D(
                [xs[i], xs[i + 1]],
                [ys[i], ys[i + 1]],
                [zs[i], zs[i + 1]],
                color=colors[i],
                linewidth=1.5,
                alpha=0.85,
            )
        ax.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=90, zorder=10, label="start")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="yellow", s=90, zorder=10, label="end")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.set_zlabel("z (m)", color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title(
        f"{phase_display_name} '{safe_label}' / {wind_mode} (red=LED ON, cyan=OFF)",
        color="white",
        fontsize=11,
    )
    if len(pos_list) >= 2:
        ax.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    plt.tight_layout()
    plt.savefig(str(png_3d), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"{log_prefix} 3D PNG 저장 위치: {png_3d}", flush=True)

    # ---------- Per-frame snapshots ----------
    has_drone_view = snapshots is not None and len(snapshots) > 0
    snapshot_steps = [s[0] for s in snapshots] if has_drone_view else []
    snapshot_imgs = [s[1] for s in snapshots] if has_drone_view else []

    def _latest_snapshot(step_idx: int):
        """Return the snapshot with the largest captured step <= step_idx."""
        if not has_drone_view:
            return None
        import bisect
        i = bisect.bisect_right(snapshot_steps, step_idx) - 1
        if i < 0:
            i = 0
        return snapshot_imgs[i]

    running_paint = np.zeros((64, 64), dtype=np.float32)
    n_frames = 0
    n_steps = len(pos_list)
    ref_pos_arr = None
    if ref_pos_list is not None:
        ref_pos_arr = np.asarray(ref_pos_list, dtype=np.float32).reshape(-1, 3)
        if len(ref_pos_arr) < n_steps:
            ref_pos_arr = None
    # Sub-sample for performance (cap ~120 frames)
    stride = max(1, n_steps // 120)
    n_cols = 3 if has_drone_view else 2
    for t in range(0, n_steps, stride):
        pos = pos_list[t]
        b = float(brightness_list[t])
        if b > 0.0:
            col, row = world_to_pixel(float(pos[0]), float(pos[2]))
            r = LED_STAMP_RADIUS_PX
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr = int(np.clip(row + dr, 0, 63))
                    cc = int(np.clip(col + dc, 0, 63))
                    running_paint[rr, cc] = max(float(running_paint[rr, cc]), b)

        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        fig.patch.set_facecolor("#111111")
        if n_cols == 2:
            ax_l, ax_r = axes
            ax_view = None
        else:
            ax_l, ax_r, ax_view = axes

        ax_l.set_facecolor("#111111")
        xs_t = [float(p[0]) for p in pos_list[: t + 1]]
        zs_t = [float(p[2]) for p in pos_list[: t + 1]]
        cs_t = ["red" if (b > 0.5) else "steelblue" for b in brightness_list[: t + 1]]
        if ref_pos_arr is not None:
            ax_l.plot(
                ref_pos_arr[: t + 1, 0],
                ref_pos_arr[: t + 1, 2],
                color="yellow",
                linewidth=1.0,
                alpha=0.8,
                label="commanded p_ref",
            )
        if xs_t:
            ax_l.scatter(xs_t, zs_t, c=cs_t, s=3, alpha=0.7)
            ax_l.scatter([xs_t[-1]], [zs_t[-1]], c="lime", s=40, zorder=5)
            if ref_pos_arr is not None:
                ref_now = ref_pos_arr[t]
                ax_l.scatter(
                    [float(ref_now[0])],
                    [float(ref_now[2])],
                    c="yellow",
                    marker="x",
                    s=70,
                    linewidths=2.0,
                    zorder=6,
                )
                ax_l.plot(
                    [xs_t[-1], float(ref_now[0])],
                    [zs_t[-1], float(ref_now[2])],
                    color="white",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.75,
                )
        ax_l.set_xlim(X_MIN, X_MAX)
        ax_l.set_ylim(Z_MIN, Z_MAX)
        ax_l.set_title(f"XZ t={t}", color="white", fontsize=8)
        ax_l.tick_params(colors="white", labelsize=6)
        for sp in ax_l.spines.values():
            sp.set_color("gray")
        ax_l.grid(True, alpha=0.2, color="gray")

        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :, 1] = (target_mask * 150).astype(np.uint8)
        rgb[:, :, 0] = (running_paint * 255).clip(0, 255).astype(np.uint8)
        ax_r.imshow(rgb, origin="upper")
        ax_r.set_title(f"Paint {int((running_paint > 0.3).sum())}px", color="white", fontsize=8)
        ax_r.axis("off")

        if ax_view is not None:
            ax_view.set_facecolor("#111111")
            shot = _latest_snapshot(t)
            if shot is not None:
                ax_view.imshow(shot, origin="upper")
            ax_view.set_title("PyBullet drone view", color="white", fontsize=8)
            ax_view.axis("off")

        plt.tight_layout(pad=0.5)
        frame_path = frames_dir / f"frame_{n_frames:04d}.png"
        plt.savefig(str(frame_path), dpi=80, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        n_frames += 1
    print(f"{log_prefix} frame PNG {n_frames}개 저장 위치: {frames_dir} (drone_view={has_drone_view})",
          flush=True)

    # ---------- mp4 mux (full simulation start-to-end) ----------
    written_mp4 = False
    written_gif = False
    gif_path = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}.gif"
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    try:
        import imageio
        import imageio_ffmpeg  # noqa: F401  (imports backend)
        sample = np.array(Image.open(str(frame_files[0])).convert("RGB"))
        h, w = sample.shape[:2]
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        writer = imageio.get_writer(
            str(mp4_path), fps=fps, codec="libx264", quality=6, macro_block_size=1
        )
        for fp in frame_files:
            frame = np.array(Image.open(str(fp)).convert("RGB"))
            writer.append_data(frame[:h_even, :w_even, :])
        writer.close()
        written_mp4 = True
        print(f"{log_prefix} mp4 저장 위치: {mp4_path}", flush=True)
    except Exception as e_mp4:
        print(f"{log_prefix} mp4 저장 실패: {e_mp4}", flush=True)

    # ---------- GIF mux (always, sibling of mp4) ----------
    try:
        import imageio
        # Down-sample large frame sets to keep GIF size manageable
        if len(frame_files) > 80:
            idx = np.linspace(0, len(frame_files) - 1, 80, dtype=int)
            sel = [frame_files[i] for i in idx]
        else:
            sel = frame_files
        arrs = [np.array(Image.open(str(fp)).convert("RGB")) for fp in sel]
        # duration in seconds per frame; loop=0 = infinite
        imageio.mimsave(str(gif_path), arrs, duration=max(0.05, 1.0 / fps), loop=0)
        written_gif = True
        print(f"{log_prefix} GIF 저장 위치: {gif_path} ({len(sel)} frames)", flush=True)
    except Exception as e_gif:
        print(f"{log_prefix} GIF 저장 실패: {e_gif}", flush=True)

    # ---------- Summary figure ----------
    summary_path = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}_summary.png"
    try:
        save_phase_a_summary(
            pos_list=pos_list,
            brightness_list=brightness_list,
            target_mask=target_mask,
            cumulative=cumulative if cumulative is not None else running_paint,
            label=safe_label,
            wind_mode=wind_mode,
            tracking_errs=tracking_errs,
            ref_waypoints=ref_waypoints,
            ref_segment_led=ref_segment_led,
            out_path=summary_path,
            phase_display_name=phase_display_name,
        )
        print(f"{log_prefix} summary figure 저장 위치: {summary_path}", flush=True)
    except Exception as e_sum:
        print(f"{log_prefix} summary figure 저장 실패: {e_sum}", flush=True)

    # ---------- Reference-path diagnostic figure ----------
    refpath_path = out_dir / f"{phase_prefix}_{safe_label}_{wind_mode}_refpath.png"
    if ref_waypoints is not None and ref_cumlen is not None:
        try:
            save_reference_path_diagnostic(
                target_mask=target_mask,
                ref_waypoints=ref_waypoints,
                ref_cumlen=ref_cumlen,
                ref_segment_led=ref_segment_led,
                pos_list=pos_list,
                label=safe_label,
                wind_mode=wind_mode,
                out_path=refpath_path,
                phase_display_name=phase_display_name,
                reference_speed_mps=reference_speed_mps,
                ctrl_dt_s=ctrl_dt_s,
            )
            print(f"{log_prefix} refpath figure 저장 위치: {refpath_path}", flush=True)
        except Exception as e_ref:
            print(f"{log_prefix} refpath figure 저장 실패: {e_ref}", flush=True)

    return {
        "png_3d": str(png_3d),
        "frames_dir": str(frames_dir),
        "mp4": str(mp4_path) if written_mp4 else "",
        "gif": str(gif_path) if written_gif else "",
        "summary": str(summary_path),
        "refpath": str(refpath_path) if ref_waypoints is not None else "",
        "n_frames": n_frames,
    }


def save_phase_a_summary(
    pos_list: list,
    brightness_list: list,
    target_mask: np.ndarray,
    cumulative: np.ndarray,
    label: str,
    wind_mode: str,
    tracking_errs: Optional[list],
    out_path: Path,
    ref_waypoints: Optional[np.ndarray] = None,
    ref_segment_led: Optional[np.ndarray] = None,
    phase_display_name: str = "Phase A",
) -> None:
    """Produce phase_a_<label>_<wind_mode>_summary.png as a single 4-panel figure."""
    from src.env.lightpaint_geometry import X_MIN, X_MAX, Z_MIN, Z_MAX

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.patch.set_facecolor("#111111")
    target_bin = (target_mask > 0.5).astype(np.float32)
    painted_bin = (cumulative > 0.3).astype(np.float32)
    target_px = max(int(target_bin.sum()), 1)
    on_target_paint = int((painted_bin * target_bin).sum())
    coverage = on_target_paint / target_px

    # Panel [0,0]: target mask
    ax = axes[0, 0]
    ax.set_facecolor("#111111")
    target_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    target_rgb[:, :, 1] = (target_mask * 200).astype(np.uint8)
    ax.imshow(target_rgb, origin="upper")
    ax.set_title(f"Target '{label}'", color="white", fontsize=11)
    ax.axis("off")

    # Panel [0,1]: painted result overlay
    ax = axes[0, 1]
    ax.set_facecolor("#111111")
    overlay_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay_rgb[:, :, 1] = (target_mask * 150).astype(np.uint8)  # green = target
    overlay_rgb[:, :, 0] = (cumulative * 255).clip(0, 255).astype(np.uint8)  # red = paint
    ax.imshow(overlay_rgb, origin="upper")
    ax.set_title(
        f"Painted (red on green)  coverage={coverage:.3f}",
        color="white", fontsize=11,
    )
    ax.axis("off")

    # Panel [1,0]: full XZ trajectory with LED color coding + reference overlay
    ax = axes[1, 0]
    ax.set_facecolor("#111111")
    # Reference path (cyan thin line) underneath actual trajectory
    if ref_waypoints is not None and len(ref_waypoints) >= 2:
        led = None if ref_segment_led is None else np.asarray(ref_segment_led, dtype=np.float32).reshape(-1)
        if led is not None and len(led) >= len(ref_waypoints) - 1:
            for i in range(len(ref_waypoints) - 1):
                color = "red" if float(led[i]) > 0.5 else "steelblue"
                alpha = 0.45 if float(led[i]) > 0.5 else 0.9
                ax.plot(
                    ref_waypoints[i:i + 2, 0],
                    ref_waypoints[i:i + 2, 2],
                    color=color,
                    linewidth=0.8,
                    alpha=alpha,
                )
            ax.plot([], [], color="red", linewidth=1.0, label="planned LED ON")
            ax.plot([], [], color="steelblue", linewidth=1.0, label="planned LED OFF")
        else:
            rx = ref_waypoints[:, 0]
            rz = ref_waypoints[:, 2]
            ax.plot(rx, rz, color="cyan", linewidth=0.8, alpha=0.6, label="reference path")
    if len(pos_list) >= 2:
        xs = [float(p[0]) for p in pos_list]
        zs = [float(p[2]) for p in pos_list]
        cs = ["red" if b > 0.5 else "steelblue" for b in brightness_list]
        ax.scatter(xs, zs, c=cs, s=4, alpha=0.7)
        ax.scatter([xs[0]], [zs[0]], c="lime", s=60, zorder=5, label="start")
        ax.scatter([xs[-1]], [zs[-1]], c="yellow", s=60, zorder=5, label="end")
        ax.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("z (m)", color="white")
    ax.set_title(
        f"XZ trajectory  ({len(pos_list)} steps; red=LED ON, cyan=ref)",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.grid(True, alpha=0.2, color="gray")

    # Panel [1,1]: tracking error curve (or stats text if not provided)
    ax = axes[1, 1]
    ax.set_facecolor("#111111")
    if tracking_errs is not None and len(tracking_errs) > 0:
        ax.plot(tracking_errs, color="orange", linewidth=1.0)
        ax.axhline(0.15, color="red", linestyle="--", linewidth=0.7, label="threshold 0.15 m")
        rmse = float(np.mean(tracking_errs))
        ax.set_title(f"tracking error (mean RMSE={rmse:.4f} m)", color="white", fontsize=10)
        ax.set_xlabel("step", color="white")
        ax.set_ylabel("‖p − p_ref‖₂ (m)", color="white")
        ax.legend(fontsize=8, facecolor="#222222", labelcolor="white")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("gray")
        ax.grid(True, alpha=0.2, color="gray")
    else:
        ax.axis("off")
        led_on_steps = sum(1 for b in brightness_list if b > 0.5)
        led_on_ratio = led_on_steps / max(len(brightness_list), 1)
        text = (
            f"label = {label!r}\n"
            f"wind_mode = {wind_mode}\n"
            f"steps = {len(pos_list)}\n"
            f"painted (binarized) = {int(painted_bin.sum())} px\n"
            f"target = {target_px} px\n"
            f"coverage = {coverage:.4f}\n"
            f"LED ON ratio = {led_on_ratio:.4f}"
        )
        ax.text(0.05, 0.95, text, color="white", fontsize=11,
                transform=ax.transAxes, verticalalignment="top",
                family="monospace")

    fig.suptitle(
        f"{phase_display_name} summary - label={label!r}  wind={wind_mode}",
        color="white", fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_reference_path_diagnostic(
    target_mask: np.ndarray,
    ref_waypoints: np.ndarray,
    ref_cumlen: np.ndarray,
    ref_segment_led: Optional[np.ndarray],
    pos_list: list,
    label: str,
    wind_mode: str,
    out_path: Path,
    phase_display_name: str = "Phase A",
    reference_speed_mps: Optional[float] = None,
    ctrl_dt_s: Optional[float] = None,
) -> None:
    """
    Diagnostic figure showing how the reference path was constructed and
    how the drone tracked it.

    3 panels:
      [0] Target mask (faded green) + reference waypoints colored by visit
          order (viridis: dark→start, bright→end). Drone trajectory overlaid
          in faint red dots.
      [1] arc-length cumulen vs waypoint index, verifies monotonic ordering.
      [2] arc-length traversed vs simulation step, shows V_REF=0.5 m/s ramp
          and final clamp at total_len.
    """
    from src.env.lightpaint_geometry import (
        X_MIN, X_MAX, Z_MIN, Z_MAX, V_REF, DT,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#111111")

    rx = ref_waypoints[:, 0]
    rz = ref_waypoints[:, 2]
    n = len(rx)

    # ----- Panel 0: target + ordered waypoints + drone path -----
    ax = axes[0]
    ax.set_facecolor("#111111")
    # Faded target mask in z-x extent
    ax.imshow(
        target_mask,
        extent=(X_MIN, X_MAX, Z_MIN, Z_MAX),
        origin="upper",
        cmap="Greens",
        alpha=0.35,
        aspect="auto",
    )
    # Ref waypoints colored by visit order
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, n))
    led = None if ref_segment_led is None else np.asarray(ref_segment_led, dtype=np.float32).reshape(-1)
    if led is not None and len(led) >= n - 1:
        for i in range(n - 1):
            seg_color = "red" if float(led[i]) > 0.5 else "steelblue"
            seg_alpha = 0.5 if float(led[i]) > 0.5 else 0.95
            ax.plot(rx[i:i + 2], rz[i:i + 2], color=seg_color, linewidth=1.0, alpha=seg_alpha, zorder=2)
        ax.plot([], [], color="red", linewidth=1.2, label="planned LED ON")
        ax.plot([], [], color="steelblue", linewidth=1.2, label="planned LED OFF")
    ax.scatter(rx, rz, c=colors, s=8, zorder=3)
    ax.scatter([rx[0]], [rz[0]], c="lime", s=90, marker="o", zorder=5, label="ref start")
    ax.scatter([rx[-1]], [rz[-1]], c="red", s=90, marker="X", zorder=5, label="ref end")
    # Drone actual path (faint red)
    if len(pos_list) >= 2:
        dx = [float(p[0]) for p in pos_list]
        dz = [float(p[2]) for p in pos_list]
        ax.plot(dx, dz, color="orange", linewidth=0.6, alpha=0.5, zorder=4, label="drone path")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.invert_yaxis()  # match imshow y direction (top of mask = high z)
    ax.invert_yaxis()  # double invert restores: top = high z (matplotlib default)
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("z (m)", color="white")
    ax.set_title(
        f"Reference path '{label}' - {n} waypoints (color = visit order)",
        color="white", fontsize=10,
    )
    ax.legend(fontsize=7, facecolor="#222222", labelcolor="white", loc="upper right")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.grid(True, alpha=0.2, color="gray")

    # ----- Panel 1: cumulative arc length vs waypoint index -----
    ax = axes[1]
    ax.set_facecolor("#111111")
    ax.plot(np.arange(n), ref_cumlen, color="cyan", linewidth=1.2)
    ax.scatter([0], [ref_cumlen[0]], c="lime", s=60, zorder=5)
    ax.scatter([n - 1], [ref_cumlen[-1]], c="red", s=60, marker="X", zorder=5)
    ax.set_xlabel("waypoint index", color="white")
    ax.set_ylabel("cumulative arc length (m)", color="white")
    ax.set_title(
        f"arc length per waypoint (total = {float(ref_cumlen[-1]):.3f} m)",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.grid(True, alpha=0.2, color="gray")

    # ----- Panel 2: arc-length progression vs sim step -----
    ax = axes[2]
    ax.set_facecolor("#111111")
    n_steps = len(pos_list)
    steps = np.arange(n_steps)
    ref_speed = float(reference_speed_mps) if reference_speed_mps is not None else float(V_REF)
    ctrl_dt = float(ctrl_dt_s) if ctrl_dt_s is not None else float(DT)
    s_target = steps * ctrl_dt * ref_speed
    total_len = float(ref_cumlen[-1])
    s_target_clamped = np.minimum(s_target, total_len)
    ax.plot(steps, s_target_clamped, color="orange", linewidth=1.0,
            label=f"s = step*dt*speed (speed={ref_speed:.3f} m/s)")
    ax.axhline(total_len, color="red", linestyle="--", linewidth=0.7,
               label=f"total length = {total_len:.3f} m")
    # Mark when reference clamps at endpoint
    clamp_step = int(np.ceil(total_len / max(ctrl_dt * ref_speed, 1e-9)))
    if 0 < clamp_step < n_steps:
        ax.axvline(clamp_step, color="cyan", linestyle=":", linewidth=0.6,
                   label=f"clamp @ step {clamp_step}")
    ax.set_xlabel("sim step", color="white")
    ax.set_ylabel("arc length s (m)", color="white")
    ax.set_title("reference progression over time", color="white", fontsize=10)
    ax.legend(fontsize=7, facecolor="#222222", labelcolor="white", loc="lower right")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("gray")
    ax.grid(True, alpha=0.2, color="gray")

    fig.suptitle(
        f"{phase_display_name} reference-path diagnostic - label={label!r}  wind={wind_mode}",
        color="white", fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


from src.env.lightpaint_geometry import X_MIN, X_MAX, Z_MIN, Z_MAX, world_to_pixel, LED_STAMP_RADIUS_PX

# Maximum steps per eval episode
MAX_EVAL_STEPS = 600
FRAME_FPS = 30


def run_eval_episode(
    model: "PPO",  # type: ignore[name-defined]
    env_fn: callable,
    seed: int = 0,
) -> dict:
    """
    Run a single deterministic episode and collect trajectory data.

    Returns dict with keys:
        pos_list      : list of [x, y, z] positions
        led_list      : list of bool (LED state per step)
        cumulative    : np.ndarray (64, 64) cumulative paint mask
        rewards       : list of floats
        target_mask   : np.ndarray (64, 64) target letter mask
    """
    env = env_fn()
    obs, _ = env.reset(seed=seed)

    pos_list = []
    led_list = []
    rewards = []
    cumulative = np.zeros((64, 64), dtype=np.float32)

    for t in range(MAX_EVAL_STEPS):
        # DummyVecEnv wraps obs in a batch dim; use model.predict on dict
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Unwrap info from Monitor/DummyVecEnv
        if isinstance(info, (list, tuple)):
            info = info[0] if len(info) > 0 else {}

        pos = info.get("pos", [0.0, 0.0, 1.5])
        led_on = bool(info.get("led_on", False))
        pos_list.append(list(pos))
        led_list.append(led_on)
        rewards.append(float(reward) if not isinstance(reward, (list, tuple)) else float(reward[0]))

        # Stamp paint
        if led_on:
            col, row = world_to_pixel(float(pos[0]), float(pos[2]))
            r = LED_STAMP_RADIUS_PX
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr = int(np.clip(row + dr, 0, 63))
                    cc = int(np.clip(col + dc, 0, 63))
                    cumulative[rr, cc] = 1.0

        if (isinstance(terminated, (list, tuple)) and terminated[0]) or \
           (isinstance(truncated, (list, tuple)) and truncated[0]) or \
           (isinstance(terminated, bool) and terminated) or \
           (isinstance(truncated, bool) and truncated):
            break

    # Get target mask from env
    target_mask = env.G_letter if hasattr(env, "G_letter") else np.zeros((64, 64), dtype=np.float32)
    env.close()

    return {
        "pos_list": pos_list,
        "led_list": led_list,
        "cumulative": cumulative,
        "rewards": rewards,
        "target_mask": target_mask,
    }


def save_3d_flight_png(
    pos_list: list,
    led_list: list,
    out_path: Path,
    letter: str,
) -> None:
    """
    Save 3D matplotlib flight trajectory PNG.

    Colored by LED state: red = LED ON, cyan = LED OFF.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("#111111")

    if len(pos_list) < 2:
        ax.text(0, 0, 1.5, "No trajectory data", color="orange")
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    xs = [p[0] for p in pos_list]
    ys = [p[1] for p in pos_list]
    zs = [p[2] for p in pos_list]
    # Color each segment by LED state
    colors = ["red" if l else "cyan" for l in led_list]

    for i in range(len(xs) - 1):
        ax.plot3D(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            [zs[i], zs[i + 1]],
            color=colors[i], linewidth=1.5, alpha=0.8,
        )

    # Mark start (green) and end (yellow)
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=80, zorder=10, label="start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="yellow", s=80, zorder=10, label="end")
    # Current drone marker
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="lime", s=60, zorder=11)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_xlabel("x (m)", color="white", fontsize=9)
    ax.set_ylabel("y (m)", color="white", fontsize=9)
    ax.set_zlabel("z (m)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title(
        f"RL Light Painting - Letter '{letter}' Flight Trajectory\n"
        f"(red=LED ON, cyan=LED OFF)  n_steps={len(pos_list)}",
        color="white", fontsize=11,
    )
    legend = ax.legend(fontsize=8, facecolor="#222222", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] 3D PNG 저장 위치: {out_path}", flush=True)


def save_frame_snapshots(
    pos_list: list,
    led_list: list,
    cumulative: np.ndarray,
    target_mask: np.ndarray,
    frames_dir: Path,
    fps_subsample: int = 1,
) -> int:
    """
    Save per-frame PNG snapshots of the flight.

    Each frame shows: XZ trajectory + current drone position + painted mask.
    Returns the number of frames saved.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_steps = len(pos_list)
    frame_count = 0
    running_paint = np.zeros((64, 64), dtype=np.float32)

    for t in range(n_steps):
        if t % fps_subsample != 0:
            continue

        pos = pos_list[t]
        led_on = led_list[t]
        if led_on:
            col, row = world_to_pixel(float(pos[0]), float(pos[2]))
            r = LED_STAMP_RADIUS_PX
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr = int(np.clip(row + dr, 0, 63))
                    cc = int(np.clip(col + dc, 0, 63))
                    running_paint[rr, cc] = 1.0

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.patch.set_facecolor("#111111")

        # Left panel: XZ trajectory up to t
        ax_l = axes[0]
        ax_l.set_facecolor("#111111")
        xs_t = [p[0] for p in pos_list[:t + 1]]
        zs_t = [p[2] for p in pos_list[:t + 1]]
        cs_t = ["red" if l else "steelblue" for l in led_list[:t + 1]]
        if xs_t:
            ax_l.scatter(xs_t, zs_t, c=cs_t, s=3, alpha=0.7)
            ax_l.scatter([xs_t[-1]], [zs_t[-1]], c="lime", s=40, zorder=5)
        ax_l.set_xlim(X_MIN, X_MAX)
        ax_l.set_ylim(Z_MIN, Z_MAX)
        ax_l.set_xlabel("x (m)", color="white", fontsize=7)
        ax_l.set_ylabel("z (m)", color="white", fontsize=7)
        ax_l.set_title(f"XZ trajectory t={t}", color="white", fontsize=8)
        ax_l.tick_params(colors="white", labelsize=6)
        for sp in ax_l.spines.values():
            sp.set_color("gray")
        ax_l.grid(True, alpha=0.2, color="gray")

        # Right panel: painted mask vs target
        ax_r = axes[1]
        ax_r.set_facecolor("#111111")
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :, 1] = (target_mask * 150).astype(np.uint8)    # green = target
        rgb[:, :, 0] = (running_paint * 255).astype(np.uint8)  # red = painted
        ax_r.imshow(rgb, origin="upper")
        painted_px = int(running_paint.sum())
        ax_r.set_title(f"Paint mask  painted={painted_px}px", color="white", fontsize=8)
        ax_r.axis("off")

        plt.tight_layout(pad=0.5)
        frame_path = frames_dir / f"frame_{frame_count:04d}.png"
        plt.savefig(str(frame_path), dpi=80, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        frame_count += 1

    print(f"[viz] frame snapshot {frame_count}개 저장 위치: {frames_dir}", flush=True)
    return frame_count


def save_mp4(frames_dir: Path, out_path: Path, fps: int = 10) -> bool:
    """
    Mux frame PNGs into mp4 using imageio-ffmpeg.

    Falls back to GIF if ffmpeg is unavailable. Returns True if mp4 saved.
    """
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_paths) == 0:
        print("[viz] mp4 생성에 사용할 frame이 없습니다.", flush=True)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try imageio with ffmpeg
    try:
        import imageio
        import imageio_ffmpeg
        # Load a sample frame to check dimensions; libx264 requires even width+height
        sample = np.array(Image.open(str(frame_paths[0])).convert("RGB"))
        h, w = sample.shape[:2]
        # Round to nearest even number (libx264 requirement)
        h_even = h if h % 2 == 0 else h - 1
        w_even = w if w % 2 == 0 else w - 1
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                    quality=6, macro_block_size=1)
        for frame_path in frame_paths:
            frame = np.array(Image.open(str(frame_path)).convert("RGB"))
            # Crop to even dimensions
            frame = frame[:h_even, :w_even, :]
            writer.append_data(frame)
        writer.close()
        duration = len(frame_paths) / fps
        print(f"[viz] mp4 저장 위치: {out_path} ({len(frame_paths)} frames, {duration:.1f}s)",
              flush=True)
        return True
    except Exception as e_mp4:
        print(f"[viz] mp4 저장 실패: {e_mp4}. GIF 저장을 시도합니다.", flush=True)

    # Save GIF when mp4 encoding is unavailable.
    try:
        import imageio
        gif_path = out_path.with_suffix(".gif")
        frames_arr = [np.array(Image.open(str(p)).convert("RGB")) for p in frame_paths[:60]]
        imageio.mimsave(str(gif_path), frames_arr, fps=fps, loop=0)
        # Replace out_path with .gif content (rename with overwrite)
        if out_path.exists():
            out_path.unlink()
        gif_path.rename(out_path)
        print(f"[viz] GIF 저장 위치: {out_path} ({len(frames_arr)} frames)", flush=True)
        return True
    except Exception as e_gif:
        print(f"[viz] GIF 저장 실패: {e_gif}", flush=True)
        return False


def main(args: argparse.Namespace) -> None:
    """Main visualization entry point."""
    letter = args.letter.upper()
    wind_mode = _normalize_wind_arg(args.wind)
    wind = wind_mode
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path relative to PKG_ROOT if not absolute
    if not model_path.is_absolute():
        model_path = _PKG_ROOT / model_path
    if not model_path.is_file():
        # Try without extension
        alt = model_path.with_suffix(".zip")
        if alt.is_file():
            model_path = alt
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    print(f"[viz] 모델을 불러옵니다: {model_path}", flush=True)
    print(f"[viz] 글자={letter}, 외란={wind}", flush=True)

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.env.light_paint_aviary_standalone import LightPaintAviaryW1

    # Build single-env DummyVecEnv (NOT SubprocVecEnv for eval)
    def _make_eval_env() -> Monitor:
        env = LightPaintAviaryW1(letter=letter, wind_mode=wind_mode, gui=False,
                                  init_box_size=0.0, led_always_on=False)
        return Monitor(env)

    eval_env = DummyVecEnv([_make_eval_env])
    PPO = _lazy_import_ppo()
    model = PPO.load(str(model_path), env=eval_env)
    print("[viz] 모델 로드 완료", flush=True)

    # Run eval episode
    print("[viz] deterministic rollout을 1 episode 실행합니다.", flush=True)
    data = run_eval_episode(model, _make_eval_env, seed=42)
    pos_list = data["pos_list"]
    led_list = data["led_list"]
    cumulative = data["cumulative"]
    target_mask = data["target_mask"]
    print(f"[viz] episode 결과: {len(pos_list)} steps, "
          f"LED on {sum(led_list)}/{len(led_list)} steps, "
          f"painted {int(cumulative.sum())}px", flush=True)

    # Save 3D flight PNG
    png_3d = out_dir / "rl_lp_w1_flight_3d.png"
    save_3d_flight_png(pos_list, led_list, png_3d, letter)

    # Save frame snapshots
    frames_dir = out_dir / "frames"
    n_frames = save_frame_snapshots(
        pos_list, led_list, cumulative, target_mask,
        frames_dir, fps_subsample=1
    )

    # Save mp4, or GIF when mp4 encoding is unavailable.
    mp4_path = out_dir / "rl_lp_w1_flight.mp4"
    mp4_ok = save_mp4(frames_dir, mp4_path, fps=min(FRAME_FPS, max(1, n_frames // 10)))

    # Summary
    print(f"\n[viz] 완료:", flush=True)
    print(f"  3D PNG:   {png_3d}", flush=True)
    print(f"  Frames:   {frames_dir} ({n_frames} files)", flush=True)
    print(f"  mp4:      {mp4_path} (ok={mp4_ok})", flush=True)

    eval_env.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for visualize_flight."""
    parser = argparse.ArgumentParser(
        description="Visualize trained LightPaint policy flight."
    )
    parser.add_argument("--model", default="models/rl_lp_w1.zip",
                        help="Path to trained PPO model (.zip).")
    parser.add_argument("--letter", default="L",
                        help="Letter to visualize (default: L).")
    parser.add_argument("--wind", default="M0",
                        help="Wind mode M0/M1/M2, or W0/W1/W2 alias (default: M0).")
    parser.add_argument("--out-dir", default=str(_PKG_ROOT / "artifacts" / "visualization"),
                        help="Output directory for visualization artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    sys.exit(0)
