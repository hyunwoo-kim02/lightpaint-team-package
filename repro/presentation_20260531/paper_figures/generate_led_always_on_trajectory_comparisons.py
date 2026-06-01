from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = PACKAGE / "data"
OUT = HERE / "generated_20260531"

MODES = ["M0", "M1", "M2"]

SUBTITLE = (
    "Ground Truth is rendered as the ground-truth light-painting result "
    "using its intended binary LED ON/OFF schedule."
)
FOOTNOTE = (
    "LED Always On shows the full flight path illuminated; Ground Truth and PPO use binary LED scheduling "
    "to hide transit strokes."
)


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    title: str
    color: str


COLUMNS = [
    ColumnSpec("reference", "Ground Truth", "#55d8f0"),
    ColumnSpec("always_on", "LED Always On", "#edf1f7"),
    ColumnSpec("ppo", "PID + Residual PPO RL", "#f4b612"),
]


def user_drawn_ground_truth_path() -> Path:
    return user_drawn_ppo_path("M0")


def user_drawn_ppo_path(mode: str) -> Path:
    return (
        DATA
        / "user_drawn_eval"
        / mode.lower()
        / "artifacts"
        / f"phase_b_user_drawn_path_{mode}_phaseB_trained_trajectory.csv"
    )


def dg_ground_truth_path() -> Path:
    return dg_ppo_path("M0")


def dg_ppo_path(mode: str) -> Path:
    return DATA / "eval" / mode.lower() / "artifacts" / f"phase_b_DG_{mode}_phaseB_trained_trajectory.csv"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def with_binary_brightness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["brightness"] = (df["brightness"].to_numpy(dtype=np.float32) > 0.5).astype(np.float32)
    return df


def segments_from_xy(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def add_glow(
    ax,
    segments: np.ndarray,
    color: str,
    *,
    core_width: float = 1.5,
    glow_scale: float = 1.0,
    highlight: bool = True,
) -> None:
    if len(segments) == 0:
        return
    for width, alpha in ((16.0 * glow_scale, 0.045), (10.0 * glow_scale, 0.12), (5.5 * glow_scale, 0.27)):
        ax.add_collection(
            LineCollection(
                segments,
                colors=[color],
                linewidths=width,
                alpha=alpha,
                capstyle="round",
                joinstyle="round",
            )
        )
    ax.add_collection(
        LineCollection(
            segments,
            colors=[color],
            linewidths=core_width,
            alpha=0.96,
            capstyle="round",
            joinstyle="round",
        )
    )
    if highlight:
        ax.add_collection(
            LineCollection(
                segments,
                colors=["#fff8df"],
                linewidths=max(core_width * 0.38, 0.6),
                alpha=0.42,
                capstyle="round",
                joinstyle="round",
            )
        )


def add_shadow(ax, segments: np.ndarray) -> None:
    ax.add_collection(
        LineCollection(
            segments,
            colors=["#dbe3ea"],
            linewidths=0.52,
            alpha=0.055,
            capstyle="round",
            joinstyle="round",
        )
    )


def add_reference_from_traj(ax, df: pd.DataFrame, color: str) -> None:
    segments = segments_from_xy(
        df["ref_x"].to_numpy(dtype=np.float32),
        df["ref_z"].to_numpy(dtype=np.float32),
    )
    on = df["brightness"].to_numpy(dtype=np.float32)[1:] > 0.5
    add_glow(ax, segments[on], color, core_width=1.65, glow_scale=1.03)


def add_full_reference_path(ax, df: pd.DataFrame, color: str) -> None:
    segments = segments_from_xy(
        df["ref_x"].to_numpy(dtype=np.float32),
        df["ref_z"].to_numpy(dtype=np.float32),
    )
    add_glow(ax, segments, color, core_width=1.35, glow_scale=0.98, highlight=True)


def add_led_scheduled_trajectory(ax, df: pd.DataFrame, color: str) -> None:
    segments = segments_from_xy(
        df["actual_x"].to_numpy(dtype=np.float32),
        df["actual_z"].to_numpy(dtype=np.float32),
    )
    add_shadow(ax, segments)
    on = df["brightness"].to_numpy(dtype=np.float32)[1:] > 0.5
    add_glow(ax, segments[on], color, core_width=1.35, glow_scale=1.0)


def style_axis(
    ax,
    *,
    row: int,
    col: int,
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    equal_aspect: bool,
) -> None:
    ax.set_facecolor("#070b12")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal" if equal_aspect else "auto", adjustable="box")
    ax.grid(True, color="#253143", alpha=0.55, linewidth=0.65)
    ax.tick_params(colors="#aeb6c4", labelsize=8.5, length=3.0, width=0.75)
    for spine in ax.spines.values():
        spine.set_color("#3d4758")
        spine.set_linewidth(0.8)
    if row < len(MODES) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("x (m)", color="#e5e7eb", fontsize=11, labelpad=7)
    if col != 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("z (m)", color="#e5e7eb", fontsize=11, fontweight="bold", labelpad=2)


def square_bounds(xs: list[np.ndarray], zs: list[np.ndarray], *, pad_ratio: float = 0.07) -> tuple[tuple[float, float], tuple[float, float]]:
    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    x_mid = 0.5 * (float(np.min(x_all)) + float(np.max(x_all)))
    z_mid = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    span = max(float(np.max(x_all) - np.min(x_all)), float(np.max(z_all) - np.min(z_all)))
    half = span * (0.5 + pad_ratio)
    return (x_mid - half, x_mid + half), (z_mid - half, z_mid + half)


def padded_bounds(xs: list[np.ndarray], zs: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    x_pad = max(float(np.max(x_all) - np.min(x_all)) * 0.075, 0.035)
    z_pad = max(float(np.max(z_all) - np.min(z_all)) * 0.075, 0.025)
    return (float(np.min(x_all)) - x_pad, float(np.max(x_all)) + x_pad), (
        float(np.min(z_all)) - z_pad,
        float(np.max(z_all)) + z_pad,
    )


def collect_user_drawn_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    gt = load_csv(user_drawn_ground_truth_path())
    xs.append(gt["ref_x"].to_numpy(dtype=np.float32))
    zs.append(gt["ref_z"].to_numpy(dtype=np.float32))
    for mode in MODES:
        df = load_csv(user_drawn_ppo_path(mode))
        xs.append(df["actual_x"].to_numpy(dtype=np.float32))
        zs.append(df["actual_z"].to_numpy(dtype=np.float32))
    return square_bounds(xs, zs, pad_ratio=0.075)


def collect_dg_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    gt = load_csv(dg_ground_truth_path())
    xs.append(gt["ref_x"].to_numpy(dtype=np.float32))
    zs.append(gt["ref_z"].to_numpy(dtype=np.float32))
    for mode in MODES:
        df = load_csv(dg_ppo_path(mode))
        xs.append(df["actual_x"].to_numpy(dtype=np.float32))
        zs.append(df["actual_z"].to_numpy(dtype=np.float32))
    return padded_bounds(xs, zs)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for target_dir in (OUT, HERE):
        fig.savefig(target_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(target_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())


def write_sources(stem: str, lines: list[str]) -> None:
    target = HERE / f"{stem}_sources.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(target, OUT / f"{stem}_sources.md")


def add_common_labels(fig: plt.Figure, title: str) -> None:
    fig.suptitle(title, color="#f8fafc", fontsize=24, fontweight="bold", y=0.992)
    fig.text(0.5, 0.925, SUBTITLE, color="#c6c9d1", fontsize=13.5, ha="center")
    handles = [
        Line2D([0], [0], color=col.color, lw=3.5, solid_capstyle="round", label=col.title)
        for col in COLUMNS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.066),
        ncol=3,
        frameon=True,
        facecolor="#111827",
        edgecolor="#e5e7eb",
        labelcolor="#e5e7eb",
        fontsize=11.5,
        handlelength=2.0,
        columnspacing=2.0,
        borderpad=0.55,
    )
    legend.get_frame().set_alpha(0.92)
    fig.text(0.5, 0.020, FOOTNOTE, color="#a7acb7", fontsize=9.8, ha="center")


def draw_user_drawn() -> None:
    xlim, zlim = collect_user_drawn_bounds()
    fig, axes = plt.subplots(len(MODES), len(COLUMNS), figsize=(12.7, 12.0), constrained_layout=False)
    fig.patch.set_facecolor("#030917")
    gt_df = with_binary_brightness(load_csv(user_drawn_ground_truth_path()))

    for r, mode in enumerate(MODES):
        ppo_df = with_binary_brightness(load_csv(user_drawn_ppo_path(mode)))
        for c, col in enumerate(COLUMNS):
            ax = axes[r, c]
            if col.key == "reference":
                add_reference_from_traj(ax, gt_df, col.color)
            elif col.key == "always_on":
                add_full_reference_path(ax, gt_df, col.color)
            else:
                add_led_scheduled_trajectory(ax, ppo_df, col.color)
            style_axis(ax, row=r, col=c, xlim=xlim, zlim=zlim, equal_aspect=True)
            if r == 0:
                ax.set_title(col.title, color="#f8fafc", fontsize=14.5, pad=10, fontweight="bold")
            if c == 0:
                ax.text(
                    -0.30,
                    0.5,
                    mode,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    color="#f8fafc",
                    fontsize=13.5,
                    fontweight="bold",
                )

    add_common_labels(fig, "Light-painting trajectory comparison: user-drawn path")
    fig.subplots_adjust(left=0.075, right=0.982, top=0.855, bottom=0.170, wspace=0.45, hspace=0.18)
    stem = "fig7_user_drawn_led_always_on_comparison"
    save(fig, stem)
    plt.close(fig)

    lines = [
        "# User Drawn LED Always On Trajectory Comparison Sources",
        "",
        f"Figure files: `{stem}.png`, `{stem}.pdf`",
        "",
        "| Panel | Source |",
        "| --- | --- |",
    ]
    for mode in MODES:
        lines.extend(
            [
                f"| {mode} / Ground Truth | `{user_drawn_ground_truth_path().relative_to(PACKAGE).as_posix()}` (`ref_x`, `ref_z`, canonical M0 binary `brightness > 0.5` ON/OFF mask) |",
                f"| {mode} / LED Always On | `{user_drawn_ground_truth_path().relative_to(PACKAGE).as_posix()}` (`ref_x`, `ref_z`, all segments illuminated) |",
                f"| {mode} / PID + Residual PPO RL | `{user_drawn_ppo_path(mode).relative_to(PACKAGE).as_posix()}` (binary `brightness > 0.5` ON/OFF mask) |",
            ]
        )
    write_sources(stem, lines)


def draw_dg() -> None:
    xlim, zlim = collect_dg_bounds()
    fig, axes = plt.subplots(len(MODES), len(COLUMNS), figsize=(14.7, 10.4), constrained_layout=False)
    fig.patch.set_facecolor("#030917")
    gt_df = with_binary_brightness(load_csv(dg_ground_truth_path()))

    for r, mode in enumerate(MODES):
        ppo_df = with_binary_brightness(load_csv(dg_ppo_path(mode)))
        for c, col in enumerate(COLUMNS):
            ax = axes[r, c]
            if col.key == "reference":
                add_reference_from_traj(ax, gt_df, col.color)
            elif col.key == "always_on":
                add_full_reference_path(ax, gt_df, col.color)
            else:
                add_led_scheduled_trajectory(ax, ppo_df, col.color)
            style_axis(ax, row=r, col=c, xlim=xlim, zlim=zlim, equal_aspect=False)
            if r == 0:
                ax.set_title(col.title, color="#f8fafc", fontsize=14.5, pad=10, fontweight="bold")
            if c == 0:
                ax.text(
                    -0.17,
                    0.5,
                    mode,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    color="#f8fafc",
                    fontsize=13.5,
                    fontweight="bold",
                )

    add_common_labels(fig, "Light-painting trajectory comparison: DG path")
    fig.subplots_adjust(left=0.058, right=0.982, top=0.850, bottom=0.170, wspace=0.16, hspace=0.34)
    stem = "fig8_dg_led_always_on_comparison"
    save(fig, stem)
    plt.close(fig)

    lines = [
        "# DG LED Always On Trajectory Comparison Sources",
        "",
        f"Figure files: `{stem}.png`, `{stem}.pdf`",
        "",
        "| Panel | Source |",
        "| --- | --- |",
    ]
    for mode in MODES:
        lines.extend(
            [
                f"| {mode} / Ground Truth | `{dg_ground_truth_path().relative_to(PACKAGE).as_posix()}` (`ref_x`, `ref_z`, canonical M0 binary `brightness > 0.5` ON/OFF mask) |",
                f"| {mode} / LED Always On | `{dg_ground_truth_path().relative_to(PACKAGE).as_posix()}` (`ref_x`, `ref_z`, all segments illuminated) |",
                f"| {mode} / PID + Residual PPO RL | `{dg_ppo_path(mode).relative_to(PACKAGE).as_posix()}` (binary `brightness > 0.5` ON/OFF mask) |",
            ]
        )
    write_sources(stem, lines)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "regular",
            "axes.titleweight": "bold",
            "figure.dpi": 150,
            "savefig.pad_inches": 0.08,
        }
    )
    draw_user_drawn()
    draw_dg()


if __name__ == "__main__":
    main()
