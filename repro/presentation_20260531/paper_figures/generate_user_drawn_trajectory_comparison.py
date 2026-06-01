from __future__ import annotations

import json
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

TITLE = "Light-painting trajectory comparison: user-drawn path"
SUBTITLE = "Only LED-ON trajectory segments are rendered as thick glowing traces on a dark long-exposure background"
FOOTNOTE = (
    "LED-OFF portions are intentionally hidden except for a very faint path shadow; "
    "the visible strokes approximate a camera long-exposure light-painting result."
)

MODES = ["M0", "M1", "M2"]


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    title: str
    color: str
    source: str


COLUMNS = [
    ColumnSpec("reference", "Reference", "#22d3ee", "data/waypoints/user_drawn_path.json"),
    ColumnSpec("phaseB_zero", "PID only", "#e7e7e3", "data/user_drawn_eval/{mode}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_zero_trajectory.csv"),
    ColumnSpec("phaseB_teacher", "PID + corner hints", "#65f3dc", "data/user_drawn_eval/{mode}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_teacher_trajectory.csv"),
    ColumnSpec("phaseB_trained", "PID + Residual PPO RL", "#f6b400", "data/user_drawn_eval/{mode}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_trained_trajectory.csv"),
]


def normalized_to_world(point: list[float], ref: dict) -> tuple[float, float]:
    x = float(ref["center_x"]) + (float(point[0]) - 0.5) * float(ref["width_m"])
    z = float(ref["center_z"]) + (0.5 - float(point[1])) * float(ref["height_m"])
    return x, z


def load_reference_strokes() -> list[np.ndarray]:
    ref_path = DATA / "waypoints" / "user_drawn_path.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    strokes: list[np.ndarray] = []
    for stroke in ref.get("strokes", []):
        pts = [normalized_to_world(point, ref) for point in stroke.get("points", [])]
        if pts:
            strokes.append(np.asarray(pts, dtype=np.float32))
    return strokes


def trajectory_path(mode: str, phase: str) -> Path:
    return DATA / "user_drawn_eval" / mode.lower() / "artifacts" / f"phase_b_user_drawn_path_{mode}_{phase}_trajectory.csv"


def load_traj(mode: str, phase: str) -> pd.DataFrame:
    return pd.read_csv(trajectory_path(mode, phase))


def segments_from_xy(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def add_glow(ax, segments: np.ndarray, color: str, *, core_width: float = 1.55, glow_scale: float = 1.0) -> None:
    if len(segments) == 0:
        return
    for width, alpha in ((14.0 * glow_scale, 0.055), (8.0 * glow_scale, 0.16), (4.0 * glow_scale, 0.34)):
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
            alpha=0.98,
            capstyle="round",
            joinstyle="round",
        )
    )
    ax.add_collection(
        LineCollection(
            segments,
            colors=["#fff8df"],
            linewidths=max(core_width * 0.45, 0.7),
            alpha=0.55,
            capstyle="round",
            joinstyle="round",
        )
    )


def add_reference_panel(ax, strokes: list[np.ndarray], color: str) -> None:
    for stroke in strokes:
        add_glow(ax, segments_from_xy(stroke[:, 0], stroke[:, 1]), color, core_width=1.9, glow_scale=1.08)


def add_trajectory_panel(ax, df: pd.DataFrame, color: str) -> None:
    x = df["actual_x"].to_numpy(dtype=np.float32)
    z = df["actual_z"].to_numpy(dtype=np.float32)
    shadow = segments_from_xy(x, z)
    ax.add_collection(
        LineCollection(
            shadow,
            colors=["#dbe3ea"],
            linewidths=0.55,
            alpha=0.075,
            capstyle="round",
            joinstyle="round",
        )
    )
    on = df["brightness"].to_numpy(dtype=np.float32)[1:] > 0.5
    add_glow(ax, shadow[on], color, core_width=1.45, glow_scale=1.0)


def style_axis(ax, *, row: int, col: int, xlim: tuple[float, float], zlim: tuple[float, float]) -> None:
    ax.set_facecolor("#070b12")
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal", adjustable="box")
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


def collect_bounds(strokes: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[np.ndarray] = [stroke[:, 0] for stroke in strokes]
    zs: list[np.ndarray] = [stroke[:, 1] for stroke in strokes]
    for mode in MODES:
        for phase in ("phaseB_zero", "phaseB_teacher", "phaseB_trained"):
            df = load_traj(mode, phase)
            xs.append(df["actual_x"].to_numpy(dtype=np.float32))
            zs.append(df["actual_z"].to_numpy(dtype=np.float32))
    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    x_mid = 0.5 * (float(np.min(x_all)) + float(np.max(x_all)))
    z_mid = 0.5 * (float(np.min(z_all)) + float(np.max(z_all)))
    span = max(float(np.max(x_all) - np.min(x_all)), float(np.max(z_all) - np.min(z_all)))
    half = span * 0.56
    return (x_mid - half, x_mid + half), (z_mid - half, z_mid + half)


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for target_dir in (OUT, HERE):
        fig.savefig(target_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(target_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=fig.get_facecolor())


def write_sources(stem: str) -> None:
    lines = [
        "# User Drawn Trajectory Comparison Sources",
        "",
        f"Figure files: `{stem}.png`, `{stem}.pdf`",
        "",
        "| Panel | Source |",
        "| --- | --- |",
        "| Reference | `data/waypoints/user_drawn_path.json` |",
    ]
    for mode in MODES:
        lines.extend(
            [
                f"| {mode} / PID only | `data/user_drawn_eval/{mode.lower()}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_zero_trajectory.csv` |",
                f"| {mode} / PID + corner hints | `data/user_drawn_eval/{mode.lower()}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_teacher_trajectory.csv` |",
                f"| {mode} / PID + Residual PPO RL | `data/user_drawn_eval/{mode.lower()}/artifacts/phase_b_user_drawn_path_{mode}_phaseB_trained_trajectory.csv` |",
            ]
        )
    (HERE / f"{stem}_sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(HERE / f"{stem}_sources.md", OUT / f"{stem}_sources.md")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "regular",
            "axes.titleweight": "bold",
            "figure.dpi": 150,
        }
    )
    strokes = load_reference_strokes()
    xlim, zlim = collect_bounds(strokes)
    fig, axes = plt.subplots(len(MODES), len(COLUMNS), figsize=(14.7, 11.9), constrained_layout=False)
    fig.patch.set_facecolor("#030917")

    for r, mode in enumerate(MODES):
        for c, col in enumerate(COLUMNS):
            ax = axes[r, c]
            if col.key == "reference":
                add_reference_panel(ax, strokes, col.color)
            else:
                add_trajectory_panel(ax, load_traj(mode, col.key), col.color)
            style_axis(ax, row=r, col=c, xlim=xlim, zlim=zlim)
            if r == 0:
                ax.set_title(col.title, color="#f8fafc", fontsize=13.5, pad=9, fontweight="bold")
            if c == 0:
                ax.text(
                    -0.27,
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

    fig.suptitle(TITLE, color="#f8fafc", fontsize=22, fontweight="bold", y=0.977)
    fig.text(0.5, 0.94, SUBTITLE, color="#c6c9d1", fontsize=13.5, ha="center")

    legend_handles = [
        Line2D([0], [0], color=col.color, lw=3.0, solid_capstyle="round", label=col.title)
        for col in COLUMNS
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.070),
        ncol=4,
        frameon=True,
        facecolor="#111827",
        edgecolor="#e5e7eb",
        labelcolor="#e5e7eb",
        fontsize=11,
        handlelength=2.0,
        columnspacing=1.9,
        borderpad=0.55,
    )
    legend.get_frame().set_alpha(0.92)
    fig.text(0.5, 0.020, FOOTNOTE, color="#a7acb7", fontsize=9.8, ha="center")
    fig.subplots_adjust(left=0.065, right=0.982, top=0.855, bottom=0.175, wspace=0.34, hspace=0.18)

    stem = "fig7_user_drawn_trajectory_comparison"
    save(fig, stem)
    plt.close(fig)
    write_sources(stem)


if __name__ == "__main__":
    main()
