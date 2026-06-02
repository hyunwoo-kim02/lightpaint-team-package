from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "metric_charts"

MODES = ["M0", "M1", "M2"]
PHASES = [
    ("phaseB_zero", "PID Only", "#1f77b4"),
    ("phaseB_teacher", "PID + corner hints", "#ff7f0e"),
    ("phaseB_trained", "PID + Residual PPO RL", "#2ca02c"),
]


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(DATA / "eval" / "selected_phase_metrics.csv", encoding="utf-8-sig")
    numeric_cols = [
        "path_rmse_m",
        "corner_path_rmse_m",
        "painted_pixel_iou",
        "off_target_pixel_ratio",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def metric_row(metrics: pd.DataFrame, path_group: str, mode: str, phase: str) -> pd.Series:
    rows = metrics[
        (metrics["path_group"] == path_group)
        & (metrics["mode"] == mode)
        & (metrics["phase"] == phase)
    ]
    if rows.empty:
        raise KeyError((path_group, mode, phase))
    return rows.iloc[0]


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")


def plot_rmse(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    x = np.arange(len(MODES))
    width = 0.24
    offsets = [-width, 0.0, width]
    specs = [
        ("path_rmse_m", "Path RMSE"),
        ("corner_path_rmse_m", "Corner Path RMSE"),
    ]

    for ax, (metric_name, title) in zip(axes, specs):
        for offset, (phase, label, color) in zip(offsets, PHASES):
            values = [float(metric_row(metrics, "DG", mode, phase)[metric_name]) for mode in MODES]
            bars = ax.bar(x + offset, values, width=width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_title(title)
        ax.set_xticks(x, MODES)
        ax.set_xlabel("Disturbance mode")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE (m)")
    axes[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(1.08, 1.24))
    fig.subplots_adjust(top=0.80, wspace=0.04)
    save(fig, "rmse_bars")
    plt.close(fig)


def plot_iou_offtarget(metrics: pd.DataFrame) -> None:
    cases = [("DG", mode) for mode in MODES] + [("User Drawn", mode) for mode in MODES]
    labels = [f"{'Drawn' if path == 'User Drawn' else path}\n{mode}" for path, mode in cases]
    x = np.arange(len(cases))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    specs = [
        ("painted_pixel_iou", "Painted Pixel IoU", "higher is better"),
        ("off_target_pixel_ratio", "Off-target Pixel Ratio", "lower is better"),
    ]
    phases = [
        ("led_always_on", "LED Always ON", "#1f77b4"),
        ("phaseB_trained", "PID + Residual PPO RL", "#ff7f0e"),
    ]
    for ax, (metric_name, title, subtitle) in zip(axes, specs):
        for offset, (phase, label, color) in zip([-width / 2, width / 2], phases):
            values = [float(metric_row(metrics, path, mode, phase)[metric_name]) for path, mode in cases]
            bars = ax.bar(x + offset, values, width=width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_title(f"{title}\n({subtitle})")
        ax.set_xticks(x, labels)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Score / Ratio")
    axes[0].legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(1.05, 1.24))
    fig.subplots_adjust(top=0.78, wspace=0.22, bottom=0.20)
    save(fig, "iou_offtarget_bars")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150})
    metrics = load_metrics()
    plot_rmse(metrics)
    plot_iou_offtarget(metrics)


if __name__ == "__main__":
    main()
