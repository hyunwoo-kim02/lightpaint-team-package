"""Export a reference snapshot for tools/corner_hint_editor.html."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.train.reference_factory import add_reference_args, build_reference_from_args


def _as_float_list(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


def _plane_axes(plane: str) -> tuple[int, int, str, str]:
    if plane == "xy":
        return 0, 1, "x (m)", "y (m)"
    return 0, 2, "x (m)", "z (m)"


def _save_preview(snapshot: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts = np.asarray(snapshot["waypoints"], dtype=np.float32)
    led = np.asarray(snapshot.get("segment_led", []), dtype=np.float32)
    x_axis, y_axis, x_label, y_label = _plane_axes(str(snapshot.get("plane", "xz")))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0b1220")

    for idx in range(len(pts) - 1):
        is_on = idx < len(led) and float(led[idx]) > 0.5
        ax.plot(
            pts[idx:idx + 2, x_axis],
            pts[idx:idx + 2, y_axis],
            color="#34d399" if is_on else "#64748b",
            linestyle="-" if is_on else "--",
            linewidth=2.2 if is_on else 1.4,
            alpha=0.95,
        )

    if len(pts):
        ax.scatter(
            pts[:, x_axis],
            pts[:, y_axis],
            s=10,
            c=np.linspace(0.0, 1.0, len(pts)),
            cmap="viridis",
            alpha=0.8,
        )

    for order, corner in enumerate(snapshot.get("auto_corners", []), start=1):
        point = np.asarray(corner["point"], dtype=np.float32)
        ax.scatter(
            [point[x_axis]],
            [point[y_axis]],
            s=90,
            c="#ef4444",
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        ax.text(
            float(point[x_axis]),
            float(point[y_axis]) + 0.025,
            str(order),
            color="white",
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=6,
        )

    ax.set_title(
        f"{snapshot.get('name', 'reference')} | {len(pts)} waypoints | "
        f"{len(snapshot.get('auto_corners', []))} corners",
        color="white",
        pad=12,
    )
    ax.set_xlabel(x_label, color="#d1d5db")
    ax.set_ylabel(y_label, color="#d1d5db")
    ax.tick_params(colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.grid(True, color="#334155", alpha=0.45, linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#34d399", lw=2.2, label="LED on path"),
            Line2D([0], [0], color="#64748b", lw=1.4, ls="--", label="LED off connector"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#ef4444",
                markeredgecolor="white",
                markersize=7,
                label="auto corner",
            ),
        ],
        loc="best",
        facecolor="#111827",
        edgecolor="#475569",
        labelcolor="white",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_args(parser)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=Path("data/corner_hints/DG_reference.json"))
    parser.add_argument(
        "--preview-output",
        type=Path,
        default=None,
        help="Optional PNG path for an end-to-end reference preview.",
    )
    args = parser.parse_args()

    built = build_reference_from_args(args)
    ref = built.reference
    out = {
        "schema_version": 1,
        "source": "tools/export_corner_reference.py",
        "metadata": built.metadata,
        "name": ref.name,
        "plane": ref.plane,
        "speed": float(ref.speed),
        "length_m": float(ref.length),
        "duration_s": float(ref.duration),
        "waypoints": [[float(v) for v in row] for row in ref.waypoints],
        "segment_led": _as_float_list(ref.segment_led),
        "auto_corner_distance_hints_m": _as_float_list(ref.cumlen[ref.corner_indices]),
        "auto_corners": [
            {
                "s_m": float(ref.cumlen[int(idx)]),
                "t_s": float(ref.waypoint_times[int(idx)]),
                "point": [float(v) for v in ref.waypoints[int(idx)]],
                "sharpness": float(sharpness),
            }
            for idx, sharpness in zip(ref.corner_indices, ref.corner_sharpness)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(args.output)
    if args.preview_output is not None:
        _save_preview(out, args.preview_output)
        print(args.preview_output)


if __name__ == "__main__":
    main()
