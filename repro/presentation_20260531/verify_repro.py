from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


EXPECTED_WEIGHT_HASHES = {
    "M0.zip": "B026F5F1FEFFE8005383E3F327596B9082A5E5A354438FDF3C572244420E1D4A",
    "M1.zip": "E7B8BE34E61750662298208AFCA3588447509AC3490BB46BB1852BDAF673565C",
    "M2.zip": "644088D1638675E97861FE42B8DC336C7005E8B1E8ADDFDAA2CEBB3BB4DDE9FF",
}

EXPECTED_METRICS = {
    ("DG", "M0", "phaseB_zero"): {"path_rmse_m": 0.029616402, "corner_path_rmse_m": 0.035241097},
    ("DG", "M0", "phaseB_teacher"): {"path_rmse_m": 0.016340187, "corner_path_rmse_m": 0.013342242},
    ("DG", "M0", "phaseB_trained"): {
        "path_rmse_m": 0.018883608,
        "corner_path_rmse_m": 0.017202072,
        "painted_pixel_iou": 0.855856,
        "off_target_pixel_ratio": 0.064039,
    },
    ("DG", "M1", "phaseB_trained"): {
        "path_rmse_m": 0.018132929,
        "corner_path_rmse_m": 0.023274474,
        "painted_pixel_iou": 0.828947,
        "off_target_pixel_ratio": 0.091346,
    },
    ("DG", "M2", "phaseB_trained"): {
        "path_rmse_m": 0.024412898,
        "corner_path_rmse_m": 0.020273641,
        "painted_pixel_iou": 0.816594,
        "off_target_pixel_ratio": 0.096618,
    },
    ("User Drawn", "M0", "phaseB_trained"): {
        "painted_pixel_iou": 0.807927,
        "off_target_pixel_ratio": 0.107744,
    },
    ("User Drawn", "M1", "phaseB_trained"): {
        "painted_pixel_iou": 0.835913,
        "off_target_pixel_ratio": 0.090909,
    },
    ("User Drawn", "M2", "phaseB_trained"): {
        "painted_pixel_iou": 0.800613,
        "off_target_pixel_ratio": 0.103093,
    },
    ("DG", "M0", "led_always_on"): {"painted_pixel_iou": 0.730496, "off_target_pixel_ratio": 0.231343},
    ("DG", "M1", "led_always_on"): {"painted_pixel_iou": 0.710145, "off_target_pixel_ratio": 0.254753},
    ("DG", "M2", "led_always_on"): {"painted_pixel_iou": 0.700730, "off_target_pixel_ratio": 0.252918},
}

REQUIRED_FILES = [
    "data/eval/selected_phase_metrics.csv",
    "data/eval/final_trained_metrics.csv",
    "data/eval/pid_teacher_loaded_trained_metrics.csv",
    "data/waypoints/DG_reference.json",
    "data/waypoints/user_drawn_path.json",
]

GENERATED_FILES = [
    "paper_figures/generated_20260531/fig8_dg_trajectory_comparison.png",
    "paper_figures/generated_20260531/fig7_user_drawn_trajectory_comparison.png",
    "paper_figures/generated_20260531/fig8_dg_led_always_on_comparison.png",
    "paper_figures/generated_20260531/fig7_user_drawn_led_always_on_comparison.png",
    "paper_figures/metric_charts/rmse_bars.png",
    "paper_figures/metric_charts/iou_offtarget_bars.png",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.exists():
        raise SystemExit(f"missing required file: {relative_path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise SystemExit(f"empty required file: {relative_path}")
    return path


def load_metrics() -> dict[tuple[str, str, str], dict[str, str]]:
    path = require_file("data/eval/selected_phase_metrics.csv")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {(row["path_group"], row["mode"], row["phase"]): row for row in rows}


def check_required_files() -> None:
    for relative in REQUIRED_FILES:
        require_file(relative)
    for group in ("eval", "user_drawn_eval"):
        token = "DG" if group == "eval" else "user_drawn_path"
        for mode in ("m0", "m1", "m2"):
            mode_upper = mode.upper()
            require_file(f"data/{group}/{mode}/summary.json")
            for phase in ("phaseB_zero", "phaseB_teacher", "phaseB_trained"):
                prefix = "phase_b_" + token + "_" + mode_upper + "_" + phase
                require_file(f"data/{group}/{mode}/artifacts/{prefix}_trajectory.csv")
                require_file(f"data/{group}/{mode}/artifacts/{prefix}_corners.csv")
                require_file(f"data/{group}/{mode}/artifacts/{prefix}_corner_diagnostics.csv")


def check_weights() -> None:
    for filename, expected in EXPECTED_WEIGHT_HASHES.items():
        actual = sha256(require_file(f"weights/{filename}"))
        if actual != expected:
            raise SystemExit(f"weight hash mismatch for {filename}: {actual} != {expected}")
    manifest = json.loads(require_file("weights/manifest.json").read_text(encoding="utf-8"))
    if len(manifest.get("weights", [])) != 3:
        raise SystemExit("weights/manifest.json must list three weights")


def check_metrics() -> None:
    rows = load_metrics()
    for key, expected_values in EXPECTED_METRICS.items():
        if key not in rows:
            raise SystemExit(f"missing metric row: {key}")
        row = rows[key]
        for metric_name, expected in expected_values.items():
            actual = float(row[metric_name])
            if abs(actual - expected) > 5e-7:
                raise SystemExit(
                    f"metric mismatch {key} {metric_name}: {actual:.9f} != {expected:.9f}"
                )


def check_generated() -> None:
    for relative in GENERATED_FILES:
        require_file(relative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-generated", action="store_true")
    args = parser.parse_args()

    check_required_files()
    check_weights()
    check_metrics()
    if args.check_generated:
        check_generated()
    print("repro package verification passed")


if __name__ == "__main__":
    main()
