"""Batch runner for the documented final LightPaint RL goal.

This script orchestrates full-matrix Phase B training/evaluation runs.  It does
not relax metrics or replace PyBullet evaluation; it only schedules canonical
`src.train.train_phase_b` invocations and aggregates their summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from src.train.train_phase_b_m0_corner import _final_goal_criteria as _compute_final_goal_criteria

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent

DEFAULT_LETTER_TRAJECTORIES = ("DG",)
DEFAULT_TRAJECTORIES = (*DEFAULT_LETTER_TRAJECTORIES, "drawn")
DEFAULT_WINDS = ("M0", "M1", "M2")
DEFAULT_SEEDS = (7, 11, 17)
FINAL_SEEDS = (7, 11, 17, 23, 29)
CANONICAL_DRAWN_PATH = "data/drawn_paths/user/user_drawn_path.json"
CANONICAL_DRAWN_SHA256 = "0f02e7c4b875bcc204f053e02fed82f85711ee24c9b559d06da0d5f764e74b98"
DEFAULT_TRAINED_ACTION_FILTER = "corner_tangent_decel"
PROFILE_DEFAULTS = {
    "sanity": {
        "trajectories": "DG,drawn",
        "wind_modes": "M0",
        "seeds": "7",
        "total_timesteps": 20_000,
    },
    "letters": {
        "trajectories": ",".join(DEFAULT_LETTER_TRAJECTORIES),
        "wind_modes": ",".join(DEFAULT_WINDS),
        "seeds": ",".join(str(seed) for seed in DEFAULT_SEEDS),
        "total_timesteps": 100_000,
    },
    "standard": {
        "trajectories": ",".join(DEFAULT_TRAJECTORIES),
        "wind_modes": ",".join(DEFAULT_WINDS),
        "seeds": ",".join(str(seed) for seed in DEFAULT_SEEDS),
        "total_timesteps": 100_000,
    },
    "final": {
        "trajectories": ",".join(DEFAULT_TRAJECTORIES),
        "wind_modes": ",".join(DEFAULT_WINDS),
        "seeds": ",".join(str(seed) for seed in FINAL_SEEDS),
        "total_timesteps": 300_000,
    },
}
AGGREGATE_METRICS = (
    "painting_iou",
    "painting_precision",
    "painting_recall",
    "painting_dice",
    "off_target_ratio",
    "path_rmse_m",
    "path_max_m",
    "corner_path_rmse_m",
    "corner_speed_ratio",
    "corner_speed_vs_straight_ratio",
    "corner_overshoot_m",
    "led_precision",
    "led_recall",
    "led_flicker_rate",
    "action_norm_mean",
    "action_norm_max",
    "action_rate_mean",
    "action_rate_max",
    "rpm_saturation_ratio",
    "crash_rate",
    "out_of_bounds_rate",
)
ROBUST_IOU_REL_MIN = 0.80
ROBUST_PATH_RMSE_REL_MAX = 1.30
REQUIRED_RUNTIME_DEPENDENCIES = {
    "pybullet": "pybullet",
    "pybullet_data": "pybullet_data",
    "gym_pybullet_drones": "gym-pybullet-drones",
    "pkg_resources": "setuptools<81",
    "stable_baselines3": "stable-baselines3",
    "torch": "torch",
}


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "package_root": str(_PKG_ROOT),
        "argv": list(sys.argv),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_runtime_dependencies() -> list[str]:
    return [
        package
        for module, package in REQUIRED_RUNTIME_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]


def _dependency_error_message(missing: list[str]) -> str:
    return (
        "PyBullet batch 학습 실행에 필요한 의존성이 없습니다: "
        + ", ".join(missing)
        + f". 현재 Python: {sys.executable}. 프로젝트 환경에서 `pip install -r requirements.txt`를 실행하거나 "
        + "PyBullet 의존성이 설치된 venv의 python으로 batch를 실행하세요."
    )


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _split_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _split_csv(raw)]


def _resolve_package_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = _PKG_ROOT / path
    return path


def _package_relative(path: Path) -> str | None:
    try:
        return path.resolve(strict=False).relative_to(_PKG_ROOT.resolve(strict=False)).as_posix()
    except ValueError:
        return None


def _drawn_input_metadata(args: argparse.Namespace, trajectories: list[str]) -> dict[str, Any] | None:
    if "drawn" not in trajectories:
        return None
    path = _resolve_package_path(args.drawn_path)
    metadata: dict[str, Any] = {
        "input_arg": str(args.drawn_path),
        "path": str(path),
        "relative_path": _package_relative(path),
        "exists": path.exists(),
    }
    if path.exists():
        metadata["sha256"] = _file_sha256(path)
        metadata["size_bytes"] = path.stat().st_size
    return metadata


def _apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile = str(args.profile)
    defaults = PROFILE_DEFAULTS[profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def _safe_token(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "run"


def _reference_args(trajectory: str, args: argparse.Namespace) -> list[str]:
    if trajectory == "square":
        return ["--trajectory", "square", "--square-side", str(args.square_side)]
    if trajectory == "drawn":
        drawn_path = _resolve_package_path(args.drawn_path)
        if not drawn_path.exists():
            raise FileNotFoundError(f"drawn path file not found: {drawn_path}")
        return ["--trajectory", "drawn", "--drawn-path", str(drawn_path)]
    return ["--trajectory", "letter", "--label", trajectory, "--letter-plane", args.letter_plane]


def _reference_token_for_plan(trajectory: str, args: argparse.Namespace) -> str:
    if trajectory == "square":
        return "square"
    if trajectory == "drawn":
        return _safe_token(Path(args.drawn_path).stem)
    return _safe_token(trajectory)


def _batch_preflight_issues(args: argparse.Namespace, trajectories: list[str], wind_modes: list[str]) -> list[str]:
    issues: list[str] = []
    invalid_winds = [wind_mode for wind_mode in wind_modes if wind_mode not in DEFAULT_WINDS]
    if invalid_winds:
        issues.append(
            "지원하지 않는 wind mode가 있습니다: "
            + ", ".join(invalid_winds)
            + f". 허용값: {', '.join(DEFAULT_WINDS)}"
        )

    if "drawn" in trajectories:
        drawn_path = _resolve_package_path(args.drawn_path)
        if not drawn_path.exists():
            issues.append(f"drawn trajectory JSON 파일이 없습니다: {drawn_path}")

    if bool(args.curriculum) and not bool(args.eval_only):
        positions = {wind_mode: index for index, wind_mode in enumerate(wind_modes)}
        if "M1" in positions and ("M0" not in positions or positions["M0"] > positions["M1"]):
            issues.append("curriculum에서 M1은 같은 trajectory/seed의 M0 학습 이후에만 실행할 수 있습니다.")
        if "M2" in positions and ("M1" not in positions or positions["M1"] > positions["M2"]):
            issues.append("curriculum에서 M2는 같은 trajectory/seed의 M1 학습 이후에만 실행할 수 있습니다.")
    device = str(args.device).strip().lower()
    if device.startswith("cuda"):
        try:
            import torch
        except Exception as exc:
            issues.append(f"--device {args.device} requested, but torch import failed: {exc}")
        else:
            if not torch.cuda.is_available():
                issues.append(f"--device {args.device} requested, but torch.cuda.is_available() is false")
            elif ":" in device:
                try:
                    index = int(device.split(":", 1)[1])
                except ValueError:
                    issues.append(f"invalid CUDA device specifier: {args.device!r}")
                else:
                    if index < 0 or index >= torch.cuda.device_count():
                        issues.append(
                            f"--device {args.device} requested, but only {torch.cuda.device_count()} CUDA device(s) are visible"
                        )
    return issues


def _append_failed_preflight_runs(
    manifest: dict[str, Any],
    output_root: Path,
    trajectories: list[str],
    wind_modes: list[str],
    seeds: list[int],
    *,
    error_type: str,
    message: str,
) -> None:
    for trajectory in trajectories:
        for seed in seeds:
            for wind_mode in wind_modes:
                run_id = f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}"
                run_dir = output_root / run_id
                manifest["runs"].append(
                    {
                        "run_id": run_id,
                        "trajectory": trajectory,
                        "wind_mode": wind_mode,
                        "seed": int(seed),
                        "output_dir": str(run_dir),
                        "load_model": None,
                        "command": "",
                        "status": "failed",
                        "return_code": 2,
                        "error_type": error_type,
                        "error": message,
                    }
                )


def _build_command(
    *,
    trajectory: str,
    wind_mode: str,
    seed: int,
    output_dir: Path,
    load_model: str | None,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train.train_phase_b",
        "--output-dir",
        str(output_dir),
        *_reference_args(trajectory, args),
        "--wind-mode",
        wind_mode,
        "--seed",
        str(seed),
        "--speed",
        str(args.speed),
        "--settle-time",
        str(args.settle_time),
        "--corner-window-m",
        str(args.corner_window_m),
        "--init-box-size",
        str(args.init_box_size),
        "--ctrl-freq",
        str(args.ctrl_freq),
        "--pyb-freq",
        str(args.pyb_freq),
        "--total-timesteps",
        str(args.total_timesteps),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--ent-coef",
        str(args.ent_coef),
        "--clip-range",
        str(args.clip_range),
        "--log-std-init",
        str(args.log_std_init),
        "--teacher-gain",
        str(args.teacher_gain),
        "--teacher-window-m",
        str(args.teacher_window_m),
        "--bc-epochs",
        str(args.bc_epochs),
        "--bc-episodes",
        str(args.bc_episodes),
        "--bc-batch-size",
        str(args.bc_batch_size),
        "--bc-learning-rate",
        str(args.bc_learning_rate),
        "--bc-nonzero-weight",
        str(args.bc_nonzero_weight),
        "--trained-action-filter",
        str(args.trained_action_filter),
        "--device",
        str(args.device),
        "--verbose",
        str(args.verbose),
    ]
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.save_video:
        command.append("--save-video")
    if args.eval_only:
        command.append("--eval-only")
    if load_model:
        command.extend(["--load-model", load_model])
    return command


def _summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in {"final_goal_pass", "overall_pass"}:
        source = summary.get("final_goal_criteria" if key == "final_goal_pass" else "success_criteria") or {}
        return source.get(key)
    rows = summary.get("metrics") or []
    trained = next((row for row in rows if row.get("tag") == "phaseB_trained"), {})
    return trained.get(key)


def _failed_final_goal_criteria(summary: dict[str, Any]) -> list[str]:
    criteria = summary.get("final_goal_criteria") or {}
    failed = [
        str(key)
        for key, value in criteria.items()
        if key != "final_goal_pass" and value is False
    ]
    return sorted(failed)


def _summary_artifact_mismatches(summary: dict[str, Any]) -> list[str]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    model_path = artifacts.get("model_path")
    if not model_path:
        return ["artifacts.model_path 누락"]
    if _resolve_existing_model_path(model_path) is None:
        return [f"artifacts.model_path 파일 없음 {model_path!r}"]
    return []


def _summary_final_goal_metric_mismatches(summary: dict[str, Any]) -> list[str]:
    criteria = summary.get("final_goal_criteria")
    metrics_rows = summary.get("metrics")
    if not isinstance(criteria, dict):
        return ["final_goal_criteria 누락"]
    if not isinstance(metrics_rows, list):
        return ["metrics 누락"]
    try:
        recomputed = _compute_final_goal_criteria(metrics_rows)
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        return [f"final_goal_criteria 재계산 불가 {exc}"]
    mismatches: list[str] = []
    for key, value in recomputed.items():
        if criteria.get(key) is not value:
            mismatches.append(f"final_goal_criteria 재계산 불일치 {key}: {criteria.get(key)!r} != {value!r}")
    return mismatches


def _row_from_summary(
    *,
    run_id: str,
    trajectory: str,
    wind_mode: str,
    seed: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    model_path = (summary.get("artifacts") or {}).get("model_path")
    return {
        "run_id": run_id,
        "trajectory": trajectory,
        "wind_mode": wind_mode,
        "seed": seed,
        "overall_pass": _summary_value(summary, "overall_pass"),
        "final_goal_pass": _summary_value(summary, "final_goal_pass"),
        "failed_final_goal_criteria": "|".join(_failed_final_goal_criteria(summary)),
        "painting_iou": _summary_value(summary, "painted_pixel_iou"),
        "painting_precision": _summary_value(summary, "painted_pixel_precision"),
        "painting_recall": _summary_value(summary, "painted_pixel_recall"),
        "painting_dice": _summary_value(summary, "painted_pixel_dice"),
        "off_target_ratio": _summary_value(summary, "off_target_pixel_ratio"),
        "path_rmse_m": _summary_value(summary, "path_rmse_m"),
        "path_max_m": _summary_value(summary, "path_max_m"),
        "corner_path_rmse_m": _summary_value(summary, "corner_path_rmse_m"),
        "corner_speed_ratio": _summary_value(summary, "corner_speed_ratio"),
        "corner_speed_vs_straight_ratio": _summary_value(summary, "corner_speed_vs_straight_ratio"),
        "corner_overshoot_m": _summary_value(summary, "corner_overshoot_m"),
        "led_precision": _summary_value(summary, "led_precision"),
        "led_recall": _summary_value(summary, "led_recall"),
        "led_flicker_rate": _summary_value(summary, "led_flicker_rate"),
        "action_norm_mean": _summary_value(summary, "action_norm_mean"),
        "action_norm_max": _summary_value(summary, "action_norm_max"),
        "action_rate_mean": _summary_value(summary, "action_rate_mean"),
        "action_rate_max": _summary_value(summary, "action_rate_max"),
        "rpm_saturation_ratio": _summary_value(summary, "rpm_saturation_ratio"),
        "crash_rate": _summary_value(summary, "crash_rate"),
        "out_of_bounds_rate": _summary_value(summary, "out_of_bounds_rate"),
        "model_path": model_path,
    }


def _read_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _resolved_path_text(raw: Any) -> str | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = _PKG_ROOT / path
    return str(path.resolve(strict=False))


def _resolve_existing_model_path(raw: Any) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = _PKG_ROOT / path
    return path if path.exists() else None


def _resolve_manifest_model_path(raw: Any, manifest_path: Path) -> str:
    if raw is None or str(raw).strip() == "":
        return ""
    path = Path(str(raw))
    if path.is_absolute():
        return str(path.resolve(strict=False))

    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return str(manifest_relative.resolve(strict=False))

    package_relative = _PKG_ROOT / path
    if package_relative.exists():
        return str(package_relative.resolve(strict=False))

    return str(manifest_relative.resolve(strict=False))


def _same_optional_path(left: Any, right: Any) -> bool:
    return _resolved_path_text(left) == _resolved_path_text(right)


def _close_float(left: Any, right: Any, tol: float = 1e-9) -> bool:
    parsed_left = _float_or_none(left)
    parsed_right = _float_or_none(right)
    return parsed_left is not None and parsed_right is not None and abs(parsed_left - parsed_right) <= tol


def _expected_reference_mismatch(trajectory: str, args: argparse.Namespace, config: dict[str, Any]) -> str | None:
    reference = config.get("reference")
    if not isinstance(reference, dict):
        return "config.reference가 없습니다"
    actual_trajectory = str(reference.get("trajectory", ""))
    if trajectory == "square":
        if actual_trajectory != "square":
            return f"reference trajectory 불일치: {actual_trajectory!r} != 'square'"
        expected_side = float(args.square_side)
        actual_side = reference.get("unscaled_square_side_m", reference.get("square_side_m"))
        if not _close_float(actual_side, expected_side):
            return f"square_side 불일치: {actual_side!r} != {expected_side}"
        return None
    if trajectory == "drawn":
        if actual_trajectory != "drawn":
            return f"reference trajectory 불일치: {actual_trajectory!r} != 'drawn'"
        expected_path = Path(args.drawn_path)
        if not expected_path.is_absolute():
            expected_path = _PKG_ROOT / expected_path
        actual_path = reference.get("resolved_drawn_path", reference.get("drawn_path"))
        if actual_path is None or not _same_optional_path(actual_path, expected_path):
            return f"drawn_path 불일치: {actual_path!r} != {str(expected_path)!r}"
        return None
    if actual_trajectory != "letter":
        return f"reference trajectory 불일치: {actual_trajectory!r} != 'letter'"
    if str(reference.get("label", "")) != str(trajectory):
        return f"letter label 불일치: {reference.get('label')!r} != {trajectory!r}"
    if str(reference.get("plane", "")) != str(args.letter_plane):
        return f"letter plane 불일치: {reference.get('plane')!r} != {args.letter_plane!r}"
    return None


def _summary_resume_mismatches(
    summary: dict[str, Any],
    *,
    trajectory: str,
    wind_mode: str,
    seed: int,
    load_model: str | None,
    args: argparse.Namespace,
) -> list[str]:
    config = summary.get("config")
    if not isinstance(config, dict):
        return ["summary.config가 없어 현재 실행 인자와 비교할 수 없습니다"]

    mismatches: list[str] = []
    if str(config.get("wind_mode", "")) != str(wind_mode):
        mismatches.append(f"wind_mode 불일치: {config.get('wind_mode')!r} != {wind_mode!r}")
    if _int_or_none(config.get("seed")) != int(seed):
        mismatches.append(f"seed 불일치: {config.get('seed')!r} != {seed}")
    if _bool_or_none(config.get("eval_only")) != bool(args.eval_only):
        mismatches.append(f"eval_only 불일치: {config.get('eval_only')!r} != {bool(args.eval_only)}")
    if not _same_optional_path(config.get("load_model"), load_model):
        mismatches.append("load_model 불일치")

    reference_mismatch = _expected_reference_mismatch(trajectory, args, config)
    if reference_mismatch:
        mismatches.append(reference_mismatch)

    int_fields = (
        ("requested_total_timesteps", int(args.total_timesteps)),
        ("total_timesteps", 0 if bool(args.eval_only) else int(args.total_timesteps)),
        ("ctrl_freq", int(args.ctrl_freq)),
        ("pyb_freq", int(args.pyb_freq)),
        ("n_steps", int(args.n_steps)),
        ("batch_size", int(args.batch_size)),
        ("n_epochs", int(args.n_epochs)),
        ("bc_epochs", int(args.bc_epochs)),
        ("bc_episodes", int(args.bc_episodes)),
        ("bc_batch_size", int(args.bc_batch_size)),
    )
    for key, expected in int_fields:
        if _int_or_none(config.get(key)) != expected:
            mismatches.append(f"{key} 불일치: {config.get(key)!r} != {expected}")
    if args.max_steps is None:
        if "max_steps_override" not in config or config.get("max_steps_override") is not None:
            mismatches.append(f"max_steps_override 불일치: {config.get('max_steps_override')!r} != None")
    else:
        expected_max_steps = int(args.max_steps)
        if _int_or_none(config.get("max_steps_override")) != expected_max_steps:
            mismatches.append(f"max_steps_override 불일치: {config.get('max_steps_override')!r} != {expected_max_steps}")
        if _int_or_none(config.get("max_steps")) != expected_max_steps:
            mismatches.append(f"max_steps 불일치: {config.get('max_steps')!r} != {expected_max_steps}")

    float_fields = (
        ("speed", float(args.speed)),
        ("corner_window_m", float(args.corner_window_m)),
        ("learning_rate", float(args.learning_rate)),
        ("gamma", float(args.gamma)),
        ("gae_lambda", float(args.gae_lambda)),
        ("ent_coef", float(args.ent_coef)),
        ("clip_range", float(args.clip_range)),
        ("log_std_init", float(args.log_std_init)),
        ("teacher_gain", float(args.teacher_gain)),
        ("teacher_window_m", float(args.teacher_window_m)),
        ("bc_learning_rate", float(args.bc_learning_rate)),
        ("bc_nonzero_weight", float(args.bc_nonzero_weight)),
    )
    for key, expected in float_fields:
        if not _close_float(config.get(key), expected):
            mismatches.append(f"{key} 불일치: {config.get(key)!r} != {expected}")

    if str(config.get("trained_action_filter", "")) != str(args.trained_action_filter):
        mismatches.append(
            f"trained_action_filter 불일치: {config.get('trained_action_filter')!r} != {args.trained_action_filter!r}"
        )
    if str(config.get("device", "")) != str(args.device):
        mismatches.append(f"device 불일치: {config.get('device')!r} != {args.device!r}")
    if _bool_or_none(config.get("save_video")) != bool(args.save_video):
        mismatches.append(f"save_video 불일치: {config.get('save_video')!r} != {bool(args.save_video)}")
    mismatches.extend(_summary_artifact_mismatches(summary))
    if str(args.profile) == "final":
        mismatches.extend(_summary_final_goal_metric_mismatches(summary))
    return mismatches


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _aggregate_one(group_name: str, group_value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group": group_name,
        "value": group_value,
        "n": len(rows),
        "final_goal_pass_count": sum(1 for row in rows if row.get("final_goal_pass") is True),
        "overall_pass_count": sum(1 for row in rows if row.get("overall_pass") is True),
    }
    out["final_goal_pass_rate"] = out["final_goal_pass_count"] / max(len(rows), 1)
    out["overall_pass_rate"] = out["overall_pass_count"] / max(len(rows), 1)
    for metric in AGGREGATE_METRICS:
        vals = [_float_or_none(row.get(metric)) for row in rows]
        finite_vals = [val for val in vals if val is not None]
        if not finite_vals:
            out[f"{metric}_mean"] = ""
            out[f"{metric}_std"] = ""
            out[f"{metric}_min"] = ""
            out[f"{metric}_max"] = ""
            continue
        mean = sum(finite_vals) / len(finite_vals)
        var = sum((val - mean) ** 2 for val in finite_vals) / len(finite_vals)
        out[f"{metric}_mean"] = f"{mean:.6f}"
        out[f"{metric}_std"] = f"{math.sqrt(var):.6f}"
        out[f"{metric}_min"] = f"{min(finite_vals):.6f}"
        out[f"{metric}_max"] = f"{max(finite_vals):.6f}"
    return out


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for trajectory in sorted({str(row["trajectory"]) for row in rows}):
        groups.append(("trajectory", trajectory, [row for row in rows if str(row["trajectory"]) == trajectory]))
    for wind_mode in sorted({str(row["wind_mode"]) for row in rows}):
        groups.append(("wind_mode", wind_mode, [row for row in rows if str(row["wind_mode"]) == wind_mode]))
    for trajectory in sorted({str(row["trajectory"]) for row in rows}):
        for wind_mode in sorted({str(row["wind_mode"]) for row in rows}):
            subset = [
                row
                for row in rows
                if str(row["trajectory"]) == trajectory and str(row["wind_mode"]) == wind_mode
            ]
            if subset:
                groups.append(("trajectory_wind", f"{trajectory}/{wind_mode}", subset))
    return [_aggregate_one(name, value, subset) for name, value, subset in groups]


def _matrix_result_rows(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    trajectories: list[str],
    wind_modes: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    row_lookup = {
        (str(row.get("trajectory")), str(row.get("wind_mode")), int(row.get("seed"))): row
        for row in rows
        if row.get("trajectory") is not None and row.get("wind_mode") is not None and row.get("seed") is not None
    }
    run_lookup = {
        (str(run.get("trajectory")), str(run.get("wind_mode")), int(run.get("seed"))): run
        for run in manifest.get("runs", [])
        if run.get("trajectory") is not None and run.get("wind_mode") is not None and run.get("seed") is not None
    }
    table: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for wind_mode in wind_modes:
            for seed in seeds:
                key = (str(trajectory), str(wind_mode), int(seed))
                result_row = row_lookup.get(key)
                run_record = run_lookup.get(key, {})
                table.append(
                    {
                        "trajectory": trajectory,
                        "wind_mode": wind_mode,
                        "seed": int(seed),
                        "run_id": (result_row or run_record).get("run_id", ""),
                        "status": run_record.get("status", "missing"),
                        "overall_pass": "" if result_row is None else result_row.get("overall_pass"),
                        "final_goal_pass": "" if result_row is None else result_row.get("final_goal_pass"),
                        "failed_final_goal_criteria": ""
                        if result_row is None
                        else result_row.get("failed_final_goal_criteria", ""),
                        **{
                            metric: "" if result_row is None else result_row.get(metric, "")
                            for metric in AGGREGATE_METRICS
                        },
                        "model_path": "" if result_row is None else result_row.get("model_path", ""),
                        "output_dir": run_record.get("output_dir", ""),
                        "load_model": run_record.get("load_model", ""),
                        "command": run_record.get("command", ""),
                    }
                )
    return table


def _write_batch_outputs(
    output_root: Path,
    manifest: dict[str, Any],
    best_model_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    trajectories: list[str],
    wind_modes: list[str],
    seeds: list[int],
) -> None:
    manifest["batch_final_goal_criteria"] = _batch_final_goal_criteria(rows, trajectories, wind_modes, seeds)
    _write_json(output_root / "batch_summary.json", manifest)
    _write_json(output_root / "best_model_manifest.json", best_model_manifest)
    _write_csv(output_root / "metrics.csv", rows)
    _write_csv(output_root / "aggregate_metrics.csv", _aggregate_rows(rows))
    _write_csv(output_root / "matrix_results.csv", _matrix_result_rows(manifest, rows, trajectories, wind_modes, seeds))


def _finalize_manifest(manifest: dict[str, Any], rows: list[dict[str, Any]], *, status: str) -> None:
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["status"] = status
    manifest["completed_runs"] = sum(
        1 for run_record in manifest["runs"] if run_record.get("status") in {"complete", "reused"}
    )
    manifest["failed_runs"] = sum(1 for run_record in manifest["runs"] if run_record.get("status") == "failed")
    manifest["reused_runs"] = sum(1 for run_record in manifest["runs"] if run_record.get("status") == "reused")
    manifest["planned_runs"] = len(manifest["runs"])
    if rows:
        manifest["final_goal_pass_count"] = sum(1 for row in rows if row.get("final_goal_pass") is True)
        manifest["final_goal_pass_rate"] = manifest["final_goal_pass_count"] / max(len(rows), 1)
        manifest["aggregate"] = _aggregate_rows(rows)


def _expected_matrix_keys(trajectories: list[str], wind_modes: list[str], seeds: list[int]) -> set[tuple[str, str, int]]:
    return {
        (str(trajectory), str(wind_mode), int(seed))
        for trajectory in trajectories
        for wind_mode in wind_modes
        for seed in seeds
    }


def _batch_final_goal_criteria(
    rows: list[dict[str, Any]],
    trajectories: list[str],
    wind_modes: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    expected = _expected_matrix_keys(trajectories, wind_modes, seeds)
    observed = {
        (str(row.get("trajectory")), str(row.get("wind_mode")), int(row.get("seed")))
        for row in rows
        if row.get("trajectory") is not None and row.get("wind_mode") is not None and row.get("seed") is not None
    }
    missing = sorted(expected - observed)
    lookup = {
        (str(row.get("trajectory")), str(row.get("wind_mode")), int(row.get("seed"))): row
        for row in rows
        if row.get("trajectory") is not None and row.get("wind_mode") is not None and row.get("seed") is not None
    }

    robustness_failures: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for seed in seeds:
            base = lookup.get((trajectory, "M0", int(seed)))
            if base is None:
                continue
            base_iou = _float_or_none(base.get("painting_iou"))
            base_path = _float_or_none(base.get("path_rmse_m"))
            for wind_mode in ("M1", "M2"):
                if wind_mode not in wind_modes:
                    continue
                row = lookup.get((trajectory, wind_mode, int(seed)))
                if row is None:
                    continue
                iou = _float_or_none(row.get("painting_iou"))
                path = _float_or_none(row.get("path_rmse_m"))
                iou_ok = base_iou is not None and iou is not None and iou >= base_iou * ROBUST_IOU_REL_MIN
                path_ok = base_path is not None and path is not None and path <= base_path * ROBUST_PATH_RMSE_REL_MAX
                if not iou_ok or not path_ok:
                    robustness_failures.append(
                        {
                            "trajectory": trajectory,
                            "seed": int(seed),
                            "wind_mode": wind_mode,
                            "m0_painting_iou": base_iou,
                            "painting_iou": iou,
                            "m0_path_rmse_m": base_path,
                            "path_rmse_m": path,
                            "iou_ok": bool(iou_ok),
                            "path_rmse_ok": bool(path_ok),
                        }
                    )

    all_run_final = bool(rows) and all(row.get("final_goal_pass") is True for row in rows)
    failed_final_goal_runs = [
        {
            "run_id": row.get("run_id"),
            "trajectory": row.get("trajectory"),
            "wind_mode": row.get("wind_mode"),
            "seed": row.get("seed"),
            "failed_criteria": [
                item
                for item in str(row.get("failed_final_goal_criteria") or "").split("|")
                if item
            ],
            "painting_iou": _float_or_none(row.get("painting_iou")),
            "off_target_ratio": _float_or_none(row.get("off_target_ratio")),
            "path_rmse_m": _float_or_none(row.get("path_rmse_m")),
            "corner_speed_ratio": _float_or_none(row.get("corner_speed_ratio")),
            "crash_rate": _float_or_none(row.get("crash_rate")),
            "out_of_bounds_rate": _float_or_none(row.get("out_of_bounds_rate")),
        }
        for row in rows
        if row.get("final_goal_pass") is not True
    ]
    matrix_complete = len(missing) == 0 and len(observed) >= len(expected)
    robustness_pass = len(robustness_failures) == 0
    criteria = {
        "matrix_complete": matrix_complete,
        "expected_runs": len(expected),
        "observed_runs": len(observed),
        "missing_runs": [
            {"trajectory": trajectory, "wind_mode": wind_mode, "seed": seed}
            for trajectory, wind_mode, seed in missing
        ],
        "all_runs_final_goal_pass": all_run_final,
        "failed_final_goal_run_count": len(failed_final_goal_runs),
        "failed_final_goal_runs": failed_final_goal_runs,
        "robustness_relative_to_m0_pass": robustness_pass,
        "robustness_thresholds": {
            "m1_m2_painting_iou_min_relative_to_m0": ROBUST_IOU_REL_MIN,
            "m1_m2_path_rmse_max_relative_to_m0": ROBUST_PATH_RMSE_REL_MAX,
        },
        "robustness_failures": robustness_failures,
    }
    criteria["batch_final_goal_pass"] = bool(matrix_complete and all_run_final and robustness_pass)
    return criteria


def run(args: argparse.Namespace) -> int:
    args = _apply_profile_defaults(args)
    trajectories = _split_csv(args.trajectories)
    wind_modes = _split_csv(args.wind_modes)
    seeds = _split_int_csv(args.seeds)
    if not trajectories:
        raise ValueError("--trajectories must contain at least one entry")
    if not wind_modes:
        raise ValueError("--wind-modes must contain at least one entry")
    if not seeds:
        raise ValueError("--seeds must contain at least one entry")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _PKG_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    eval_models: dict[str, str] = {}
    model_manifest_path: Path | None = None
    if args.model_manifest:
        model_manifest_path = Path(args.model_manifest)
        if not model_manifest_path.is_absolute():
            model_manifest_path = _PKG_ROOT / model_manifest_path

    started_at = datetime.now().isoformat(timespec="seconds")
    drawn_input = _drawn_input_metadata(args, trajectories)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "final_goal_spec": "final_goal_thresholds_v1",
        "profile": str(args.profile),
        "started_at": started_at,
        "dry_run": bool(args.dry_run),
        "python_executable": sys.executable,
        "runtime": _runtime_metadata(),
        "eval_only": bool(args.eval_only),
        "model_manifest": str(model_manifest_path) if model_manifest_path is not None else None,
        "trajectories": trajectories,
        "wind_modes": wind_modes,
        "seeds": seeds,
        "total_timesteps": int(args.total_timesteps),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "teacher_window_m": float(args.teacher_window_m),
        "corner_window_m": float(args.corner_window_m),
        "trained_action_filter": str(args.trained_action_filter),
        "drawn_input": drawn_input,
        "evaluation_policy": (
            "phaseB_trained rollout applies the selected trained_action_filter; "
            "set --trained-action-filter none only when an unfiltered rollout is needed."
        ),
        "runs": [],
    }
    best_model_manifest: dict[str, Any] = {
        "schema_version": 2,
        "final_goal_spec": "final_goal_thresholds_v1",
        "profile": str(args.profile),
        "dry_run": bool(args.dry_run),
        "eval_only": bool(args.eval_only),
        "source_model_manifest": str(model_manifest_path) if model_manifest_path is not None else None,
        "batch_summary": str(output_root / "batch_summary.json"),
        "trajectories": trajectories,
        "wind_modes": wind_modes,
        "seeds": seeds,
        "total_timesteps": int(args.total_timesteps),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "teacher_window_m": float(args.teacher_window_m),
        "trained_action_filter": str(args.trained_action_filter),
        "drawn_input": drawn_input,
        "created_at": started_at,
        "runtime": _runtime_metadata(),
        "models": {},
    }
    rows: list[dict[str, Any]] = []
    prior_model: dict[tuple[str, int], str] = {}
    dependency_preflight_checked = False

    batch_preflight_issues = _batch_preflight_issues(args, trajectories, wind_modes)
    if batch_preflight_issues:
        message = " ".join(batch_preflight_issues)
        manifest["batch_preflight"] = {
            "ok": False,
            "issues": batch_preflight_issues,
        }
        _append_failed_preflight_runs(
            manifest,
            output_root,
            trajectories,
            wind_modes,
            seeds,
            error_type="batch_preflight",
            message=message,
        )
        _finalize_manifest(manifest, rows, status="failed")
        _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
        print(f"[batch] preflight error: {message}", file=sys.stderr, flush=True)
        return 2
    manifest["batch_preflight"] = {
        "ok": True,
        "issues": [],
    }

    if args.eval_only:
        if model_manifest_path is None or not model_manifest_path.exists():
            message = (
                "--eval-only batch는 기존 학습 batch의 best_model_manifest.json 경로가 필요합니다. "
                f"입력 경로: {model_manifest_path}"
            )
            manifest["model_manifest_preflight"] = {
                "ok": False,
                "message": message,
                "model_manifest": str(model_manifest_path) if model_manifest_path is not None else None,
            }
            _finalize_manifest(manifest, rows, status="failed")
            _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
            print(f"[batch] model manifest error: {message}", file=sys.stderr, flush=True)
            return 2
        data = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        eval_models = {
            str(key): _resolve_manifest_model_path(value, model_manifest_path)
            for key, value in (data.get("models") or {}).items()
        }
        expected_model_keys = [
            f"{trajectory}/{wind_mode}/seed{seed}"
            for trajectory in trajectories
            for wind_mode in wind_modes
            for seed in seeds
        ]
        missing_model_keys = [
            key
            for key in expected_model_keys
            if key not in eval_models
        ]
        missing_model_files = [
            {"key": key, "path": value}
            for key, value in eval_models.items()
            if key in expected_model_keys and not bool(args.dry_run) and _resolve_existing_model_path(value) is None
        ]
        if missing_model_keys:
            manifest["model_manifest_preflight"] = {
                "ok": False,
                "model_manifest": str(model_manifest_path),
                "missing_models": missing_model_keys,
            }
            for key in missing_model_keys:
                trajectory, wind_mode, seed_token = key.split("/")
                seed = int(seed_token.replace("seed", ""))
                run_dir = output_root / f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}"
                command = _build_command(
                    trajectory=trajectory,
                    wind_mode=wind_mode,
                    seed=seed,
                    output_dir=run_dir,
                    load_model=None,
                    args=args,
                )
                manifest["runs"].append(
                    {
                        "run_id": f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}",
                        "trajectory": trajectory,
                        "wind_mode": wind_mode,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "load_model": None,
                        "command": subprocess.list2cmdline(command),
                        "status": "failed",
                        "return_code": 2,
                        "error_type": "model_manifest_preflight",
                        "error": f"model manifest에 {key} 항목이 없습니다",
                    }
                )
            _finalize_manifest(manifest, rows, status="failed")
            _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
            print(
                "[batch] model manifest error: missing models: " + ", ".join(missing_model_keys),
                file=sys.stderr,
                flush=True,
            )
            return 2
        if missing_model_files:
            manifest["model_manifest_preflight"] = {
                "ok": False,
                "model_manifest": str(model_manifest_path),
                "missing_models": [],
                "missing_model_files": missing_model_files,
            }
            for item in missing_model_files:
                trajectory, wind_mode, seed_token = str(item["key"]).split("/")
                seed = int(seed_token.replace("seed", ""))
                run_dir = output_root / f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}"
                command = _build_command(
                    trajectory=trajectory,
                    wind_mode=wind_mode,
                    seed=seed,
                    output_dir=run_dir,
                    load_model=str(item["path"]),
                    args=args,
                )
                manifest["runs"].append(
                    {
                        "run_id": f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}",
                        "trajectory": trajectory,
                        "wind_mode": wind_mode,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "load_model": str(item["path"]),
                        "command": subprocess.list2cmdline(command),
                        "status": "failed",
                        "return_code": 2,
                        "error_type": "model_manifest_preflight",
                        "error": f"model manifest의 모델 파일이 없습니다: {item['path']}",
                    }
                )
            _finalize_manifest(manifest, rows, status="failed")
            _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
            print(
                "[batch] model manifest error: missing model files: "
                + ", ".join(f"{item['key']}={item['path']}" for item in missing_model_files),
                file=sys.stderr,
                flush=True,
            )
            return 2
        manifest["model_manifest_preflight"] = {
            "ok": True,
            "model_manifest": str(model_manifest_path),
            "missing_models": [],
            "missing_model_files": [],
        }

    total_runs = len(trajectories) * len(seeds) * len(wind_modes)
    run_index = 0
    for trajectory in trajectories:
        for seed in seeds:
            for wind_mode in wind_modes:
                run_index += 1
                run_id = f"{_safe_token(trajectory)}_{wind_mode}_seed{seed}"
                run_dir = output_root / run_id
                load_model = None
                if args.eval_only:
                    load_model = eval_models.get(f"{trajectory}/{wind_mode}/seed{seed}")
                elif args.curriculum and wind_mode != "M0":
                    load_model = prior_model.get((trajectory, seed))
                    if not load_model:
                        message = (
                            "curriculum source model이 없습니다: "
                            f"{trajectory}/{wind_mode}/seed{seed}. "
                            "이전 wind 단계가 실패했거나 model_path가 기록되지 않았습니다."
                        )
                        run_record = {
                            "run_id": run_id,
                            "run_index": run_index,
                            "total_runs": total_runs,
                            "trajectory": trajectory,
                            "wind_mode": wind_mode,
                            "seed": seed,
                            "output_dir": str(run_dir),
                            "load_model": None,
                            "command": "",
                            "status": "failed",
                            "return_code": 2,
                            "error_type": "curriculum_source_missing",
                            "error": message,
                        }
                        manifest["runs"].append(run_record)
                        _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                        print(f"[batch] curriculum error: {message}", file=sys.stderr, flush=True)
                        if not args.continue_on_error:
                            _finalize_manifest(manifest, rows, status="failed")
                            _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                            return 2
                        continue
                command = _build_command(
                    trajectory=trajectory,
                    wind_mode=wind_mode,
                    seed=seed,
                    output_dir=run_dir,
                    load_model=load_model,
                    args=args,
                )
                run_record = {
                    "run_id": run_id,
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "trajectory": trajectory,
                    "wind_mode": wind_mode,
                    "seed": seed,
                    "output_dir": str(run_dir),
                    "load_model": load_model,
                    "command": subprocess.list2cmdline(command),
                    "status": "planned" if args.dry_run else "running",
                }
                manifest["runs"].append(run_record)
                if args.dry_run:
                    planned_model = (
                        str(load_model)
                        if args.eval_only and load_model
                        else run_dir / "models" / f"ppo_phaseB_{_reference_token_for_plan(trajectory, args)}_{wind_mode}_corner.zip"
                    )
                    prior_model[(trajectory, seed)] = str(planned_model)
                    best_model_manifest["models"][f"{trajectory}/{wind_mode}/seed{seed}"] = str(planned_model)
                    continue
                existing_summary = _read_summary(run_dir) if args.resume else None
                if existing_summary is not None:
                    resume_mismatches = _summary_resume_mismatches(
                        existing_summary,
                        trajectory=trajectory,
                        wind_mode=wind_mode,
                        seed=seed,
                        load_model=load_model,
                        args=args,
                    )
                    if not resume_mismatches:
                        row = _row_from_summary(
                            run_id=run_id,
                            trajectory=trajectory,
                            wind_mode=wind_mode,
                            seed=seed,
                            summary=existing_summary,
                        )
                        rows.append(row)
                        if row.get("model_path"):
                            prior_model[(trajectory, seed)] = str(row["model_path"])
                            best_model_manifest["models"][f"{trajectory}/{wind_mode}/seed{seed}"] = str(row["model_path"])
                        run_record["status"] = "reused"
                        run_record["return_code"] = 0
                        run_record["finished_at"] = datetime.now().isoformat(timespec="seconds")
                        _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                        continue
                    run_record["resume_skip_reason"] = "; ".join(resume_mismatches[:8])

                if not dependency_preflight_checked:
                    missing_dependencies = _missing_runtime_dependencies()
                    if missing_dependencies:
                        message = _dependency_error_message(missing_dependencies)
                        run_dir.mkdir(parents=True, exist_ok=True)
                        log_path = run_dir / "batch_run.log"
                        log_path.write_text(f"[batch] runtime dependency error: {message}\n", encoding="utf-8")
                        run_record["status"] = "failed"
                        run_record["return_code"] = 2
                        run_record["error_type"] = "dependency_preflight"
                        run_record["error"] = message
                        run_record["log_path"] = str(log_path)
                        manifest["dependency_preflight"] = {
                            "ok": False,
                            "python_executable": sys.executable,
                            "missing_dependencies": missing_dependencies,
                            "message": message,
                        }
                        _finalize_manifest(manifest, rows, status="failed")
                        _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                        print(f"[batch] runtime dependency error: {message}", file=sys.stderr, flush=True)
                        return 2
                    manifest["dependency_preflight"] = {
                        "ok": True,
                        "python_executable": sys.executable,
                        "missing_dependencies": [],
                    }
                    dependency_preflight_checked = True

                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = run_dir / "batch_run.log"
                run_record["log_path"] = str(log_path)
                run_record["started_at"] = datetime.now().isoformat(timespec="seconds")
                run_record["status"] = "running"
                _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                print(f"[batch] running {run_index}/{total_runs}: {run_id}", flush=True)
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                run_started = monotonic()
                with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                    log_file.write(f"[batch] run_id={run_id}\n")
                    log_file.write(f"[batch] index={run_index}/{total_runs}\n")
                    log_file.write(f"[batch] command={subprocess.list2cmdline(command)}\n")
                    log_file.flush()
                    proc = subprocess.run(
                        command,
                        cwd=str(_PKG_ROOT),
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=env,
                    )
                run_record["return_code"] = proc.returncode
                run_record["finished_at"] = datetime.now().isoformat(timespec="seconds")
                run_record["duration_s"] = round(monotonic() - run_started, 3)
                summary = _read_summary(run_dir)
                run_record["status"] = "complete" if proc.returncode == 0 and summary else "failed"
                if summary:
                    row = _row_from_summary(
                        run_id=run_id,
                        trajectory=trajectory,
                        wind_mode=wind_mode,
                        seed=seed,
                        summary=summary,
                    )
                    if row.get("model_path"):
                        prior_model[(trajectory, seed)] = str(row["model_path"])
                        best_model_manifest["models"][f"{trajectory}/{wind_mode}/seed{seed}"] = str(row["model_path"])
                    rows.append(row)
                _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                if proc.returncode != 0:
                    if not args.continue_on_error:
                        _finalize_manifest(manifest, rows, status="failed")
                        _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
                        return proc.returncode

    _finalize_manifest(manifest, rows, status="planned" if args.dry_run else "complete")
    _write_batch_outputs(output_root, manifest, best_model_manifest, rows, trajectories, wind_modes, seeds)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full final-goal LightPaint Phase B matrix.")
    parser.add_argument("--output-dir", default=str(_PKG_ROOT / "artifacts" / "final_goal_batch"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="standard",
        help=(
            "sanity is only an execution check; letters runs canonical DG only; "
            "standard runs canonical DG plus the user drawn path with 3 seeds; "
            "final runs the same dual-path matrix with 5 seeds and longer training."
        ),
    )
    parser.add_argument("--trajectories", default=None)
    parser.add_argument("--wind-modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--square-side", type=float, default=0.8)
    parser.add_argument("--letter-plane", default="xz")
    parser.add_argument("--drawn-path", default=CANONICAL_DRAWN_PATH)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--corner-window-m", type=float, default=0.18)
    parser.add_argument("--init-box-size", type=float, default=0.0)
    parser.add_argument("--ctrl-freq", type=int, default=30)
    parser.add_argument("--pyb-freq", type=int, default=240)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--teacher-gain", type=float, default=0.1)
    parser.add_argument("--teacher-window-m", type=float, default=0.15)
    parser.add_argument("--bc-epochs", type=int, default=2)
    parser.add_argument("--bc-episodes", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-nonzero-weight", type=float, default=1.0)
    parser.add_argument(
        "--trained-action-filter",
        choices=("none", "corner_tangent_decel"),
        default=DEFAULT_TRAINED_ACTION_FILTER,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
