"""Verify that a LightPaint batch artifact set proves the final goal."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.train import run_final_goal_batch
from src.train.train_phase_b_m0_corner import _final_goal_criteria as _compute_final_goal_criteria

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent

REQUIRED_ARTIFACTS = (
    "batch_summary.json",
    "metrics.csv",
    "aggregate_metrics.csv",
    "matrix_results.csv",
    "best_model_manifest.json",
)
CANONICAL_TEACHER_WINDOW_M = 0.15
FLOAT_TOL = 1e-9
AGGREGATE_FLOAT_TOL = 1e-6
FINAL_GOAL_CRITERIA_KEYS = (
    "painting_iou_abs",
    "painting_precision_abs",
    "painting_recall_abs",
    "painting_dice_abs",
    "off_target_ratio_abs",
    "path_rmse_abs",
    "path_max_abs",
    "straight_path_rmse_relative",
    "corner_path_rmse_relative",
    "corner_speed_ratio_abs",
    "corner_overshoot_relative",
    "led_precision_abs",
    "led_recall_abs",
    "led_flicker_not_worse",
    "crash_free",
    "out_of_bounds_free",
    "rpm_saturation_ratio_abs",
    "action_norm_bounded",
    "action_rate_bounded",
)
MATRIX_SUMMARY_METRIC_KEYS = (
    ("painting_iou", "painted_pixel_iou"),
    ("painting_precision", "painted_pixel_precision"),
    ("painting_recall", "painted_pixel_recall"),
    ("painting_dice", "painted_pixel_dice"),
    ("off_target_ratio", "off_target_pixel_ratio"),
    ("path_rmse_m", "path_rmse_m"),
    ("path_max_m", "path_max_m"),
    ("corner_path_rmse_m", "corner_path_rmse_m"),
    ("corner_speed_ratio", "corner_speed_ratio"),
    ("corner_speed_vs_straight_ratio", "corner_speed_vs_straight_ratio"),
    ("corner_overshoot_m", "corner_overshoot_m"),
    ("led_precision", "led_precision"),
    ("led_recall", "led_recall"),
    ("led_flicker_rate", "led_flicker_rate"),
    ("action_norm_mean", "action_norm_mean"),
    ("action_norm_max", "action_norm_max"),
    ("action_rate_mean", "action_rate_mean"),
    ("action_rate_max", "action_rate_max"),
    ("rpm_saturation_ratio", "rpm_saturation_ratio"),
    ("crash_rate", "crash_rate"),
    ("out_of_bounds_rate", "out_of_bounds_rate"),
)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    issues: list[str]
    summary: dict[str, Any]
    matrix_rows: list[dict[str, str]]


def _resolve_batch_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _PKG_ROOT / candidate
    if candidate.name == "batch_summary.json":
        candidate = candidate.parent
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix_key(row: dict[str, str]) -> str:
    return f"{row.get('trajectory')}/{row.get('wind_mode')}/seed{row.get('seed')}"


def _row_truth(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"true", "1", "yes"}:
        return "True"
    if text.lower() in {"false", "0", "no"}:
        return "False"
    return text


def _resolve_artifact_path(root: Path, raw: str | None) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def _same_resolved_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return str(left.resolve(strict=False)) == str(right.resolve(strict=False))


@lru_cache(maxsize=4096)
def _model_archive_issue_messages(path_text: str, mtime_ns: int, size_bytes: int) -> tuple[str, ...]:
    path = Path(path_text)
    del mtime_ns, size_bytes
    if not zipfile.is_zipfile(path):
        return (f"Stable-Baselines 모델 zip이 아닙니다: {path}",)
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        return (f"모델 zip을 읽을 수 없습니다: {path} ({exc})",)
    messages: list[str] = []
    if bad_member is not None:
        messages.append(f"모델 zip에 손상된 파일이 있습니다: {path} ({bad_member})")
    required_members = {"data", "policy.pth"}
    missing_members = sorted(required_members - names)
    if missing_members:
        messages.append(f"모델 zip에 필수 항목이 없습니다: {path} ({', '.join(missing_members)})")
    return tuple(messages)


def _validate_model_archive(path: Path | None, label: str, issues: list[str]) -> None:
    if path is None or not path.exists():
        return
    try:
        stat = path.stat()
    except OSError as exc:
        issues.append(f"{label} 모델 zip 상태를 확인할 수 없습니다: {path} ({exc})")
        return
    path_text = str(path.resolve(strict=False))
    for message in _model_archive_issue_messages(path_text, stat.st_mtime_ns, stat.st_size):
        issues.append(f"{label} {message}")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _split_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _split_csv(raw)]


def _validate_canonical_drawn_input(owner: str, drawn_input: Any, issues: list[str]) -> None:
    if not isinstance(drawn_input, dict):
        issues.append(f"{owner} drawn_input이 없습니다")
        return
    expected_path = run_final_goal_batch.CANONICAL_DRAWN_PATH
    expected_sha = run_final_goal_batch.CANONICAL_DRAWN_SHA256
    relative_path = drawn_input.get("relative_path")
    if relative_path != expected_path:
        issues.append(f"{owner} drawn_input relative_path가 canonical과 다릅니다: {relative_path!r}")
    if drawn_input.get("sha256") != expected_sha:
        issues.append(f"{owner} drawn_input sha256이 canonical과 다릅니다: {drawn_input.get('sha256')!r}")
    if drawn_input.get("exists") is not True:
        issues.append(f"{owner} drawn_input 파일 존재가 확인되지 않았습니다")


def _validate_final_profile_matrix(summary: dict[str, Any], criteria: dict[str, Any], issues: list[str]) -> None:
    final_defaults = run_final_goal_batch.PROFILE_DEFAULTS["final"]
    expected_trajectories = _split_csv(final_defaults["trajectories"])
    expected_winds = _split_csv(final_defaults["wind_modes"])
    expected_seeds = _split_int_csv(final_defaults["seeds"])
    expected_steps = int(final_defaults["total_timesteps"])
    expected_runs = len(expected_trajectories) * len(expected_winds) * len(expected_seeds)

    if list(summary.get("trajectories") or []) != expected_trajectories:
        issues.append(f"final profile trajectory matrix가 canonical과 다릅니다: {summary.get('trajectories')!r}")
    if list(summary.get("wind_modes") or []) != expected_winds:
        issues.append(f"final profile wind matrix가 canonical과 다릅니다: {summary.get('wind_modes')!r}")
    summary_seeds = [_int_or_none(seed) for seed in (summary.get("seeds") or [])]
    if summary_seeds != expected_seeds:
        issues.append(f"final profile seed matrix가 canonical과 다릅니다: {summary.get('seeds')!r}")
    if _int_or_none(summary.get("total_timesteps")) != expected_steps:
        issues.append(f"final profile total_timesteps가 canonical과 다릅니다: {summary.get('total_timesteps')!r}")
    if "max_steps" not in summary or summary.get("max_steps") is not None:
        issues.append(f"final profile max_steps override가 없어야 합니다: {summary.get('max_steps')!r}")
    _validate_canonical_drawn_input("final profile", summary.get("drawn_input"), issues)
    if _int_or_none(criteria.get("expected_runs")) != expected_runs:
        issues.append(f"final profile expected_runs가 canonical과 다릅니다: {criteria.get('expected_runs')!r} != {expected_runs}")


def _validate_source_training_manifest(
    root: Path,
    summary: dict[str, Any],
    issues: list[str],
) -> tuple[Path | None, dict[str, Any]]:
    final_defaults = run_final_goal_batch.PROFILE_DEFAULTS["final"]
    expected_trajectories = _split_csv(final_defaults["trajectories"])
    expected_winds = _split_csv(final_defaults["wind_modes"])
    expected_seeds = _split_int_csv(final_defaults["seeds"])
    expected_steps = int(final_defaults["total_timesteps"])
    expected_keys = _expected_final_matrix_keys()
    expected_runs = len(expected_keys)

    preflight = summary.get("model_manifest_preflight")
    source_manifest_raw = preflight.get("model_manifest") if isinstance(preflight, dict) else None
    if not source_manifest_raw:
        issues.append("final eval의 source training model manifest 경로가 없습니다")
        return None, {}
    source_manifest_path = _resolve_artifact_path(root, str(source_manifest_raw))
    if source_manifest_path is None or not source_manifest_path.exists():
        issues.append(f"final eval의 source training model manifest 파일이 없습니다: {source_manifest_raw}")
        return None, {}
    try:
        source_manifest = _read_json(source_manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"source training model manifest를 읽을 수 없습니다: {exc}")
        return None, {}

    if source_manifest.get("profile") != "final":
        issues.append(f"source training manifest profile이 final이 아닙니다: {source_manifest.get('profile')!r}")
    if source_manifest.get("dry_run") is True:
        issues.append("source training manifest가 dry-run 산출물입니다")
    if source_manifest.get("eval_only") is not False:
        issues.append("source training manifest가 학습 batch 산출물이 아닙니다")
    if list(source_manifest.get("trajectories") or []) != expected_trajectories:
        issues.append(f"source training manifest trajectory matrix가 canonical과 다릅니다: {source_manifest.get('trajectories')!r}")
    if list(source_manifest.get("wind_modes") or []) != expected_winds:
        issues.append(f"source training manifest wind matrix가 canonical과 다릅니다: {source_manifest.get('wind_modes')!r}")
    source_seeds = [_int_or_none(seed) for seed in (source_manifest.get("seeds") or [])]
    if source_seeds != expected_seeds:
        issues.append(f"source training manifest seed matrix가 canonical과 다릅니다: {source_manifest.get('seeds')!r}")
    if _int_or_none(source_manifest.get("total_timesteps")) != expected_steps:
        issues.append(
            f"source training manifest total_timesteps가 canonical과 다릅니다: {source_manifest.get('total_timesteps')!r}"
        )
    if "max_steps" not in source_manifest or source_manifest.get("max_steps") is not None:
        issues.append(f"source training manifest max_steps override가 없어야 합니다: {source_manifest.get('max_steps')!r}")
    _validate_canonical_drawn_input("source training manifest", source_manifest.get("drawn_input"), issues)
    if str(source_manifest.get("trained_action_filter", "")) != run_final_goal_batch.DEFAULT_TRAINED_ACTION_FILTER:
        issues.append(
            "source training manifest action filter가 canonical과 다릅니다: "
            f"{source_manifest.get('trained_action_filter')!r}"
        )
    teacher_window = _float_or_none(source_manifest.get("teacher_window_m"))
    if teacher_window is None or abs(teacher_window - CANONICAL_TEACHER_WINDOW_M) > FLOAT_TOL:
        issues.append(
            "source training manifest teacher_window_m이 canonical과 다릅니다: "
            f"{source_manifest.get('teacher_window_m')!r}"
        )

    source_models = source_manifest.get("models") if isinstance(source_manifest, dict) else None
    if not isinstance(source_models, dict):
        issues.append("source training manifest의 models가 올바른 객체가 아닙니다")
        source_models = {}
    source_model_keys = set(str(key) for key in source_models)
    missing_model_keys = sorted(expected_keys - source_model_keys)
    extra_model_keys = sorted(source_model_keys - expected_keys)
    if missing_model_keys:
        issues.append(f"source training manifest에 canonical model key가 누락되었습니다: {', '.join(missing_model_keys[:5])}")
    if extra_model_keys:
        issues.append(f"source training manifest에 canonical 밖 model key가 있습니다: {', '.join(extra_model_keys[:5])}")
    for key in sorted(expected_keys):
        if key not in source_models:
            continue
        model_path = _resolve_artifact_path(source_manifest_path.parent, str(source_models[key]))
        if model_path is None or not model_path.exists():
            issues.append(f"source training manifest의 model 파일이 없습니다: {key} ({source_models[key]})")
        else:
            _validate_model_archive(model_path, f"source training manifest model {key}", issues)

    source_summary_raw = source_manifest.get("batch_summary")
    if not source_summary_raw:
        issues.append("source training manifest에 batch_summary 경로가 없습니다")
        return source_manifest_path, source_models
    source_summary_path = _resolve_artifact_path(source_manifest_path.parent, str(source_summary_raw))
    if source_summary_path is None or not source_summary_path.exists():
        issues.append(f"source training batch_summary.json 파일이 없습니다: {source_summary_raw}")
        return source_manifest_path, source_models
    try:
        source_summary = _read_json(source_summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"source training batch_summary.json을 읽을 수 없습니다: {exc}")
        return source_manifest_path, source_models

    if source_summary.get("status") != "complete":
        issues.append(f"source training batch status가 complete가 아닙니다: {source_summary.get('status')!r}")
    if source_summary.get("dry_run") is True:
        issues.append("source training batch_summary가 dry-run 산출물입니다")
    if source_summary.get("eval_only") is not False:
        issues.append("source training batch_summary가 학습 batch 산출물이 아닙니다")
    _validate_final_profile_matrix(source_summary, source_summary.get("batch_final_goal_criteria") or {}, issues)
    if _int_or_none(source_summary.get("planned_runs")) != expected_runs:
        issues.append(f"source training planned_runs가 canonical과 다릅니다: {source_summary.get('planned_runs')!r}")
    if _int_or_none(source_summary.get("completed_runs")) != expected_runs:
        issues.append(f"source training completed_runs가 canonical과 다릅니다: {source_summary.get('completed_runs')!r}")
    if _int_or_none(source_summary.get("failed_runs")) not in {0, None}:
        issues.append(f"source training failed_runs가 0이 아닙니다: {source_summary.get('failed_runs')!r}")
    _validate_source_training_matrix(source_summary_path.parent, source_manifest_path, source_models, expected_keys, issues)
    return source_manifest_path, source_models


def _expected_final_matrix_keys() -> set[str]:
    final_defaults = run_final_goal_batch.PROFILE_DEFAULTS["final"]
    trajectories = _split_csv(final_defaults["trajectories"])
    wind_modes = _split_csv(final_defaults["wind_modes"])
    seeds = _split_int_csv(final_defaults["seeds"])
    return {
        f"{trajectory}/{wind_mode}/seed{seed}"
        for trajectory in trajectories
        for wind_mode in wind_modes
        for seed in seeds
    }


def _previous_curriculum_wind(wind_mode: str) -> str | None:
    if wind_mode == "M1":
        return "M0"
    if wind_mode == "M2":
        return "M1"
    return None


def _validate_source_training_curriculum(
    row: dict[str, str],
    source_root: Path,
    source_manifest_path: Path,
    source_models: dict[str, Any],
    issues: list[str],
) -> Path | None:
    key = _matrix_key(row)
    wind_mode = str(row.get("wind_mode", ""))
    row_load_model = _resolve_artifact_path(source_root, row.get("load_model"))
    previous_wind = _previous_curriculum_wind(wind_mode)

    if previous_wind is None:
        if row_load_model is not None:
            issues.append(f"source training M0 row에는 load_model이 없어야 합니다: {key}")
        return None

    if row_load_model is None:
        issues.append(f"source training curriculum load_model이 없습니다: {key}")
        return None

    previous_key = f"{row.get('trajectory')}/{previous_wind}/seed{row.get('seed')}"
    previous_model = source_models.get(previous_key)
    if previous_model is None:
        issues.append(f"source training curriculum 이전 wind model manifest 항목이 없습니다: {key} <- {previous_key}")
        return row_load_model

    expected_model = _resolve_artifact_path(source_manifest_path.parent, str(previous_model))
    if expected_model is None or not _same_resolved_path(row_load_model, expected_model):
        issues.append(f"source training curriculum load_model이 이전 wind 모델과 다릅니다: {key} <- {previous_key}")
    return expected_model


def _validate_source_training_matrix(
    source_root: Path,
    source_manifest_path: Path,
    source_models: dict[str, Any],
    expected_keys: set[str],
    issues: list[str],
) -> None:
    matrix_path = source_root / "matrix_results.csv"
    if not matrix_path.exists():
        issues.append(f"source training matrix_results.csv가 없습니다: {matrix_path}")
        return
    try:
        rows = _read_csv(matrix_path)
    except (OSError, csv.Error) as exc:
        issues.append(f"source training matrix_results.csv를 읽을 수 없습니다: {exc}")
        return

    row_keys = [_matrix_key(row) for row in rows]
    row_key_set = set(row_keys)
    missing_keys = sorted(expected_keys - row_key_set)
    extra_keys = sorted(row_key_set - expected_keys)
    if missing_keys:
        issues.append(f"source training matrix_results.csv에 canonical key가 누락되었습니다: {', '.join(missing_keys[:5])}")
    if extra_keys:
        issues.append(f"source training matrix_results.csv에 canonical 밖 key가 있습니다: {', '.join(extra_keys[:5])}")
    duplicates = sorted({key for key in row_keys if row_keys.count(key) > 1})
    if duplicates:
        issues.append(f"source training matrix_results.csv에 중복 key가 있습니다: {', '.join(duplicates[:5])}")

    for row in rows:
        key = _matrix_key(row)
        if key not in expected_keys:
            continue
        expected_curriculum_model = _validate_source_training_curriculum(
            row,
            source_root,
            source_manifest_path,
            source_models,
            issues,
        )
        if row.get("status") not in {"complete", "reused"}:
            issues.append(f"source training matrix row가 완료 상태가 아닙니다: {key} ({row.get('status')!r})")

        row_model = _resolve_artifact_path(source_root, row.get("model_path"))
        manifest_model = source_models.get(key)
        manifest_model_path = (
            _resolve_artifact_path(source_manifest_path.parent, str(manifest_model))
            if manifest_model is not None
            else None
        )
        if row_model is None:
            issues.append(f"source training matrix_results.csv에 model_path가 없습니다: {key}")
        elif not row_model.exists():
            issues.append(f"source training matrix_results.csv의 model_path 파일이 없습니다: {key} ({row.get('model_path')})")
        else:
            _validate_model_archive(row_model, f"source training matrix_results.csv model {key}", issues)
        if manifest_model_path is not None and row_model is not None and not _same_resolved_path(
            manifest_model_path, row_model
        ):
            issues.append(f"source training manifest model과 matrix_results.csv model_path가 다릅니다: {key}")

        output_dir = _resolve_artifact_path(source_root, row.get("output_dir"))
        if output_dir is None:
            issues.append(f"source training matrix_results.csv에 output_dir이 없습니다: {key}")
            continue
        run_summary_path = output_dir / "summary.json"
        if not run_summary_path.exists():
            issues.append(f"source training per-run summary.json이 없습니다: {key} ({run_summary_path})")
            continue
        try:
            run_summary = _read_json(run_summary_path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"source training per-run summary.json을 읽을 수 없습니다: {key}: {exc}")
            continue
        _validate_source_training_run_config(
            row,
            run_summary,
            issues,
            source_root=source_root,
            expected_curriculum_model=expected_curriculum_model,
        )
        artifacts = run_summary.get("artifacts") if isinstance(run_summary.get("artifacts"), dict) else {}
        artifact_model = _resolve_artifact_path(source_root, artifacts.get("model_path"))
        if artifact_model is None:
            issues.append(f"source training per-run artifacts.model_path가 없습니다: {key}")
        elif not artifact_model.exists():
            issues.append(f"source training per-run artifacts.model_path 파일이 없습니다: {key} ({artifacts.get('model_path')})")
        else:
            _validate_model_archive(artifact_model, f"source training per-run artifacts.model_path {key}", issues)
        if manifest_model_path is not None and artifact_model is not None and not _same_resolved_path(
            manifest_model_path, artifact_model
        ):
            issues.append(f"source training manifest model과 per-run artifacts.model_path가 다릅니다: {key}")


def _validate_final_policy(summary: dict[str, Any], issues: list[str]) -> None:
    expected_filter = run_final_goal_batch.DEFAULT_TRAINED_ACTION_FILTER
    if str(summary.get("trained_action_filter", "")) != expected_filter:
        issues.append(f"final 평가 action filter가 canonical과 다릅니다: {summary.get('trained_action_filter')!r}")
    teacher_window = _float_or_none(summary.get("teacher_window_m"))
    if teacher_window is None or abs(teacher_window - CANONICAL_TEACHER_WINDOW_M) > FLOAT_TOL:
        issues.append(f"final 평가 teacher_window_m이 canonical과 다릅니다: {summary.get('teacher_window_m')!r}")


def _validate_run_config(row: dict[str, str], run_summary: dict[str, Any], issues: list[str]) -> None:
    key = _matrix_key(row)
    config = run_summary.get("config")
    if not isinstance(config, dict):
        issues.append(f"per-run summary config가 없습니다: {key}")
        return

    wind_mode = str(config.get("wind_mode", ""))
    if wind_mode != str(row.get("wind_mode", "")):
        issues.append(f"per-run summary wind_mode가 matrix와 다릅니다: {key} ({wind_mode!r})")

    config_seed = _int_or_none(config.get("seed"))
    row_seed = _int_or_none(row.get("seed"))
    if config_seed != row_seed:
        issues.append(f"per-run summary seed가 matrix와 다릅니다: {key} ({config_seed!r})")

    reference = config.get("reference")
    if not isinstance(reference, dict):
        issues.append(f"per-run summary reference가 없습니다: {key}")
        return
    row_trajectory = str(row.get("trajectory", ""))
    reference_trajectory = str(reference.get("trajectory", ""))
    reference_label = str(reference.get("label", ""))
    if row_trajectory == "square":
        if reference_trajectory != "square":
            issues.append(f"per-run summary reference trajectory가 matrix와 다릅니다: {key} ({reference_trajectory!r})")
    elif row_trajectory == "drawn":
        if reference_trajectory != "drawn":
            issues.append(f"per-run summary reference trajectory가 matrix와 다릅니다: {key} ({reference_trajectory!r})")
        if reference.get("drawn_relative_path") != run_final_goal_batch.CANONICAL_DRAWN_PATH:
            issues.append(
                f"per-run summary drawn path가 canonical과 다릅니다: {key} ({reference.get('drawn_relative_path')!r})"
            )
        if reference.get("drawn_sha256") != run_final_goal_batch.CANONICAL_DRAWN_SHA256:
            issues.append(
                f"per-run summary drawn sha256이 canonical과 다릅니다: {key} ({reference.get('drawn_sha256')!r})"
            )
    elif reference_trajectory != "letter" or reference_label != row_trajectory:
        issues.append(
            "per-run summary letter reference가 matrix와 다릅니다: "
            f"{key} (trajectory={reference_trajectory!r}, label={reference_label!r})"
        )


def _validate_run_policy(row: dict[str, str], config: dict[str, Any], issues: list[str]) -> None:
    key = _matrix_key(row)
    expected_steps = int(run_final_goal_batch.PROFILE_DEFAULTS["final"]["total_timesteps"])
    if _int_or_none(config.get("requested_total_timesteps")) != expected_steps:
        issues.append(
            f"per-run summary requested_total_timesteps가 canonical과 다릅니다: "
            f"{key} ({config.get('requested_total_timesteps')!r})"
        )
    if _int_or_none(config.get("total_timesteps")) != 0:
        issues.append(f"per-run summary eval total_timesteps는 0이어야 합니다: {key} ({config.get('total_timesteps')!r})")
    expected_filter = run_final_goal_batch.DEFAULT_TRAINED_ACTION_FILTER
    if str(config.get("trained_action_filter", "")) != expected_filter:
        issues.append(f"per-run summary action filter가 canonical과 다릅니다: {key} ({config.get('trained_action_filter')!r})")
    teacher_window = _float_or_none(config.get("teacher_window_m"))
    if teacher_window is None or abs(teacher_window - CANONICAL_TEACHER_WINDOW_M) > FLOAT_TOL:
        issues.append(f"per-run summary teacher_window_m이 canonical과 다릅니다: {key} ({config.get('teacher_window_m')!r})")
    if "max_steps_override" not in config or config.get("max_steps_override") is not None:
        issues.append(f"per-run summary max_steps override가 없어야 합니다: {key} ({config.get('max_steps_override')!r})")


def _validate_source_training_run_config(
    row: dict[str, str],
    run_summary: dict[str, Any],
    issues: list[str],
    *,
    source_root: Path,
    expected_curriculum_model: Path | None,
) -> None:
    key = _matrix_key(row)
    config = run_summary.get("config") if isinstance(run_summary.get("config"), dict) else {}
    _validate_run_config(row, run_summary, issues)
    expected_steps = int(run_final_goal_batch.PROFILE_DEFAULTS["final"]["total_timesteps"])
    if config.get("eval_only") is not False:
        issues.append(f"source training per-run summary가 학습 실행이 아닙니다: {key}")
    if _int_or_none(config.get("requested_total_timesteps")) != expected_steps:
        issues.append(
            f"source training per-run requested_total_timesteps가 canonical과 다릅니다: "
            f"{key} ({config.get('requested_total_timesteps')!r})"
        )
    if _int_or_none(config.get("total_timesteps")) != expected_steps:
        issues.append(
            f"source training per-run total_timesteps가 canonical과 다릅니다: {key} ({config.get('total_timesteps')!r})"
        )
    if "max_steps_override" not in config or config.get("max_steps_override") is not None:
        issues.append(
            f"source training per-run max_steps override가 없어야 합니다: {key} ({config.get('max_steps_override')!r})"
        )
    if str(config.get("trained_action_filter", "")) != run_final_goal_batch.DEFAULT_TRAINED_ACTION_FILTER:
        issues.append(
            f"source training per-run action filter가 canonical과 다릅니다: {key} "
            f"({config.get('trained_action_filter')!r})"
        )
    teacher_window = _float_or_none(config.get("teacher_window_m"))
    if teacher_window is None or abs(teacher_window - CANONICAL_TEACHER_WINDOW_M) > FLOAT_TOL:
        issues.append(
            f"source training per-run teacher_window_m이 canonical과 다릅니다: {key} "
            f"({config.get('teacher_window_m')!r})"
        )

    wind_mode = str(row.get("wind_mode", ""))
    config_load_model = _resolve_artifact_path(source_root, config.get("load_model"))
    if _previous_curriculum_wind(wind_mode) is None:
        if config_load_model is not None:
            issues.append(f"source training M0 per-run config.load_model은 비어 있어야 합니다: {key}")
    else:
        if config_load_model is None:
            issues.append(f"source training per-run config.load_model이 없습니다: {key}")
        elif expected_curriculum_model is not None and not _same_resolved_path(config_load_model, expected_curriculum_model):
            issues.append(f"source training per-run config.load_model이 이전 wind 모델과 다릅니다: {key}")


def _validate_run_final_goal_criteria(row: dict[str, str], run_summary: dict[str, Any], issues: list[str]) -> None:
    key = _matrix_key(row)
    criteria = run_summary.get("final_goal_criteria")
    if not isinstance(criteria, dict):
        issues.append(f"per-run summary final_goal_criteria가 없습니다: {key}")
        return
    for criterion_key in FINAL_GOAL_CRITERIA_KEYS:
        if criterion_key not in criteria:
            issues.append(f"per-run summary final criterion이 없습니다: {key} ({criterion_key})")
        elif criteria.get(criterion_key) is not True:
            issues.append(
                f"per-run summary final criterion이 true가 아닙니다: {key} ({criterion_key}={criteria.get(criterion_key)!r})"
            )


def _validate_run_final_goal_metrics(row: dict[str, str], run_summary: dict[str, Any], issues: list[str]) -> None:
    key = _matrix_key(row)
    metrics_rows = run_summary.get("metrics")
    criteria = run_summary.get("final_goal_criteria")
    if not isinstance(metrics_rows, list):
        issues.append(f"per-run summary metrics가 없습니다: {key}")
        return
    if not isinstance(criteria, dict):
        return
    try:
        recomputed = _compute_final_goal_criteria(metrics_rows)
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        issues.append(f"per-run summary metrics로 final_goal_criteria를 재계산할 수 없습니다: {key} ({exc})")
        return
    for criterion_key in (*FINAL_GOAL_CRITERIA_KEYS, "final_goal_pass"):
        if criteria.get(criterion_key) is not recomputed.get(criterion_key):
            issues.append(
                "per-run summary final_goal_criteria가 metrics 재계산 결과와 다릅니다: "
                f"{key} ({criterion_key}: summary={criteria.get(criterion_key)!r}, "
                f"metrics={recomputed.get(criterion_key)!r})"
            )
    if recomputed.get("final_goal_pass") is not True:
        failed = [
            criterion_key
            for criterion_key in FINAL_GOAL_CRITERIA_KEYS
            if recomputed.get(criterion_key) is not True
        ]
        issues.append(f"per-run summary metrics 기준 final_goal_pass가 true가 아닙니다: {key} ({', '.join(failed)})")


def _phase_b_trained_metrics(run_summary: dict[str, Any]) -> dict[str, Any] | None:
    metrics_rows = run_summary.get("metrics")
    if not isinstance(metrics_rows, list):
        return None
    trained = next(
        (item for item in metrics_rows if isinstance(item, dict) and item.get("tag") == "phaseB_trained"),
        None,
    )
    return trained if isinstance(trained, dict) else None


def _validate_matrix_row_against_run_summary(
    row: dict[str, str],
    run_summary: dict[str, Any],
    issues: list[str],
) -> None:
    key = _matrix_key(row)
    if row.get("status") not in {"complete", "reused"}:
        return
    trained = _phase_b_trained_metrics(run_summary)
    if trained is None:
        issues.append(f"per-run summary phaseB_trained metrics가 없습니다: {key}")
        return
    for matrix_key, summary_key in MATRIX_SUMMARY_METRIC_KEYS:
        matrix_value = _float_or_none(row.get(matrix_key))
        summary_value = _float_or_none(trained.get(summary_key))
        if matrix_value is None:
            issues.append(f"matrix_results.csv metric이 없습니다: {key} ({matrix_key})")
            continue
        if summary_value is None:
            issues.append(f"per-run summary phaseB_trained metric이 없습니다: {key} ({summary_key})")
            continue
        if abs(matrix_value - summary_value) > 1e-6:
            issues.append(
                f"matrix_results.csv metric이 per-run summary와 다릅니다: "
                f"{key} ({matrix_key}: matrix={matrix_value!r}, summary={summary_value!r})"
            )


def _validate_eval_source_model(
    root: Path,
    row: dict[str, str],
    run_summary: dict[str, Any],
    issues: list[str],
    source_manifest_path: Path | None = None,
    source_models: dict[str, Any] | None = None,
) -> None:
    key = _matrix_key(row)
    config = run_summary.get("config") if isinstance(run_summary.get("config"), dict) else {}
    artifacts = run_summary.get("artifacts") if isinstance(run_summary.get("artifacts"), dict) else {}

    row_load_model = _resolve_artifact_path(root, row.get("load_model"))
    config_load_model = _resolve_artifact_path(root, config.get("load_model"))
    source_model = _resolve_artifact_path(root, artifacts.get("source_model_path"))

    if row_load_model is None:
        issues.append(f"matrix_results.csv에 eval load_model이 없습니다: {key}")
    elif not row_load_model.exists():
        issues.append(f"matrix_results.csv의 eval load_model 파일이 없습니다: {key} ({row.get('load_model')})")
    else:
        _validate_model_archive(row_load_model, f"matrix_results.csv eval load_model {key}", issues)

    if config_load_model is None:
        issues.append(f"per-run summary config.load_model이 없습니다: {key}")
    elif not config_load_model.exists():
        issues.append(f"per-run summary config.load_model 파일이 없습니다: {key} ({config.get('load_model')})")
    else:
        _validate_model_archive(config_load_model, f"per-run summary config.load_model {key}", issues)

    if source_model is None:
        issues.append(f"per-run summary artifacts.source_model_path가 없습니다: {key}")
    elif not source_model.exists():
        issues.append(f"per-run summary artifacts.source_model_path 파일이 없습니다: {key} ({artifacts.get('source_model_path')})")
    else:
        _validate_model_archive(source_model, f"per-run summary artifacts.source_model_path {key}", issues)

    if row_load_model is not None and config_load_model is not None and not _same_resolved_path(row_load_model, config_load_model):
        issues.append(f"matrix_results.csv load_model과 per-run config.load_model이 다릅니다: {key}")
    if config_load_model is not None and source_model is not None and not _same_resolved_path(config_load_model, source_model):
        issues.append(f"per-run config.load_model과 artifacts.source_model_path가 다릅니다: {key}")

    if source_manifest_path is None or source_models is None:
        return
    source_manifest_model = source_models.get(key)
    if source_manifest_model is None:
        issues.append(f"source training manifest에 eval source model 항목이 없습니다: {key}")
        return
    expected_source_model = _resolve_artifact_path(source_manifest_path.parent, str(source_manifest_model))
    if expected_source_model is None:
        issues.append(f"source training manifest의 model 경로가 비어 있습니다: {key}")
        return
    if row_load_model is not None and not _same_resolved_path(row_load_model, expected_source_model):
        issues.append(f"source training manifest model과 matrix_results.csv load_model이 다릅니다: {key}")
    if config_load_model is not None and not _same_resolved_path(config_load_model, expected_source_model):
        issues.append(f"source training manifest model과 per-run config.load_model이 다릅니다: {key}")
    if source_model is not None and not _same_resolved_path(source_model, expected_source_model):
        issues.append(f"source training manifest model과 artifacts.source_model_path가 다릅니다: {key}")


def _metrics_key(row: dict[str, str], matrix_by_run_id: dict[str, dict[str, str]]) -> str:
    if row.get("trajectory") and row.get("wind_mode") and row.get("seed"):
        return _matrix_key(row)
    run_id = str(row.get("run_id", ""))
    if run_id in matrix_by_run_id:
        return _matrix_key(matrix_by_run_id[run_id])
    return f"<unknown:{run_id or 'missing_run_id'}>"


def _validate_metrics_rows(
    root: Path,
    matrix_rows: list[dict[str, str]],
    metrics_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    matrix_by_key = {_matrix_key(row): row for row in matrix_rows}
    matrix_by_run_id = {str(row.get("run_id", "")): row for row in matrix_rows if row.get("run_id")}
    metric_keys = [_metrics_key(row, matrix_by_run_id) for row in metrics_rows]
    duplicate_metric_keys = sorted({key for key in metric_keys if metric_keys.count(key) > 1})
    if duplicate_metric_keys:
        issues.append(f"metrics.csv에 중복 matrix key가 있습니다: {', '.join(duplicate_metric_keys[:5])}")

    matrix_keys = set(matrix_by_key)
    metric_key_set = set(metric_keys)
    missing_metric_keys = sorted(matrix_keys - metric_key_set)
    extra_metric_keys = sorted(metric_key_set - matrix_keys)
    if missing_metric_keys:
        issues.append(f"metrics.csv에 matrix row가 누락되었습니다: {', '.join(missing_metric_keys[:5])}")
    if extra_metric_keys:
        issues.append(f"metrics.csv에 matrix 밖 row가 있습니다: {', '.join(extra_metric_keys[:5])}")

    for row, key in zip(metrics_rows, metric_keys):
        matrix_row = matrix_by_key.get(key)
        if matrix_row is None:
            continue
        if _row_truth(row.get("final_goal_pass")) != _row_truth(matrix_row.get("final_goal_pass")):
            issues.append(f"metrics.csv final_goal_pass가 matrix_results.csv와 다릅니다: {key}")
        for matrix_key, _summary_key in MATRIX_SUMMARY_METRIC_KEYS:
            metric_value = _float_or_none(row.get(matrix_key))
            matrix_value = _float_or_none(matrix_row.get(matrix_key))
            if metric_value is None and matrix_value is None:
                continue
            if metric_value is None or matrix_value is None or abs(metric_value - matrix_value) > 1e-6:
                issues.append(f"metrics.csv metric이 matrix_results.csv와 다릅니다: {key} ({matrix_key})")
        metric_model = _resolve_artifact_path(root, row.get("model_path"))
        matrix_model = _resolve_artifact_path(root, matrix_row.get("model_path"))
        if metric_model is not None and matrix_model is not None and not _same_resolved_path(metric_model, matrix_model):
            issues.append(f"metrics.csv model_path가 matrix_results.csv와 다릅니다: {key}")


def _expected_aggregate_one(group_name: str, group_value: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group": group_name,
        "value": group_value,
        "n": len(rows),
        "final_goal_pass_count": sum(1 for row in rows if _row_truth(row.get("final_goal_pass")) == "True"),
        "overall_pass_count": sum(1 for row in rows if _row_truth(row.get("overall_pass")) == "True"),
    }
    out["final_goal_pass_rate"] = out["final_goal_pass_count"] / max(len(rows), 1)
    out["overall_pass_rate"] = out["overall_pass_count"] / max(len(rows), 1)
    for metric in run_final_goal_batch.AGGREGATE_METRICS:
        vals = [_float_or_none(row.get(metric)) for row in rows]
        finite_vals = [val for val in vals if val is not None]
        if not finite_vals:
            out[f"{metric}_mean"] = None
            out[f"{metric}_std"] = None
            out[f"{metric}_min"] = None
            out[f"{metric}_max"] = None
            continue
        mean = sum(finite_vals) / len(finite_vals)
        var = sum((val - mean) ** 2 for val in finite_vals) / len(finite_vals)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = var ** 0.5
        out[f"{metric}_min"] = min(finite_vals)
        out[f"{metric}_max"] = max(finite_vals)
    return out


def _expected_aggregate_rows(matrix_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    if not matrix_rows:
        return {}
    groups: list[tuple[str, str, list[dict[str, str]]]] = [("overall", "all", matrix_rows)]
    for trajectory in sorted({str(row.get("trajectory")) for row in matrix_rows}):
        groups.append(("trajectory", trajectory, [row for row in matrix_rows if str(row.get("trajectory")) == trajectory]))
    for wind_mode in sorted({str(row.get("wind_mode")) for row in matrix_rows}):
        groups.append(("wind_mode", wind_mode, [row for row in matrix_rows if str(row.get("wind_mode")) == wind_mode]))
    for trajectory in sorted({str(row.get("trajectory")) for row in matrix_rows}):
        for wind_mode in sorted({str(row.get("wind_mode")) for row in matrix_rows}):
            subset = [
                row
                for row in matrix_rows
                if str(row.get("trajectory")) == trajectory and str(row.get("wind_mode")) == wind_mode
            ]
            if subset:
                groups.append(("trajectory_wind", f"{trajectory}/{wind_mode}", subset))
    return {
        (group_name, group_value): _expected_aggregate_one(group_name, group_value, rows)
        for group_name, group_value, rows in groups
    }


def _validate_aggregate_metric_values(
    matrix_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    expected_by_key = _expected_aggregate_rows(matrix_rows)
    actual_by_key = {
        (str(row.get("group", "")).strip(), str(row.get("value", "")).strip()): row
        for row in aggregate_rows
    }
    missing_keys = sorted(set(expected_by_key) - set(actual_by_key))
    extra_keys = sorted(set(actual_by_key) - set(expected_by_key))
    if missing_keys:
        preview = ", ".join(f"{group}/{value}" for group, value in missing_keys[:5])
        issues.append(f"aggregate_metrics.csv에 필요한 group row가 없습니다: {preview}")
    if extra_keys:
        preview = ", ".join(f"{group}/{value}" for group, value in extra_keys[:5])
        issues.append(f"aggregate_metrics.csv에 matrix 기준 외 group row가 있습니다: {preview}")

    for key, expected in expected_by_key.items():
        actual = actual_by_key.get(key)
        if actual is None:
            continue
        label = f"{key[0]}/{key[1]}"
        int_fields = ("n", "final_goal_pass_count", "overall_pass_count")
        for field in int_fields:
            actual_value = _int_or_none(actual.get(field))
            if actual_value != expected[field]:
                issues.append(
                    f"aggregate_metrics.csv {label} {field}가 matrix_results.csv 재계산 결과와 다릅니다: "
                    f"{actual_value!r} != {expected[field]!r}"
                )
        rate_fields = ("final_goal_pass_rate", "overall_pass_rate")
        for field in rate_fields:
            actual_value = _float_or_none(actual.get(field))
            if actual_value is None or abs(actual_value - float(expected[field])) > AGGREGATE_FLOAT_TOL:
                issues.append(
                    f"aggregate_metrics.csv {label} {field}가 matrix_results.csv 재계산 결과와 다릅니다: "
                    f"{actual_value!r} != {expected[field]!r}"
                )
        for metric in run_final_goal_batch.AGGREGATE_METRICS:
            for suffix in ("mean", "std", "min", "max"):
                field = f"{metric}_{suffix}"
                expected_value = expected[field]
                actual_raw = actual.get(field)
                if expected_value is None:
                    if str(actual_raw or "").strip():
                        issues.append(f"aggregate_metrics.csv {label} {field}는 비어 있어야 합니다: {actual_raw!r}")
                    continue
                actual_value = _float_or_none(actual_raw)
                if actual_value is None or abs(actual_value - float(expected_value)) > AGGREGATE_FLOAT_TOL:
                    issues.append(
                        f"aggregate_metrics.csv {label} {field}가 matrix_results.csv 재계산 결과와 다릅니다: "
                        f"{actual_value!r} != {expected_value!r}"
                    )


def _validate_aggregate_rows(
    matrix_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    overall_rows = [
        row
        for row in aggregate_rows
        if str(row.get("group", "")).strip() == "overall" and str(row.get("value", "")).strip() == "all"
    ]
    if not overall_rows:
        issues.append("aggregate_metrics.csv에 overall/all row가 없습니다")
        return
    if len(overall_rows) > 1:
        issues.append("aggregate_metrics.csv에 overall/all row가 중복되었습니다")

    overall = overall_rows[0]
    expected_n = len(matrix_rows)
    expected_final_count = sum(1 for row in matrix_rows if _row_truth(row.get("final_goal_pass")) == "True")
    expected_overall_count = sum(1 for row in matrix_rows if _row_truth(row.get("overall_pass")) == "True")
    expected_final_rate = expected_final_count / max(expected_n, 1)
    expected_overall_rate = expected_overall_count / max(expected_n, 1)

    actual_n = _int_or_none(overall.get("n"))
    actual_final_count = _int_or_none(overall.get("final_goal_pass_count"))
    actual_overall_count = _int_or_none(overall.get("overall_pass_count"))
    actual_final_rate = _float_or_none(overall.get("final_goal_pass_rate"))
    actual_overall_rate = _float_or_none(overall.get("overall_pass_rate"))

    if actual_n != expected_n:
        issues.append(f"aggregate_metrics.csv overall n이 matrix_results.csv와 다릅니다: {actual_n!r} != {expected_n}")
    if actual_final_count != expected_final_count:
        issues.append(
            "aggregate_metrics.csv final_goal_pass_count가 matrix_results.csv와 다릅니다: "
            f"{actual_final_count!r} != {expected_final_count}"
        )
    if actual_overall_count != expected_overall_count:
        issues.append(
            "aggregate_metrics.csv overall_pass_count가 matrix_results.csv와 다릅니다: "
            f"{actual_overall_count!r} != {expected_overall_count}"
        )
    if actual_final_rate is None or abs(actual_final_rate - expected_final_rate) > FLOAT_TOL:
        issues.append(
            "aggregate_metrics.csv final_goal_pass_rate가 matrix_results.csv와 다릅니다: "
            f"{actual_final_rate!r} != {expected_final_rate}"
        )
    if actual_overall_rate is None or abs(actual_overall_rate - expected_overall_rate) > FLOAT_TOL:
        issues.append(
            "aggregate_metrics.csv overall_pass_rate가 matrix_results.csv와 다릅니다: "
            f"{actual_overall_rate!r} != {expected_overall_rate}"
        )
    _validate_aggregate_metric_values(matrix_rows, aggregate_rows, issues)


def _recompute_batch_criteria_from_matrix(matrix_rows: list[dict[str, str]]) -> dict[str, Any]:
    final_defaults = run_final_goal_batch.PROFILE_DEFAULTS["final"]
    trajectories = _split_csv(final_defaults["trajectories"])
    wind_modes = _split_csv(final_defaults["wind_modes"])
    seeds = _split_int_csv(final_defaults["seeds"])
    expected_keys = {
        (trajectory, wind_mode, int(seed))
        for trajectory in trajectories
        for wind_mode in wind_modes
        for seed in seeds
    }
    completed_rows = [row for row in matrix_rows if row.get("status") in {"complete", "reused"}]
    lookup: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in completed_rows:
        seed = _int_or_none(row.get("seed"))
        if seed is None:
            continue
        lookup[(str(row.get("trajectory")), str(row.get("wind_mode")), int(seed))] = row
    observed_keys = set(lookup)
    missing = sorted(expected_keys - observed_keys)

    robustness_failures: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for seed in seeds:
            base = lookup.get((trajectory, "M0", int(seed)))
            if base is None:
                continue
            base_iou = _float_or_none(base.get("painting_iou"))
            base_path = _float_or_none(base.get("path_rmse_m"))
            for wind_mode in ("M1", "M2"):
                row = lookup.get((trajectory, wind_mode, int(seed)))
                if row is None:
                    continue
                iou = _float_or_none(row.get("painting_iou"))
                path = _float_or_none(row.get("path_rmse_m"))
                iou_ok = base_iou is not None and iou is not None and iou >= base_iou * run_final_goal_batch.ROBUST_IOU_REL_MIN
                path_ok = base_path is not None and path is not None and path <= base_path * run_final_goal_batch.ROBUST_PATH_RMSE_REL_MAX
                if not iou_ok or not path_ok:
                    robustness_failures.append(
                        {
                            "trajectory": trajectory,
                            "wind_mode": wind_mode,
                            "seed": int(seed),
                            "m0_painting_iou": base_iou,
                            "painting_iou": iou,
                            "m0_path_rmse_m": base_path,
                            "path_rmse_m": path,
                        }
                    )

    matrix_complete = len(missing) == 0 and len(observed_keys) == len(expected_keys)
    all_run_final = bool(completed_rows) and all(
        _row_truth(row.get("final_goal_pass")) == "True" for row in completed_rows
    )
    robustness_pass = len(robustness_failures) == 0
    return {
        "matrix_complete": matrix_complete,
        "expected_runs": len(expected_keys),
        "observed_runs": len(observed_keys),
        "missing_runs": missing,
        "all_runs_final_goal_pass": all_run_final,
        "robustness_relative_to_m0_pass": robustness_pass,
        "robustness_failures": robustness_failures,
        "batch_final_goal_pass": bool(matrix_complete and all_run_final and robustness_pass),
    }


def _validate_batch_final_goal_criteria_from_matrix(
    criteria: dict[str, Any],
    matrix_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    recomputed = _recompute_batch_criteria_from_matrix(matrix_rows)
    bool_fields = (
        "matrix_complete",
        "all_runs_final_goal_pass",
        "robustness_relative_to_m0_pass",
        "batch_final_goal_pass",
    )
    for field in bool_fields:
        if criteria.get(field) is not recomputed[field]:
            issues.append(
                f"batch_final_goal_criteria.{field}가 matrix_results.csv 재계산 결과와 다릅니다: "
                f"{criteria.get(field)!r} != {recomputed[field]!r}"
            )
    for field in ("expected_runs", "observed_runs"):
        if _int_or_none(criteria.get(field)) != recomputed[field]:
            issues.append(
                f"batch_final_goal_criteria.{field}가 matrix_results.csv 재계산 결과와 다릅니다: "
                f"{criteria.get(field)!r} != {recomputed[field]!r}"
            )
    if recomputed["robustness_failures"]:
        preview = ", ".join(
            f"{item['trajectory']}/{item['wind_mode']}/seed{item['seed']}"
            for item in recomputed["robustness_failures"][:5]
        )
        issues.append(f"matrix_results.csv 기준 robustness failure가 있습니다: {preview}")


def _validate_summary_counters(
    summary: dict[str, Any],
    matrix_rows: list[dict[str, str]],
    issues: list[str],
) -> None:
    if summary.get("status") != "complete":
        return

    expected_n = len(matrix_rows)
    completed_count = sum(1 for row in matrix_rows if row.get("status") in {"complete", "reused"})
    failed_count = sum(1 for row in matrix_rows if row.get("status") == "failed")
    final_count = sum(1 for row in matrix_rows if _row_truth(row.get("final_goal_pass")) == "True")
    final_rate = final_count / max(expected_n, 1)

    planned_runs = _int_or_none(summary.get("planned_runs"))
    completed_runs = _int_or_none(summary.get("completed_runs"))
    failed_runs = _int_or_none(summary.get("failed_runs"))
    final_goal_pass_count = _int_or_none(summary.get("final_goal_pass_count"))
    final_goal_pass_rate = _float_or_none(summary.get("final_goal_pass_rate"))

    if planned_runs != expected_n:
        issues.append(f"batch_summary.json planned_runs가 matrix_results.csv 행 수와 다릅니다: {planned_runs!r} != {expected_n}")
    if completed_runs != completed_count:
        issues.append(
            f"batch_summary.json completed_runs가 matrix_results.csv 상태와 다릅니다: {completed_runs!r} != {completed_count}"
        )
    if failed_runs != failed_count:
        issues.append(f"batch_summary.json failed_runs가 matrix_results.csv 상태와 다릅니다: {failed_runs!r} != {failed_count}")
    if final_goal_pass_count != final_count:
        issues.append(
            "batch_summary.json final_goal_pass_count가 matrix_results.csv와 다릅니다: "
            f"{final_goal_pass_count!r} != {final_count}"
        )
    if final_goal_pass_rate is None or abs(final_goal_pass_rate - final_rate) > FLOAT_TOL:
        issues.append(
            "batch_summary.json final_goal_pass_rate가 matrix_results.csv와 다릅니다: "
            f"{final_goal_pass_rate!r} != {final_rate}"
        )


def verify_batch_artifacts(batch_dir: str | Path, required_profile: str = "final") -> VerificationResult:
    root = _resolve_batch_dir(batch_dir)
    issues: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            issues.append(f"필수 산출물이 없습니다: {name}")

    summary_path = root / "batch_summary.json"
    matrix_path = root / "matrix_results.csv"
    metrics_path = root / "metrics.csv"
    aggregate_path = root / "aggregate_metrics.csv"
    summary: dict[str, Any] = {}
    matrix_rows: list[dict[str, str]] = []
    metrics_rows: list[dict[str, str]] = []
    aggregate_rows: list[dict[str, str]] = []
    model_manifest: dict[str, Any] = {}
    source_manifest_path: Path | None = None
    source_models: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = _read_json(summary_path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"batch_summary.json을 읽을 수 없습니다: {exc}")
    if matrix_path.exists():
        try:
            matrix_rows = _read_csv(matrix_path)
        except (OSError, csv.Error) as exc:
            issues.append(f"matrix_results.csv를 읽을 수 없습니다: {exc}")
    if metrics_path.exists():
        try:
            metrics_rows = _read_csv(metrics_path)
        except (OSError, csv.Error) as exc:
            issues.append(f"metrics.csv를 읽을 수 없습니다: {exc}")
    if aggregate_path.exists():
        try:
            aggregate_rows = _read_csv(aggregate_path)
        except (OSError, csv.Error) as exc:
            issues.append(f"aggregate_metrics.csv를 읽을 수 없습니다: {exc}")
    manifest_path = root / "best_model_manifest.json"
    if manifest_path.exists():
        try:
            model_manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"best_model_manifest.json을 읽을 수 없습니다: {exc}")

    profile = str(summary.get("profile", ""))
    if required_profile != "any" and profile != required_profile:
        issues.append(f"profile이 {required_profile!r}이 아닙니다: {profile!r}")
    if summary.get("dry_run") is True:
        issues.append("dry_run 산출물은 최종 목표 달성 증거가 아닙니다")
    if summary.get("status") != "complete":
        issues.append(f"batch status가 complete가 아닙니다: {summary.get('status')!r}")
    if required_profile == "final" and summary.get("eval_only") is not True:
        issues.append("최종 목표 검증은 eval_only batch 산출물이어야 합니다")
    if required_profile == "final":
        _validate_final_policy(summary, issues)

    dep = summary.get("dependency_preflight")
    if isinstance(dep, dict) and dep.get("ok") is False:
        issues.append("dependency_preflight가 실패했습니다")
    batch_preflight = summary.get("batch_preflight")
    if isinstance(batch_preflight, dict) and batch_preflight.get("ok") is False:
        issues.append("batch_preflight가 실패했습니다")
    manifest_dep = summary.get("model_manifest_preflight")
    if isinstance(manifest_dep, dict) and manifest_dep.get("ok") is False:
        issues.append("model_manifest_preflight가 실패했습니다")
    if required_profile == "final":
        source_manifest_path, source_models = _validate_source_training_manifest(root, summary, issues)

    criteria = summary.get("batch_final_goal_criteria") or {}
    if required_profile == "final":
        _validate_final_profile_matrix(summary, criteria, issues)
        _validate_batch_final_goal_criteria_from_matrix(criteria, matrix_rows, issues)
    if criteria.get("matrix_complete") is not True:
        issues.append("matrix_complete가 true가 아닙니다")
    if criteria.get("all_runs_final_goal_pass") is not True:
        issues.append("all_runs_final_goal_pass가 true가 아닙니다")
    if criteria.get("robustness_relative_to_m0_pass") is not True:
        issues.append("robustness_relative_to_m0_pass가 true가 아닙니다")
    if criteria.get("batch_final_goal_pass") is not True:
        issues.append("batch_final_goal_pass가 true가 아닙니다")

    expected = int(criteria.get("expected_runs") or 0)
    observed = int(criteria.get("observed_runs") or 0)
    if expected <= 0:
        issues.append("expected_runs가 0보다 커야 합니다")
    if observed != expected:
        issues.append(f"observed_runs({observed})가 expected_runs({expected})와 다릅니다")
    if len(matrix_rows) != expected:
        issues.append(f"matrix_results.csv 행 수({len(matrix_rows)})가 expected_runs({expected})와 다릅니다")
    _validate_summary_counters(summary, matrix_rows, issues)

    bad_matrix_rows = [
        row
        for row in matrix_rows
        if row.get("status") not in {"complete", "reused"} or row.get("final_goal_pass") != "True"
    ]
    if bad_matrix_rows:
        preview = ", ".join(
            f"{_matrix_key(row)}:{row.get('status')}/{row.get('final_goal_pass')}"
            for row in bad_matrix_rows[:5]
        )
        issues.append(f"matrix_results.csv에 통과하지 못한 행이 있습니다: {preview}")

    row_keys = [_matrix_key(row) for row in matrix_rows]
    duplicate_keys = sorted({key for key in row_keys if row_keys.count(key) > 1})
    if duplicate_keys:
        issues.append(f"matrix_results.csv에 중복 matrix key가 있습니다: {', '.join(duplicate_keys[:5])}")
    if required_profile == "final":
        expected_final_keys = _expected_final_matrix_keys()
        actual_keys = set(row_keys)
        missing_keys = sorted(expected_final_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_final_keys)
        if missing_keys:
            issues.append(f"matrix_results.csv에 canonical final key가 누락되었습니다: {', '.join(missing_keys[:5])}")
        if extra_keys:
            issues.append(f"matrix_results.csv에 canonical 밖 key가 있습니다: {', '.join(extra_keys[:5])}")
    if metrics_rows:
        _validate_metrics_rows(root, matrix_rows, metrics_rows, issues)
    elif matrix_rows and (root / "metrics.csv").exists():
        issues.append("metrics.csv에 row가 없습니다")
    if aggregate_rows:
        _validate_aggregate_rows(matrix_rows, aggregate_rows, issues)
    elif matrix_rows and aggregate_path.exists():
        issues.append("aggregate_metrics.csv에 row가 없습니다")

    models = model_manifest.get("models") if isinstance(model_manifest, dict) else None
    if not isinstance(models, dict):
        issues.append("best_model_manifest.json의 models가 올바른 객체가 아닙니다")
        models = {}

    for row in matrix_rows:
        key = _matrix_key(row)
        should_crosscheck_run = row.get("status") in {"complete", "reused"} or row.get("final_goal_pass") == "True"
        if not should_crosscheck_run:
            continue
        output_dir = _resolve_artifact_path(root, row.get("output_dir"))
        if output_dir is None:
            issues.append(f"matrix_results.csv에 output_dir가 없습니다: {key}")
            continue
        run_summary_path = output_dir / "summary.json"
        if not run_summary_path.exists():
            issues.append(f"per-run summary.json이 없습니다: {key} ({run_summary_path})")
            continue
        try:
            run_summary = _read_json(run_summary_path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"per-run summary.json을 읽을 수 없습니다: {key}: {exc}")
            continue
        _validate_run_config(row, run_summary, issues)
        config = run_summary.get("config") if isinstance(run_summary.get("config"), dict) else {}
        if required_profile == "final" and config.get("eval_only") is not True:
            issues.append(f"per-run summary가 eval_only 실행이 아닙니다: {key}")
        if required_profile == "final":
            _validate_run_policy(row, config, issues)
            _validate_eval_source_model(root, row, run_summary, issues, source_manifest_path, source_models)
            _validate_run_final_goal_criteria(row, run_summary, issues)
            _validate_run_final_goal_metrics(row, run_summary, issues)
            _validate_matrix_row_against_run_summary(row, run_summary, issues)
        run_final = (run_summary.get("final_goal_criteria") or {}).get("final_goal_pass")
        if row.get("final_goal_pass") == "True" and run_final is not True:
            issues.append(f"matrix_results.csv와 per-run summary final_goal_pass가 다릅니다: {key}")
        if row.get("status") in {"complete", "reused"} and row.get("final_goal_pass") == "True" and run_final is not True:
            issues.append(f"per-run summary final_goal_pass가 true가 아닙니다: {key}")

        manifest_model = models.get(key)
        manifest_model_path: Path | None = None
        if not manifest_model:
            issues.append(f"best_model_manifest.json에 model 항목이 없습니다: {key}")
        else:
            manifest_model_path = _resolve_artifact_path(root, str(manifest_model))
            if manifest_model_path is None or not manifest_model_path.exists():
                issues.append(f"best_model_manifest.json의 model 파일이 없습니다: {key} ({manifest_model})")
            else:
                _validate_model_archive(manifest_model_path, f"best_model_manifest.json model {key}", issues)

        row_model = _resolve_artifact_path(root, row.get("model_path"))
        artifacts = run_summary.get("artifacts") if isinstance(run_summary.get("artifacts"), dict) else {}
        artifact_model = _resolve_artifact_path(root, artifacts.get("model_path"))
        if row.get("status") in {"complete", "reused"} and row.get("final_goal_pass") == "True":
            if row_model is None:
                issues.append(f"matrix_results.csv에 model_path가 없습니다: {key}")
            elif not row_model.exists():
                issues.append(f"matrix_results.csv의 model_path 파일이 없습니다: {key} ({row.get('model_path')})")
            else:
                _validate_model_archive(row_model, f"matrix_results.csv model_path {key}", issues)
            if artifact_model is None:
                issues.append(f"per-run summary artifacts.model_path가 없습니다: {key}")
            elif not artifact_model.exists():
                issues.append(f"per-run summary artifacts.model_path 파일이 없습니다: {key} ({artifacts.get('model_path')})")
            else:
                _validate_model_archive(artifact_model, f"per-run summary artifacts.model_path {key}", issues)
            if manifest_model_path is not None and row_model is not None and not _same_resolved_path(
                manifest_model_path, row_model
            ):
                issues.append(f"best_model_manifest.json model과 matrix_results.csv model_path가 다릅니다: {key}")
            if manifest_model_path is not None and artifact_model is not None and not _same_resolved_path(
                manifest_model_path, artifact_model
            ):
                issues.append(f"best_model_manifest.json model과 per-run artifacts.model_path가 다릅니다: {key}")
            if row_model is not None and artifact_model is not None and not _same_resolved_path(row_model, artifact_model):
                issues.append(f"matrix_results.csv model_path와 per-run artifacts.model_path가 다릅니다: {key}")

    return VerificationResult(ok=not issues, issues=issues, summary=summary, matrix_rows=matrix_rows)


def _print_report(result: VerificationResult, batch_dir: Path) -> None:
    summary = result.summary
    criteria = summary.get("batch_final_goal_criteria") or {}
    print(f"[verify] batch dir: {batch_dir}")
    print(f"[verify] profile: {summary.get('profile', '-')}")
    print(f"[verify] status: {summary.get('status', '-')}")
    print(f"[verify] planned/observed/expected: {summary.get('planned_runs', '-')}/{criteria.get('observed_runs', '-')}/{criteria.get('expected_runs', '-')}")
    print(f"[verify] batch_final_goal_pass: {criteria.get('batch_final_goal_pass', '-')}")
    if result.ok:
        print("[verify] 최종 목표 산출물 검증 통과")
    else:
        print("[verify] 최종 목표 산출물 검증 실패")
        for issue in result.issues:
            print(f"[verify] - {issue}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LightPaint final-goal batch artifacts.")
    parser.add_argument("batch_dir", help="Batch output directory or batch_summary.json path.")
    parser.add_argument(
        "--required-profile",
        default="final",
        help="Required batch profile. Use 'any' for intermediate diagnostics.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    root = _resolve_batch_dir(args.batch_dir)
    result = verify_batch_artifacts(root, required_profile=str(args.required_profile))
    _print_report(result, root)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
