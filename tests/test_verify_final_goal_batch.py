from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from src.train import run_final_goal_batch as batch
from src.train import verify_final_goal_batch as verify
from src.train.verify_final_goal_batch import verify_batch_artifacts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_model_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "{}")
        archive.writestr("policy.pth", b"placeholder")
        archive.writestr("_stable_baselines3_version", "test")


def _passing_final_goal_criteria() -> dict[str, bool]:
    criteria = {key: True for key in verify.FINAL_GOAL_CRITERIA_KEYS}
    criteria["final_goal_pass"] = True
    return criteria


def _passing_summary_metrics(
    *,
    painting_iou: str = "0.800000",
    path_rmse_m: str = "0.040000",
) -> list[dict[str, object]]:
    return [
        {
            "tag": "phaseB_zero",
            "painted_pixel_iou": "0.760000",
            "painted_pixel_precision": "0.860000",
            "painted_pixel_recall": "0.860000",
            "painted_pixel_dice": "0.810000",
            "off_target_pixel_ratio": "0.090000",
            "path_rmse_m": "0.050000",
            "path_max_m": "0.150000",
            "straight_path_rmse_m": "0.040000",
            "corner_path_rmse_m": "0.100000",
            "corner_speed_ratio": "1.000000",
            "corner_overshoot_m": "0.100000",
            "led_precision": "0.900000",
            "led_recall": "0.900000",
            "led_flicker_rate": "0.050000",
            "crash_rate": "0.000000",
            "out_of_bounds_rate": "0.000000",
            "rpm_saturation_ratio": "0.000000",
            "action_norm_max": "0.000000",
            "action_rate_max": "0.000000",
        },
        {
            "tag": "phaseB_trained",
            "painted_pixel_iou": painting_iou,
            "painted_pixel_precision": "0.900000",
            "painted_pixel_recall": "0.900000",
            "painted_pixel_dice": "0.850000",
            "off_target_pixel_ratio": "0.050000",
            "path_rmse_m": path_rmse_m,
            "path_max_m": "0.120000",
            "straight_path_rmse_m": "0.040000",
            "corner_path_rmse_m": "0.070000",
            "corner_speed_ratio": "0.750000",
            "corner_speed_vs_straight_ratio": "0.750000",
            "corner_overshoot_m": "0.070000",
            "led_precision": "0.950000",
            "led_recall": "0.950000",
            "led_flicker_rate": "0.060000",
            "crash_rate": "0.000000",
            "out_of_bounds_rate": "0.000000",
            "rpm_saturation_ratio": "0.050000",
            "action_norm_mean": "0.800000",
            "action_norm_max": "1.500000",
            "action_rate_mean": "1.200000",
            "action_rate_max": "3.000000",
        },
    ]


def _write_batch_fixture(root: Path, *, pass_batch: bool = True, profile: str = "final") -> None:
    run_id = "L_M0_seed7"
    run_dir = root / run_id
    model_path = run_dir / "models" / "ppo_phaseB_L_M0_corner.zip"
    _write_model_zip(model_path)
    source_model_path = root / "source_models" / "ppo_phaseB_L_M0_corner.zip"
    _write_model_zip(source_model_path)
    run_summary = {
        "config": {
            "phase": "B",
            "wind_mode": "M0",
            "reference": {"trajectory": "letter", "label": "L"},
            "seed": 7,
            "load_model": str(source_model_path),
            "eval_only": True,
            "total_timesteps": 0,
            "requested_total_timesteps": int(batch.PROFILE_DEFAULTS["final"]["total_timesteps"]),
            "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
            "teacher_window_m": 0.15,
            "max_steps_override": None,
        },
        "final_goal_criteria": _passing_final_goal_criteria() if pass_batch else {"final_goal_pass": False},
        "metrics": _passing_summary_metrics(),
        "artifacts": {
            "model_path": str(model_path),
            "source_model_path": str(source_model_path),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
    criteria = {
        "matrix_complete": pass_batch,
        "expected_runs": 1,
        "observed_runs": 1 if pass_batch else 0,
        "missing_runs": [] if pass_batch else [{"trajectory": "L", "wind_mode": "M0", "seed": 7}],
        "all_runs_final_goal_pass": pass_batch,
        "failed_final_goal_run_count": 0 if pass_batch else 1,
        "failed_final_goal_runs": [] if pass_batch else [{"run_id": "L_M0_seed7"}],
        "robustness_relative_to_m0_pass": pass_batch,
        "robustness_failures": [],
        "batch_final_goal_pass": pass_batch,
    }
    summary = {
        "schema_version": 1,
        "profile": profile,
        "dry_run": False,
        "status": "complete",
        "eval_only": True,
        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
        "teacher_window_m": 0.15,
        "planned_runs": 1,
        "completed_runs": 1 if pass_batch else 0,
        "failed_runs": 0 if pass_batch else 1,
        "final_goal_pass_count": 1 if pass_batch else 0,
        "final_goal_pass_rate": 1.0 if pass_batch else 0.0,
        "dependency_preflight": {"ok": True},
        "batch_final_goal_criteria": criteria,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    matrix_pass = pass_batch
    matrix_rows = [
        {
            "trajectory": "L",
            "wind_mode": "M0",
            "seed": 7,
            "run_id": run_id,
            "status": "complete" if matrix_pass else "failed",
            "overall_pass": "True" if matrix_pass else "False",
            "final_goal_pass": "True" if matrix_pass else "False",
            "model_path": str(model_path),
            "output_dir": str(run_dir),
            "load_model": str(source_model_path),
        }
    ]
    _write_csv(root / "matrix_results.csv", matrix_rows)
    _write_csv(
        root / "metrics.csv",
        [
            {
                "run_id": "L_M0_seed7",
                "overall_pass": matrix_rows[0]["overall_pass"],
                "final_goal_pass": matrix_rows[0]["final_goal_pass"],
            }
        ],
    )
    _write_csv(
        root / "aggregate_metrics.csv",
        list(verify._expected_aggregate_rows(matrix_rows).values()),
    )
    (root / "best_model_manifest.json").write_text(
        json.dumps({"schema_version": 1, "models": {"L/M0/seed7": str(model_path)}}),
        encoding="utf-8",
    )


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _split_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _split_csv(raw)]


def _write_canonical_final_fixture(root: Path) -> None:
    defaults = batch.PROFILE_DEFAULTS["final"]
    trajectories = _split_csv(defaults["trajectories"])
    wind_modes = _split_csv(defaults["wind_modes"])
    seeds = _split_int_csv(defaults["seeds"])
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    models: dict[str, str] = {}
    source_models: dict[str, str] = {}
    drawn_input = {
        "input_arg": batch.CANONICAL_DRAWN_PATH,
        "path": str(root / batch.CANONICAL_DRAWN_PATH),
        "relative_path": batch.CANONICAL_DRAWN_PATH,
        "exists": True,
        "sha256": batch.CANONICAL_DRAWN_SHA256,
        "size_bytes": 123,
    }
    for trajectory in trajectories:
        for wind_mode in wind_modes:
            for seed in seeds:
                run_id = f"{trajectory}_{wind_mode}_seed{seed}"
                run_dir = root / run_id
                model_path = run_dir / "models" / f"ppo_phaseB_{trajectory}_{wind_mode}_corner.zip"
                _write_model_zip(model_path)
                source_model_path = root / "source_models" / f"ppo_phaseB_{trajectory}_{wind_mode}_seed{seed}_corner.zip"
                _write_model_zip(source_model_path)
                source_run_dir = root / "source_train" / run_id
                reference = (
                    {"trajectory": "square", "label": "square"}
                    if trajectory == "square"
                    else {
                        "trajectory": "drawn",
                        "label": "drawn",
                        "drawn_relative_path": batch.CANONICAL_DRAWN_PATH,
                        "drawn_sha256": batch.CANONICAL_DRAWN_SHA256,
                    }
                    if trajectory == "drawn"
                    else {"trajectory": "letter", "label": trajectory}
                )
                previous_wind = {"M1": "M0", "M2": "M1"}.get(wind_mode)
                source_load_model = (
                    source_models.get(f"{trajectory}/{previous_wind}/seed{seed}") if previous_wind else None
                )
                source_run_summary = {
                    "config": {
                        "phase": "B",
                        "wind_mode": wind_mode,
                        "reference": reference,
                        "seed": seed,
                        "load_model": source_load_model,
                        "eval_only": False,
                        "total_timesteps": int(defaults["total_timesteps"]),
                        "requested_total_timesteps": int(defaults["total_timesteps"]),
                        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
                        "teacher_window_m": 0.15,
                        "max_steps_override": None,
                    },
                    "artifacts": {
                        "model_path": str(source_model_path),
                    },
                }
                source_run_dir.mkdir(parents=True, exist_ok=True)
                (source_run_dir / "summary.json").write_text(json.dumps(source_run_summary), encoding="utf-8")
                matrix_iou = "0.800000" if wind_mode == "M0" else "0.780000"
                matrix_path = "0.040000" if wind_mode == "M0" else "0.045000"
                run_summary = {
                    "config": {
                        "phase": "B",
                        "wind_mode": wind_mode,
                        "reference": reference,
                        "seed": seed,
                        "load_model": str(source_model_path),
                        "eval_only": True,
                        "total_timesteps": 0,
                        "requested_total_timesteps": int(defaults["total_timesteps"]),
                        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
                        "teacher_window_m": 0.15,
                        "max_steps_override": None,
                    },
                    "final_goal_criteria": _passing_final_goal_criteria(),
                    "metrics": _passing_summary_metrics(painting_iou=matrix_iou, path_rmse_m=matrix_path),
                    "artifacts": {
                        "model_path": str(model_path),
                        "source_model_path": str(source_model_path),
                    },
                }
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
                matrix_row = {
                    "trajectory": trajectory,
                    "wind_mode": wind_mode,
                    "seed": seed,
                    "run_id": run_id,
                    "status": "complete",
                    "overall_pass": "True",
                    "final_goal_pass": "True",
                    "model_path": str(model_path),
                    "output_dir": str(run_dir),
                    "load_model": str(source_model_path),
                    "painting_iou": matrix_iou,
                    "painting_precision": "0.900000",
                    "painting_recall": "0.900000",
                    "painting_dice": "0.850000",
                    "off_target_ratio": "0.050000",
                    "path_rmse_m": matrix_path,
                    "path_max_m": "0.120000",
                    "corner_path_rmse_m": "0.070000",
                    "corner_speed_ratio": "0.750000",
                    "corner_speed_vs_straight_ratio": "0.750000",
                    "corner_overshoot_m": "0.070000",
                    "led_precision": "0.950000",
                    "led_recall": "0.950000",
                    "led_flicker_rate": "0.060000",
                    "action_norm_mean": "0.800000",
                    "action_norm_max": "1.500000",
                    "action_rate_mean": "1.200000",
                    "action_rate_max": "3.000000",
                    "rpm_saturation_ratio": "0.050000",
                    "crash_rate": "0.000000",
                    "out_of_bounds_rate": "0.000000",
                }
                rows.append(matrix_row)
                source_rows.append(
                    {
                        "trajectory": trajectory,
                        "wind_mode": wind_mode,
                        "seed": seed,
                        "run_id": run_id,
                        "status": "complete",
                        "overall_pass": "True",
                        "final_goal_pass": "True",
                        "model_path": str(source_model_path),
                        "output_dir": str(source_run_dir),
                        "load_model": source_load_model or "",
                    }
                )
                metrics.append(dict(matrix_row))
                models[f"{trajectory}/{wind_mode}/seed{seed}"] = str(model_path)
                source_models[f"{trajectory}/{wind_mode}/seed{seed}"] = str(source_model_path)

    criteria = {
        "matrix_complete": True,
        "expected_runs": len(rows),
        "observed_runs": len(rows),
        "missing_runs": [],
        "all_runs_final_goal_pass": True,
        "failed_final_goal_run_count": 0,
        "failed_final_goal_runs": [],
        "robustness_relative_to_m0_pass": True,
        "robustness_failures": [],
        "batch_final_goal_pass": True,
    }
    summary = {
        "schema_version": 1,
        "profile": "final",
        "dry_run": False,
        "status": "complete",
        "eval_only": True,
        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
        "teacher_window_m": 0.15,
        "trajectories": trajectories,
        "wind_modes": wind_modes,
        "seeds": seeds,
        "total_timesteps": int(defaults["total_timesteps"]),
        "max_steps": None,
        "drawn_input": drawn_input,
        "planned_runs": len(rows),
        "completed_runs": len(rows),
        "failed_runs": 0,
        "final_goal_pass_count": len(rows),
        "final_goal_pass_rate": 1.0,
        "dependency_preflight": {"ok": True},
        "model_manifest_preflight": {
            "ok": True,
            "model_manifest": str(root / "source_train" / "best_model_manifest.json"),
        },
        "batch_final_goal_criteria": criteria,
    }
    root.mkdir(parents=True, exist_ok=True)
    source_train = root / "source_train"
    source_train.mkdir(parents=True, exist_ok=True)
    source_summary = {
        "schema_version": 1,
        "profile": "final",
        "dry_run": False,
        "status": "complete",
        "eval_only": False,
        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
        "teacher_window_m": 0.15,
        "trajectories": trajectories,
        "wind_modes": wind_modes,
        "seeds": seeds,
        "total_timesteps": int(defaults["total_timesteps"]),
        "max_steps": None,
        "drawn_input": drawn_input,
        "planned_runs": len(rows),
        "completed_runs": len(rows),
        "failed_runs": 0,
        "batch_final_goal_criteria": criteria,
    }
    source_summary_path = source_train / "batch_summary.json"
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")
    _write_csv(source_train / "matrix_results.csv", source_rows)
    source_manifest = {
        "schema_version": 2,
        "profile": "final",
        "dry_run": False,
        "eval_only": False,
        "batch_summary": str(source_summary_path),
        "trajectories": trajectories,
        "wind_modes": wind_modes,
        "seeds": seeds,
        "total_timesteps": int(defaults["total_timesteps"]),
        "max_steps": None,
        "drawn_input": drawn_input,
        "trained_action_filter": batch.DEFAULT_TRAINED_ACTION_FILTER,
        "teacher_window_m": 0.15,
        "models": source_models,
    }
    (source_train / "best_model_manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    (root / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_csv(root / "matrix_results.csv", rows)
    _write_csv(root / "metrics.csv", metrics)
    _write_csv(
        root / "aggregate_metrics.csv",
        list(verify._expected_aggregate_rows(rows).values()),
    )
    (root / "best_model_manifest.json").write_text(
        json.dumps({"schema_version": 1, "models": models}),
        encoding="utf-8",
    )


def test_verify_final_goal_batch_accepts_complete_final_artifacts(tmp_path):
    _write_canonical_final_fixture(tmp_path)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is True
    assert result.issues == []


def test_verify_final_goal_batch_rejects_short_source_training_manifest(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    source_manifest_path = tmp_path / "source_train" / "best_model_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["total_timesteps"] = 16
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    source_summary_path = tmp_path / "source_train" / "batch_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["total_timesteps"] = 16
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("source training manifest total_timesteps가 canonical과 다릅니다" in issue for issue in result.issues)
    assert any("final profile total_timesteps가 canonical과 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_source_training_per_run_step_mismatch(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    source_run_summary = tmp_path / "source_train" / "L_M0_seed7" / "summary.json"
    data = json.loads(source_run_summary.read_text(encoding="utf-8"))
    data["config"]["eval_only"] = True
    data["config"]["requested_total_timesteps"] = 16
    data["config"]["total_timesteps"] = 16
    source_run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("source training per-run summary가 학습 실행이 아닙니다: L/M0/seed7" in issue for issue in result.issues)
    assert any(
        "source training per-run requested_total_timesteps가 canonical과 다릅니다: L/M0/seed7 (16)" in issue
        for issue in result.issues
    )
    assert any(
        "source training per-run total_timesteps가 canonical과 다릅니다: L/M0/seed7 (16)" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_source_training_curriculum_matrix_gap(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    matrix_path = tmp_path / "source_train" / "matrix_results.csv"
    rows = verify._read_csv(matrix_path)
    for row in rows:
        if row["trajectory"] == "L" and row["wind_mode"] == "M1" and row["seed"] == "7":
            row["load_model"] = ""
    _write_csv(matrix_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("source training curriculum load_model이 없습니다: L/M1/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_source_training_per_run_curriculum_mismatch(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    source_manifest = json.loads((tmp_path / "source_train" / "best_model_manifest.json").read_text(encoding="utf-8"))
    wrong_model = source_manifest["models"]["L/M0/seed7"]
    source_run_summary = tmp_path / "source_train" / "L_M2_seed7" / "summary.json"
    data = json.loads(source_run_summary.read_text(encoding="utf-8"))
    data["config"]["load_model"] = wrong_model
    source_run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "source training per-run config.load_model이 이전 wind 모델과 다릅니다: L/M2/seed7" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_dry_run_source_training_artifacts(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    source_manifest_path = tmp_path / "source_train" / "best_model_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["dry_run"] = True
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    source_summary_path = tmp_path / "source_train" / "batch_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["dry_run"] = True
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "source training manifest가 dry-run 산출물입니다" in result.issues
    assert "source training batch_summary가 dry-run 산출물입니다" in result.issues


def test_verify_final_goal_batch_rejects_max_steps_override(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    batch_summary = tmp_path / "batch_summary.json"
    summary = json.loads(batch_summary.read_text(encoding="utf-8"))
    summary["max_steps"] = 12
    batch_summary.write_text(json.dumps(summary), encoding="utf-8")

    source_manifest_path = tmp_path / "source_train" / "best_model_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["max_steps"] = 12
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    source_summary_path = tmp_path / "source_train" / "batch_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["max_steps"] = 12
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    run_data = json.loads(run_summary.read_text(encoding="utf-8"))
    run_data["config"]["max_steps_override"] = 12
    run_summary.write_text(json.dumps(run_data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "final profile max_steps override가 없어야 합니다: 12" in result.issues
    assert "source training manifest max_steps override가 없어야 합니다: 12" in result.issues
    assert any("per-run summary max_steps override가 없어야 합니다: L/M0/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_noncanonical_drawn_input(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    bad_drawn = {
        "input_arg": "data/drawn_paths/user/easy.json",
        "path": str(tmp_path / "data" / "drawn_paths" / "user" / "easy.json"),
        "relative_path": "data/drawn_paths/user/easy.json",
        "exists": True,
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    batch_summary = tmp_path / "batch_summary.json"
    summary = json.loads(batch_summary.read_text(encoding="utf-8"))
    summary["drawn_input"] = bad_drawn
    batch_summary.write_text(json.dumps(summary), encoding="utf-8")

    source_manifest_path = tmp_path / "source_train" / "best_model_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["drawn_input"] = bad_drawn
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    source_summary_path = tmp_path / "source_train" / "batch_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_summary["drawn_input"] = bad_drawn
    source_summary_path.write_text(json.dumps(source_summary), encoding="utf-8")

    run_summary = tmp_path / "drawn_M0_seed7" / "summary.json"
    run_data = json.loads(run_summary.read_text(encoding="utf-8"))
    run_data["config"]["reference"]["drawn_relative_path"] = "data/drawn_paths/user/easy.json"
    run_data["config"]["reference"]["drawn_sha256"] = "0" * 64
    run_summary.write_text(json.dumps(run_data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("final profile drawn_input relative_path가 canonical과 다릅니다" in issue for issue in result.issues)
    assert any("source training manifest drawn_input sha256이 canonical과 다릅니다" in issue for issue in result.issues)
    assert any("per-run summary drawn path가 canonical과 다릅니다: drawn/M0/seed7" in issue for issue in result.issues)
    assert any("per-run summary drawn sha256이 canonical과 다릅니다: drawn/M0/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_reduced_final_matrix(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("final profile trajectory matrix가 canonical과 다릅니다" in issue for issue in result.issues)
    assert any("final profile expected_runs가 canonical과 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_wrong_matrix_key_set(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    matrix_path = tmp_path / "matrix_results.csv"
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["trajectory"] = "bogus"
    _write_csv(matrix_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("matrix_results.csv에 canonical final key가 누락되었습니다" in issue for issue in result.issues)
    assert any("matrix_results.csv에 canonical 밖 key가 있습니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_nonfinal_profile_by_default(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="standard")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "profile이 'final'이 아닙니다: 'standard'" in result.issues


def test_verify_final_goal_batch_reports_failed_matrix(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=False, profile="final")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "batch_final_goal_pass가 true가 아닙니다" in result.issues
    assert any("matrix_results.csv에 통과하지 못한 행" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_failed_batch_preflight(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=False, profile="final")
    batch_summary = tmp_path / "batch_summary.json"
    data = json.loads(batch_summary.read_text(encoding="utf-8"))
    data["batch_preflight"] = {"ok": False, "issues": ["invalid curriculum"]}
    batch_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "batch_preflight가 실패했습니다" in result.issues


def test_verify_final_goal_batch_rejects_missing_per_run_summary(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    run_summary.unlink()

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("per-run summary.json이 없습니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_manifest_model_mismatch(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    model_manifest = tmp_path / "best_model_manifest.json"
    model_manifest.write_text(
        json.dumps({"schema_version": 1, "models": {"L/M0/seed7": "missing_model.zip"}}),
        encoding="utf-8",
    )

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("best_model_manifest.json의 model 파일이 없습니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_output_model_path_disagreement(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    alt_model = tmp_path / "alt_models" / "alt_eval_model.zip"
    _write_model_zip(alt_model)

    model_manifest = tmp_path / "best_model_manifest.json"
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    manifest["models"]["L/M0/seed7"] = str(alt_model)
    model_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["artifacts"]["model_path"] = str(alt_model)
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "best_model_manifest.json model과 matrix_results.csv model_path가 다릅니다: L/M0/seed7" in issue
        for issue in result.issues
    )
    assert any(
        "matrix_results.csv model_path와 per-run artifacts.model_path가 다릅니다: L/M0/seed7" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_corrupt_output_model_archive(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    model_path = tmp_path / "L_M0_seed7" / "models" / "ppo_phaseB_L_M0_corner.zip"
    model_path.write_bytes(b"not a stable-baselines zip")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("Stable-Baselines 모델 zip이 아닙니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_matrix_summary_disagreement(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["final_goal_criteria"]["final_goal_pass"] = False
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("matrix_results.csv와 per-run summary final_goal_pass가 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_missing_or_false_final_criteria(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["final_goal_criteria"].pop("painting_iou_abs")
    data["final_goal_criteria"]["led_recall_abs"] = False
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("per-run summary final criterion이 없습니다: L/M0/seed7 (painting_iou_abs)" in issue for issue in result.issues)
    assert any(
        "per-run summary final criterion이 true가 아닙니다: L/M0/seed7 (led_recall_abs=False)" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_final_criteria_not_backed_by_metrics(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    for row in data["metrics"]:
        if row["tag"] == "phaseB_trained":
            row["painted_pixel_iou"] = "0.100000"
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "per-run summary final_goal_criteria가 metrics 재계산 결과와 다릅니다: "
        "L/M0/seed7 (painting_iou_abs: summary=True, metrics=False)" in issue
        for issue in result.issues
    )
    assert any(
        "per-run summary metrics 기준 final_goal_pass가 true가 아닙니다: L/M0/seed7" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_training_batch_for_final_claim(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    batch_summary = tmp_path / "batch_summary.json"
    data = json.loads(batch_summary.read_text(encoding="utf-8"))
    data["eval_only"] = False
    batch_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "최종 목표 검증은 eval_only batch 산출물이어야 합니다" in result.issues


def test_verify_final_goal_batch_rejects_per_run_config_mismatch(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["config"]["reference"]["label"] = "DG"
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("per-run summary letter reference가 matrix와 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_per_run_non_eval_summary(tmp_path):
    _write_batch_fixture(tmp_path, pass_batch=True, profile="final")
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["config"]["eval_only"] = False
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("per-run summary가 eval_only 실행이 아닙니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_per_run_step_mismatch(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["config"]["requested_total_timesteps"] = 16
    data["config"]["total_timesteps"] = 16
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "per-run summary requested_total_timesteps가 canonical과 다릅니다: L/M0/seed7 (16)" in issue
        for issue in result.issues
    )
    assert any(
        "per-run summary eval total_timesteps는 0이어야 합니다: L/M0/seed7 (16)" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_noncanonical_batch_policy(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    batch_summary = tmp_path / "batch_summary.json"
    data = json.loads(batch_summary.read_text(encoding="utf-8"))
    data["trained_action_filter"] = "none"
    data["teacher_window_m"] = 0.2
    batch_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "final 평가 action filter가 canonical과 다릅니다: 'none'" in result.issues
    assert "final 평가 teacher_window_m이 canonical과 다릅니다: 0.2" in result.issues


def test_verify_final_goal_batch_rejects_noncanonical_per_run_policy(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    data["config"]["trained_action_filter"] = "none"
    data["config"]["teacher_window_m"] = 0.2
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("per-run summary action filter가 canonical과 다릅니다: L/M0/seed7" in issue for issue in result.issues)
    assert any("per-run summary teacher_window_m이 canonical과 다릅니다: L/M0/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_missing_eval_source_model(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    source_path = Path(data["artifacts"]["source_model_path"])
    source_path.unlink()

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("matrix_results.csv의 eval load_model 파일이 없습니다: L/M0/seed7" in issue for issue in result.issues)
    assert any("per-run summary config.load_model 파일이 없습니다: L/M0/seed7" in issue for issue in result.issues)
    assert any("per-run summary artifacts.source_model_path 파일이 없습니다: L/M0/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_eval_source_model_disagreement(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    run_summary = tmp_path / "L_M0_seed7" / "summary.json"
    data = json.loads(run_summary.read_text(encoding="utf-8"))
    alt_source = tmp_path / "source_models" / "alt_model.zip"
    _write_model_zip(alt_source)
    data["config"]["load_model"] = str(alt_source)
    run_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("matrix_results.csv load_model과 per-run config.load_model이 다릅니다: L/M0/seed7" in issue for issue in result.issues)
    assert any("per-run config.load_model과 artifacts.source_model_path가 다릅니다: L/M0/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_eval_model_not_from_source_training_manifest(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    source_manifest_path = tmp_path / "source_train" / "best_model_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    alt_model = tmp_path / "source_train" / "models" / "alt_source_model.zip"
    _write_model_zip(alt_model)
    source_manifest["models"]["L/M0/seed7"] = str(alt_model)
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "source training manifest model과 matrix_results.csv load_model이 다릅니다: L/M0/seed7" in issue
        for issue in result.issues
    )
    assert any(
        "source training manifest model과 per-run config.load_model이 다릅니다: L/M0/seed7" in issue
        for issue in result.issues
    )
    assert any(
        "source training manifest model과 artifacts.source_model_path가 다릅니다: L/M0/seed7" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_stale_metrics_final_goal(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["final_goal_pass"] = "False"
    _write_csv(metrics_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("metrics.csv final_goal_pass가 matrix_results.csv와 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_matrix_metric_not_backed_by_run_summary(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    matrix_path = tmp_path / "matrix_results.csv"
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["trajectory"] == "L" and row["wind_mode"] == "M0" and row["seed"] == "7":
            row["path_rmse_m"] = "0.001000"
    _write_csv(matrix_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "matrix_results.csv metric이 per-run summary와 다릅니다: "
        "L/M0/seed7 (path_rmse_m: matrix=0.001, summary=0.04)" in issue
        for issue in result.issues
    )
    assert any(
        "metrics.csv metric이 matrix_results.csv와 다릅니다: L/M0/seed7 (path_rmse_m)" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_missing_metrics_row(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    metrics_path = tmp_path / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _write_csv(metrics_path, rows[1:])

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("metrics.csv에 matrix row가 누락되었습니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_stale_aggregate_counts(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    aggregate_path = tmp_path / "aggregate_metrics.csv"
    with aggregate_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["final_goal_pass_count"] = "0"
    rows[0]["final_goal_pass_rate"] = "0.0"
    _write_csv(aggregate_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("aggregate_metrics.csv final_goal_pass_count가 matrix_results.csv와 다릅니다" in issue for issue in result.issues)
    assert any("aggregate_metrics.csv final_goal_pass_rate가 matrix_results.csv와 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_stale_aggregate_metric_values(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    aggregate_path = tmp_path / "aggregate_metrics.csv"
    with aggregate_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["group"] == "overall" and row["value"] == "all":
            row["painting_iou_mean"] = "0.000000"
    _write_csv(aggregate_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "aggregate_metrics.csv overall/all painting_iou_mean가 matrix_results.csv 재계산 결과와 다릅니다" in issue
        for issue in result.issues
    )


def test_verify_final_goal_batch_rejects_stale_summary_counters(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    batch_summary = tmp_path / "batch_summary.json"
    data = json.loads(batch_summary.read_text(encoding="utf-8"))
    data["planned_runs"] = 1
    data["completed_runs"] = 1
    data["failed_runs"] = 1
    data["final_goal_pass_count"] = 1
    data["final_goal_pass_rate"] = 0.01
    batch_summary.write_text(json.dumps(data), encoding="utf-8")

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any("batch_summary.json planned_runs가 matrix_results.csv 행 수와 다릅니다" in issue for issue in result.issues)
    assert any("batch_summary.json completed_runs가 matrix_results.csv 상태와 다릅니다" in issue for issue in result.issues)
    assert any("batch_summary.json failed_runs가 matrix_results.csv 상태와 다릅니다" in issue for issue in result.issues)
    assert any("batch_summary.json final_goal_pass_count가 matrix_results.csv와 다릅니다" in issue for issue in result.issues)
    assert any("batch_summary.json final_goal_pass_rate가 matrix_results.csv와 다릅니다" in issue for issue in result.issues)


def test_verify_final_goal_batch_recomputes_batch_robustness_from_matrix(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    matrix_path = tmp_path / "matrix_results.csv"
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["trajectory"] == "L" and row["wind_mode"] == "M1" and row["seed"] == "7":
            row["painting_iou"] = "0.100000"
    _write_csv(matrix_path, rows)

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert any(
        "batch_final_goal_criteria.robustness_relative_to_m0_pass가 matrix_results.csv 재계산 결과와 다릅니다: "
        "True != False" in issue
        for issue in result.issues
    )
    assert any("matrix_results.csv 기준 robustness failure가 있습니다: L/M1/seed7" in issue for issue in result.issues)


def test_verify_final_goal_batch_rejects_missing_aggregate_overall_row(tmp_path):
    _write_canonical_final_fixture(tmp_path)
    _write_csv(
        tmp_path / "aggregate_metrics.csv",
        [
            {
                "group": "trajectory",
                "value": "L",
                "n": 15,
                "final_goal_pass_count": 15,
                "overall_pass_count": 15,
                "final_goal_pass_rate": 1.0,
                "overall_pass_rate": 1.0,
            }
        ],
    )

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "aggregate_metrics.csv에 overall/all row가 없습니다" in result.issues


def test_verify_final_goal_batch_keeps_dry_run_plan_report_concise(tmp_path):
    criteria = {
        "matrix_complete": False,
        "expected_runs": 1,
        "observed_runs": 0,
        "missing_runs": [{"trajectory": "L", "wind_mode": "M0", "seed": 7}],
        "all_runs_final_goal_pass": False,
        "failed_final_goal_run_count": 0,
        "failed_final_goal_runs": [],
        "robustness_relative_to_m0_pass": True,
        "robustness_failures": [],
        "batch_final_goal_pass": False,
    }
    summary = {
        "schema_version": 1,
        "profile": "final",
        "dry_run": True,
        "status": "planned",
        "planned_runs": 1,
        "batch_final_goal_criteria": criteria,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_csv(
        tmp_path / "matrix_results.csv",
        [
            {
                "trajectory": "L",
                "wind_mode": "M0",
                "seed": 7,
                "run_id": "L_M0_seed7",
                "status": "planned",
                "overall_pass": "",
                "final_goal_pass": "",
                "output_dir": str(tmp_path / "L_M0_seed7"),
            }
        ],
    )
    _write_csv(tmp_path / "metrics.csv", [{"run_id": "", "final_goal_pass": ""}])
    _write_csv(
        tmp_path / "aggregate_metrics.csv",
        [
            {
                "group": "overall",
                "value": "all",
                "n": 1,
                "final_goal_pass_count": 0,
                "overall_pass_count": 0,
                "final_goal_pass_rate": 0.0,
                "overall_pass_rate": 0.0,
            }
        ],
    )
    (tmp_path / "best_model_manifest.json").write_text(
        json.dumps({"schema_version": 1, "models": {"L/M0/seed7": str(tmp_path / "missing.zip")}}),
        encoding="utf-8",
    )

    result = verify_batch_artifacts(tmp_path)

    assert result.ok is False
    assert "dry_run 산출물은 최종 목표 달성 증거가 아닙니다" in result.issues
    assert not any("per-run summary.json이 없습니다" in issue for issue in result.issues)
