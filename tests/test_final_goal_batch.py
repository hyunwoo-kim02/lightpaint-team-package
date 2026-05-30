from __future__ import annotations

import json
import sys
import csv

from src.train import run_final_goal_batch as batch


def test_final_goal_batch_dry_run_plans_multi_trajectory_curriculum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square,L,drawn",
            "--wind-modes",
            "M0,M1,M2",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["planned_runs"] == 9
    assert summary["total_timesteps"] == 100000
    assert summary["max_steps"] is None
    assert summary["drawn_input"]["relative_path"] == batch.CANONICAL_DRAWN_PATH
    assert summary["drawn_input"]["sha256"] == batch.CANONICAL_DRAWN_SHA256
    assert summary["final_goal_spec"] == "docs/FINAL_GOAL_SPEC.md"
    assert summary["runtime"]["python_executable"] == sys.executable
    assert summary["runtime"]["argv"][0] == "prog"
    assert summary["runtime"]["package_root"]
    assert summary["trained_action_filter"] == "corner_tangent_decel"
    assert summary["teacher_window_m"] == 0.15
    assert all("--total-timesteps 100000" in run["command"] for run in summary["runs"])
    assert all("--trained-action-filter corner_tangent_decel" in run["command"] for run in summary["runs"])
    assert any(run["trajectory"] == "L" and "--trajectory letter" in run["command"] for run in summary["runs"])
    assert any(run["trajectory"] == "drawn" and "--trajectory drawn" in run["command"] for run in summary["runs"])
    assert any(run["wind_mode"] == "M1" and "--load-model" in run["command"] for run in summary["runs"])

    manifest = json.loads((tmp_path / "best_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["profile"] == "standard"
    assert manifest["dry_run"] is True
    assert manifest["eval_only"] is False
    assert manifest["total_timesteps"] == 100000
    assert manifest["max_steps"] is None
    assert manifest["drawn_input"]["relative_path"] == batch.CANONICAL_DRAWN_PATH
    assert manifest["drawn_input"]["sha256"] == batch.CANONICAL_DRAWN_SHA256
    assert manifest["batch_summary"] == str(tmp_path / "batch_summary.json")
    assert "square/M0/seed7" in manifest["models"]
    assert "L/M2/seed7" in manifest["models"]
    with (tmp_path / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert len(matrix_rows) == 9
    assert {row["status"] for row in matrix_rows} == {"planned"}
    assert any(row["trajectory"] == "drawn" and row["wind_mode"] == "M2" for row in matrix_rows)


def test_final_goal_batch_eval_only_uses_model_manifest(tmp_path, monkeypatch):
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text(
        json.dumps({"models": {"square/M0/seed7": "artifacts/model.zip"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--eval-only",
            "--model-manifest",
            str(model_manifest),
            "--output-dir",
            str(tmp_path / "eval"),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "eval" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_runs"] == 1
    command = summary["runs"][0]["command"]
    expected_model = (tmp_path / "artifacts" / "model.zip").resolve(strict=False)
    assert "--eval-only" in command
    assert str(expected_model) in command
    assert summary["model_manifest_preflight"]["ok"] is True
    output_manifest = json.loads((tmp_path / "eval" / "best_model_manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["eval_only"] is True
    assert output_manifest["source_model_manifest"] == str(model_manifest)
    assert output_manifest["models"]["square/M0/seed7"] == str(expected_model)


def test_final_goal_batch_eval_only_resolves_manifest_relative_model_files(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    model_dir = manifest_dir / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.zip"
    model_path.write_bytes(b"placeholder")
    model_manifest = manifest_dir / "models.json"
    model_manifest.write_text(
        json.dumps({"models": {"square/M0/seed7": "models/model.zip"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: ["pybullet"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--eval-only",
            "--model-manifest",
            str(model_manifest),
            "--output-dir",
            str(tmp_path / "eval_existing_file"),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "eval_existing_file" / "batch_summary.json").read_text(encoding="utf-8"))
    expected_model = model_path.resolve(strict=False)
    assert summary["model_manifest_preflight"]["ok"] is True
    assert summary["dependency_preflight"]["ok"] is False
    assert summary["runs"][0]["load_model"] == str(expected_model)
    assert str(expected_model) in summary["runs"][0]["command"]


def test_final_goal_batch_eval_only_reports_missing_manifest_entries(tmp_path, monkeypatch):
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text(
        json.dumps({"models": {"square/M0/seed7": "artifacts/model.zip"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--eval-only",
            "--model-manifest",
            str(model_manifest),
            "--output-dir",
            str(tmp_path / "eval_missing"),
            "--trajectories",
            "L",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "eval_missing" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["model_manifest_preflight"]["ok"] is False
    assert summary["model_manifest_preflight"]["missing_models"] == ["L/M0/seed7"]
    with (tmp_path / "eval_missing" / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert matrix_rows[0]["status"] == "failed"
    assert "model manifest에 L/M0/seed7 항목이 없습니다" in summary["runs"][0]["error"]


def test_final_goal_batch_eval_only_reports_missing_model_files(tmp_path, monkeypatch):
    model_manifest = tmp_path / "models.json"
    missing_model = tmp_path / "missing_model.zip"
    model_manifest.write_text(
        json.dumps({"models": {"square/M0/seed7": str(missing_model)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--eval-only",
            "--model-manifest",
            str(model_manifest),
            "--output-dir",
            str(tmp_path / "eval_missing_file"),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "eval_missing_file" / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["model_manifest_preflight"]["ok"] is False
    assert summary["model_manifest_preflight"]["missing_model_files"] == [
        {"key": "square/M0/seed7", "path": str(missing_model)}
    ]
    assert summary["failed_runs"] == 1
    assert "모델 파일이 없습니다" in summary["runs"][0]["error"]


def test_final_goal_batch_preflight_reports_missing_runtime_dependency(tmp_path, monkeypatch):
    def fake_find_spec(name: str):
        if name == "pybullet":
            return None
        return object()

    monkeypatch.setattr(batch.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["dependency_preflight"]["ok"] is False
    assert summary["dependency_preflight"]["missing_dependencies"] == ["pybullet"]
    assert "pip install -r requirements.txt" in summary["dependency_preflight"]["message"]
    with (tmp_path / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert matrix_rows[0]["status"] == "failed"
    assert (tmp_path / "square_M0_seed7" / "batch_run.log").exists()


def test_final_goal_batch_preflight_rejects_invalid_curriculum_wind_order(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0,M2",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["batch_preflight"]["ok"] is False
    assert summary["failed_runs"] == 2
    assert any("M2는 같은 trajectory/seed의 M1 학습 이후" in issue for issue in summary["batch_preflight"]["issues"])
    with (tmp_path / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert {row["status"] for row in matrix_rows} == {"failed"}


def test_final_goal_batch_preflight_rejects_missing_drawn_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "drawn",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--drawn-path",
            "data/drawn_paths/user/missing_for_batch_test.json",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["batch_preflight"]["ok"] is False
    assert any("drawn trajectory JSON 파일이 없습니다" in issue for issue in summary["batch_preflight"]["issues"])


def test_final_goal_batch_records_subprocess_failure_status(tmp_path, monkeypatch):
    class Result:
        returncode = 9

    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(batch.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 9
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["completed_runs"] == 0
    assert summary["failed_runs"] == 1
    assert summary["runs"][0]["status"] == "failed"


def test_final_goal_batch_writes_running_status_before_subprocess_returns(tmp_path, monkeypatch):
    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        running_summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
        running_record = running_summary["runs"][0]
        assert running_record["status"] == "running"
        assert running_record["run_index"] == 1
        assert running_record["total_runs"] == 1
        assert running_record["log_path"]
        output_dir = command[command.index("--output-dir") + 1]
        summary = {
            "artifacts": {"model_path": str(tmp_path / "model.zip")},
            "final_goal_criteria": {"final_goal_pass": False},
            "success_criteria": {"overall_pass": False},
            "metrics": [{"tag": "phaseB_trained"}],
        }
        (tmp_path / "model.zip").write_bytes(b"model")
        (tmp_path / "square_M0_seed7" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        assert output_dir.endswith("square_M0_seed7")
        return Result()

    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    run = summary["runs"][0]
    assert run["status"] == "complete"
    assert run["duration_s"] >= 0
    assert run["started_at"]
    assert run["finished_at"]
    log_text = (tmp_path / "square_M0_seed7" / "batch_run.log").read_text(encoding="utf-8")
    assert "[batch] run_id=square_M0_seed7" in log_text
    assert "[batch] index=1/1" in log_text


def test_final_goal_batch_continue_on_error_records_missing_curriculum_source(tmp_path, monkeypatch):
    calls: list[object] = []

    class Result:
        returncode = 9

    def fake_run(*args, **kwargs):
        calls.append(args)
        return Result()

    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--continue-on-error",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0,M1,M2",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    assert len(calls) == 1
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["failed_runs"] == 3
    assert [run["status"] for run in summary["runs"]] == ["failed", "failed", "failed"]
    assert summary["runs"][1]["error_type"] == "curriculum_source_missing"
    assert summary["runs"][2]["error_type"] == "curriculum_source_missing"
    with (tmp_path / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert len(matrix_rows) == 3
    assert {row["status"] for row in matrix_rows} == {"failed"}


def test_final_goal_batch_stops_cleanly_on_missing_curriculum_source_without_continue(tmp_path, monkeypatch):
    class Result:
        returncode = 9

    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(batch.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0,M1,M2",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 9
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_runs"] == 1
    assert summary["runs"][0]["status"] == "failed"


def test_final_goal_batch_aggregates_seed_metrics():
    rows = [
        {
            "trajectory": "L",
            "wind_mode": "M0",
            "seed": 7,
            "final_goal_pass": True,
            "failed_final_goal_criteria": "",
            "overall_pass": True,
            "painting_iou": "0.80",
            "path_rmse_m": "0.04",
        },
        {
            "trajectory": "L",
            "wind_mode": "M0",
            "seed": 11,
            "final_goal_pass": False,
            "failed_final_goal_criteria": "painting_iou_min|corner_speed_ratio_range",
            "overall_pass": True,
            "painting_iou": "0.70",
            "path_rmse_m": "0.06",
        },
    ]

    aggregate = batch._aggregate_rows(rows)

    overall = next(row for row in aggregate if row["group"] == "overall")
    assert overall["n"] == 2
    assert overall["final_goal_pass_rate"] == 0.5
    assert overall["overall_pass_rate"] == 1.0
    assert overall["painting_iou_mean"] == "0.750000"
    assert overall["painting_iou_min"] == "0.700000"
    assert overall["path_rmse_m_max"] == "0.060000"


def test_batch_final_goal_criteria_checks_matrix_and_disturbance_robustness():
    rows = [
        {
            "trajectory": "L",
            "wind_mode": "M0",
            "seed": 7,
            "final_goal_pass": True,
            "failed_final_goal_criteria": "",
            "painting_iou": "0.80",
            "path_rmse_m": "0.040",
        },
        {
            "trajectory": "L",
            "wind_mode": "M1",
            "seed": 7,
            "final_goal_pass": True,
            "failed_final_goal_criteria": "",
            "painting_iou": "0.70",
            "path_rmse_m": "0.050",
        },
        {
            "trajectory": "L",
            "wind_mode": "M2",
            "seed": 7,
            "final_goal_pass": True,
            "failed_final_goal_criteria": "",
            "painting_iou": "0.65",
            "path_rmse_m": "0.051",
        },
    ]

    criteria = batch._batch_final_goal_criteria(rows, ["L"], ["M0", "M1", "M2"], [7])

    assert criteria["matrix_complete"] is True
    assert criteria["all_runs_final_goal_pass"] is True
    assert criteria["robustness_relative_to_m0_pass"] is True
    assert criteria["batch_final_goal_pass"] is True

    rows[2]["path_rmse_m"] = "0.080"
    failed = batch._batch_final_goal_criteria(rows, ["L"], ["M0", "M1", "M2"], [7])

    assert failed["robustness_relative_to_m0_pass"] is False
    assert failed["batch_final_goal_pass"] is False
    assert failed["robustness_failures"][0]["wind_mode"] == "M2"


def test_batch_final_goal_criteria_reports_missing_matrix_entries():
    rows = [
        {
            "trajectory": "L",
            "wind_mode": "M0",
            "seed": 7,
            "final_goal_pass": True,
            "failed_final_goal_criteria": "",
            "painting_iou": "0.80",
            "path_rmse_m": "0.040",
        }
    ]

    criteria = batch._batch_final_goal_criteria(rows, ["L"], ["M0", "M1"], [7])

    assert criteria["matrix_complete"] is False
    assert criteria["batch_final_goal_pass"] is False
    assert criteria["missing_runs"] == [{"trajectory": "L", "wind_mode": "M1", "seed": 7}]


def test_final_goal_batch_resume_reuses_existing_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "square_M0_seed7"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "models" / "ppo_phaseB_square_M0_corner.zip"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    summary = {
        "config": {
            "phase": "B",
            "wind_mode": "M0",
            "reference": {
                "trajectory": "square",
                "label": "square",
                "path_scale": 1.0,
                "square_side_m": 0.8,
                "unscaled_square_side_m": 0.8,
            },
            "seed": 7,
            "load_model": None,
            "eval_only": False,
            "speed": 0.35,
            "ctrl_freq": 30,
            "pyb_freq": 240,
            "requested_total_timesteps": 100000,
            "total_timesteps": 100000,
            "max_steps_override": None,
            "corner_window_m": 0.18,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "log_std_init": -2.0,
            "teacher_gain": 0.1,
            "teacher_window_m": 0.15,
            "bc_epochs": 2,
            "bc_episodes": 8,
            "bc_batch_size": 64,
            "bc_learning_rate": 1e-3,
            "bc_nonzero_weight": 1.0,
            "trained_action_filter": "corner_tangent_decel",
            "device": "cpu",
            "save_video": False,
        },
        "success_criteria": {"overall_pass": False},
        "final_goal_criteria": {
            "final_goal_pass": False,
            "painting_iou_min": False,
            "path_rmse_m_max": True,
        },
        "metrics": [
            {
                "tag": "phaseB_trained",
                "painted_pixel_iou": "0.1",
                "painted_pixel_precision": "1.0",
                "painted_pixel_recall": "0.1",
                "painted_pixel_dice": "0.2",
                "off_target_pixel_ratio": "0.0",
                "path_rmse_m": "0.01",
                "path_max_m": "0.02",
                "corner_path_rmse_m": "nan",
                "corner_speed_ratio": "nan",
                "corner_overshoot_m": "nan",
                "led_precision": "1.0",
                "led_recall": "1.0",
                "led_flicker_rate": "0.0",
                "action_norm_mean": "0.0",
                "action_norm_max": "0.0",
                "action_rate_mean": "0.0",
                "action_rate_max": "0.0",
                "rpm_saturation_ratio": "0.0",
                "crash_rate": "0.0",
                "out_of_bounds_rate": "0.0",
            }
        ],
        "artifacts": {"model_path": str(model_path)},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    batch_summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["status"] == "reused"
    assert batch_summary["reused_runs"] == 1
    assert batch_summary["completed_runs"] == 1
    metrics = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert "square_M0_seed7" in metrics
    assert "painting_iou_min" in metrics
    with (tmp_path / "matrix_results.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert matrix_rows[0]["status"] == "reused"
    assert matrix_rows[0]["final_goal_pass"] == "False"
    assert matrix_rows[0]["failed_final_goal_criteria"] == "painting_iou_min"
    assert matrix_rows[0]["painting_precision"] == "1.0"
    assert matrix_rows[0]["path_max_m"] == "0.02"
    assert matrix_rows[0]["action_rate_mean"] == "0.0"
    criteria = batch_summary["batch_final_goal_criteria"]
    assert criteria["failed_final_goal_run_count"] == 1
    assert criteria["failed_final_goal_runs"][0]["failed_criteria"] == ["painting_iou_min"]


def test_final_goal_batch_resume_skips_incompatible_existing_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "square_M0_seed7"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "phase": "B",
                    "wind_mode": "M0",
                    "reference": {
                        "trajectory": "square",
                        "label": "square",
                        "path_scale": 1.0,
                        "square_side_m": 0.8,
                        "unscaled_square_side_m": 0.8,
                    },
                    "seed": 7,
                    "load_model": None,
                    "eval_only": False,
                    "speed": 0.35,
                    "ctrl_freq": 30,
                    "pyb_freq": 240,
                    "requested_total_timesteps": 16,
                    "total_timesteps": 16,
                    "max_steps_override": None,
                    "corner_window_m": 0.18,
                    "n_steps": 256,
                    "batch_size": 64,
                    "n_epochs": 4,
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.0,
                    "clip_range": 0.2,
                    "log_std_init": -2.0,
                    "teacher_gain": 0.1,
                    "teacher_window_m": 0.15,
                    "bc_epochs": 2,
                    "bc_episodes": 8,
                    "bc_batch_size": 64,
                    "bc_learning_rate": 1e-3,
                    "bc_nonzero_weight": 1.0,
                    "trained_action_filter": "corner_tangent_decel",
                    "device": "cpu",
                    "save_video": False,
                },
                "success_criteria": {"overall_pass": True},
                "final_goal_criteria": {"final_goal_pass": True},
                "metrics": [],
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: ["pybullet"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    batch_summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["status"] == "failed"
    assert "requested_total_timesteps 불일치" in batch_summary["runs"][0]["resume_skip_reason"]
    assert batch_summary["reused_runs"] == 0


def test_final_goal_batch_resume_skips_shortened_max_steps_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "square_M0_seed7"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "models" / "ppo_phaseB_square_M0_corner.zip"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {
                    "phase": "B",
                    "wind_mode": "M0",
                    "reference": {
                        "trajectory": "square",
                        "label": "square",
                        "path_scale": 1.0,
                        "square_side_m": 0.8,
                        "unscaled_square_side_m": 0.8,
                    },
                    "seed": 7,
                    "load_model": None,
                    "eval_only": False,
                    "speed": 0.35,
                    "ctrl_freq": 30,
                    "pyb_freq": 240,
                    "requested_total_timesteps": 100000,
                    "total_timesteps": 100000,
                    "max_steps": 3,
                    "max_steps_override": 3,
                    "corner_window_m": 0.18,
                    "n_steps": 256,
                    "batch_size": 64,
                    "n_epochs": 4,
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.0,
                    "clip_range": 0.2,
                    "log_std_init": -2.0,
                    "teacher_gain": 0.1,
                    "teacher_window_m": 0.15,
                    "bc_epochs": 2,
                    "bc_episodes": 8,
                    "bc_batch_size": 64,
                    "bc_learning_rate": 1e-3,
                    "bc_nonzero_weight": 1.0,
                    "trained_action_filter": "corner_tangent_decel",
                    "device": "cpu",
                    "save_video": False,
                },
                "success_criteria": {"overall_pass": False},
                "final_goal_criteria": {"final_goal_pass": False},
                "metrics": [],
                "artifacts": {"model_path": str(model_path)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: ["pybullet"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "100000",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    batch_summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["status"] == "failed"
    assert "max_steps_override 불일치" in batch_summary["runs"][0]["resume_skip_reason"]
    assert batch_summary["reused_runs"] == 0


def test_final_goal_batch_final_resume_skips_stale_final_goal_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "square_M0_seed7"
    run_dir.mkdir(parents=True)
    model_path = run_dir / "models" / "ppo_phaseB_square_M0_corner.zip"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    base_metrics = {
        "painted_pixel_precision": "0.90",
        "painted_pixel_recall": "0.90",
        "painted_pixel_dice": "0.85",
        "off_target_pixel_ratio": "0.05",
        "path_rmse_m": "0.04",
        "path_max_m": "0.12",
        "straight_path_rmse_m": "0.04",
        "corner_path_rmse_m": "0.10",
        "corner_speed_ratio": "0.75",
        "corner_overshoot_m": "0.10",
        "led_precision": "0.95",
        "led_recall": "0.95",
        "led_flicker_rate": "0.05",
        "crash_rate": "0.0",
        "out_of_bounds_rate": "0.0",
        "rpm_saturation_ratio": "0.05",
        "action_norm_max": "1.5",
        "action_rate_max": "3.0",
    }
    summary = {
        "config": {
            "phase": "B",
            "wind_mode": "M0",
            "reference": {
                "trajectory": "square",
                "label": "square",
                "path_scale": 1.0,
                "square_side_m": 0.8,
                "unscaled_square_side_m": 0.8,
            },
            "seed": 7,
            "load_model": None,
            "eval_only": False,
            "speed": 0.35,
            "ctrl_freq": 30,
            "pyb_freq": 240,
            "requested_total_timesteps": 300000,
            "total_timesteps": 300000,
            "max_steps_override": None,
            "corner_window_m": 0.18,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "log_std_init": -2.0,
            "teacher_gain": 0.1,
            "teacher_window_m": 0.15,
            "bc_epochs": 2,
            "bc_episodes": 8,
            "bc_batch_size": 64,
            "bc_learning_rate": 1e-3,
            "bc_nonzero_weight": 1.0,
            "trained_action_filter": "corner_tangent_decel",
            "device": "cpu",
            "save_video": False,
        },
        "success_criteria": {"overall_pass": True},
        "final_goal_criteria": {"final_goal_pass": True, "painting_iou_abs": True},
        "metrics": [
            {"tag": "phaseB_zero", "painted_pixel_iou": "0.80", **base_metrics},
            {"tag": "phaseB_trained", "painted_pixel_iou": "0.10", **base_metrics},
        ],
        "artifacts": {"model_path": str(model_path)},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(batch, "_missing_runtime_dependencies", lambda: ["pybullet"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "final",
            "--output-dir",
            str(tmp_path),
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 2
    batch_summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert batch_summary["runs"][0]["status"] == "failed"
    assert "final_goal_criteria 재계산 불일치" in batch_summary["runs"][0]["resume_skip_reason"]
    assert batch_summary["reused_runs"] == 0


def test_final_goal_batch_final_profile_uses_long_multi_seed_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--profile",
            "final",
            "--output-dir",
            str(tmp_path),
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "final"
    assert summary["trajectories"] == ["square", "L", "DG", "CAT", "Pig", "RL", "drawn"]
    assert summary["wind_modes"] == ["M0", "M1", "M2"]
    assert summary["seeds"] == [7, 11, 17, 23, 29]
    assert summary["total_timesteps"] == 300000
    assert summary["planned_runs"] == 105
    assert all("--total-timesteps 300000" in run["command"] for run in summary["runs"])


def test_final_goal_batch_letters_profile_uses_long_letter_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--profile",
            "letters",
            "--output-dir",
            str(tmp_path),
        ],
    )

    rc = batch.run(batch.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "batch_summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "letters"
    assert summary["trajectories"] == ["L", "DG", "CAT", "Pig", "RL"]
    assert summary["wind_modes"] == ["M0", "M1", "M2"]
    assert summary["seeds"] == [7, 11, 17]
    assert summary["total_timesteps"] == 100000
    assert summary["planned_runs"] == 45
    assert all("--total-timesteps 100000" in run["command"] for run in summary["runs"])
    assert all("--trained-action-filter corner_tangent_decel" in run["command"] for run in summary["runs"])
    assert all("--trajectory letter" in run["command"] for run in summary["runs"])
