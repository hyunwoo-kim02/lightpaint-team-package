from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src.train import run_final_goal_pipeline as pipeline


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_failed_eval_fixture(root: Path) -> None:
    criteria = {
        "matrix_complete": True,
        "expected_runs": 1,
        "observed_runs": 1,
        "missing_runs": [],
        "all_runs_final_goal_pass": False,
        "failed_final_goal_run_count": 1,
        "failed_final_goal_runs": [{"run_id": "L_M0_seed7"}],
        "robustness_relative_to_m0_pass": True,
        "robustness_failures": [],
        "batch_final_goal_pass": False,
    }
    summary = {
        "schema_version": 1,
        "profile": "final",
        "dry_run": False,
        "status": "complete",
        "planned_runs": 1,
        "dependency_preflight": {"ok": True},
        "batch_final_goal_criteria": criteria,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    _write_csv(
        root / "matrix_results.csv",
        [
            {
                "trajectory": "L",
                "wind_mode": "M0",
                "seed": 7,
                "run_id": "L_M0_seed7",
                "status": "complete",
                "overall_pass": "False",
                "final_goal_pass": "False",
            }
        ],
    )
    _write_csv(root / "metrics.csv", [{"run_id": "L_M0_seed7", "overall_pass": "False", "final_goal_pass": "False"}])
    _write_csv(
        root / "aggregate_metrics.csv",
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
    (root / "best_model_manifest.json").write_text(json.dumps({"schema_version": 1, "models": {}}), encoding="utf-8")


def test_final_goal_pipeline_dry_run_creates_train_and_eval_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "sanity",
            "--required-profile",
            "any",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "unit_pipeline",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 0
    root = tmp_path / "unit_pipeline"
    summary = json.loads((root / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "planned"
    assert summary["preflight"]["ok"] is True
    assert summary["preflight"]["runtime_dependency_check"] == "skipped_for_dry_run"
    assert summary["runtime"]["python_executable"] == sys.executable
    assert summary["runtime"]["argv"][0] == "prog"
    assert summary["runtime"]["package_root"]
    assert summary["verify_skipped"] == "dry_run"
    assert len(summary["steps"]) == 2
    assert (root / "train" / "batch_summary.json").exists()
    assert (root / "eval" / "batch_summary.json").exists()
    eval_summary = json.loads((root / "eval" / "batch_summary.json").read_text(encoding="utf-8"))
    assert eval_summary["eval_only"] is True
    assert eval_summary["model_manifest_preflight"]["ok"] is True


def test_final_goal_pipeline_command_summary_uses_final_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "final_plan",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "final_plan" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["profile"] == "final"
    assert "--profile final" in summary["commands"]["train"]
    assert "--eval-only" in summary["commands"]["eval"]
    assert "src.train.verify_final_goal_batch" in summary["commands"]["verify"]


def test_final_goal_pipeline_passes_batch_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "letters",
            "--required-profile",
            "any",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "letters_probe",
            "--trajectories",
            "L,DG",
            "--wind-modes",
            "M0",
            "--seeds",
            "7,11",
            "--total-timesteps",
            "1234",
            "--max-steps",
            "5",
            "--n-steps",
            "16",
            "--batch-size",
            "8",
            "--bc-epochs",
            "0",
            "--drawn-path",
            "data/drawn_paths/user/user_drawn_path.json",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 0
    summary = json.loads((tmp_path / "letters_probe" / "pipeline_summary.json").read_text(encoding="utf-8"))
    train_command = summary["commands"]["train"]
    eval_command = summary["commands"]["eval"]
    assert "--trajectories L,DG" in train_command
    assert "--wind-modes M0" in train_command
    assert "--seeds 7,11" in train_command
    assert "--total-timesteps 1234" in train_command
    assert "--max-steps 5" in train_command
    assert "--n-steps 16" in train_command
    assert "--batch-size 8" in train_command
    assert "--bc-epochs 0" in train_command
    assert "--drawn-path data/drawn_paths/user/user_drawn_path.json" in train_command
    assert "--eval-only" in eval_command
    train_summary = json.loads((tmp_path / "letters_probe" / "train" / "batch_summary.json").read_text(encoding="utf-8"))
    assert train_summary["planned_runs"] == 4
    assert train_summary["trajectories"] == ["L", "DG"]


def test_final_goal_pipeline_preflight_rejects_missing_runtime_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.run_final_goal_batch, "_missing_runtime_dependencies", lambda: ["pybullet"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "sanity",
            "--required-profile",
            "any",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "missing_deps",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "missing_deps" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "preflight"
    assert summary["preflight"]["ok"] is False
    assert summary["preflight"]["missing_runtime_dependencies"] == ["pybullet"]
    assert summary["steps"] == []


def test_final_goal_pipeline_preflight_rejects_missing_drawn_path(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.run_final_goal_batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "standard",
            "--required-profile",
            "any",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "missing_drawn",
            "--drawn-path",
            "data/drawn_paths/user/does_not_exist.json",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 2
    summary = json.loads((tmp_path / "missing_drawn" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "preflight"
    assert summary["preflight"]["ok"] is False
    assert any("drawn trajectory JSON 파일이 없습니다" in issue for issue in summary["preflight"]["issues"])


def test_final_goal_pipeline_verification_summary_reports_failed_criteria(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_failed_eval_fixture(eval_dir)

    summary = pipeline._verification_summary(eval_dir, required_profile="final")

    assert summary["ok"] is False
    assert summary["profile"] == "final"
    assert summary["batch_final_goal_pass"] is False
    assert summary["expected_runs"] == 1
    assert summary["observed_runs"] == 1
    assert summary["matrix_rows"] == 1
    assert "batch_final_goal_pass가 true가 아닙니다" in summary["issues"]
    assert any("matrix_results.csv에 통과하지 못한 행" in issue for issue in summary["issues"])


def test_final_goal_pipeline_stops_before_eval_when_train_manifest_incomplete(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_run_step(name, command, log_path):
        calls.append(name)
        if name == "train":
            train_dir = Path(command[command.index("--output-dir") + 1])
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "batch_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_runs": 0,
                        "failed_runs": 1,
                        "batch_final_goal_criteria": {"expected_runs": 1},
                    }
                ),
                encoding="utf-8",
            )
            (train_dir / "best_model_manifest.json").write_text(
                json.dumps({"schema_version": 1, "models": {}}),
                encoding="utf-8",
            )
        return {
            "name": name,
            "command": pipeline._quote_command(command),
            "return_code": 0,
            "log_path": str(log_path),
            "started_at": "2026-05-23T00:00:00",
            "finished_at": "2026-05-23T00:00:01",
        }

    monkeypatch.setattr(pipeline.run_final_goal_batch, "_missing_runtime_dependencies", lambda: [])
    monkeypatch.setattr(pipeline, "_run_step", fake_run_step)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--profile",
            "sanity",
            "--required-profile",
            "any",
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "incomplete_train",
            "--trajectories",
            "square",
            "--wind-modes",
            "M0",
            "--seeds",
            "7",
            "--total-timesteps",
            "16",
        ],
    )

    rc = pipeline.run(pipeline.parse_args())

    assert rc == 2
    assert calls == ["train"]
    summary = json.loads((tmp_path / "incomplete_train" / "pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failed_step"] == "train_manifest_preflight"
    assert summary["train_manifest_preflight"]["ok"] is False
    assert "square/M0/seed7" in summary["train_manifest_preflight"]["missing_models"]
    assert any("실패 run" in issue for issue in summary["train_manifest_preflight"]["issues"])


def test_final_goal_pipeline_run_step_forces_utf8_child_output(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    step = pipeline._run_step("verify", ["python", "-c", "print('검증')"], tmp_path / "verify.log")

    assert step["return_code"] == 0
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
