from __future__ import annotations

import sys

from src.train import experiment_dashboard as dash


def test_dashboard_preview_builds_training_command(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    payload = {
        "config": {
            "run_name": "unit smoke",
            "trajectory": "square",
            "wind_mode": "M0",
            "max_steps": 3,
            "total_timesteps": 0,
            "n_steps": 8,
            "batch_size": 8,
        },
        "reward": dash._reward_defaults(),
    }

    preview = dash._preview(payload)

    assert preview["issues"] == []
    assert sys.executable in preview["command"]
    assert "-m src.train.train_phase_b" in preview["command"]
    assert "--trajectory square" in preview["command"]
    assert "--wind-mode M0" in preview["command"]
    assert "--max-steps 3" in preview["command"]
    assert "--total-timesteps 0" in preview["command"]
    assert "--gui" not in preview["command"]


def test_dashboard_preview_can_enable_pybullet_gui(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    payload = {
        "config": {
            "run_name": "gui smoke",
            "trajectory": "square",
            "wind_mode": "M0",
            "max_steps": 3,
            "total_timesteps": 0,
            "n_steps": 8,
            "batch_size": 8,
            "gui": True,
        },
        "reward": dash._reward_defaults(),
    }

    preview = dash._preview(payload)

    assert preview["issues"] == []
    assert "--gui" in preview["command"]
    assert "--gui-hold-seconds 5.0" in preview["command"]


def test_dashboard_validation_catches_single_env_batch_shape():
    config = dict(dash.DEFAULT_CONFIG)
    config.update({"n_steps": 8, "batch_size": 16})

    issues = dash._validate_config(config, dash._reward_defaults())

    assert "batch_size는 n_steps 이하이어야 합니다" in issues


def test_dashboard_validation_catches_reward_range():
    reward = dash._reward_defaults()
    reward["LED_ON_THRESHOLD"] = 2.0

    issues = dash._validate_config(dict(dash.DEFAULT_CONFIG), reward)

    assert "LED_ON_THRESHOLD은 [0.0, 1.0] 범위여야 합니다" in issues


def test_dashboard_validation_catches_optimizer_and_runtime_ranges():
    config = dict(dash.DEFAULT_CONFIG)
    config.update(
        {
            "learning_rate": -1.0,
            "gamma": 1.5,
            "gae_lambda": -0.1,
            "clip_range": 0.0,
            "pyb_freq": 241,
            "eval_only": True,
            "load_model": "",
        }
    )

    issues = dash._validate_config(config, dash._reward_defaults())

    assert "learning_rate는 0보다 커야 합니다" in issues
    assert "gamma는 0과 1 사이여야 합니다" in issues
    assert "gae_lambda는 0과 1 사이여야 합니다" in issues
    assert "clip_range는 0보다 커야 합니다" in issues
    assert "pyb_freq는 ctrl_freq 이상의 정수배여야 합니다" in issues
    assert "eval_only는 load_model 경로가 필요합니다" in issues


def test_dashboard_html_exposes_korean_status_and_help():
    assert "실행 상태" in dash.INDEX_HTML
    assert "현재 설정 저장" in dash.INDEX_HTML
    assert "실험 로드맵" in dash.INDEX_HTML
    assert "전체 Batch" in dash.INDEX_HTML
    assert "통과 기준 계산" in dash.INDEX_HTML
    assert "선택 모델로 이어서 학습" in dash.INDEX_HTML
    assert "Load model" in dash.INDEX_HTML
    assert "추천 기본값" in dash.INDEX_HTML
    assert "Settle time" in dash.INDEX_HTML
    assert "0.30~0.35" in dash.INDEX_HTML
    assert "Smoke preset" in dash.INDEX_HTML
    assert "Run 시작 요청을 보냈습니다" in dash.INDEX_HTML
    assert "M0 full" in dash.INDEX_HTML
    assert "최종 평가 적용" in dash.INDEX_HTML
    assert "Painting 결과 reward" in dash.INDEX_HTML
    assert "Action norm max" in dash.INDEX_HTML
    assert "Action rate max" in dash.INDEX_HTML
    assert "PyBullet GUI 보기" in dash.INDEX_HTML
    assert "letters" in dash.INDEX_HTML
    assert "L/DG/CAT/Pig/RL 전체" in dash.INDEX_HTML
    assert "letters 장시간 학습 적용" in dash.INDEX_HTML
    assert "standard 전체 학습 적용" in dash.INDEX_HTML
    assert "final 최종 계획 적용" in dash.INDEX_HTML
    assert 'applyBatchPreset("letters", false' in dash.INDEX_HTML
    assert 'applyBatchPreset("standard", false' in dash.INDEX_HTML
    assert 'applyBatchPreset("final", true' in dash.INDEX_HTML
    assert "평가 Action filter" in dash.INDEX_HTML
    assert "Eval-only 최종 평가" in dash.INDEX_HTML
    assert "batch_model_manifest" in dash.INDEX_HTML
    assert "batch_trained_action_filter" in dash.INDEX_HTML
    assert "대시보드 서버에 연결할 수 없습니다" in dash.INDEX_HTML
    assert "서버 연결 끊김" in dash.INDEX_HTML
    assert all(field.get("help") for field in dash.REWARD_FIELDS)
    assert "rollout 확인용" in dash.INDEX_HTML
    assert "PPO/BC 학습 env는 DIRECT" in dash.INDEX_HTML


def test_dashboard_batch_preview_builds_multi_letter_command(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    payload = {
        "batch": {
            "batch_run_name": "final_matrix",
            "batch_profile": "final",
            "batch_dry_run": True,
            "batch_resume": True,
            "batch_continue_on_error": True,
        },
        "reward": dash._reward_defaults(),
    }

    preview = dash._preview_batch(payload)

    assert preview["issues"] == []
    assert sys.executable in preview["command"]
    assert "-m src.train.run_final_goal_batch" in preview["command"]
    assert "--profile final" in preview["command"]
    assert "--trained-action-filter corner_tangent_decel" in preview["command"]
    assert "--teacher-window-m 0.15" in preview["command"]
    assert "--device cuda" in preview["command"]
    assert "--dry-run" in preview["command"]
    assert "--continue-on-error" in preview["command"]
    assert preview["profile_defaults"]["trajectories"] == "square,L,DG,CAT,Pig,RL,drawn"
    assert preview["profile_defaults"]["seeds"] == "7,11,17,23,29"
    assert preview["profile_defaults"]["total_timesteps"] == 300000


def test_dashboard_batch_preview_supports_letters_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    payload = {
        "batch": {
            "batch_run_name": "letter_matrix",
            "batch_profile": "letters",
            "batch_dry_run": True,
            "batch_resume": True,
            "batch_continue_on_error": True,
        },
        "reward": dash._reward_defaults(),
    }

    preview = dash._preview_batch(payload)

    assert preview["issues"] == []
    assert "--profile letters" in preview["command"]
    assert preview["profile_defaults"]["trajectories"] == "L,DG,CAT,Pig,RL"
    assert preview["profile_defaults"]["wind_modes"] == "M0,M1,M2"
    assert preview["profile_defaults"]["seeds"] == "7,11,17"
    assert preview["profile_defaults"]["total_timesteps"] == 100000


def test_dashboard_batch_preview_supports_eval_only_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    manifest = tmp_path / "best_model_manifest.json"
    payload = {
        "batch": {
            "batch_run_name": "final_eval",
            "batch_profile": "final",
            "batch_dry_run": True,
            "batch_resume": True,
            "batch_continue_on_error": False,
            "batch_eval_only": True,
            "batch_model_manifest": str(manifest),
        },
        "reward": dash._reward_defaults(),
    }

    preview = dash._preview_batch(payload)

    assert preview["issues"] == []
    assert "--profile final" in preview["command"]
    assert "--eval-only" in preview["command"]
    assert "--model-manifest" in preview["command"]
    assert str(manifest) in preview["command"]


def test_dashboard_batch_validation_requires_eval_manifest():
    config = dict(dash.DEFAULT_BATCH_CONFIG)
    config.update({"batch_eval_only": True, "batch_model_manifest": ""})

    issues = dash._validate_batch_config(config, dash._reward_defaults())

    assert "batch_eval_only는 batch_model_manifest 경로가 필요합니다" in issues


def test_dashboard_json_safe_converts_nonfinite_values():
    data = {"score": float("nan"), "nested": [float("inf"), 1.0]}

    safe = dash._json_safe(data)

    assert safe == {"score": None, "nested": [None, 1.0]}


def test_dashboard_write_json_preserves_korean_text(tmp_path):
    path = tmp_path / "dashboard_run.json"

    dash._write_json(path, {"message": "대시보드 실행 로그", "path": "강화학습"})

    text = path.read_text(encoding="utf-8")
    assert "대시보드 실행 로그" in text
    assert "강화학습" in text
    assert "\\uac15" not in text


def test_quality_score_returns_none_for_nan_metric():
    summary = {
        "metrics": [
            {
                "tag": "phaseB_trained",
                "painted_pixel_coverage": "0.1",
                "tracking_rmse_m": "0.1",
                "corner_path_rmse_m": "nan",
                "crash": 0,
            }
        ]
    }

    assert dash._quality_score(summary) is None


def test_quality_score_uses_overlap_and_off_target_metrics():
    base = {
        "metrics": [
            {
                "tag": "phaseB_trained",
                "painted_pixel_coverage": "0.8",
                "painted_pixel_precision": "0.9",
                "painted_pixel_iou": "0.72",
                "off_target_pixel_ratio": "0.1",
                "tracking_rmse_m": "0.05",
                "corner_path_rmse_m": "0.04",
                "crash": 0,
            }
        ]
    }
    worse = {
        "metrics": [
            {
                "tag": "phaseB_trained",
                "painted_pixel_coverage": "0.8",
                "painted_pixel_precision": "0.5",
                "painted_pixel_iou": "0.42",
                "off_target_pixel_ratio": "0.5",
                "tracking_rmse_m": "0.05",
                "corner_path_rmse_m": "0.04",
                "crash": 0,
            }
        ]
    }

    assert dash._quality_score(base) > dash._quality_score(worse)


def test_dashboard_command_uses_selected_reference_options(tmp_path):
    config = dict(dash.DEFAULT_CONFIG)
    config.update(
        {
            "trajectory": "letter",
            "label": "DG",
            "letter_plane": "xy",
            "width_m": 0.7,
            "height_m": 0.6,
            "path_scale": 0.9,
        }
    )

    command = dash._build_command(config, tmp_path / "run")

    assert "--trajectory" in command
    assert "letter" in command
    assert "--label" in command
    assert "DG" in command
    assert "--letter-plane" in command
    assert "xy" in command
    assert "--width-m" in command
    assert "--height-m" in command
    assert "--square-side" not in command


def test_dashboard_options_define_roadmap_and_success_criteria():
    stage_ids = {stage["id"] for stage in dash.EXPERIMENT_STAGES}
    criterion_keys = {criterion["key"] for criterion in dash.SUCCESS_CRITERIA}

    assert {
        "env_smoke",
        "m0_corner",
        "m0_full_train",
        "m1_disturbance",
        "m2_disturbance",
        "final_painting",
    } <= stage_ids
    assert "overall_pass" in criterion_keys
    assert {
        "painting_coverage_kept_90pct",
        "painting_precision_kept_90pct",
        "painting_iou_kept_90pct",
        "off_target_ratio_not_worse_5pct",
    } <= criterion_keys
    assert any("load_model" in stage["next"] for stage in dash.EXPERIMENT_STAGES)
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m0_full_train")["recommended"]["max_steps"] == ""
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m0_full_train")["recommended"]["total_timesteps"] >= 100000
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m2_disturbance")["recommended"]["total_timesteps"] >= 200000
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m1_disturbance")["recommended"]["max_steps"] == ""
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m2_disturbance")["recommended"]["max_steps"] == ""
    assert dash.DEFAULT_BATCH_CONFIG["batch_profile"] == "standard"
    assert dash.DEFAULT_BATCH_CONFIG["batch_dry_run"] is True
    assert dash.DEFAULT_BATCH_CONFIG["batch_trained_action_filter"] == "corner_tangent_decel"
    assert dash.DEFAULT_BATCH_CONFIG["batch_device"] == "cuda"
    assert dash.DEFAULT_BATCH_CONFIG["batch_eval_only"] is False


def test_dashboard_summary_exposes_reusable_model_path(tmp_path):
    run_dir = tmp_path / "20260521_000000_unit"
    run_dir.mkdir()
    (run_dir / "dashboard_run.json").write_text(
        '{"created_at":"2026-05-21T00:00:00","title":"unit"}',
        encoding="utf-8",
    )
    model_path = tmp_path / "model.zip"
    (run_dir / "summary.json").write_text(
        '{"artifacts":{"model_path":"' + str(model_path).replace("\\", "\\\\") + '"},"metrics":[]}',
        encoding="utf-8",
    )

    summary = dash._summarize_run(run_dir)

    assert summary["model_path"] == str(model_path)


def test_dashboard_summary_exposes_visual_artifacts_and_skips_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    run_dir = tmp_path / "20260521_000001_visual"
    vis_dir = run_dir / "artifacts" / "visualization"
    frames_dir = vis_dir / "phase_b_frames"
    frames_dir.mkdir(parents=True)
    (run_dir / "dashboard_run.json").write_text('{"title":"visual"}', encoding="utf-8")
    (vis_dir / "summary.png").write_bytes(b"png")
    (vis_dir / "flight.mp4").write_bytes(b"mp4")
    (frames_dir / "frame_000001.png").write_bytes(b"frame")
    (run_dir / "summary.json").write_text('{"metrics":[]}', encoding="utf-8")

    summary = dash._summarize_run(run_dir)

    rel_paths = {item["rel_path"] for item in summary["artifacts"]}
    assert "artifacts/visualization/summary.png" in rel_paths
    assert "artifacts/visualization/flight.mp4" in rel_paths
    assert "artifacts/visualization/phase_b_frames/frame_000001.png" not in rel_paths
    kinds = {item["rel_path"]: item["kind"] for item in summary["artifacts"]}
    assert kinds["artifacts/visualization/summary.png"] == "image"
    assert kinds["artifacts/visualization/flight.mp4"] == "video"
    assert all(item["url"].startswith("/api/artifacts/20260521_000001_visual/") for item in summary["artifacts"])


def test_dashboard_summary_exposes_batch_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    run_dir = tmp_path / "20260521_000003_batch"
    run_dir.mkdir()
    (run_dir / "dashboard_run.json").write_text(
        '{"title":"batch","kind":"batch","created_at":"2026-05-21T00:00:00"}',
        encoding="utf-8",
    )
    (run_dir / "batch_summary.json").write_text(
        '{"profile":"standard","planned_runs":63,"batch_final_goal_criteria":{"batch_final_goal_pass":false}}',
        encoding="utf-8",
    )
    (run_dir / "metrics.csv").write_text("run_id,painting_iou\n", encoding="utf-8")

    summary = dash._summarize_run(run_dir)

    assert summary["status"] == "complete"
    assert summary["kind"] == "batch"
    assert summary["batch_summary"]["planned_runs"] == 63
    rel_paths = {item["rel_path"] for item in summary["artifacts"]}
    assert "batch_summary.json" in rel_paths
    assert "metrics.csv" in rel_paths


def test_dashboard_artifact_resolver_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "_RUNS_ROOT", tmp_path)
    run_dir = tmp_path / "20260521_000002_safe"
    run_dir.mkdir()
    (run_dir / "result.png").write_bytes(b"png")

    assert dash._resolve_run_artifact("20260521_000002_safe", "result.png") == run_dir.resolve() / "result.png"

    try:
        dash._resolve_run_artifact("20260521_000002_safe", "../outside.png")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")
