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
    assert "PyBullet GUI 보기" in dash.INDEX_HTML
    assert "대시보드 서버에 연결할 수 없습니다" in dash.INDEX_HTML
    assert "서버 연결 끊김" in dash.INDEX_HTML
    assert all(field.get("help") for field in dash.REWARD_FIELDS)
    assert "rollout 확인용" in dash.INDEX_HTML
    assert "PPO/BC 학습 env는 DIRECT" in dash.INDEX_HTML


def test_dashboard_json_safe_converts_nonfinite_values():
    data = {"score": float("nan"), "nested": [float("inf"), 1.0]}

    safe = dash._json_safe(data)

    assert safe == {"score": None, "nested": [None, 1.0]}


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
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m0_full_train")["recommended"]["total_timesteps"] >= 20000
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m1_disturbance")["recommended"]["max_steps"] == ""
    assert next(stage for stage in dash.EXPERIMENT_STAGES if stage["id"] == "m2_disturbance")["recommended"]["max_steps"] == ""


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
