"""Local experiment dashboard for Phase B light-painting runs.

The dashboard intentionally uses only the Python standard library.  Team
members can launch it from whatever virtual environment they already use, and
training subprocesses will reuse that same Python executable.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from src.train import run_final_goal_batch as final_batch

_INHERITED_REWARD_CONFIG = os.environ.pop("LIGHTPAINT_REWARD_CONFIG", None)
try:
    from src.env import reward_config as reward_defaults
finally:
    if _INHERITED_REWARD_CONFIG is not None:
        os.environ["LIGHTPAINT_REWARD_CONFIG"] = _INHERITED_REWARD_CONFIG

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
_DASHBOARD_ROOT = _PKG_ROOT / "artifacts" / "dashboard"
_RUNS_ROOT = _DASHBOARD_ROOT / "runs"
_ARTIFACT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm", ".html", ".json", ".csv", ".txt", ".log", ".zip"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
_VIDEO_EXTENSIONS = {".mp4", ".webm"}


REWARD_FIELDS: tuple[dict[str, Any], ...] = (
    {"key": "RESIDUAL_DELTA_MAX", "group": "action", "label": "RESIDUAL_DELTA_MAX", "min": 0.05, "max": 1.5, "step": 0.01, "help": "Phase B action[0:3]이 PID target velocity에 더하는 최대 residual 속도입니다. 값이 크면 정책이 속도를 강하게 바꿀 수 있지만 불안정해질 수 있습니다."},
    {"key": "LED_RESIDUAL_SCALE", "group": "action", "label": "LED_RESIDUAL_SCALE", "min": 0.0, "max": 2.0, "step": 0.01, "help": "Phase B action[3]을 scripted LED reference에 더할 때 쓰는 배율입니다. 값이 크면 LED on/off 보정 폭이 커집니다."},
    {"key": "LED_ON_THRESHOLD", "group": "led", "label": "LED_ON_THRESHOLD", "min": 0.0, "max": 1.0, "step": 0.01, "help": "brightness가 이 값보다 클 때 LED가 켜진 것으로 판정합니다. painting, LED miss, flicker 해석에 영향을 줍니다."},
    {"key": "PATH_SIGMA_M", "group": "tracking", "label": "PATH_SIGMA_M", "min": 0.01, "max": 0.3, "step": 0.005, "help": "현재 위치가 reference path에서 벗어난 거리를 reward로 바꿀 때 쓰는 폭입니다. 작을수록 path 이탈을 더 강하게 벌합니다."},
    {"key": "SCHEDULE_SIGMA_M", "group": "tracking", "label": "SCHEDULE_SIGMA_M", "min": 0.01, "max": 0.5, "step": 0.005, "help": "시간 t에서 기대되는 reference point와의 오차를 reward로 바꿀 때 쓰는 폭입니다. 작을수록 정해진 시간표를 더 강하게 따릅니다."},
    {"key": "W_PATH", "group": "tracking", "label": "W_PATH", "min": 0.0, "max": 5.0, "step": 0.05, "help": "경로 자체에 가까이 머무는 reward 가중치입니다. 올리면 path 이탈을 줄이는 방향으로 학습 압력이 커집니다."},
    {"key": "W_SCHEDULE", "group": "tracking", "label": "W_SCHEDULE", "min": 0.0, "max": 5.0, "step": 0.05, "help": "시간별 reference point를 따라가게 하는 reward 가중치입니다. 올리면 일정 속도/일정 시간표 유지 압력이 커집니다."},
    {"key": "W_ACTION_MAG", "group": "regularization", "label": "W_ACTION_MAG", "min": 0.0, "max": 5.0, "step": 0.05, "help": "RL residual action 크기에 대한 penalty입니다. 올리면 PID baseline에서 크게 벗어나는 행동을 억제합니다."},
    {"key": "W_ACTION_RATE", "group": "regularization", "label": "W_ACTION_RATE", "min": 0.0, "max": 3.0, "step": 0.05, "help": "이전 step 대비 residual action 변화량 penalty입니다. 올리면 급격한 속도/LED 명령 변화를 줄입니다."},
    {"key": "LAMBDA_SMOOTH", "group": "regularization", "label": "LAMBDA_SMOOTH", "min": 0.0, "max": 2.0, "step": 0.05, "help": "최종 제어 명령의 smoothness penalty입니다. PyBullet에서는 RPM 변화, fallback에서는 velocity command 변화에 영향을 줍니다."},
    {"key": "W_NEW_TARGET", "group": "painting", "label": "W_NEW_TARGET", "min": 0.0, "max": 5.0, "step": 0.05, "help": "아직 칠하지 않은 target pixel을 새로 칠했을 때의 보상입니다. 올리면 coverage를 더 적극적으로 늘리려 합니다."},
    {"key": "W_OFF_TARGET", "group": "painting", "label": "W_OFF_TARGET", "min": 0.0, "max": 5.0, "step": 0.05, "help": "target mask 밖을 칠했을 때의 penalty입니다. 올리면 글자 밖으로 빛이 새는 행동을 더 강하게 벌합니다."},
    {"key": "W_REPAINT", "group": "painting", "label": "W_REPAINT", "min": 0.0, "max": 3.0, "step": 0.05, "help": "이미 칠한 영역을 다시 칠하는 penalty입니다. 올리면 같은 부분을 반복해서 칠하는 행동을 억제합니다."},
    {"key": "W_LED_MISS", "group": "led", "label": "W_LED_MISS", "min": 0.0, "max": 5.0, "step": 0.05, "help": "one-sided LED miss prior 가중치입니다. scripted LED reference보다 어두울 때만 벌하고, led_ref=0인데 켜는 false-on은 off-target/repaint 같은 painting outcome 항에서 판단합니다. residual 단계에서는 1.0~1.5가 안전하고, full RL LED 단계로 갈수록 0에 가깝게 낮춥니다."},
    {"key": "W_LED_FLICKER", "group": "led", "label": "W_LED_FLICKER", "min": 0.0, "max": 1.0, "step": 0.01, "help": "연속 step 사이 LED brightness 변화 penalty입니다. 올리면 LED가 자주 깜빡이는 정책을 억제합니다."},
    {"key": "CORNER_WINDOW_M", "group": "corner", "label": "CORNER_WINDOW_M", "min": 0.02, "max": 0.8, "step": 0.01, "help": "corner 전후 몇 m 구간을 corner reward 영향권으로 볼지 정합니다. 너무 크면 직선 구간까지 corner로 취급될 수 있습니다."},
    {"key": "CORNER_SPEED_FRACTION", "group": "corner", "label": "CORNER_SPEED_FRACTION", "min": 0.1, "max": 1.2, "step": 0.01, "help": "corner에서 목표로 삼는 속도 비율입니다. 0.65면 reference speed의 65% 수준으로 감속하는 방향입니다."},
    {"key": "CORNER_SIGMA_M", "group": "corner", "label": "CORNER_SIGMA_M", "min": 0.01, "max": 0.3, "step": 0.005, "help": "corner 근처 tracking/path error를 reward로 바꿀 때 쓰는 폭입니다. 작을수록 corner 오차에 민감합니다."},
    {"key": "W_CORNER_SPEED", "group": "corner", "label": "W_CORNER_SPEED", "min": 0.0, "max": 40.0, "step": 0.5, "help": "corner 영향권에서 목표 감속 속도를 맞추는 reward 가중치입니다. 코너 감속 학습에 직접적인 영향을 줍니다."},
    {"key": "W_CORNER_TRACK", "group": "corner", "label": "W_CORNER_TRACK", "min": 0.0, "max": 5.0, "step": 0.05, "help": "corner 근처에서 시간별 reference point를 따라가는 reward 가중치입니다. 올리면 코너 시간표 추종 압력이 커집니다."},
    {"key": "W_CORNER_PATH", "group": "corner", "label": "W_CORNER_PATH", "min": 0.0, "max": 5.0, "step": 0.05, "help": "corner 근처에서 path 자체에 붙어 있도록 하는 reward 가중치입니다. 코너에서 바깥으로 튀는 현상을 줄이는 데 관련됩니다."},
    {"key": "W_CORNER_DECEL", "group": "corner", "label": "W_CORNER_DECEL", "min": 0.0, "max": 10.0, "step": 0.1, "help": "corner 접근 시 reference tangent 반대 방향 residual, 즉 감속 residual을 장려하는 항입니다."},
    {"key": "W_CORNER_ACCEL", "group": "corner", "label": "W_CORNER_ACCEL", "min": 0.0, "max": 10.0, "step": 0.1, "help": "corner 접근 구간에서 reference tangent 방향으로 더 가속하는 residual을 벌하는 항입니다."},
    {"key": "W_CORNER_LATERAL", "group": "corner", "label": "W_CORNER_LATERAL", "min": 0.0, "max": 10.0, "step": 0.1, "help": "corner 근처에서 tangent와 무관한 옆 방향 residual을 벌하는 항입니다. 불필요한 lateral 흔들림을 줄입니다."},
    {"key": "W_COMPLETION", "group": "terminal", "label": "W_COMPLETION", "min": 0.0, "max": 10.0, "step": 0.1, "help": "episode 종료 시 painting coverage가 좋을 때 주는 bonus입니다. 최종 결과물 품질을 직접 밀어줍니다."},
    {"key": "W_INCOMPLETE", "group": "terminal", "label": "W_INCOMPLETE", "min": 0.0, "max": 10.0, "step": 0.1, "help": "episode 종료 시 target coverage가 부족할 때 주는 penalty입니다. 미완성 light painting을 줄입니다."},
    {"key": "W_TERMINAL_BOUNDS", "group": "terminal", "label": "W_TERMINAL_BOUNDS", "min": 0.0, "max": 300.0, "step": 5.0, "help": "드론이 허용 bounds 밖으로 나가 episode가 종료될 때의 penalty입니다. 너무 낮으면 crash 회피 학습이 약해질 수 있습니다."},
)


DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "",
    "trajectory": "letter",
    "label": "DG",
    "square_side": 0.8,
    "letter_plane": "xz",
    "drawn_path": "data/drawn_paths/user/user_drawn_path.json",
    "drawn_plane": "",
    "drawn_space": "",
    "width_m": "",
    "height_m": "",
    "path_scale": "",
    "max_waypoints": "",
    "smooth_ref": "",
    "smooth_window_m": "",
    "wind_mode": "M0",
    "speed": 0.35,
    "settle_time": 2.0,
    "max_steps": "",
    "corner_window_m": 0.18,
    "init_box_size": 0.0,
    "gui": False,
    "gui_hold_seconds": 5.0,
    "led_always_on": False,
    "ctrl_freq": 30,
    "pyb_freq": 240,
    "seed": 7,
    "total_timesteps": 100000,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 2,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "clip_range": 0.2,
    "log_std_init": -2.0,
    "teacher_gain": 0.10,
    "teacher_window_m": 0.15,
    "bc_epochs": 0,
    "bc_episodes": 4,
    "bc_batch_size": 64,
    "bc_learning_rate": 0.001,
    "bc_nonzero_weight": 1.0,
    "trained_action_filter": final_batch.DEFAULT_TRAINED_ACTION_FILTER,
    "device": "cuda",
    "verbose": 0,
    "save_video": False,
    "load_model": "",
    "eval_only": False,
    "reset_num_timesteps": False,
}

DEFAULT_BATCH_CONFIG: dict[str, Any] = {
    "batch_run_name": "dual_path_final_plan",
    "batch_profile": "final",
    "batch_dry_run": True,
    "batch_resume": True,
    "batch_continue_on_error": True,
    "batch_trained_action_filter": final_batch.DEFAULT_TRAINED_ACTION_FILTER,
    "batch_teacher_window_m": 0.15,
    "batch_device": "cuda",
    "batch_eval_only": False,
    "batch_model_manifest": "",
}

EXPERIMENT_STAGES: tuple[dict[str, Any], ...] = (
    {
        "id": "env_smoke",
        "title": "0. 환경 / API smoke",
        "goal": "PyBullet dependency, reset/step, dashboard subprocess 실행이 정상인지 확인합니다.",
        "preset": "smoke",
        "wind_mode": "M0",
        "recommended": {"trajectory": "square", "max_steps": 3, "total_timesteps": 0},
        "pass": "Run 상태가 complete이고 summary.json이 생성되어야 합니다. 이 단계의 overall_pass=false는 학습 성능 실패가 아니라 너무 짧은 smoke의 정상적인 결과일 수 있습니다.",
        "next": "문제가 없으면 M0 corner PPO smoke로 이동합니다.",
    },
    {
        "id": "m0_corner",
        "title": "1. M0 corner sanity",
        "goal": "외란 없는 상태에서 짧은 PPO 설정으로 Phase B residual velocity와 reward/metric 흐름이 동작하는지 확인합니다.",
        "preset": "short",
        "wind_mode": "M0",
        "recommended": {"trajectory": "square", "max_steps": 90, "total_timesteps": 2048},
        "pass": "Run 상태가 complete이고 summary/metrics/model_path가 생성되어야 합니다. 이 단계는 빠른 sanity라 overall_pass가 false일 수 있습니다.",
        "next": "문제가 없으면 M0 full train 단계로 이동합니다.",
    },
    {
        "id": "m0_full_train",
        "title": "2. M0 full corner 학습",
        "goal": "외란 없는 full horizon에서 코너 감속과 path tracking 개선을 실제 학습 목표로 확인합니다.",
        "preset": "short",
        "wind_mode": "M0",
        "recommended": {"trajectory": "square", "max_steps": "", "total_timesteps": 100000, "n_steps": 256, "batch_size": 64, "bc_epochs": 2, "bc_episodes": 8},
        "pass": "corner path RMSE가 zero-action보다 20% 이상 줄고, 직선 path RMSE가 10% 이상 악화되지 않아야 합니다.",
        "next": "완료된 model_path를 다음 M1 실험의 load_model로 재사용합니다.",
    },
    {
        "id": "m1_disturbance",
        "title": "3. M1 외란 강건성",
        "goal": "M0에서 얻은 weight를 이어받아 약한 external force 환경에서 tracking과 painting coverage를 유지합니다.",
        "preset": "robust",
        "wind_mode": "M1",
        "recommended": {"trajectory": "letter", "label": "L", "max_steps": "", "total_timesteps": 100000, "n_steps": 256, "batch_size": 64, "bc_epochs": 2, "bc_episodes": 8},
        "pass": "crash=0, painting coverage가 M0 zero baseline의 90% 이상, straight path RMSE가 허용 범위 내여야 합니다.",
        "next": "M1 model_path를 M2 실험의 load_model로 재사용합니다.",
    },
    {
        "id": "m2_disturbance",
        "title": "4. M2 외란 강건성",
        "goal": "더 강한 external force 조건에서도 경로 이탈과 off-target painting을 억제합니다.",
        "preset": "robust",
        "wind_mode": "M2",
        "recommended": {"trajectory": "letter", "label": "L", "max_steps": "", "total_timesteps": 200000, "n_steps": 256, "batch_size": 64, "bc_epochs": 2, "bc_episodes": 8},
        "pass": "crash=0, coverage 유지, corner/straight metric 악화가 허용 범위 안이어야 합니다.",
        "next": "letter/drawn final painting 실험으로 이동합니다.",
    },
    {
        "id": "final_painting",
        "title": "5. 최종 light painting",
        "goal": "letter 또는 drawn path에서 최종 목표인 정확한 드론 라이트 페인팅 품질을 평가합니다.",
        "preset": "robust",
        "wind_mode": "M2",
        "recommended": {"trajectory": "letter", "label": "L", "max_steps": "", "total_timesteps": 0, "save_video": True, "eval_only": True},
        "pass": "painted pixel coverage, off-target 억제, tracking RMSE, crash=0, 영상/trajectory artifact를 함께 확인합니다.",
        "next": "좋은 run의 dashboard_run.json, reward_overrides.json, summary.json, model zip을 공유합니다.",
    },
)

SUCCESS_CRITERIA: tuple[dict[str, str], ...] = (
    {
        "key": "corner_path_rmse_reduced_20pct",
        "label": "Corner path RMSE 20% 개선",
        "formula": "phaseB_trained.corner_path_rmse_m <= phaseB_zero.corner_path_rmse_m * 0.80",
        "why": "코너에서 path 밖으로 튀는 문제를 residual policy가 실제로 줄였는지 보는 핵심 기준입니다.",
    },
    {
        "key": "corner_speed_reduced_5_to_20pct",
        "label": "Corner 속도 5~20% 감속",
        "formula": "phaseB_zero.corner_mean_speed_mps * 0.80 <= phaseB_trained.corner_mean_speed_mps <= phaseB_zero.corner_mean_speed_mps * 0.95",
        "why": "무조건 느리게 가는 정책이 아니라, corner에서만 적절히 감속하는지 확인합니다.",
    },
    {
        "key": "straight_path_rmse_not_worse_10pct",
        "label": "직선 path RMSE 10% 이내 유지",
        "formula": "phaseB_trained.straight_path_rmse_m <= phaseB_zero.straight_path_rmse_m * 1.10",
        "why": "코너를 개선하면서 직선 구간의 기존 PID baseline 성능을 망치지 않아야 합니다.",
    },
    {
        "key": "painting_coverage_kept_90pct",
        "label": "Painting coverage 90% 이상 유지",
        "formula": "phaseB_trained.painted_pixel_coverage >= phaseB_zero.painted_pixel_coverage * 0.90",
        "why": "경로 추종만 좋아지고 실제 light painting 결과가 나빠지는 것을 막는 기준입니다.",
    },
    {
        "key": "painting_precision_kept_90pct",
        "label": "Painting precision 90% 이상 유지",
        "formula": "phaseB_trained.painted_pixel_precision >= phaseB_zero.painted_pixel_precision * 0.90",
        "why": "target 밖을 많이 칠하면서 coverage만 유지하는 정책을 걸러냅니다.",
    },
    {
        "key": "painting_iou_kept_90pct",
        "label": "Painting IoU 90% 이상 유지",
        "formula": "phaseB_trained.painted_pixel_iou >= phaseB_zero.painted_pixel_iou * 0.90",
        "why": "target coverage와 off-target painting을 하나의 overlap 품질 기준으로 함께 봅니다.",
    },
    {
        "key": "off_target_ratio_not_worse_5pct",
        "label": "Off-target ratio +5%p 이내",
        "formula": "phaseB_trained.off_target_pixel_ratio <= phaseB_zero.off_target_pixel_ratio + 0.05",
        "why": "coverage가 좋아 보여도 target 밖에 빛이 새는 run을 통과시키지 않기 위한 기준입니다.",
    },
    {
        "key": "overall_pass",
        "label": "Overall pass",
        "formula": "위 기준을 모두 만족",
        "why": "실행 성공과 학습 성능 통과를 구분합니다. complete는 subprocess 성공이고, overall_pass는 실험 목표 통과입니다.",
    },
)


NUMERIC_ARGS: dict[str, tuple[str, type]] = {
    "square_side": ("--square-side", float),
    "width_m": ("--width-m", float),
    "height_m": ("--height-m", float),
    "path_scale": ("--path-scale", float),
    "max_waypoints": ("--max-waypoints", int),
    "smooth_window_m": ("--smooth-window-m", float),
    "speed": ("--speed", float),
    "settle_time": ("--settle-time", float),
    "max_steps": ("--max-steps", int),
    "corner_window_m": ("--corner-window-m", float),
    "init_box_size": ("--init-box-size", float),
    "gui_hold_seconds": ("--gui-hold-seconds", float),
    "ctrl_freq": ("--ctrl-freq", int),
    "pyb_freq": ("--pyb-freq", int),
    "seed": ("--seed", int),
    "total_timesteps": ("--total-timesteps", int),
    "n_steps": ("--n-steps", int),
    "batch_size": ("--batch-size", int),
    "n_epochs": ("--n-epochs", int),
    "learning_rate": ("--learning-rate", float),
    "gamma": ("--gamma", float),
    "gae_lambda": ("--gae-lambda", float),
    "ent_coef": ("--ent-coef", float),
    "clip_range": ("--clip-range", float),
    "log_std_init": ("--log-std-init", float),
    "teacher_gain": ("--teacher-gain", float),
    "teacher_window_m": ("--teacher-window-m", float),
    "bc_epochs": ("--bc-epochs", int),
    "bc_episodes": ("--bc-episodes", int),
    "bc_batch_size": ("--bc-batch-size", int),
    "bc_learning_rate": ("--bc-learning-rate", float),
    "bc_nonzero_weight": ("--bc-nonzero-weight", float),
    "verbose": ("--verbose", int),
}


STRING_ARGS: dict[str, str] = {
    "trajectory": "--trajectory",
    "label": "--label",
    "letter_plane": "--letter-plane",
    "drawn_path": "--drawn-path",
    "drawn_plane": "--drawn-plane",
    "drawn_space": "--drawn-space",
    "wind_mode": "--wind-mode",
    "trained_action_filter": "--trained-action-filter",
    "device": "--device",
    "load_model": "--load-model",
}


@dataclass
class Job:
    run_id: str
    process: subprocess.Popen[str]
    output_dir: Path
    log_path: Path
    stopped: bool = False


class DashboardState:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.run_id] = job

    def get(self, run_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(run_id)


STATE = DashboardState()


def _now_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._-") or "run"


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _coerce_number(raw: Any, kind: type, name: str) -> int | float:
    if kind is int:
        value = int(raw)
    else:
        value = float(raw)
    if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다")
    return value


def _reward_defaults() -> dict[str, float]:
    return {field["key"]: float(getattr(reward_defaults, field["key"])) for field in REWARD_FIELDS}


def _normalize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    config = dict(DEFAULT_CONFIG)
    config.update(payload.get("config") or {})
    reward_raw = payload.get("reward") or _reward_defaults()
    reward: dict[str, float] = {}
    allowed = {field["key"] for field in REWARD_FIELDS}
    for key, value in reward_raw.items():
        if key in allowed:
            reward[key] = float(value)
    for key, value in _reward_defaults().items():
        reward.setdefault(key, value)
    return config, reward


def _normalize_batch_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    config = dict(DEFAULT_BATCH_CONFIG)
    config.update(payload.get("batch") or {})
    reward_raw = payload.get("reward") or _reward_defaults()
    reward: dict[str, float] = {}
    allowed = {field["key"] for field in REWARD_FIELDS}
    for key, value in reward_raw.items():
        if key in allowed:
            reward[key] = float(value)
    for key, value in _reward_defaults().items():
        reward.setdefault(key, value)
    return config, reward


def _validate_config(config: dict[str, Any], reward: dict[str, float]) -> list[str]:
    issues: list[str] = []
    trajectory = str(config.get("trajectory", "square"))
    if trajectory not in {"square", "letter", "drawn"}:
        issues.append("trajectory는 square, letter, drawn 중 하나여야 합니다")
    if trajectory == "drawn" and not str(config.get("drawn_path") or "").strip():
        issues.append("drawn trajectory는 drawn_path가 필요합니다")
    if str(config.get("wind_mode", "M0")) not in {"M0", "M1", "M2"}:
        issues.append("wind_mode는 M0, M1, M2 중 하나여야 합니다")
    if str(config.get("trained_action_filter", "none")) not in {"none", "corner_tangent_decel"}:
        issues.append("trained_action_filter 값이 올바르지 않습니다")

    for key, (_, kind) in NUMERIC_ARGS.items():
        if _is_empty(config.get(key)):
            continue
        try:
            _coerce_number(config[key], kind, key)
        except (TypeError, ValueError) as exc:
            issues.append(str(exc))

    def _num(key: str, default: Any | None = None) -> float | None:
        raw = config.get(key, default)
        if _is_empty(raw):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _require_positive(key: str, label: str) -> None:
        value = _num(key)
        if value is not None and value <= 0.0:
            issues.append(f"{label}는 0보다 커야 합니다")

    def _require_nonnegative(key: str, label: str) -> None:
        value = _num(key)
        if value is not None and value < 0.0:
            issues.append(f"{label}는 0 이상이어야 합니다")

    _require_positive("speed", "speed")
    _require_positive("corner_window_m", "corner_window_m")
    _require_positive("learning_rate", "learning_rate")
    _require_positive("clip_range", "clip_range")
    _require_positive("teacher_window_m", "teacher_window_m")
    _require_positive("bc_learning_rate", "bc_learning_rate")
    _require_nonnegative("settle_time", "settle_time")
    _require_nonnegative("init_box_size", "init_box_size")
    _require_nonnegative("gui_hold_seconds", "gui_hold_seconds")
    _require_nonnegative("total_timesteps", "total_timesteps")
    _require_nonnegative("ent_coef", "ent_coef")
    _require_nonnegative("teacher_gain", "teacher_gain")
    _require_nonnegative("bc_epochs", "bc_epochs")
    _require_nonnegative("bc_nonzero_weight", "bc_nonzero_weight")
    gamma = _num("gamma")
    if gamma is not None and not (0.0 <= gamma <= 1.0):
        issues.append("gamma는 0과 1 사이여야 합니다")
    gae_lambda = _num("gae_lambda")
    if gae_lambda is not None and not (0.0 <= gae_lambda <= 1.0):
        issues.append("gae_lambda는 0과 1 사이여야 합니다")

    try:
        n_steps = int(config.get("n_steps"))
        batch_size = int(config.get("batch_size"))
        if n_steps <= 0:
            issues.append("n_steps는 0보다 커야 합니다")
        if batch_size <= 0:
            issues.append("batch_size는 0보다 커야 합니다")
        if batch_size > n_steps:
            issues.append("batch_size는 n_steps 이하이어야 합니다")
    except (TypeError, ValueError):
        pass
    try:
        n_epochs = int(config.get("n_epochs"))
        if n_epochs <= 0:
            issues.append("n_epochs는 0보다 커야 합니다")
    except (TypeError, ValueError):
        pass
    try:
        bc_epochs = int(config.get("bc_epochs"))
        bc_episodes = int(config.get("bc_episodes"))
        if bc_epochs > 0 and bc_episodes <= 0:
            issues.append("bc_epochs가 0보다 크면 bc_episodes도 0보다 커야 합니다")
    except (TypeError, ValueError):
        pass
    try:
        bc_batch_size = int(config.get("bc_batch_size"))
        if bc_batch_size <= 0:
            issues.append("bc_batch_size는 0보다 커야 합니다")
    except (TypeError, ValueError):
        pass
    try:
        pyb_freq = int(config.get("pyb_freq"))
        ctrl_freq = int(config.get("ctrl_freq"))
        if ctrl_freq <= 0:
            issues.append("ctrl_freq는 0보다 커야 합니다")
        if pyb_freq < ctrl_freq or pyb_freq % ctrl_freq != 0:
            issues.append("pyb_freq는 ctrl_freq 이상의 정수배여야 합니다")
    except (TypeError, ValueError):
        pass
    if bool(config.get("eval_only")) and not str(config.get("load_model") or "").strip():
        issues.append("eval_only는 load_model 경로가 필요합니다")

    reward_ranges = {field["key"]: field for field in REWARD_FIELDS}
    for key, value in reward.items():
        if not isinstance(value, (int, float)) or value != value:
            issues.append(f"{key}은 유한한 숫자여야 합니다")
            continue
        field = reward_ranges.get(key)
        if field and not (float(field["min"]) <= float(value) <= float(field["max"])):
            issues.append(f"{key}은 [{field['min']}, {field['max']}] 범위여야 합니다")
    return issues


def _validate_batch_config(config: dict[str, Any], reward: dict[str, float]) -> list[str]:
    issues: list[str] = []
    if str(config.get("batch_profile")) not in final_batch.PROFILE_DEFAULTS:
        allowed = ", ".join(final_batch.PROFILE_DEFAULTS)
        issues.append(f"batch_profile은 {allowed} 중 하나여야 합니다")
    if str(config.get("batch_trained_action_filter", "none")) not in {"none", "corner_tangent_decel"}:
        issues.append("batch_trained_action_filter 값이 올바르지 않습니다")
    if str(config.get("batch_device", "cuda")) not in {"cuda", "cpu"}:
        issues.append("batch_device는 cuda 또는 cpu여야 합니다")
    try:
        teacher_window_m = float(config.get("batch_teacher_window_m"))
        if teacher_window_m <= 0.0:
            issues.append("batch_teacher_window_m은 0보다 커야 합니다")
    except (TypeError, ValueError):
        issues.append("batch_teacher_window_m은 유효한 숫자여야 합니다")
    if not str(config.get("batch_run_name") or "").strip():
        issues.append("batch_run_name이 필요합니다")
    if bool(config.get("batch_eval_only")) and not str(config.get("batch_model_manifest") or "").strip():
        issues.append("batch_eval_only는 batch_model_manifest 경로가 필요합니다")
    reward_ranges = {field["key"]: field for field in REWARD_FIELDS}
    for key, value in reward.items():
        if not isinstance(value, (int, float)) or value != value:
            issues.append(f"{key}는 유효한 숫자이어야 합니다")
            continue
        field = reward_ranges.get(key)
        if field and not (float(field["min"]) <= float(value) <= float(field["max"])):
            issues.append(f"{key}는 [{field['min']}, {field['max']}] 범위여야 합니다")
    return issues


def _append_arg(command: list[str], flag: str, value: Any) -> None:
    if _is_empty(value):
        return
    command.extend([flag, str(value)])


def _build_command(config: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train.train_phase_b",
        "--output-dir",
        str(output_dir),
    ]

    for key, flag in STRING_ARGS.items():
        value = config.get(key)
        if _is_empty(value):
            continue
        if key == "label" and config.get("trajectory") != "letter":
            continue
        if key == "letter_plane" and config.get("trajectory") != "letter":
            continue
        if key in {"drawn_path", "drawn_plane", "drawn_space"} and config.get("trajectory") != "drawn":
            continue
        command.extend([flag, str(value)])

    for key, (flag, kind) in NUMERIC_ARGS.items():
        value = config.get(key)
        if _is_empty(value):
            continue
        if key == "square_side" and config.get("trajectory") != "square":
            continue
        if key in {"width_m", "height_m"} and config.get("trajectory") == "square":
            continue
        if key == "gui_hold_seconds" and not bool(config.get("gui")):
            continue
        command.extend([flag, str(_coerce_number(value, kind, key))])

    smooth_ref = config.get("smooth_ref")
    if smooth_ref == "true":
        command.append("--smooth-ref")
    elif smooth_ref == "false":
        command.append("--no-smooth-ref")
    if bool(config.get("led_always_on")):
        command.append("--led-always-on")
    if bool(config.get("gui")):
        command.append("--gui")
    if bool(config.get("save_video")):
        command.append("--save-video")
    if bool(config.get("eval_only")):
        command.append("--eval-only")
    if bool(config.get("reset_num_timesteps")):
        command.append("--reset-num-timesteps")
    return command


def _build_batch_command(config: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train.run_final_goal_batch",
        "--profile",
        str(config.get("batch_profile") or "final"),
        "--output-dir",
        str(output_dir),
    ]
    if bool(config.get("batch_dry_run")):
        command.append("--dry-run")
    if not bool(config.get("batch_resume")):
        command.append("--no-resume")
    if bool(config.get("batch_continue_on_error")):
        command.append("--continue-on-error")
    if bool(config.get("batch_eval_only")):
        command.append("--eval-only")
        command.extend(["--model-manifest", str(config.get("batch_model_manifest") or "")])
    command.extend(["--trained-action-filter", str(config.get("batch_trained_action_filter") or "none")])
    command.extend(["--teacher-window-m", str(config.get("batch_teacher_window_m") or 0.15)])
    command.extend(["--device", str(config.get("batch_device") or "cuda")])
    return command


def _quote_command(command: list[str]) -> str:
    parts = []
    for item in command:
        text = str(item)
        if re.search(r"\s|[()&^]", text):
            parts.append('"' + text.replace('"', '\\"') + '"')
        else:
            parts.append(text)
    return " ".join(parts)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json_file(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _tail_text(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def _run_status(run_id: str, output_dir: Path) -> str:
    job = STATE.get(run_id)
    if job is not None:
        poll = job.process.poll()
        if poll is None:
            return "running"
        if job.stopped:
            return "stopped"
        return "complete" if poll == 0 else "failed"
    meta = _read_json_file(output_dir / "dashboard_run.json") or {}
    if (output_dir / "summary.json").exists() or (output_dir / "batch_summary.json").exists():
        return "complete"
    return str(meta.get("status", "unknown"))


def _quality_score(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    rows = summary.get("metrics") or []
    trained = next((row for row in rows if row.get("tag") == "phaseB_trained"), None)
    if not trained:
        return None
    try:
        coverage = float(trained.get("painted_pixel_coverage", 0.0))
        precision = float(trained.get("painted_pixel_precision", coverage))
        iou = float(trained.get("painted_pixel_iou", coverage))
        off_target = float(trained.get("off_target_pixel_ratio", max(0.0, 1.0 - precision)))
        tracking = float(trained.get("tracking_rmse_m", 1.0))
        corner = float(trained.get("corner_path_rmse_m", tracking))
        crash = float(trained.get("crash", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (coverage, precision, iou, off_target, tracking, corner, crash)):
        return None
    score = 120.0 * iou + 40.0 * precision + 20.0 * coverage - 80.0 * off_target
    score -= 120.0 * tracking + 80.0 * corner + 50.0 * crash
    return round(score, 3)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _model_path_from_summary(summary: dict[str, Any] | None) -> str | None:
    if not summary:
        return None
    artifacts = summary.get("artifacts") or {}
    model_path = artifacts.get("model_path")
    return str(model_path) if model_path else None


def _artifact_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext == ".html":
        return "html"
    return "file"


def _is_frame_artifact(rel_path: Path) -> bool:
    return any(part.endswith("_frames") for part in rel_path.parts[:-1])


def _artifact_url(run_id: str, rel_path: Path) -> str:
    rel = rel_path.as_posix()
    return f"/api/artifacts/{quote(run_id)}/{quote(rel, safe='/')}"


def _list_run_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _ARTIFACT_EXTENSIONS:
            continue
        rel_path = path.relative_to(run_dir)
        if _is_frame_artifact(rel_path):
            continue
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "rel_path": rel_path.as_posix(),
                "kind": _artifact_kind(path),
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "url": _artifact_url(run_dir.name, rel_path),
            }
        )
    return items


def _resolve_run_artifact(run_id: str, rel_path: str) -> Path:
    run_dir = (_RUNS_ROOT / run_id).resolve()
    file_path = (run_dir / rel_path).resolve()
    try:
        file_path.relative_to(run_dir)
    except ValueError as exc:
        raise FileNotFoundError("artifact path escapes run directory") from exc
    if not file_path.is_file():
        raise FileNotFoundError("artifact not found")
    return file_path


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    run_id = run_dir.name
    meta = _read_json_file(run_dir / "dashboard_run.json") or {}
    summary = _read_json_file(run_dir / "summary.json")
    batch_summary = _read_json_file(run_dir / "batch_summary.json")
    status = _run_status(run_id, run_dir)
    log_path = run_dir / "run.log"
    log_updated_at = None
    if log_path.exists():
        log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds")
    result = {
        "run_id": run_id,
        "status": status,
        "created_at": meta.get("created_at"),
        "finished_at": meta.get("finished_at"),
        "return_code": meta.get("return_code"),
        "kind": meta.get("kind", "single"),
        "log_updated_at": log_updated_at,
        "title": meta.get("title", run_id),
        "output_dir": str(run_dir),
        "command": meta.get("command", ""),
        "model_path": _model_path_from_summary(summary),
        "quality_score": _quality_score(summary),
        "artifacts": _list_run_artifacts(run_dir),
        "summary": summary,
        "batch_summary": batch_summary,
    }
    return result


def _list_runs() -> list[dict[str, Any]]:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    runs = [_summarize_run(path) for path in _RUNS_ROOT.iterdir() if path.is_dir()]
    runs.sort(key=lambda item: str(item.get("created_at") or item["run_id"]), reverse=True)
    return runs


def _preview(payload: dict[str, Any]) -> dict[str, Any]:
    config, reward = _normalize_payload(payload)
    issues = _validate_config(config, reward)
    run_title = _safe_slug(str(config.get("run_name") or f"{config.get('trajectory')}_{config.get('wind_mode')}"))
    preview_dir = _RUNS_ROOT / f"{_now_token()}_{run_title}"
    command = _build_command(config, preview_dir)
    return {
        "issues": issues,
        "command": _quote_command(command),
        "python": sys.executable,
        "cwd": str(_PKG_ROOT),
        "reward_overrides": reward,
    }


def _preview_batch(payload: dict[str, Any]) -> dict[str, Any]:
    config, reward = _normalize_batch_payload(payload)
    issues = _validate_batch_config(config, reward)
    run_title = _safe_slug(str(config.get("batch_run_name") or "final_goal_batch"))
    preview_dir = _RUNS_ROOT / f"{_now_token()}_{run_title}"
    command = _build_batch_command(config, preview_dir)
    profile = str(config.get("batch_profile") or "final")
    defaults = final_batch.PROFILE_DEFAULTS[profile]
    return {
        "issues": issues,
        "command": _quote_command(command),
        "python": sys.executable,
        "cwd": str(_PKG_ROOT),
        "profile": profile,
        "profile_defaults": defaults,
        "reward_overrides": reward,
    }


def _start_run(payload: dict[str, Any]) -> dict[str, Any]:
    config, reward = _normalize_payload(payload)
    issues = _validate_config(config, reward)
    if issues:
        return {"ok": False, "issues": issues}

    title = _safe_slug(str(config.get("run_name") or f"{config.get('trajectory')}_{config.get('wind_mode')}"))
    run_id = f"{_now_token()}_{title}"
    output_dir = _RUNS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    reward_path = output_dir / "reward_overrides.json"
    log_path = output_dir / "run.log"
    command = _build_command(config, output_dir)
    _write_json(reward_path, reward)

    meta = {
        "run_id": run_id,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "python": sys.executable,
        "cwd": str(_PKG_ROOT),
        "command": _quote_command(command),
        "config": config,
        "reward_config": str(reward_path),
        "reward_overrides": reward,
    }
    _write_json(output_dir / "dashboard_run.json", meta)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LIGHTPAINT_REWARD_CONFIG"] = str(reward_path)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(_PKG_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )
    job = Job(run_id=run_id, process=process, output_dir=output_dir, log_path=log_path)
    STATE.add(job)

    def _pipe_log() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
            meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            if job.stopped:
                meta["status"] = "stopped"
            else:
                meta["status"] = "complete" if return_code == 0 else "failed"
            meta["return_code"] = return_code
            _write_json(output_dir / "dashboard_run.json", meta)
        finally:
            log_file.close()

    threading.Thread(target=_pipe_log, name=f"dashboard-log-{run_id}", daemon=True).start()
    return {
        "ok": True,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "command": meta["command"],
    }


def _start_batch_run(payload: dict[str, Any]) -> dict[str, Any]:
    config, reward = _normalize_batch_payload(payload)
    issues = _validate_batch_config(config, reward)
    if issues:
        return {"ok": False, "issues": issues}

    title = _safe_slug(str(config.get("batch_run_name") or "final_goal_batch"))
    run_id = f"{_now_token()}_{title}"
    output_dir = _RUNS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    reward_path = output_dir / "reward_overrides.json"
    log_path = output_dir / "run.log"
    command = _build_batch_command(config, output_dir)
    _write_json(reward_path, reward)

    meta = {
        "run_id": run_id,
        "title": title,
        "kind": "batch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "python": sys.executable,
        "cwd": str(_PKG_ROOT),
        "command": _quote_command(command),
        "batch_config": config,
        "reward_config": str(reward_path),
        "reward_overrides": reward,
    }
    _write_json(output_dir / "dashboard_run.json", meta)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LIGHTPAINT_REWARD_CONFIG"] = str(reward_path)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(_PKG_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )
    job = Job(run_id=run_id, process=process, output_dir=output_dir, log_path=log_path)
    STATE.add(job)

    def _pipe_log() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
            meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            if job.stopped:
                meta["status"] = "stopped"
            else:
                meta["status"] = "complete" if return_code == 0 else "failed"
            meta["return_code"] = return_code
            _write_json(output_dir / "dashboard_run.json", meta)
        finally:
            log_file.close()

    threading.Thread(target=_pipe_log, name=f"dashboard-batch-log-{run_id}", daemon=True).start()
    return {
        "ok": True,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "command": meta["command"],
    }


def _stop_run(run_id: str) -> dict[str, Any]:
    job = STATE.get(run_id)
    if job is None:
        return {"ok": False, "message": "run is not active in this dashboard process"}
    if job.process.poll() is not None:
        return {"ok": True, "message": "run already finished"}
    job.stopped = True
    job.process.terminate()
    return {"ok": True, "message": "termination requested"}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LightPaintDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(_json_safe(data), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
        self.end_headers()
        with file_path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self._send_html()
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path == "/api/options":
            self._send_json(
                {
                    "defaults": DEFAULT_CONFIG,
                    "batch_defaults": DEFAULT_BATCH_CONFIG,
                    "batch_profiles": final_batch.PROFILE_DEFAULTS,
                    "reward_defaults": _reward_defaults(),
                    "reward_fields": REWARD_FIELDS,
                    "experiment_stages": EXPERIMENT_STAGES,
                    "success_criteria": SUCCESS_CRITERIA,
                    "python": sys.executable,
                    "cwd": str(_PKG_ROOT),
                }
            )
            return
        if path == "/api/runs":
            self._send_json({"runs": _list_runs()})
            return
        if path.startswith("/api/artifacts/"):
            rest = path[len("/api/artifacts/") :]
            if "/" not in rest:
                self.send_error(404)
                return
            run_id_raw, rel_raw = rest.split("/", 1)
            run_id = _safe_slug(unquote(run_id_raw))
            try:
                file_path = _resolve_run_artifact(run_id, unquote(rel_raw))
            except FileNotFoundError:
                self.send_error(404)
                return
            self._send_file(file_path)
            return
        if path.startswith("/api/runs/"):
            run_id = _safe_slug(unquote(path.rsplit("/", 1)[-1]))
            run_dir = _RUNS_ROOT / run_id
            if not run_dir.exists():
                self._send_json({"error": "run not found"}, status=404)
                return
            data = _summarize_run(run_dir)
            data["log_tail"] = _tail_text(run_dir / "run.log")
            self._send_json(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/preview":
                self._send_json(_preview(self._read_body()))
                return
            if path == "/api/batch-preview":
                self._send_json(_preview_batch(self._read_body()))
                return
            if path == "/api/runs":
                result = _start_run(self._read_body())
                self._send_json(result, status=200 if result.get("ok") else 400)
                return
            if path == "/api/batches":
                result = _start_batch_run(self._read_body())
                self._send_json(result, status=200 if result.get("ok") else 400)
                return
            if path.startswith("/api/runs/") and path.endswith("/stop"):
                run_id = _safe_slug(unquote(path.split("/")[-2]))
                self._send_json(_stop_run(run_id))
                return
        except Exception as exc:
            self._send_json({"ok": False, "issues": [str(exc)]}, status=500)
            return
        self.send_error(404)


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LightPaint RL Dashboard</title>
  <style>
    :root {
      --paper: #f4f1ea;
      --ink: #1f2421;
      --muted: #687069;
      --line: #c9c2b2;
      --panel: #fffaf0;
      --accent: #1f7a5b;
      --accent-dark: #12513d;
      --warn: #c2612c;
      --bad: #a43b3b;
      --good: #226b48;
      --code: #19211d;
      --code-ink: #dbe8d1;
      --pulse: #29a36a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Aptos, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select, textarea { font: inherit; }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #e9e2d2;
      padding: 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    main { padding: 18px 22px 28px; }
    h1 {
      margin: 0 0 4px;
      font-size: 22px;
      line-height: 1.1;
      font-weight: 800;
    }
    h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 800;
    }
    h3 {
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
    }
    .sub {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .tabs {
      display: grid;
      gap: 7px;
      margin-top: 16px;
    }
    .tab {
      width: 100%;
      border: 1px solid var(--line);
      background: transparent;
      color: var(--ink);
      padding: 10px 12px;
      border-radius: 6px;
      text-align: left;
      cursor: pointer;
    }
    .tab.active {
      background: var(--ink);
      color: #fff7e5;
      border-color: var(--ink);
    }
    .env {
      margin-top: 18px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
      font-size: 12px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .status-strip {
      display: grid;
      grid-template-columns: minmax(180px, 1.2fr) repeat(4, minmax(120px, .8fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .status-card {
      background: #fffdf7;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }
    .status-card span {
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 5px;
    }
    .status-card strong {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 14px;
      min-height: 20px;
      overflow-wrap: anywhere;
    }
    .status-dot {
      width: 10px;
      height: 10px;
      flex: 0 0 10px;
      border-radius: 50%;
      background: var(--muted);
    }
    .status-dot.running {
      background: var(--pulse);
      box-shadow: 0 0 0 0 rgba(41, 163, 106, .55);
      animation: pulse 1.2s infinite;
    }
    .status-dot.complete { background: var(--good); }
    .status-dot.failed { background: var(--bad); }
    .status-dot.stopped { background: var(--warn); }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(41, 163, 106, .55); }
      70% { box-shadow: 0 0 0 8px rgba(41, 163, 106, 0); }
      100% { box-shadow: 0 0 0 0 rgba(41, 163, 106, 0); }
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .toolbar-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .view { display: none; }
    .view.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
      position: relative;
    }
    .has-help::after {
      content: "?";
      display: inline-grid;
      place-items: center;
      width: 16px;
      height: 16px;
      border: 1px solid var(--line);
      border-radius: 50%;
      color: var(--accent-dark);
      background: #f7f1df;
      font-size: 11px;
      font-weight: 800;
      justify-self: start;
      cursor: help;
    }
    .has-help:hover::before,
    .has-help:focus-within::before {
      content: attr(data-help);
      position: absolute;
      left: 0;
      bottom: calc(100% + 8px);
      z-index: 20;
      width: min(320px, 72vw);
      background: #111814;
      color: #f9f4e7;
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 12px 30px rgba(0,0,0,.22);
      line-height: 1.45;
      font-size: 12px;
      text-transform: none;
      letter-spacing: 0;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fffdf7;
      color: var(--ink);
      padding: 8px 9px;
      min-height: 36px;
    }
    input[type="checkbox"] {
      width: 16px;
      min-height: 16px;
      accent-color: var(--accent);
    }
    .check {
      grid-template-columns: 18px 1fr;
      align-items: center;
      color: var(--ink);
      min-height: 36px;
    }
    .field-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .wide-field { grid-column: 1 / -1; }
    .btn {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: #fff7e5;
      border-radius: 6px;
      min-height: 36px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .btn.secondary {
      background: transparent;
      color: var(--ink);
      border-color: var(--line);
    }
    .btn.warn {
      background: var(--warn);
      border-color: var(--warn);
    }
    .btn:disabled {
      opacity: .55;
      cursor: default;
    }
    .btn.busy {
      position: relative;
      pointer-events: none;
    }
    .btn.busy::after {
      content: "";
      width: 12px;
      height: 12px;
      margin-left: 8px;
      border: 2px solid currentColor;
      border-right-color: transparent;
      border-radius: 50%;
      display: inline-block;
      vertical-align: -2px;
      animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .pill-row { display: flex; flex-wrap: wrap; gap: 7px; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      background: #fffdf7;
    }
    .pill.good { border-color: #8ab59d; color: var(--good); }
    .pill.bad { border-color: #c89486; color: var(--bad); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fffdf7;
      min-height: 78px;
    }
    .metric .name {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .metric .value {
      margin-top: 8px;
      font-weight: 800;
      font-size: 22px;
      overflow-wrap: anywhere;
    }
    .artifact-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .artifact-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffdf7;
      padding: 10px;
      min-width: 0;
    }
    .artifact-card img,
    .artifact-card video,
    .artifact-card iframe {
      width: 100%;
      aspect-ratio: 16 / 10;
      object-fit: contain;
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111814;
    }
    .artifact-card iframe {
      background: #fffdf7;
    }
    .artifact-title {
      margin-top: 8px;
      font-size: 12px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .artifact-meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .artifact-links {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .artifact-link {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      background: #fffdf7;
      color: var(--ink);
      text-decoration: none;
      font-size: 12px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .stage-grid, .criteria-grid {
      display: grid;
      gap: 10px;
    }
    .stage {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fffdf7;
      display: grid;
      gap: 8px;
    }
    .stage-title {
      font-weight: 800;
      font-size: 14px;
    }
    .stage-body {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .stage-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .criteria {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fffdf7;
    }
    .criteria .formula {
      margin-top: 6px;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 11px;
      color: var(--accent-dark);
      overflow-wrap: anywhere;
    }
    .compare-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .compare-table th, .compare-table td {
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      text-align: left;
    }
    .compare-table th {
      color: var(--muted);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .04em;
      font-size: 10px;
    }
    .run-list {
      display: grid;
      gap: 8px;
      max-height: 420px;
      overflow: auto;
    }
    .run-item {
      border: 1px solid var(--line);
      background: #fffdf7;
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
    }
    .run-item.active { border-color: var(--accent); box-shadow: inset 4px 0 0 var(--accent); }
    .run-title { font-weight: 800; font-size: 13px; }
    .run-meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .run-meta .status-label { font-weight: 800; }
    pre {
      margin: 0;
      background: var(--code);
      color: var(--code-ink);
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      min-height: 110px;
      max-height: 360px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .reward-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .reward-field {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 8px;
      align-items: end;
    }
    .range-row {
      display: grid;
      gap: 5px;
    }
    .range-row input[type="range"] {
      padding: 0;
      min-height: 22px;
      accent-color: var(--accent);
    }
    .notice {
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      background: #edf3e7;
      border-radius: 6px;
      color: #263328;
      font-size: 13px;
      line-height: 1.45;
    }
    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 50;
      display: grid;
      gap: 8px;
      width: min(420px, calc(100vw - 36px));
      pointer-events: none;
    }
    .toast-message {
      background: #111814;
      color: #fff7e5;
      border-radius: 8px;
      padding: 11px 13px;
      box-shadow: 0 14px 32px rgba(0,0,0,.24);
      font-size: 13px;
      line-height: 1.45;
      opacity: 0;
      transform: translateY(8px);
      animation: toastIn .18s ease-out forwards;
    }
    .toast-message.good { border-left: 4px solid var(--good); }
    .toast-message.warn { border-left: 4px solid var(--warn); }
    .toast-message.bad { border-left: 4px solid var(--bad); }
    @keyframes toastIn {
      to { opacity: 1; transform: translateY(0); }
    }
    .issues {
      color: var(--bad);
      font-size: 13px;
      display: grid;
      gap: 4px;
    }
    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .status-strip { grid-template-columns: 1fr 1fr; }
      .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 { grid-column: span 12; }
      .field-grid, .metric-grid, .reward-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 680px) {
      main { padding: 14px; }
      .status-strip { grid-template-columns: 1fr; }
      .field-grid, .metric-grid, .reward-grid { grid-template-columns: 1fr; }
      .reward-field { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>LightPaint RL</h1>
      <p class="sub">Phase B 실험 대시보드</p>
      <div class="tabs">
        <button class="tab active" data-tab="setup">실험 설정</button>
        <button class="tab" data-tab="pipeline">실험 로드맵</button>
        <button class="tab" data-tab="batch">전체 Batch</button>
        <button class="tab" data-tab="reward">Reward 튜닝</button>
        <button class="tab" data-tab="run">실행</button>
        <button class="tab" data-tab="results">결과</button>
      </div>
      <div class="env">
        <strong>Python</strong>
        <div id="pythonPath"></div>
        <strong>CWD</strong>
        <div id="cwdPath"></div>
      </div>
    </aside>
    <main>
      <div class="status-strip" aria-live="polite">
        <div class="status-card">
          <span>선택된 Run</span>
          <strong id="globalRunTitle">선택 없음</strong>
        </div>
        <div class="status-card">
          <span>실행 상태</span>
          <strong><i id="globalStatusDot" class="status-dot"></i><b id="globalStatusText">대기</b></strong>
        </div>
        <div class="status-card">
          <span>최근 로그 갱신</span>
          <strong id="globalUpdated">-</strong>
        </div>
        <div class="status-card">
          <span>실행 중 Run</span>
          <strong id="activeRunCount">0</strong>
        </div>
        <div class="status-card">
          <span>설정 저장</span>
          <strong id="draftStatus">자동 임시저장 대기</strong>
        </div>
      </div>
      <section id="pipeline" class="view">
        <div class="toolbar">
          <h2>실험 로드맵</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="applyStageSmoke">0단계 smoke 적용</button>
            <button class="btn secondary" id="applyStageM0">1단계 M0 sanity 적용</button>
            <button class="btn secondary" id="applyStageM0Full">2단계 M0 full 적용</button>
            <button class="btn secondary" id="applyStageM1">3단계 M1 적용</button>
            <button class="btn secondary" id="applyStageM2">4단계 M2 적용</button>
            <button class="btn secondary" id="applyStageFinal">5단계 최종 평가 적용</button>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-7">
            <h3>처음부터 최종 목표까지</h3>
            <div id="stageList" class="stage-grid"></div>
          </div>
          <div class="panel span-5">
            <h3>운영 원칙</h3>
            <div class="notice">
              각 단계는 이전 단계의 좋은 <code>model_path</code>를 다음 단계의 <code>load_model</code>로 넘겨서 이어갑니다.
              <code>complete</code>는 실행 성공이고, <code>overall_pass</code>는 학습 목표 통과입니다. Smoke는 실행 확인용이라
              <code>overall_pass=false</code>여도 실패로 보지 않습니다.
            </div>
            <div style="margin-top:12px" class="notice">
              권장 흐름: smoke 확인 -> DG/user drawn batch 계획 확인 -> dual-path final 평가.
            </div>
          </div>
          <div class="panel span-12">
            <h3>통과 기준</h3>
            <div id="criteriaList" class="criteria-grid"></div>
          </div>
        </div>
      </section>
      <section id="setup" class="view active">
        <div class="toolbar">
          <h2>실험 설정</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="presetSmoke" title="빠른 설정값만 적용합니다. 실제 실행은 Run 시작 버튼을 눌러야 시작됩니다.">Smoke preset</button>
            <button class="btn secondary" id="presetShort" title="짧은 PPO 학습 설정값만 적용합니다. 실제 실행은 Run 시작 버튼을 눌러야 시작됩니다.">Short PPO preset</button>
            <button class="btn secondary" id="presetRobust" title="M2 외란 강건성 실험 설정값만 적용합니다. 실제 실행은 Run 시작 버튼을 눌러야 시작됩니다.">M2 강건성 preset</button>
            <button class="btn secondary" id="saveDraft">현재 설정 저장</button>
            <button class="btn secondary" id="restoreDraft">저장 설정 복원</button>
            <button class="btn secondary" id="restoreBackup">이전 설정 복원</button>
            <button class="btn secondary" id="resetBaseline">기준값 복원</button>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-7">
            <h3>Reference</h3>
            <div class="field-grid">
              <label>Run name<input id="run_name" placeholder="square_m0_trial01"></label>
              <label>Trajectory<select id="trajectory"><option>square</option><option>letter</option><option>drawn</option></select></label>
              <label>Wind mode<select id="wind_mode"><option>M0</option><option>M1</option><option>M2</option></select></label>
              <label>Letter label<input id="label"></label>
              <label>Square side<input id="square_side" type="number" step="0.05"></label>
              <label>Letter plane<select id="letter_plane"><option>xz</option><option>xy</option></select></label>
              <label>Drawn JSON<input id="drawn_path"></label>
              <label>Path scale<input id="path_scale" type="number" step="0.05" placeholder="기본값"></label>
              <label>Speed m/s<input id="speed" type="number" step="0.01"></label>
              <label>Width m<input id="width_m" type="number" step="0.05" placeholder="선택"></label>
              <label>Height m<input id="height_m" type="number" step="0.05" placeholder="선택"></label>
              <label>Max waypoints<input id="max_waypoints" type="number" step="1" placeholder="선택"></label>
            </div>
          </div>
          <div class="panel span-5">
            <h3>Runtime</h3>
            <div class="notice" style="margin-bottom:10px;">추천 기본값: 학습/평가는 <code>Max steps</code>를 비워 자동 계산, <code>Speed</code> 0.30~0.35m/s, <code>Settle time</code> 2.0s, <code>Control Hz</code> 30Hz.</div>
            <div class="field-grid">
              <label>Seed<input id="seed" type="number" step="1"></label>
              <label>Total timesteps<input id="total_timesteps" type="number" step="1"></label>
              <label>Max steps<input id="max_steps" type="number" step="1" placeholder="자동"></label>
              <label>Settle time s<input id="settle_time" type="number" step="0.1"></label>
              <label>Control Hz<input id="ctrl_freq" type="number" step="1"></label>
              <label>PyBullet Hz<input id="pyb_freq" type="number" step="1"></label>
              <label>Device<select id="device"><option>cuda</option><option>cpu</option></select></label>
              <label class="check"><input id="gui" type="checkbox">PyBullet GUI 보기</label>
              <label>GUI hold s<input id="gui_hold_seconds" type="number" step="0.5"></label>
              <label class="check"><input id="save_video" type="checkbox">Save video 저장</label>
              <label class="check"><input id="led_always_on" type="checkbox">LED always on</label>
              <label class="check"><input id="eval_only" type="checkbox">Eval only</label>
              <label class="check"><input id="reset_num_timesteps" type="checkbox">Reset timesteps</label>
              <label class="wide-field">Load model<input id="load_model" placeholder="이전 run의 model_path 또는 .zip 경로"></label>
            </div>
          </div>
          <div class="panel span-12">
            <h3>PPO / BC</h3>
            <div class="field-grid">
              <label>n steps<input id="n_steps" type="number" step="1"></label>
              <label>Batch size<input id="batch_size" type="number" step="1"></label>
              <label>n epochs<input id="n_epochs" type="number" step="1"></label>
              <label>Learning rate<input id="learning_rate" type="number" step="0.0001"></label>
              <label>Gamma<input id="gamma" type="number" step="0.01"></label>
              <label>GAE lambda<input id="gae_lambda" type="number" step="0.01"></label>
              <label>Entropy coef<input id="ent_coef" type="number" step="0.001"></label>
              <label>Clip range<input id="clip_range" type="number" step="0.01"></label>
              <label>Log std init<input id="log_std_init" type="number" step="0.1"></label>
              <label>Teacher gain<input id="teacher_gain" type="number" step="0.01"></label>
              <label>Teacher window m<input id="teacher_window_m" type="number" step="0.01"></label>
              <label>Corner window m<input id="corner_window_m" type="number" step="0.01"></label>
              <label>BC epochs<input id="bc_epochs" type="number" step="1"></label>
              <label>BC episodes<input id="bc_episodes" type="number" step="1"></label>
              <label>BC batch<input id="bc_batch_size" type="number" step="1"></label>
              <label>BC LR<input id="bc_learning_rate" type="number" step="0.0001"></label>
              <label>BC nonzero weight<input id="bc_nonzero_weight" type="number" step="0.1"></label>
              <label>Action filter<select id="trained_action_filter"><option>none</option><option>corner_tangent_decel</option></select></label>
            </div>
          </div>
        </div>
      </section>

      <section id="batch" class="view">
        <div class="toolbar">
          <h2>전체 Batch 실행</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="applyBatchLetters">DG batch 적용</button>
            <button class="btn secondary" id="applyBatchStandard">DG+drawn batch 적용</button>
            <button class="btn secondary" id="applyBatchFinalPlan">dual-path final 계획 적용</button>
            <button class="btn secondary" id="batchPreviewBtn">Batch 명령 갱신</button>
            <button class="btn" id="batchStartBtn">Batch 시작</button>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-5">
            <h3>Batch profile</h3>
            <div class="notice" style="margin-bottom:10px;">
              최종 목표 검증은 단일 run이 아니라 retained path, 외란, seed를 모두 포함한 batch로 판단합니다.
              <code>letters</code>는 DG만 실행하고, <code>standard</code>/<code>final</code>은 DG와 user drawn path만 실행합니다.
              기본값은 안전하게 dry-run 계획 확인으로 시작합니다.
            </div>
            <div class="field-grid">
              <label class="wide-field">Batch run name<input id="batch_run_name"></label>
              <label>Profile<select id="batch_profile"><option>sanity</option><option>letters</option><option>standard</option><option>final</option></select></label>
              <label class="check"><input id="batch_dry_run" type="checkbox">Dry-run 계획만 생성</label>
              <label class="check"><input id="batch_resume" type="checkbox">완료 run 재사용</label>
              <label class="check"><input id="batch_continue_on_error" type="checkbox">실패해도 다음 run 진행</label>
              <label class="check"><input id="batch_eval_only" type="checkbox">Eval-only 최종 평가</label>
              <label class="wide-field">Model manifest<input id="batch_model_manifest" placeholder="artifacts\\...\\best_model_manifest.json"></label>
              <label>평가 Action filter<select id="batch_trained_action_filter"><option>corner_tangent_decel</option><option>none</option></select></label>
              <label>Teacher window m<input id="batch_teacher_window_m" type="number" step="0.01"></label>
            </div>
            <div id="batchProfileInfo" class="notice" style="margin-top:10px;"></div>
          </div>
          <div class="panel span-7">
            <h3>Batch command</h3>
            <pre id="batchCommandPreview"></pre>
            <h3 style="margin-top:12px;">검증</h3>
            <div id="batchIssues" class="issues"></div>
          </div>
          <div class="panel span-12">
            <h3>사용 기준</h3>
            <div class="notice">
              <code>sanity</code>는 실행 경로 확인용입니다. 글자별 장시간 sweep은 <code>letters</code>,
              팀 공유용 전체 성능 비교는 <code>standard</code>, 최종 목표 달성 주장은 <code>final</code>
              profile의 <code>batch_summary.json</code>과
              <code>batch_final_goal_criteria.batch_final_goal_pass</code>로 판단합니다.
            </div>
          </div>
        </div>
      </section>

      <section id="reward" class="view">
        <div class="toolbar">
          <h2>Reward 튜닝</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="resetReward">Reward 초기화</button>
            <button class="btn secondary" id="exportPreset">Preset 내보내기</button>
            <label class="btn secondary" style="cursor:pointer;">Preset 가져오기<input id="importPreset" type="file" accept="application/json" style="display:none"></label>
          </div>
        </div>
        <div class="notice">Reward 변경은 run별 <code>reward_overrides.json</code>에 저장됩니다. 기준 소스 파일은 실험 중 직접 바뀌지 않습니다.</div>
        <div id="rewardGroups" class="grid" style="margin-top:12px;"></div>
      </section>

      <section id="run" class="view">
        <div class="toolbar">
          <h2>실행 제어</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="previewBtn">명령 갱신</button>
            <button class="btn" id="startBtn">Run 시작</button>
            <button class="btn secondary" id="copyCommand">명령 복사</button>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-8">
            <h3>Command</h3>
            <pre id="commandPreview"></pre>
          </div>
          <div class="panel span-4">
            <h3>검증</h3>
            <div id="issues" class="issues"></div>
          </div>
          <div class="panel span-12">
            <h3>Live Log</h3>
            <div class="notice" id="liveStatusNotice" style="margin-bottom:10px;">선택된 run이 없습니다. Run을 시작하거나 Results에서 run을 선택하면 상태와 로그가 갱신됩니다.</div>
            <pre id="liveLog"></pre>
          </div>
        </div>
      </section>

      <section id="results" class="view">
        <div class="toolbar">
          <h2>결과</h2>
          <div class="toolbar-actions">
            <button class="btn secondary" id="refreshRuns">새로고침</button>
            <button class="btn secondary" id="useSelectedModel">선택 모델로 이어서 학습</button>
            <button class="btn secondary" id="evalSelectedModel">선택 모델 평가만</button>
            <button class="btn warn" id="stopRun">선택 Run 중지</button>
          </div>
        </div>
        <div class="grid">
          <div class="panel span-4">
            <h3>Runs</h3>
            <div id="runList" class="run-list"></div>
          </div>
          <div class="panel span-8">
            <h3>Quality Gate</h3>
            <div id="gate" class="pill-row"></div>
            <div id="metrics" class="metric-grid" style="margin-top:12px;"></div>
            <h3 style="margin-top:16px;">통과 기준 계산</h3>
            <div id="criteriaEval" style="margin-top:8px;"></div>
          </div>
          <div class="panel span-12">
            <h3>시각화 결과</h3>
            <div id="artifactGallery" class="artifact-gallery"></div>
            <div id="artifactLinks" class="artifact-links"></div>
          </div>
          <div class="panel span-12">
            <h3>선택 Run Log</h3>
            <pre id="selectedLog"></pre>
          </div>
        </div>
      </section>
    </main>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const DRAFT_KEY = "lightpaint_dashboard_draft_v2";
    const BACKUP_KEY = "lightpaint_dashboard_backup_v2";
    const state = {
      defaults: {},
      batchDefaults: {},
      batchProfiles: {},
      rewardDefaults: {},
      rewardFields: [],
      experimentStages: [],
      successCriteria: [],
      selectedRun: null,
      selectedRunData: null,
      pollTimer: null,
      autosaveTimer: null,
      heartbeatTimer: null,
    };
    const GROUP_LABELS = {
      action: "Action / LED 계약",
      tracking: "Tracking",
      regularization: "Regularization",
      painting: "Painting",
      led: "LED",
      corner: "Corner",
      terminal: "Terminal",
    };
    const STATUS_LABELS = {
      running: "실행 중",
      complete: "완료",
      failed: "실패",
      stopped: "중지됨",
      unknown: "상태 미확인",
      idle: "대기",
    };
    const CONFIG_HELP = {
      run_name: "대시보드와 artifacts 폴더에 기록될 실험 이름입니다. 비워두면 trajectory와 wind mode를 기준으로 자동 생성됩니다.",
      trajectory: "reference 경로 종류입니다. square는 생성 사각형, letter는 글자 이미지 기반 경로, drawn은 tools/draw_path.html에서 만든 JSON 경로입니다.",
      wind_mode: "외란 조건입니다. M0는 외란 없음, M1/M2는 PyBullet applyExternalForce 기반 외란을 적용하는 강건성 실험입니다.",
      label: "trajectory가 letter일 때 렌더링할 글자입니다. 현재 기본 목표는 DG입니다.",
      square_side: "trajectory가 square일 때 한 변의 길이입니다. 추천값: 기본 0.8m. path_scale이 있으면 최종 길이에 함께 반영됩니다.",
      letter_plane: "letter reference를 어느 평면에 배치할지 정합니다. xz는 앞에서 보는 글자 평면, xy는 수평 평면입니다.",
      drawn_path: "trajectory가 drawn일 때 사용할 JSON 파일 경로입니다. 상대 경로는 lightpaint-team-package 기준입니다.",
      path_scale: "전체 reference 경로 크기 배율입니다. 추천값: 비움 또는 1.0부터 시작. 비워두면 각 reference 생성기의 기본값 또는 drawn JSON 내부 값을 사용합니다.",
      speed: "reference가 진행되는 기준 속도입니다. 추천값: 0.30~0.35 m/s, 기본 0.35. Phase B는 이 속도 벡터에 residual velocity를 더해 PID target velocity를 만듭니다.",
      width_m: "letter/drawn reference의 목표 너비입니다. 추천값: 비워두고 시작, 필요하면 0.7~1.0m 범위에서 조정. square에는 적용되지 않습니다.",
      height_m: "letter/drawn reference의 목표 높이입니다. 추천값: 비워두고 시작, 글자가 너무 작거나 크면 0.5~0.9m 범위에서 조정. square에는 적용되지 않습니다.",
      max_waypoints: "letter/drawn 경로의 waypoint 수 상한입니다. 추천값: 보통 비움. DG가 너무 오래 걸리면 120~250 정도로 제한합니다.",
      seed: "환경 reset, 정책 초기화, BC sample 순서 등에 쓰이는 난수 seed입니다. 추천값: 비교 실험은 7로 고정, 후보가 좋아지면 3~5개 seed로 재확인하세요.",
      total_timesteps: "PPO 학습 step 수입니다. 추천값: smoke 0, 빠른 sanity 2048~8192, 단일 full 후보 10만부터 시작, M2/최종 후보는 20만~30만 이상을 사용합니다. DG/user drawn 평가는 run_final_goal_batch profile letters/standard/final을 사용하세요.",
      max_steps: "episode당 최대 step 수입니다. 추천값: 학습/평가는 비워서 자동 계산. smoke는 2~90, square 전체 평가는 대략 350~400입니다.",
      settle_time: "reference가 끝난 뒤 마지막 목표 주변에서 더 실행할 시간입니다. 추천값: 2.0s, 최종 영상/평가는 2~3s. max_steps를 비우면 자동 horizon 계산에 반영됩니다.",
      ctrl_freq: "환경 step/control frequency입니다. 추천값: 30Hz. PyBullet physics frequency보다 낮거나 같아야 합니다.",
      pyb_freq: "PyBullet physics frequency입니다. 추천값: 240Hz. ctrl_freq의 정수배여야 합니다.",
      device: "Stable-Baselines3 정책 학습 device입니다. CUDA를 사용할 수 있으면 cuda를 선택하세요.",
      gui: "PyBullet GUI 창을 rollout 확인용으로 띄울지 정합니다. 추천값: 기본 off. PyBullet은 프로세스당 GUI 1개만 허용하므로 PPO/BC 학습 env는 DIRECT로 실행되고, 평가 rollout 창이 순차적으로 열립니다.",
      gui_hold_seconds: "GUI를 켰을 때 각 rollout 창을 닫기 전에 유지하는 시간입니다. 추천값: 5초. 너무 짧으면 창이 깜빡이고 바로 닫혀 보이지 않을 수 있습니다.",
      save_video: "켜면 trained rollout 시각화 영상까지 저장합니다. smoke에서는 시간이 늘어날 수 있습니다.",
      load_model: "이전 run의 PPO model zip 경로입니다. M0에서 학습한 weight를 M1/M2 실험으로 이어갈 때 이 값을 넣습니다. Results 탭의 '선택 모델로 이어서 학습' 버튼이 자동으로 채웁니다.",
      led_always_on: "scripted LED reference를 항상 ON으로 강제합니다. 실제 LED on/off 경로 검증 때는 끄는 것이 기본입니다.",
      eval_only: "기존 load_model을 평가만 하고 BC/PPO 업데이트는 하지 않습니다. load_model 경로가 필요합니다.",
      reset_num_timesteps: "load_model로 이어서 학습할 때 SB3 timestep 카운터를 0부터 다시 셀지 정합니다. curriculum처럼 앞선 weight를 계속 키우는 실험은 보통 끕니다.",
      n_steps: "PPO rollout buffer 길이입니다. 추천값: smoke 8, 빠른 실험 64~128, 안정적 학습 256~512. batch_size보다 크거나 같아야 합니다.",
      batch_size: "PPO minibatch 크기입니다. 추천값: n_steps의 1/2 이하. 예: n_steps 64면 batch_size 32.",
      n_epochs: "수집된 rollout buffer를 PPO가 몇 번 반복 업데이트할지 정합니다. 추천값: 2~5, 처음은 2.",
      learning_rate: "PPO optimizer learning rate입니다. 추천값: 3e-4. 정책이 흔들리면 1e-4로 낮춰보세요.",
      gamma: "할인율입니다. 추천값: 0.99. 1에 가까울수록 미래 reward를 더 강하게 봅니다.",
      gae_lambda: "GAE advantage smoothing 계수입니다. 추천값: 0.95. PPO advantage 추정의 bias/variance 균형에 영향을 줍니다.",
      ent_coef: "entropy bonus 계수입니다. 추천값: 0.0부터 시작, 탐색이 너무 부족하면 0.001~0.01 범위에서만 올립니다.",
      clip_range: "PPO policy update clipping 범위입니다. 추천값: 0.2. 작을수록 업데이트가 보수적입니다.",
      log_std_init: "초기 Gaussian policy 표준편차의 log 값입니다. 추천값: -2.0. 낮을수록 초기 action 탐색 폭이 작습니다.",
      teacher_gain: "BC teacher가 corner 접근 시 얼마나 강하게 감속 residual을 줄지 정합니다. 추천값: 0.10, 후보 탐색은 0.05~0.20.",
      teacher_window_m: "teacher가 corner 앞 몇 m부터 감속 action을 줄지 정합니다. 추천값: 0.15m, 큰 글자/빠른 속도는 0.20m까지 늘릴 수 있습니다.",
      corner_window_m: "metric/teacher/reward에서 corner 영향권으로 보는 거리입니다. 추천값: 0.18m, 보통 0.15~0.25m. 너무 크면 직선 구간이 corner로 섞입니다.",
      bc_epochs: "PPO 전에 teacher action으로 behavior cloning을 수행할 epoch 수입니다. 추천값: smoke 0, M0/M1 시작점은 2, teacher가 유효하면 3~5.",
      bc_episodes: "BC dataset 수집 episode 수입니다. 추천값: 빠른 실험 4, 안정적 warm start 8~16.",
      bc_batch_size: "BC optimizer minibatch 크기입니다. 추천값: 64. dataset이 작으면 32도 가능합니다.",
      bc_learning_rate: "BC optimizer learning rate입니다. 추천값: 1e-3. loss가 흔들리면 3e-4로 낮춥니다.",
      bc_nonzero_weight: "teacher action이 0이 아닌 corner 감속 sample에 주는 추가 가중치입니다. 추천값: 1.0, corner sample이 묻히면 2~4.",
      trained_action_filter: "평가 때 trained action을 그대로 쓸지, corner tangent 감속 성분만 남길지 정합니다. 추천값: 최종/공유 평가는 corner_tangent_decel, 후처리 없는 확인은 none.",
      batch_profile: "전체 batch 범위입니다. letters는 DG만, standard/final은 DG와 user drawn path만 사용합니다. 다른 경로는 필요할 때 명시적으로 생성해서 실행합니다.",
      batch_dry_run: "체크하면 실제 학습을 시작하지 않고 실행 계획과 명령만 생성합니다. 공유 전에는 먼저 켜고 계획을 확인하세요.",
      batch_resume: "체크하면 이미 summary.json이 있는 run은 재사용합니다. 장시간 batch가 중간에 끊겨도 이어서 돌릴 때 필요합니다.",
      batch_continue_on_error: "체크하면 한 run이 실패해도 다음 trajectory/wind/seed로 넘어갑니다. 최종 batch_summary에서 실패 run을 확인합니다.",
      batch_eval_only: "체크하면 학습 없이 model manifest에 적힌 기존 PPO 모델들을 전체 matrix에서 평가만 합니다. 최종 목표 달성 주장은 이 모드의 final profile 결과로 확인하는 것이 가장 깔끔합니다.",
      batch_model_manifest: "eval-only batch에서 사용할 best_model_manifest.json 경로입니다. 이전 batch 학습 산출물의 best_model_manifest.json을 넣습니다.",
      batch_trained_action_filter: "batch 평가에서 trained rollout action을 후처리할지 정합니다. 추천값은 corner_tangent_decel이며, 후처리 없는 확인에만 none을 사용합니다.",
      batch_teacher_window_m: "batch 실행 시 teacher/action filter가 corner 전후 몇 m 구간을 볼지 정합니다. 현재 검증 기준 추천값은 0.15m입니다.",
    };
    const configKeys = [
      "run_name","trajectory","label","square_side","letter_plane","drawn_path","drawn_plane","drawn_space",
      "width_m","height_m","path_scale","max_waypoints","smooth_window_m","wind_mode","speed","settle_time",
      "max_steps","corner_window_m","init_box_size","gui","gui_hold_seconds","led_always_on","ctrl_freq","pyb_freq","seed",
      "total_timesteps","n_steps","batch_size","n_epochs","learning_rate","gamma","gae_lambda","ent_coef",
      "clip_range","log_std_init","teacher_gain","teacher_window_m","bc_epochs","bc_episodes","bc_batch_size",
      "bc_learning_rate","bc_nonzero_weight","trained_action_filter","device","verbose","save_video",
      "load_model","eval_only","reset_num_timesteps"
    ];
    const batchKeys = ["batch_run_name", "batch_profile", "batch_dry_run", "batch_resume", "batch_continue_on_error", "batch_eval_only", "batch_model_manifest", "batch_trained_action_filter", "batch_teacher_window_m"];

    function $(id) { return document.getElementById(id); }
    async function api(url, options={}) {
      let res;
      try {
        res = await fetch(url, options);
      } catch (err) {
        throw new Error("대시보드 서버에 연결할 수 없습니다. 서버 프로세스가 종료되었거나 다른 Python/포트로 열린 탭일 수 있습니다. tools\\run_dashboard.bat 또는 infra venv Python으로 다시 실행한 뒤 http://127.0.0.1:8765/ 를 새로고침하세요.");
      }
      let data;
      try {
        data = await res.json();
      } catch (err) {
        throw new Error(`서버 응답을 JSON으로 읽지 못했습니다. HTTP ${res.status} ${res.statusText}`);
      }
      if (!res.ok) throw new Error((data.issues || [data.error || res.statusText]).join("\n"));
      return data;
    }
    function setInput(id, value) {
      const el = $(id);
      if (!el) return;
      if (el.type === "checkbox") el.checked = Boolean(value);
      else el.value = value ?? "";
    }
    function getInput(id) {
      const el = $(id);
      if (!el) return "";
      if (el.type === "checkbox") return el.checked;
      return el.value;
    }
    function collectConfig() {
      const config = {};
      for (const key of configKeys) config[key] = getInput(key);
      return config;
    }
    function collectBatch() {
      const batch = {};
      for (const key of batchKeys) batch[key] = getInput(key);
      return batch;
    }
    function collectReward() {
      const reward = {};
      for (const field of state.rewardFields) reward[field.key] = Number($("reward_" + field.key).value);
      return reward;
    }
    function payload() { return {config: collectConfig(), reward: collectReward()}; }
    function batchPayload() { return {batch: collectBatch(), reward: collectReward()}; }
    function setDraftStatus(text) {
      const el = $("draftStatus");
      if (el) el.textContent = text;
    }
    function nowText() {
      return new Date().toLocaleTimeString("ko-KR", {hour12:false});
    }
    function toast(message, tone="good") {
      const host = $("toast");
      if (!host) return;
      const div = document.createElement("div");
      div.className = "toast-message " + tone;
      div.textContent = message;
      host.appendChild(div);
      setTimeout(() => {
        div.style.opacity = "0";
        div.style.transform = "translateY(8px)";
        setTimeout(() => div.remove(), 180);
      }, 3200);
    }
    async function withButtonFeedback(button, workingText, doneText, fn) {
      const original = button.textContent;
      button.disabled = true;
      button.classList.add("busy");
      button.textContent = workingText;
      try {
        const result = await fn();
        if (doneText) toast(doneText, "good");
        return result;
      } catch (err) {
        toast(err.message || String(err), "bad");
        throw err;
      } finally {
        button.classList.remove("busy");
        button.textContent = original;
        button.disabled = false;
      }
    }
    function saveDraft(reason="auto") {
      const data = {...payload(), batch: collectBatch(), savedAt: new Date().toISOString(), reason};
      localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
      setDraftStatus(`자동 임시저장 ${nowText()}`);
      return data;
    }
    function saveBackup(reason="backup") {
      const data = {...payload(), batch: collectBatch(), savedAt: new Date().toISOString(), reason};
      localStorage.setItem(BACKUP_KEY, JSON.stringify(data));
      return data;
    }
    function applyPayload(data) {
      if (!data) return false;
      if (data.config) for (const [key, value] of Object.entries(data.config)) setInput(key, value);
      if (data.batch) for (const [key, value] of Object.entries(data.batch)) setInput(key, value);
      if (data.reward) {
        for (const [key, value] of Object.entries(data.reward)) {
          const el = $("reward_" + key);
          const range = $("reward_range_" + key);
          if (el) el.value = value;
          if (range) range.value = value;
        }
      }
      refreshPreviewDebounced();
      refreshBatchPreviewDebounced();
      return true;
    }
    function restoreStored(key) {
      const raw = localStorage.getItem(key);
      if (!raw) return false;
      try {
        const data = JSON.parse(raw);
        return applyPayload(data);
      } catch {
        return false;
      }
    }
    function scheduleAutosave() {
      clearTimeout(state.autosaveTimer);
      setDraftStatus("수정됨, 자동 저장 대기");
      state.autosaveTimer = setTimeout(() => saveDraft("autosave"), 350);
    }
    function switchTab(id) {
      document.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === id));
      document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === id));
    }
    function installHelp() {
      for (const [id, text] of Object.entries(CONFIG_HELP)) {
        const el = $(id);
        if (!el) continue;
        const label = el.closest("label");
        if (!label) continue;
        label.classList.add("has-help");
        label.dataset.help = text;
        label.title = text;
        el.title = text;
      }
    }
    function statusLabel(status) {
      return STATUS_LABELS[status] || status || STATUS_LABELS.unknown;
    }
    function updateGlobalStatus(data) {
      const dot = $("globalStatusDot");
      dot.className = "status-dot " + (data?.status || "");
      $("globalStatusText").textContent = statusLabel(data?.status || "idle");
      $("globalRunTitle").textContent = data?.title || data?.run_id || "선택 없음";
      $("globalUpdated").textContent = data?.log_updated_at || data?.finished_at || data?.created_at || "-";
      if ($("liveStatusNotice")) {
        if (!data) {
          $("liveStatusNotice").textContent = "선택된 run이 없습니다. Run을 시작하거나 Results에서 run을 선택하면 상태와 로그가 갱신됩니다.";
        } else if (data.status === "running") {
          $("liveStatusNotice").textContent = "현재 선택된 run은 실행 중입니다. 로그는 약 1.5초 간격으로 자동 갱신됩니다.";
        } else {
          $("liveStatusNotice").textContent = `현재 선택된 run은 ${statusLabel(data.status)} 상태입니다. 종료 코드: ${data.return_code ?? "-"}`;
        }
      }
    }
    async function checkServerHeartbeat() {
      try {
        const data = await api("/api/options");
        if (!state.selectedRun) {
          $("globalStatusDot").className = "status-dot complete";
          $("globalStatusText").textContent = "서버 연결됨";
          $("globalUpdated").textContent = nowText();
        }
        $("pythonPath").textContent = data.python;
        $("cwdPath").textContent = data.cwd;
      } catch (err) {
        $("globalStatusDot").className = "status-dot failed";
        $("globalStatusText").textContent = "서버 연결 끊김";
        $("globalUpdated").textContent = nowText();
        if ($("liveStatusNotice")) $("liveStatusNotice").textContent = err.message;
      }
    }
    function renderReward() {
      const groups = {};
      for (const field of state.rewardFields) (groups[field.group] ||= []).push(field);
      const host = $("rewardGroups");
      host.innerHTML = "";
      for (const [group, fields] of Object.entries(groups)) {
        const panel = document.createElement("div");
        panel.className = "panel span-6";
        panel.innerHTML = `<h3>${GROUP_LABELS[group] || group}</h3><div class="reward-grid"></div>`;
        const grid = panel.querySelector(".reward-grid");
        for (const field of fields) {
          const value = state.rewardDefaults[field.key];
          const row = document.createElement("div");
          row.className = "reward-field";
          row.innerHTML = `
            <label class="range-row has-help" data-help="${field.help || ""}" title="${field.help || ""}">${field.label}
              <input id="reward_range_${field.key}" type="range" min="${field.min}" max="${field.max}" step="${field.step}" value="${value}">
            </label>
            <input id="reward_${field.key}" type="number" min="${field.min}" max="${field.max}" step="${field.step}" value="${value}" title="${field.help || ""}">
          `;
          grid.appendChild(row);
          const range = row.querySelector("#reward_range_" + field.key);
          const number = row.querySelector("#reward_" + field.key);
          range.addEventListener("input", () => { number.value = range.value; refreshPreviewDebounced(); });
          number.addEventListener("input", () => { range.value = number.value; refreshPreviewDebounced(); });
        }
        host.appendChild(panel);
      }
    }
    function renderPipeline() {
      const stageHost = $("stageList");
      const criteriaHost = $("criteriaList");
      if (stageHost) {
        stageHost.innerHTML = "";
        for (const stage of state.experimentStages) {
          const div = document.createElement("div");
          div.className = "stage";
          const action = `<div class="stage-actions"><button class="btn secondary" data-stage="${stage.id}">이 단계 설정 적용</button></div>`;
          div.innerHTML = `
            <div class="stage-title">${stage.title}</div>
            <div class="stage-body"><strong>목표</strong><br>${stage.goal}</div>
            <div class="stage-body"><strong>통과 기준</strong><br>${stage.pass}</div>
            <div class="stage-body"><strong>다음 단계</strong><br>${stage.next}</div>
            ${action}
          `;
          stageHost.appendChild(div);
        }
        stageHost.querySelectorAll("[data-stage]").forEach(btn => {
          btn.addEventListener("click", () => applyStage(btn.dataset.stage));
        });
      }
      if (criteriaHost) {
        criteriaHost.innerHTML = "";
        for (const item of state.successCriteria) {
          const div = document.createElement("div");
          div.className = "criteria";
          div.innerHTML = `<strong>${item.label}</strong><div class="formula">${item.formula}</div><p>${item.why}</p>`;
          criteriaHost.appendChild(div);
        }
      }
    }
    function renderBatchProfileInfo() {
      const host = $("batchProfileInfo");
      if (!host) return;
      const profile = getInput("batch_profile") || "final";
      const data = state.batchProfiles[profile] || {};
      const trajectories = data.trajectories || "-";
      const winds = data.wind_modes || "-";
      const seeds = data.seeds || "-";
      const steps = data.total_timesteps ?? "-";
      const filter = getInput("batch_trained_action_filter") || "corner_tangent_decel";
      const teacherWindow = getInput("batch_teacher_window_m") || "0.15";
      const mode = getInput("batch_eval_only") ? "eval-only" : "train/eval";
      host.innerHTML = `<strong>${profile}</strong><br>Mode: ${mode}<br>Trajectories: ${trajectories}<br>Wind modes: ${winds}<br>Seeds: ${seeds}<br>Run당 PPO step: ${steps}<br>평가 filter: ${filter}<br>Teacher window: ${teacherWindow} m`;
    }
    function applyBatchPreset(profile, dryRun, runName) {
      saveBackup(`before_batch_${profile}`);
      setInput("batch_run_name", runName);
      setInput("batch_profile", profile);
      setInput("batch_dry_run", dryRun);
      setInput("batch_resume", true);
      setInput("batch_continue_on_error", true);
      setInput("batch_eval_only", false);
      setInput("batch_model_manifest", "");
      setInput("batch_trained_action_filter", "corner_tangent_decel");
      setInput("batch_teacher_window_m", 0.15);
      renderBatchProfileInfo();
      saveDraft(`batch_${profile}`);
      refreshBatchPreview();
      const mode = dryRun ? "계획 확인" : "실제 학습";
      toast(`${profile} batch ${mode} 설정을 적용했습니다. 실행하려면 Batch 시작을 누르세요.`, "good");
    }
    function applyDefaults() {
      for (const [key, value] of Object.entries(state.defaults)) setInput(key, value);
      for (const [key, value] of Object.entries(state.batchDefaults)) setInput(key, value);
      for (const [key, value] of Object.entries(state.rewardDefaults)) {
        const el = $("reward_" + key);
        const range = $("reward_range_" + key);
        if (el) el.value = value;
        if (range) range.value = value;
      }
      renderBatchProfileInfo();
    }
    function setPresetValues(name) {
      if (name === "smoke") {
        setInput("run_name", "smoke_dg_m0");
        setInput("trajectory", "letter");
        setInput("label", "DG");
        setInput("wind_mode", "M0");
        setInput("max_steps", 3);
        setInput("total_timesteps", 0);
        setInput("n_steps", 8);
        setInput("batch_size", 8);
        setInput("bc_epochs", 0);
        setInput("gui", false);
        setInput("save_video", false);
      }
      if (name === "short") {
        setInput("run_name", "short_dg_m0");
        setInput("trajectory", "letter");
        setInput("label", "DG");
        setInput("wind_mode", "M0");
        setInput("max_steps", "");
        setInput("total_timesteps", 2048);
        setInput("n_steps", 64);
        setInput("batch_size", 32);
        setInput("bc_epochs", 2);
        setInput("bc_episodes", 4);
        setInput("gui", false);
      }
      if (name === "robust") {
        setInput("run_name", "robust_dg_m2");
        setInput("trajectory", "letter");
        setInput("label", "DG");
        setInput("wind_mode", "M2");
        setInput("max_steps", "");
        setInput("total_timesteps", 100000);
        setInput("n_steps", 256);
        setInput("batch_size", 64);
        setInput("bc_epochs", 2);
        setInput("bc_episodes", 8);
        setInput("trained_action_filter", "corner_tangent_decel");
        setInput("gui", false);
      }
    }
    function applyPreset(name) {
      saveBackup("before_preset_" + name);
      setPresetValues(name);
      saveDraft("preset_" + name);
      toast(`${name} preset을 적용했습니다. 아직 실행은 시작되지 않았습니다. 실행하려면 Run 시작을 누르세요.`, "warn");
      refreshPreview();
    }
    function applyStage(stageId) {
      const stage = state.experimentStages.find(item => item.id === stageId);
      if (!stage) return;
      saveBackup("before_stage_" + stageId);
      setPresetValues(stage.preset);
      setInput("wind_mode", stage.wind_mode);
      for (const [key, value] of Object.entries(stage.recommended || {})) {
        if (typeof value === "string" && value.includes("/")) continue;
        setInput(key, value);
      }
      setInput("run_name", `${stage.id}_${stage.wind_mode}`.toLowerCase());
      setInput("eval_only", false);
      if (stage.id === "env_smoke" || stage.id === "m0_corner" || stage.id === "m0_full_train") {
        setInput("load_model", "");
      } else if (state.selectedRunData?.model_path) {
        setInput("load_model", state.selectedRunData.model_path);
        setInput("reset_num_timesteps", false);
      }
      saveDraft("stage_" + stageId);
      refreshPreview();
      switchTab("setup");
      const modelText = $("load_model")?.value ? " 선택된 이전 모델 경로도 load_model에 넣었습니다." : "";
      toast(`${stage.title} 권장 설정을 적용했습니다.${modelText}`, "good");
    }
    function selectedModelPath() {
      return state.selectedRunData?.model_path || state.selectedRunData?.summary?.artifacts?.model_path || "";
    }
    function useSelectedModel(evalOnly=false) {
      const modelPath = selectedModelPath();
      if (!modelPath) {
        toast("선택한 run에서 재사용할 model_path를 찾지 못했습니다. 완료된 학습 run을 선택하세요.", "warn");
        return;
      }
      saveBackup(evalOnly ? "before_eval_selected_model" : "before_continue_selected_model");
      setInput("load_model", modelPath);
      setInput("eval_only", evalOnly);
      setInput("reset_num_timesteps", false);
      const title = state.selectedRunData?.title || state.selectedRunData?.run_id || "selected_model";
      setInput("run_name", `${evalOnly ? "eval" : "continue"}_${title}`);
      saveDraft(evalOnly ? "eval_selected_model" : "continue_selected_model");
      refreshPreview();
      switchTab("setup");
      toast(evalOnly ? "선택 모델을 평가 전용 설정으로 연결했습니다." : "선택 모델을 이어서 학습할 load_model으로 연결했습니다.", "good");
    }
    let previewTimer = null;
    let batchPreviewTimer = null;
    function refreshPreviewDebounced() {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(refreshPreview, 250);
    }
    function refreshBatchPreviewDebounced() {
      clearTimeout(batchPreviewTimer);
      batchPreviewTimer = setTimeout(refreshBatchPreview, 250);
    }
    async function refreshPreview() {
      try {
        const data = await api("/api/preview", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload())});
        if ($("commandPreview")) $("commandPreview").textContent = data.command;
        if ($("issues")) $("issues").innerHTML = data.issues.length ? data.issues.map(x => `<div>${x}</div>`).join("") : `<div style="color:var(--good)">실행을 막는 설정 문제는 없습니다.</div>`;
        if ($("startBtn")) $("startBtn").disabled = data.issues.length > 0;
      } catch (err) {
        if ($("issues")) $("issues").innerHTML = `<div>${err.message}</div>`;
        if ($("startBtn")) $("startBtn").disabled = true;
        toast(err.message || String(err), "bad");
      }
    }
    async function refreshBatchPreview() {
      try {
        renderBatchProfileInfo();
        const data = await api("/api/batch-preview", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(batchPayload())});
        if ($("batchCommandPreview")) $("batchCommandPreview").textContent = data.command;
        if ($("batchIssues")) $("batchIssues").innerHTML = data.issues.length ? data.issues.map(x => `<div>${x}</div>`).join("") : `<div style="color:var(--good)">Batch 실행을 막는 설정 문제가 없습니다.</div>`;
        if ($("batchStartBtn")) $("batchStartBtn").disabled = data.issues.length > 0;
      } catch (err) {
        if ($("batchIssues")) $("batchIssues").innerHTML = `<div>${err.message}</div>`;
        if ($("batchStartBtn")) $("batchStartBtn").disabled = true;
        toast(err.message || String(err), "bad");
      }
    }
    async function startBatchRun() {
      const btn = $("batchStartBtn");
      const original = btn.textContent;
      btn.disabled = true;
      btn.classList.add("busy");
      btn.textContent = "Batch 시작 중";
      try {
        saveDraft("before_batch_start");
        toast("Batch 시작 요청을 보냈습니다. Results 탭에서 batch_summary.json과 log를 확인하세요.", "warn");
        const data = await api("/api/batches", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(batchPayload())});
        state.selectedRun = data.run_id;
        toast(`Batch가 시작되었습니다. ${data.run_id}`, "good");
        switchTab("results");
        await refreshRuns();
        pollSelected();
      } catch (err) {
        $("batchIssues").innerHTML = `<div>${err.message}</div>`;
        toast(err.message || String(err), "bad");
      } finally {
        btn.classList.remove("busy");
        btn.textContent = original;
        btn.disabled = false;
      }
    }
    async function startRun() {
      const btn = $("startBtn");
      const original = btn.textContent;
      btn.disabled = true;
      btn.classList.add("busy");
      btn.textContent = "시작 중";
      try {
        saveDraft("before_start");
        toast("Run 시작 요청을 보냈습니다. subprocess가 생성되면 상태가 실행 중으로 바뀝니다.", "warn");
        const data = await api("/api/runs", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload())});
        state.selectedRun = data.run_id;
        toast(`Run이 시작되었습니다: ${data.run_id}`, "good");
        switchTab("results");
        await refreshRuns();
        pollSelected();
      } catch (err) {
        $("issues").innerHTML = `<div>${err.message}</div>`;
        toast(err.message || String(err), "bad");
      } finally {
        btn.classList.remove("busy");
        btn.textContent = original;
        btn.disabled = false;
      }
    }
    async function refreshRuns() {
      const data = await api("/api/runs");
      const running = data.runs.filter(run => run.status === "running").length;
      $("activeRunCount").textContent = String(running);
      const host = $("runList");
      host.innerHTML = "";
      for (const run of data.runs) {
        const div = document.createElement("div");
        div.className = "run-item" + (run.run_id === state.selectedRun ? " active" : "");
        div.innerHTML = `<div class="run-title">${run.title}</div><div class="run-meta"><span class="status-label">${statusLabel(run.status)}</span> · ${run.created_at || ""} · score ${run.quality_score ?? "-"}</div>`;
        div.addEventListener("click", () => { state.selectedRun = run.run_id; refreshRuns(); pollSelected(); });
        host.appendChild(div);
      }
      if (!state.selectedRun && data.runs.length) state.selectedRun = data.runs[0].run_id;
    }
    function metricCard(name, value) {
      return `<div class="metric"><div class="name">${name}</div><div class="value">${value ?? "-"}</div></div>`;
    }
    function asNumber(value) {
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }
    function metricValue(row, key) {
      return row ? asNumber(row[key]) : null;
    }
    function fmt(value) {
      if (value === null || value === undefined) return "-";
      if (typeof value === "number") return Math.abs(value) >= 10 ? value.toFixed(3) : value.toFixed(6);
      return value;
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }
    function formatBytes(value) {
      const size = Number(value);
      if (!Number.isFinite(size)) return "-";
      if (size < 1024) return `${size} B`;
      if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
      return `${(size / 1024 / 1024).toFixed(1)} MB`;
    }
    function renderArtifacts(data) {
      const gallery = $("artifactGallery");
      const links = $("artifactLinks");
      if (!gallery || !links) return;
      const artifacts = data?.artifacts || [];
      const previewable = artifacts.filter(item => ["image", "video", "html"].includes(item.kind));
      const linked = artifacts.filter(item => !previewable.includes(item));
      if (!artifacts.length) {
        gallery.innerHTML = `<div class="notice">아직 표시할 결과 파일이 없습니다. 시각화 이미지는 run이 끝난 뒤 자동으로 나타나고, 영상은 Save video 저장을 켠 run에서 생성됩니다.</div>`;
        links.innerHTML = "";
        return;
      }
      gallery.innerHTML = previewable.length
        ? previewable.map(item => {
            const name = escapeHtml(item.name);
            const rel = escapeHtml(item.rel_path);
            const url = item.url;
            let media = "";
            if (item.kind === "image") {
              media = `<a href="${url}" target="_blank" rel="noopener"><img src="${url}" alt="${name}" loading="lazy"></a>`;
            } else if (item.kind === "video") {
              media = `<video src="${url}" controls preload="metadata"></video>`;
            } else {
              media = `<iframe src="${url}" title="${name}" loading="lazy"></iframe>`;
            }
            return `<div class="artifact-card">${media}<div class="artifact-title">${name}</div><div class="artifact-meta">${rel} · ${formatBytes(item.size_bytes)}</div></div>`;
          }).join("")
        : `<div class="notice">이미지나 영상 미리보기 파일은 아직 없습니다. CSV/JSON/model 파일은 아래 링크에서 열 수 있습니다.</div>`;
      links.innerHTML = linked.length
        ? linked.map(item => `<a class="artifact-link" href="${item.url}" target="_blank" rel="noopener">${escapeHtml(item.name)} · ${formatBytes(item.size_bytes)}</a>`).join("")
        : "";
    }
    function passLabel(value) {
      if (value === true) return `<span class="pill good">pass</span>`;
      if (value === false) return `<span class="pill bad">fail</span>`;
      return `<span class="pill">n/a</span>`;
    }
    function renderCriteriaEval(summary) {
      const host = $("criteriaEval");
      if (!host) return;
      if (!summary) {
        host.innerHTML = `<div class="notice">summary.json이 생성되면 기준식별 zero-action 대비 trained 결과가 여기에 계산됩니다.</div>`;
        return;
      }
      const rows = summary.metrics || [];
      const zero = rows.find(row => row.tag === "phaseB_zero");
      const trained = rows.find(row => row.tag === "phaseB_trained");
      if (!zero || !trained) {
        host.innerHTML = `<div class="notice">phaseB_zero 또는 phaseB_trained metric row가 없어 기준 계산을 표시할 수 없습니다.</div>`;
        return;
      }
      const success = summary.success_criteria || {};
      const zeroCorner = metricValue(zero, "corner_path_rmse_m");
      const trainedCorner = metricValue(trained, "corner_path_rmse_m");
      const zeroSpeed = metricValue(zero, "corner_mean_speed_mps");
      const trainedSpeed = metricValue(trained, "corner_mean_speed_mps");
      const zeroStraight = metricValue(zero, "straight_path_rmse_m");
      const trainedStraight = metricValue(trained, "straight_path_rmse_m");
      const zeroCoverage = metricValue(zero, "painted_pixel_coverage");
      const trainedCoverage = metricValue(trained, "painted_pixel_coverage");
      const zeroPrecision = metricValue(zero, "painted_pixel_precision");
      const trainedPrecision = metricValue(trained, "painted_pixel_precision");
      const zeroIoU = metricValue(zero, "painted_pixel_iou");
      const trainedIoU = metricValue(trained, "painted_pixel_iou");
      const zeroOffTarget = metricValue(zero, "off_target_pixel_ratio");
      const trainedOffTarget = metricValue(trained, "off_target_pixel_ratio");
      const checks = [
        ["corner_path_rmse_reduced_20pct", "Corner path RMSE", zeroCorner, trainedCorner, zeroCorner === null ? "-" : `<= ${fmt(zeroCorner * 0.80)}`],
        ["corner_speed_reduced_5_to_20pct", "Corner mean speed", zeroSpeed, trainedSpeed, zeroSpeed === null ? "-" : `${fmt(zeroSpeed * 0.80)} ~ ${fmt(zeroSpeed * 0.95)}`],
        ["straight_path_rmse_not_worse_10pct", "Straight path RMSE", zeroStraight, trainedStraight, zeroStraight === null ? "-" : `<= ${fmt(zeroStraight * 1.10)}`],
        ["painting_coverage_kept_90pct", "Paint coverage", zeroCoverage, trainedCoverage, zeroCoverage === null ? "-" : `>= ${fmt(zeroCoverage * 0.90)}`],
        ["painting_precision_kept_90pct", "Paint precision", zeroPrecision, trainedPrecision, zeroPrecision === null ? "-" : `>= ${fmt(zeroPrecision * 0.90)}`],
        ["painting_iou_kept_90pct", "Paint IoU", zeroIoU, trainedIoU, zeroIoU === null ? "-" : `>= ${fmt(zeroIoU * 0.90)}`],
        ["off_target_ratio_not_worse_5pct", "Off-target ratio", zeroOffTarget, trainedOffTarget, zeroOffTarget === null ? "-" : `<= ${fmt(zeroOffTarget + 0.05)}`],
        ["overall_pass", "Overall pass", "-", "-", "위 기준 모두 통과"],
      ];
      host.innerHTML = `
        <table class="compare-table">
          <thead><tr><th>기준</th><th>zero-action</th><th>trained</th><th>통과 범위</th><th>결과</th></tr></thead>
          <tbody>
            ${checks.map(([key, label, z, t, rule]) => `<tr><td>${label}</td><td>${fmt(z)}</td><td>${fmt(t)}</td><td>${rule}</td><td>${passLabel(success[key])}</td></tr>`).join("")}
          </tbody>
        </table>
        <div class="notice" style="margin-top:10px;">complete는 subprocess가 끝까지 실행됐다는 뜻이고, pass/fail은 학습 목표 기준입니다. smoke run은 짧아서 overall_pass=false여도 환경/API 검증 목적상 정상일 수 있습니다.</div>
      `;
    }
    function renderBatchRun(data) {
      const batchSummary = data.batch_summary || {};
      const criteria = batchSummary.batch_final_goal_criteria || {};
      const gate = $("gate");
      const metrics = $("metrics");
      gate.innerHTML = "";
      metrics.innerHTML = "";
      for (const [key, value] of Object.entries(criteria)) {
        if (Array.isArray(value) || typeof value === "object") continue;
        const span = document.createElement("span");
        span.className = "pill " + (value === true ? "good" : value === false ? "bad" : "");
        span.textContent = `${key}: ${value}`;
        gate.appendChild(span);
      }
      metrics.innerHTML = [
        metricCard("Kind", "batch"),
        metricCard("Profile", batchSummary.profile),
        metricCard("Status", statusLabel(data.status)),
        metricCard("Planned runs", batchSummary.planned_runs),
        metricCard("Completed runs", batchSummary.completed_runs),
        metricCard("Failed runs", batchSummary.failed_runs),
        metricCard("Reused runs", batchSummary.reused_runs),
        metricCard("Final pass rate", batchSummary.final_goal_pass_rate),
        metricCard("Expected matrix", criteria.expected_runs),
        metricCard("Observed matrix", criteria.observed_runs),
        metricCard("Batch final goal", criteria.batch_final_goal_pass),
      ].join("");
      const missing = criteria.missing_runs || [];
      const robustness = criteria.robustness_failures || [];
      const failedRuns = criteria.failed_final_goal_runs || [];
      $("criteriaEval").innerHTML = `
        <div class="notice">
          batch_summary.json 기준입니다. missing_runs=${missing.length}, failed_final_goal_runs=${failedRuns.length}, robustness_failures=${robustness.length}.
          전체 완료와 최종 통과는 batch_final_goal_criteria.batch_final_goal_pass를 봅니다.
        </div>
      `;
    }
    function renderRun(data) {
      state.selectedRunData = data;
      $("selectedLog").textContent = data.log_tail || "";
      $("liveLog").textContent = data.log_tail || "";
      updateGlobalStatus(data);
      renderArtifacts(data);
      const summary = data.summary;
      const gate = $("gate");
      const metrics = $("metrics");
      gate.innerHTML = "";
      metrics.innerHTML = "";
      if (data.batch_summary) {
        renderBatchRun(data);
        return;
      }
      renderCriteriaEval(summary);
      if (!summary) {
        gate.innerHTML = `<span class="pill ${data.status === "running" ? "good" : ""}">${statusLabel(data.status)}</span>`;
        metrics.innerHTML = [
          metricCard("Status", statusLabel(data.status)),
          metricCard("Output", data.output_dir),
          metricCard("Log updated", data.log_updated_at),
          metricCard("Return code", data.return_code),
        ].join("");
        return;
      }
      for (const [key, value] of Object.entries(summary.success_criteria || {})) {
        const span = document.createElement("span");
        span.className = "pill " + (value ? "good" : "bad");
        span.textContent = `${key}: ${value}`;
        gate.appendChild(span);
      }
      const finalCriteria = summary.final_goal_criteria || {};
      if (Object.keys(finalCriteria).length) {
        const finalPass = finalCriteria.final_goal_pass === true;
        const failedCount = Object.entries(finalCriteria).filter(([key, value]) => key !== "final_goal_pass" && value !== true).length;
        const span = document.createElement("span");
        span.className = "pill " + (finalPass ? "good" : "bad");
        span.textContent = `final_goal_pass: ${finalPass} · failed ${failedCount}`;
        gate.appendChild(span);
      }
      const rows = summary.metrics || [];
      const trained = rows.find(row => row.tag === "phaseB_trained") || {};
      metrics.innerHTML = [
        metricCard("Status", statusLabel(data.status)),
        metricCard("Quality score", data.quality_score),
        metricCard("Final goal", finalCriteria.final_goal_pass ?? "-"),
        metricCard("Paint coverage", trained.painted_pixel_coverage),
        metricCard("Paint precision", trained.painted_pixel_precision),
        metricCard("Paint recall", trained.painted_pixel_recall),
        metricCard("Paint IoU", trained.painted_pixel_iou),
        metricCard("Paint Dice/F1", trained.painted_pixel_dice),
        metricCard("Off-target ratio", trained.off_target_pixel_ratio),
        metricCard("Path RMSE m", trained.path_rmse_m),
        metricCard("Path max m", trained.path_max_m),
        metricCard("Tracking RMSE m", trained.tracking_rmse_m),
        metricCard("Corner path RMSE m", trained.corner_path_rmse_m),
        metricCard("Corner overshoot m", trained.corner_overshoot_m),
        metricCard("Straight path RMSE m", trained.straight_path_rmse_m),
        metricCard("Corner speed ratio", trained.corner_speed_ratio),
        metricCard("Corner/straight speed", trained.corner_speed_vs_straight_ratio),
        metricCard("LED precision", trained.led_precision),
        metricCard("LED recall", trained.led_recall),
        metricCard("LED flicker rate", trained.led_flicker_rate),
        metricCard("RPM saturation", trained.rpm_saturation_ratio),
        metricCard("Out of bounds rate", trained.out_of_bounds_rate),
        metricCard("Action norm mean", trained.action_norm_mean),
        metricCard("Action norm max", trained.action_norm_max),
        metricCard("Action rate mean", trained.action_rate_mean),
        metricCard("Action rate max", trained.action_rate_max),
        metricCard("Path reward", trained.mean_r_path),
        metricCard("Schedule reward", trained.mean_r_schedule),
        metricCard("Action magnitude reward", trained.mean_r_action_mag),
        metricCard("Action rate reward", trained.mean_r_action_rate),
        metricCard("Painting 결과 reward", trained.mean_r_paint_outcome),
        metricCard("One-sided LED miss prior", trained.mean_r_led_prior),
        metricCard("LED flicker reward", trained.mean_r_led_flicker),
        metricCard("Reward sum", trained.reward_sum),
        metricCard("Crash", trained.crash),
      ].join("");
    }
    async function pollSelected() {
      clearInterval(state.pollTimer);
      if (!state.selectedRun) return;
      const load = async () => {
        try {
          const data = await api("/api/runs/" + encodeURIComponent(state.selectedRun));
          renderRun(data);
          await refreshRuns();
          if (!["running"].includes(data.status)) clearInterval(state.pollTimer);
        } catch (err) {
          $("selectedLog").textContent = err.message;
        }
      };
      await load();
      state.pollTimer = setInterval(load, 1500);
    }
    function exportPreset() {
      const blob = new Blob([JSON.stringify({...payload(), batch: collectBatch()}, null, 2)], {type:"application/json"});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "lightpaint_dashboard_preset.json";
      a.click();
      URL.revokeObjectURL(a.href);
      saveDraft("export_preset");
      toast("현재 설정 preset 파일을 내보냈습니다.", "good");
    }
    async function importPreset(file) {
      saveBackup("before_import");
      const text = await file.text();
      const data = JSON.parse(text);
      applyPayload(data);
      saveDraft("import_preset");
      toast("Preset을 가져왔습니다. 이전 설정은 backup에 보관했습니다.", "good");
      refreshPreview();
    }
    async function init() {
      const options = await api("/api/options");
      state.defaults = options.defaults;
      state.batchDefaults = options.batch_defaults;
      state.batchProfiles = options.batch_profiles;
      state.rewardDefaults = options.reward_defaults;
      state.rewardFields = options.reward_fields;
      state.experimentStages = options.experiment_stages || [];
      state.successCriteria = options.success_criteria || [];
      $("pythonPath").textContent = options.python;
      $("cwdPath").textContent = options.cwd;
      renderReward();
      renderPipeline();
      applyDefaults();
      const restoredDraft = restoreStored(DRAFT_KEY);
      if (restoredDraft) {
        setDraftStatus("저장된 draft 복원됨");
        toast("이 브라우저에 저장된 마지막 설정을 복원했습니다.", "good");
      } else {
        setDraftStatus("기준값 사용 중");
      }
      installHelp();
      document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
      document.querySelectorAll("input, select").forEach(el => el.addEventListener("input", () => {
        refreshPreviewDebounced();
        if (batchKeys.includes(el.id) || el.id.startsWith("reward_")) refreshBatchPreviewDebounced();
        if (el.id === "batch_profile") renderBatchProfileInfo();
        scheduleAutosave();
      }));
      $("presetSmoke").addEventListener("click", () => applyPreset("smoke"));
      $("presetShort").addEventListener("click", () => applyPreset("short"));
      $("presetRobust").addEventListener("click", () => applyPreset("robust"));
      $("applyStageSmoke").addEventListener("click", () => applyStage("env_smoke"));
      $("applyStageM0").addEventListener("click", () => applyStage("m0_corner"));
      $("applyStageM0Full").addEventListener("click", () => applyStage("m0_full_train"));
      $("applyStageM1").addEventListener("click", () => applyStage("m1_disturbance"));
      $("applyStageM2").addEventListener("click", () => applyStage("m2_disturbance"));
      $("applyStageFinal").addEventListener("click", () => applyStage("final_painting"));
      $("saveDraft").addEventListener("click", () => {
        saveDraft("manual");
        toast("현재 설정을 브라우저 draft에 저장했습니다.", "good");
      });
      $("restoreDraft").addEventListener("click", () => {
        if (restoreStored(DRAFT_KEY)) toast("저장된 설정을 복원했습니다.", "good");
        else toast("복원할 저장 설정이 없습니다.", "warn");
      });
      $("restoreBackup").addEventListener("click", () => {
        if (restoreStored(BACKUP_KEY)) {
          saveDraft("restore_backup");
          toast("preset/import/reset 직전 설정을 복원했습니다.", "good");
        } else {
          toast("복원할 이전 설정 backup이 없습니다.", "warn");
        }
      });
      $("resetBaseline").addEventListener("click", () => {
        saveBackup("before_reset_baseline");
        applyDefaults();
        saveDraft("reset_baseline");
        refreshPreview();
        toast("서버 기준 baseline 값으로 복원했습니다. 이전 설정은 backup에 보관했습니다.", "warn");
      });
      $("resetReward").addEventListener("click", () => {
        saveBackup("before_reset_reward");
        for (const [key, value] of Object.entries(state.rewardDefaults)) {
          const el = $("reward_" + key);
          const range = $("reward_range_" + key);
          if (el) el.value = value;
          if (range) range.value = value;
        }
        saveDraft("reset_reward");
        refreshPreview();
        toast("Reward 항목만 기준값으로 복원했습니다.", "warn");
      });
      $("exportPreset").addEventListener("click", exportPreset);
      $("importPreset").addEventListener("change", e => e.target.files[0] && importPreset(e.target.files[0]));
      $("previewBtn").addEventListener("click", e => withButtonFeedback(e.currentTarget, "갱신 중", "명령을 갱신했습니다.", refreshPreview));
      $("batchPreviewBtn").addEventListener("click", e => withButtonFeedback(e.currentTarget, "갱신 중", "Batch 명령을 갱신했습니다.", refreshBatchPreview));
      $("applyBatchLetters").addEventListener("click", () => applyBatchPreset("letters", false, "dg_batch_standard"));
      $("applyBatchStandard").addEventListener("click", () => applyBatchPreset("standard", false, "dual_path_batch_standard"));
      $("applyBatchFinalPlan").addEventListener("click", () => applyBatchPreset("final", true, "dual_path_batch_final"));
      $("batchStartBtn").addEventListener("click", startBatchRun);
      $("startBtn").addEventListener("click", startRun);
      $("copyCommand").addEventListener("click", e => withButtonFeedback(e.currentTarget, "복사 중", "명령을 클립보드에 복사했습니다.", async () => navigator.clipboard.writeText($("commandPreview").textContent)));
      $("refreshRuns").addEventListener("click", e => withButtonFeedback(e.currentTarget, "조회 중", "Run 목록을 새로고침했습니다.", refreshRuns));
      $("useSelectedModel").addEventListener("click", () => useSelectedModel(false));
      $("evalSelectedModel").addEventListener("click", () => useSelectedModel(true));
      $("stopRun").addEventListener("click", async () => {
        if (!state.selectedRun) return;
        toast("선택한 run 중지를 요청했습니다.", "warn");
        await api("/api/runs/" + encodeURIComponent(state.selectedRun) + "/stop", {method:"POST"});
        await pollSelected();
      });
      await refreshPreview();
      await refreshBatchPreview();
      await refreshRuns();
      if (state.selectedRun) pollSelected();
      else updateGlobalStatus(null);
      state.heartbeatTimer = setInterval(checkServerHeartbeat, 5000);
    }
    init().catch(err => { document.body.innerHTML = `<pre>${err.stack || err.message}</pre>`; });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local LightPaint RL experiment dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="기본 브라우저에서 대시보드를 엽니다.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] 대시보드 주소: {url}", flush=True)
    print(f"[dashboard] 패키지 루트: {_PKG_ROOT}", flush=True)
    print(f"[dashboard] Python 실행 파일: {sys.executable}", flush=True)
    if bool(args.open):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] 대시보드를 종료합니다.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
