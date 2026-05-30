"""Run the final LightPaint train -> eval-only -> verify pipeline."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.train import run_final_goal_batch
from src.train import verify_final_goal_batch

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent


def _safe_token(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "run"


def _quote_command(command: list[str]) -> str:
    parts: list[str] = []
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


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "package_root": str(_PKG_ROOT),
        "argv": list(sys.argv),
    }


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _profile_trajectories(args: argparse.Namespace) -> list[str]:
    if args.trajectories:
        return _split_csv(str(args.trajectories))
    defaults = run_final_goal_batch.PROFILE_DEFAULTS[str(args.profile)]
    return _split_csv(defaults["trajectories"])


def _profile_wind_modes(args: argparse.Namespace) -> list[str]:
    if args.wind_modes:
        return _split_csv(str(args.wind_modes))
    defaults = run_final_goal_batch.PROFILE_DEFAULTS[str(args.profile)]
    return _split_csv(defaults["wind_modes"])


def _profile_seeds(args: argparse.Namespace) -> list[int]:
    raw = str(args.seeds) if args.seeds else str(run_final_goal_batch.PROFILE_DEFAULTS[str(args.profile)]["seeds"])
    return [int(item) for item in _split_csv(raw)]


def _resolve_package_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _PKG_ROOT / path
    return path


def _pipeline_preflight(args: argparse.Namespace) -> dict[str, Any]:
    trajectories = _profile_trajectories(args)
    drawn_path = _resolve_package_path(str(args.drawn_path))
    missing_dependencies = [] if bool(args.dry_run) else run_final_goal_batch._missing_runtime_dependencies()
    issues: list[str] = []
    if missing_dependencies:
        issues.append("PyBullet 학습/eval 런타임 의존성이 없습니다: " + ", ".join(missing_dependencies))
    if "drawn" in trajectories and not drawn_path.exists():
        issues.append(f"drawn trajectory JSON 파일이 없습니다: {drawn_path}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "python_executable": sys.executable,
        "profile": str(args.profile),
        "trajectories": trajectories,
        "dry_run": bool(args.dry_run),
        "runtime_dependency_check": "skipped_for_dry_run" if bool(args.dry_run) else "checked",
        "missing_runtime_dependencies": missing_dependencies,
        "drawn_path": str(drawn_path),
        "drawn_path_exists": drawn_path.exists(),
    }


def _batch_common_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--profile",
        str(args.profile),
        "--trained-action-filter",
        str(args.trained_action_filter),
        "--teacher-window-m",
        str(args.teacher_window_m),
    ]
    passthrough = (
        ("trajectories", "--trajectories"),
        ("wind_modes", "--wind-modes"),
        ("seeds", "--seeds"),
        ("total_timesteps", "--total-timesteps"),
        ("max_steps", "--max-steps"),
        ("n_steps", "--n-steps"),
        ("batch_size", "--batch-size"),
        ("n_epochs", "--n-epochs"),
        ("learning_rate", "--learning-rate"),
        ("bc_epochs", "--bc-epochs"),
        ("bc_episodes", "--bc-episodes"),
        ("device", "--device"),
        ("drawn_path", "--drawn-path"),
    )
    for attr, flag in passthrough:
        value = getattr(args, attr)
        if value is not None:
            command.extend([flag, str(value)])
    if not bool(args.resume):
        command.append("--no-resume")
    if bool(args.continue_on_error):
        command.append("--continue-on-error")
    if bool(args.save_video):
        command.append("--save-video")
    return command


def _build_train_command(args: argparse.Namespace, train_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train.run_final_goal_batch",
        "--output-dir",
        str(train_dir),
        *_batch_common_args(args),
    ]
    if bool(args.dry_run):
        command.append("--dry-run")
    return command


def _build_eval_command(args: argparse.Namespace, train_dir: Path, eval_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train.run_final_goal_batch",
        "--output-dir",
        str(eval_dir),
        *_batch_common_args(args),
        "--eval-only",
        "--model-manifest",
        str(train_dir / "best_model_manifest.json"),
    ]
    if bool(args.dry_run):
        command.append("--dry-run")
    return command


def _build_verify_command(args: argparse.Namespace, eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.train.verify_final_goal_batch",
        str(eval_dir),
        "--required-profile",
        str(args.required_profile),
    ]


def _run_step(name: str, command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"[pipeline] step={name}\n")
        log_file.write(f"[pipeline] command={_quote_command(command)}\n")
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
    return {
        "name": name,
        "command": _quote_command(command),
        "return_code": int(proc.returncode),
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _verification_summary(eval_dir: Path, required_profile: str) -> dict[str, Any]:
    result = verify_final_goal_batch.verify_batch_artifacts(
        eval_dir,
        required_profile=str(required_profile),
    )
    criteria = result.summary.get("batch_final_goal_criteria") or {}
    return {
        "ok": bool(result.ok),
        "issues": list(result.issues),
        "profile": result.summary.get("profile"),
        "status": result.summary.get("status"),
        "batch_final_goal_pass": criteria.get("batch_final_goal_pass"),
        "matrix_complete": criteria.get("matrix_complete"),
        "all_runs_final_goal_pass": criteria.get("all_runs_final_goal_pass"),
        "robustness_relative_to_m0_pass": criteria.get("robustness_relative_to_m0_pass"),
        "expected_runs": criteria.get("expected_runs"),
        "observed_runs": criteria.get("observed_runs"),
        "matrix_rows": len(result.matrix_rows),
    }


def _resolve_manifest_model_path(raw: Any, manifest_path: Path) -> Path | None:
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _train_manifest_preflight(train_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    issues: list[str] = []
    batch_summary_path = train_dir / "batch_summary.json"
    model_manifest_path = train_dir / "best_model_manifest.json"
    summary: dict[str, Any] = {}
    model_manifest: dict[str, Any] = {}
    if not batch_summary_path.exists():
        issues.append(f"train batch_summary.json이 없습니다: {batch_summary_path}")
    else:
        try:
            summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"train batch_summary.json을 읽을 수 없습니다: {exc}")
    if not model_manifest_path.exists():
        issues.append(f"train best_model_manifest.json이 없습니다: {model_manifest_path}")
    else:
        try:
            model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"train best_model_manifest.json을 읽을 수 없습니다: {exc}")

    trajectories = _profile_trajectories(args)
    wind_modes = _profile_wind_modes(args)
    seeds = _profile_seeds(args)
    expected_keys = [
        f"{trajectory}/{wind_mode}/seed{seed}"
        for trajectory in trajectories
        for wind_mode in wind_modes
        for seed in seeds
    ]
    models = model_manifest.get("models") if isinstance(model_manifest, dict) else {}
    if not isinstance(models, dict):
        issues.append("train best_model_manifest.json의 models가 올바른 객체가 아닙니다")
        models = {}

    missing_models = [key for key in expected_keys if key not in models]
    missing_model_files = []
    for key in expected_keys:
        if key not in models:
            continue
        model_path = _resolve_manifest_model_path(models.get(key), model_manifest_path)
        if model_path is None or not model_path.exists():
            missing_model_files.append({"key": key, "path": str(models.get(key))})
    if missing_models:
        issues.append("train best_model_manifest.json에 모델 항목이 없습니다: " + ", ".join(missing_models[:8]))
    if missing_model_files:
        preview = ", ".join(f"{item['key']}={item['path']}" for item in missing_model_files[:8])
        issues.append("train best_model_manifest.json의 모델 파일이 없습니다: " + preview)

    expected_count = len(expected_keys)
    if summary:
        if summary.get("status") != "complete":
            issues.append(f"train batch status가 complete가 아닙니다: {summary.get('status')!r}")
        failed_runs = summary.get("failed_runs")
        if failed_runs not in {0, "0", None}:
            issues.append(f"train batch에 실패 run이 있습니다: failed_runs={failed_runs!r}")
        completed_runs = summary.get("completed_runs")
        if completed_runs is not None and int(completed_runs) != expected_count:
            issues.append(f"train completed_runs가 expected matrix와 다릅니다: {completed_runs!r} != {expected_count}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "batch_summary": str(batch_summary_path),
        "model_manifest": str(model_manifest_path),
        "expected_models": expected_count,
        "missing_models": missing_models,
        "missing_model_files": missing_model_files,
    }


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = _PKG_ROOT / output_root
    run_name = _safe_token(str(args.run_name or f"final_goal_{args.profile}"))
    root = output_root / run_name
    train_dir = root / "train"
    eval_dir = root / "eval"
    logs_dir = root / "pipeline_logs"
    root.mkdir(parents=True, exist_ok=True)

    train_command = _build_train_command(args, train_dir)
    eval_command = _build_eval_command(args, train_dir, eval_dir)
    verify_command = _build_verify_command(args, eval_dir)
    preflight = _pipeline_preflight(args)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "profile": str(args.profile),
        "required_profile": str(args.required_profile),
        "dry_run": bool(args.dry_run),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "runtime": _runtime_metadata(),
        "output_root": str(root),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_dir),
        "preflight": preflight,
        "commands": {
            "train": _quote_command(train_command),
            "eval": _quote_command(eval_command),
            "verify": _quote_command(verify_command),
        },
        "steps": [],
    }
    _write_json(root / "pipeline_summary.json", summary)

    if not preflight["ok"]:
        summary["status"] = "failed"
        summary["failed_step"] = "preflight"
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(root / "pipeline_summary.json", summary)
        for issue in preflight["issues"]:
            print(f"[pipeline] preflight error: {issue}", file=sys.stderr, flush=True)
        return 2

    for name, command in (("train", train_command), ("eval", eval_command)):
        print(f"[pipeline] running {name}: {_quote_command(command)}", flush=True)
        step = _run_step(name, command, logs_dir / f"{name}.log")
        summary["steps"].append(step)
        _write_json(root / "pipeline_summary.json", summary)
        if int(step["return_code"]) != 0:
            summary["status"] = "failed"
            summary["failed_step"] = name
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(root / "pipeline_summary.json", summary)
            return int(step["return_code"])
        if name == "train" and not bool(args.dry_run):
            train_manifest_preflight = _train_manifest_preflight(train_dir, args)
            summary["train_manifest_preflight"] = train_manifest_preflight
            _write_json(root / "pipeline_summary.json", summary)
            if not train_manifest_preflight["ok"]:
                summary["status"] = "failed"
                summary["failed_step"] = "train_manifest_preflight"
                summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(root / "pipeline_summary.json", summary)
                for issue in train_manifest_preflight["issues"]:
                    print(f"[pipeline] train manifest error: {issue}", file=sys.stderr, flush=True)
                return 2

    if bool(args.dry_run):
        summary["status"] = "planned"
        summary["verify_skipped"] = "dry_run"
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(root / "pipeline_summary.json", summary)
        return 0

    print(f"[pipeline] running verify: {_quote_command(verify_command)}", flush=True)
    step = _run_step("verify", verify_command, logs_dir / "verify.log")
    summary["steps"].append(step)
    try:
        summary["verification"] = _verification_summary(eval_dir, str(args.required_profile))
    except Exception as exc:  # pragma: no cover - defensive reporting only
        summary["verification"] = {
            "ok": False,
            "issues": [f"검증 산출물을 다시 읽는 중 오류가 발생했습니다: {exc}"],
        }
    summary["status"] = "complete" if int(step["return_code"]) == 0 else "failed"
    if int(step["return_code"]) != 0:
        summary["failed_step"] = "verify"
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(root / "pipeline_summary.json", summary)
    return int(step["return_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final LightPaint training, eval-only replay, and verification.")
    parser.add_argument("--output-dir", default=str(_PKG_ROOT / "artifacts" / "final_goal_pipeline"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--profile", choices=tuple(run_final_goal_batch.PROFILE_DEFAULTS), default="final")
    parser.add_argument("--required-profile", default="final")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trained-action-filter", choices=("none", "corner_tangent_decel"), default=run_final_goal_batch.DEFAULT_TRAINED_ACTION_FILTER)
    parser.add_argument("--teacher-window-m", type=float, default=0.15)
    parser.add_argument("--trajectories", default=None)
    parser.add_argument("--wind-modes", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--bc-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drawn-path", default="data/drawn_paths/user/user_drawn_path.json")
    parser.add_argument("--save-video", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
