# Lightpaint Team Package

Current implementation focus, updated 2026-05-13:

> Square waypoint path -> DATT-style `LightPaintRef` -> PyBullet CF2X -> `DSLPIDControl` -> M0 tracking RMSE/corner-error validation.

This package should now be treated as a PyBullet PID baseline workspace. PPO LED learning, multi-stroke connector RL, and disturbance robustness RL are deferred until the Square-PID-M0 baseline artifacts are reviewed.

Update later on 2026-05-13: image-derived letter paths now go through
arc-length resampling and smoothing before entering `LightPaintRef`. This keeps
the PID/M0 baseline from chasing pixel-grid stair steps. Letter and alphabet
rollouts should produce the same visualization family as the square gate:
tracking CSV/metrics, 3D PNG, tracking PNG, frames, MP4, GIF, summary PNG, and
reference-path diagnostic PNG.

## Active Files

| Path | Role |
|---|---|
| `src/env/lightpaint_ref.py` | DATT-style `pos/vel/acc/yaw(t)` reference adapter and square waypoint source. |
| `src/env/light_paint_aviary_pyb.py` | Active PyBullet CF2X environment using `DSLPIDControl`. |
| `src/env/pybullet_setup.py` | Korean-path PyBullet asset mirror fix. |
| `src/train/train_phase_a_pid.py` | Current PID-only rollout/visualization entrypoint. |
| `src/train/batch_alphabet_pid.py` | A-Z PID batch rollout for fixed-z x-y and z-motion x-z modes. |
| `src/train/visualize_flight.py` | Visualization helpers used by the PID rollout. |
| `tools/draw_path.html` | Browser canvas that exports user-drawn strokes as JSON. |
| `tests/test_lightpaint_ref.py` | Reference contract tests. |
| `tests/test_phase_a_pyb_env.py` | Active PyBullet environment smoke tests. |

## Archived Legacy Paths

Old PPO-first instructions and legacy code were moved to:

`../archive/2026-05-13_pre_square_pid_cleanup/`

Do not use archived `train_w1.py`, `eval_w1.py`, old `mini_script/teams/*/code`, or old Claude prompts as active implementation guidance.

## Current Gate

The DATT-style reference adapter is now wired:

`square waypoint path -> LightPaintRef -> LightPaintAviaryPyB -> DSLPIDControl`

The first concrete source is a square waypoint path. The controller consumes only the time-indexed reference, not the original input format.

## Quick Check

From `lightpaint-team-package/`:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m pytest tests/test_lightpaint_ref.py tests/test_phase_a_pyb_env.py -q
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_a_pid --trajectory square --wind-mode M0 --seed 42 --led-always-on
```

Acceptance artifacts include trajectory CSV, metrics CSV, corner-error CSV, 3D/summary/refpath/tracking plots, frames, GIF, and a PyBullet-view MP4. Painted-mask quality and learned LED behavior are not the current gate.

## Drawn Path

Open `tools/draw_path.html` in a browser to draw one or more strokes and export
a JSON file. Each stroke is treated as LED ON by default; connector motion
between strokes is inserted automatically with LED OFF.

Example rollout from a drawn JSON:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_a_pid --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --wind-mode M0 --seed 42
```

Drawn path files are organized under `data/drawn_paths/user/` and
`data/drawn_paths/examples/`.

## Alphabet Batch

From `lightpaint-team-package/`:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.batch_alphabet_pid --letters ABCDEFGHIJKLMNOPQRSTUVWXYZ --modes xy_fixed_z xz_z_motion --output-dir artifacts\alphabet_pid --snapshot-count 4 --video-frames 10 --video-fps 5
```

Outputs are organized under `artifacts/alphabet_pid/`. By default, each
letter/mode run writes the full square-style visualization family. Use
`--quick-video` only when an abbreviated debug MP4 is intentional.
