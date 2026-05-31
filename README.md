# Lightpaint Team Package

Light-painting reinforcement learning package for drone path tracking and LED
control. The active package surface is intentionally small: canonical `DG`,
one user-drawn path, PyBullet evaluation, the experiment dashboard, and the
path/corner editing tools.

```text
reference path
-> LightPaintRef
-> PyBullet CF2X + DSLPIDControl
-> Phase B velocity/LED residual policy
-> summary.json and CSV metrics
```

## Setup

Python 3.10 or 3.11 is recommended. Python 3.13 may not support the pinned
PyTorch wheel used by this project.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the default Windows `python` points to 3.13, create the environment with:

```powershell
py -3.11 -m venv .venv
```

## Verify

Run the test suite:

```powershell
python -m pytest -q --basetemp artifacts\pytest_tmp_run -p no:cacheprovider
```

Run a short smoke check:

```powershell
python -m src.train.train_phase_b --trajectory square --wind-mode M0 --max-steps 60 --total-timesteps 0 --bc-epochs 0 --output-dir artifacts\team_smoke_m0
```

The smoke run is execution-only. It may report `overall_pass=false` because it
does not train a policy.

## Dashboard

```powershell
python -m src.train.experiment_dashboard --open
```

Windows helpers:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_dashboard.ps1
```

## Path Tools

- `tools/export_corner_reference.py`: generate a letter reference JSON and optional preview.
- `tools/draw_path.html`: draw or edit a user path.
- `tools/corner_hint_editor.html`: inspect or edit corner hints.

Generate the canonical letter reference:

```powershell
python tools\export_corner_reference.py --trajectory letter --label DG --letter-plane xz --output data\corner_hints\DG_reference.json --preview-output artifacts\DG_reference_preview.png
```

The committed user path is:

```text
data/drawn_paths/user/user_drawn_path.json
```

Generated example paths and exploratory path files are intentionally excluded
from Git.

## Final Goal Batch

The default final-goal batch uses only two canonical paths:

- `DG`: generated through the canonical letter reference builder.
- `drawn`: loaded from `data/drawn_paths/user/user_drawn_path.json`.

Plan the run first:

```powershell
python -m src.train.run_final_goal_batch --profile final --dry-run --output-dir artifacts\final_goal_batch_plan
```

Remove `--dry-run` only when starting the actual run. The built-in profiles are:

- `sanity`: `DG,drawn` x `M0` x seed `7`, execution check only.
- `letters`: `DG` x `M0,M1,M2` x seeds `7,11,17`, 100000 PPO steps per run.
- `standard`: `DG,drawn` x `M0,M1,M2` x seeds `7,11,17`, 100000 PPO steps per run.
- `final`: `DG,drawn` x `M0,M1,M2` x seeds `7,11,17,23,29`, 300000 PPO steps per run.

M1 continues from the matching M0 model, and M2 continues from the matching M1
model. Final evaluation uses `--trained-action-filter corner_tangent_decel` and
`--teacher-window-m 0.15`.

After training, rerun final evaluation from the model manifest:

```powershell
python -m src.train.run_final_goal_batch --profile final --eval-only --model-manifest artifacts\final_goal_batch_final\best_model_manifest.json --output-dir artifacts\final_goal_eval_final
python -m src.train.verify_final_goal_batch artifacts\final_goal_eval_final
```

Key outputs:

- `batch_summary.json`
- `metrics.csv`
- `matrix_results.csv`
- `best_model_manifest.json`

## Commit Surface

Commit source code, tests, canonical path data, dashboard helpers, and the path
generation/editing tools. Keep generated artifacts, trained model zips, exploratory
paths, reports, local notes, and runtime caches out of Git.
