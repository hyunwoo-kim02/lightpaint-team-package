# LightPaint 드론 라이트 페인팅

드론이 기준 경로를 따라 이동하면서 LED를 켜고 끄는 방식으로 라이트 페인팅을 수행하는 강화학습 프로젝트입니다.

## 프로젝트 목표

- `DG` 문자 경로와 사용자 직접 입력 경로를 기준 경로로 사용합니다.
- PyBullet Crazyflie 시뮬레이션에서 PID baseline과 PPO residual policy를 비교합니다.
- 외란 조건 `M0`, `M1`, `M2`에서 경로 추종, 코너 감속, LED painting 품질을 평가합니다.

기본 흐름은 다음과 같습니다.

```text
reference path
-> LightPaintRef
-> PyBullet Crazyflie + PID controller
-> Phase B PPO velocity/LED residual policy
-> summary.json / metrics.csv
```

## 필요한 것

- Python 3.10 또는 3.11
- `requirements.txt`에 적힌 Python 패키지
- 기본 기준 경로:
  - `data/corner_hints/DG_reference.json`
  - `data/drawn_paths/user/user_drawn_path.json`

## 설치

아래 명령의 `python`은 Python 3.10 또는 3.11입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 사용 방법

대시보드를 실행합니다.

```powershell
python -m src.train.experiment_dashboard --open
```

Windows helper를 써도 됩니다.

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_dashboard.ps1
```

짧은 실행 확인은 아래 명령으로 합니다.

```powershell
python -m src.train.train_phase_b --trajectory square --wind-mode M0 --max-steps 60 --total-timesteps 0 --bc-epochs 0 --output-dir artifacts\team_smoke_m0
```

## 경로 도구

`DG` 문자 기준 경로를 다시 생성하려면:

```powershell
python tools\export_corner_reference.py --trajectory letter --label DG --letter-plane xz --output data\corner_hints\DG_reference.json --preview-output artifacts\DG_reference_preview.png
```

사용자 경로는 브라우저에서 `tools/draw_path.html`을 열어 만들거나 수정합니다.

코너 힌트는 브라우저에서 `tools/corner_hint_editor.html`을 열어 확인하거나 수정합니다.

## 최종 Batch 실행

최종 batch는 기본적으로 두 경로를 사용합니다.

- `DG`
- `drawn`

먼저 dry-run으로 실행 계획을 확인합니다.

```powershell
python -m src.train.run_final_goal_batch --profile final --dry-run --output-dir artifacts\final_goal_batch_plan
```

실제로 학습을 시작하려면 `--dry-run`을 제거합니다.

```powershell
python -m src.train.run_final_goal_batch --profile final --output-dir artifacts\final_goal_batch_final
```

학습이 끝난 뒤 모델 manifest로 최종 평가를 다시 실행할 수 있습니다.

```powershell
python -m src.train.run_final_goal_batch --profile final --eval-only --model-manifest artifacts\final_goal_batch_final\best_model_manifest.json --output-dir artifacts\final_goal_eval_final
python -m src.train.verify_final_goal_batch artifacts\final_goal_eval_final
```

주요 결과 파일은 실행 폴더의 `summary.json`, `metrics.csv`, `matrix_results.csv`, `best_model_manifest.json`입니다.

실행 결과와 학습 산출물은 `artifacts/` 아래에 만들고 Git 대상에서 제외합니다.

## 결과 재현

결과 화면과 지표는 아래 명령으로 재현합니다.

```powershell
powershell -ExecutionPolicy Bypass -File repro\presentation_20260531\run_reproduce_figures.ps1
```

GUI로 궤적을 확인합니다.

```powershell
python -m src.train.train_phase_b --trajectory letter --label DG --letter-plane xz --wind-mode M0 --load-model repro\presentation_20260531\weights\M0.zip --eval-only --trained-action-filter corner_tangent_decel --gui --gui-hold-seconds 10 --max-steps 300 --output-dir artifacts\gui_m0
```
