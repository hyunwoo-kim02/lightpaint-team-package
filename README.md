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

Python 3.13은 일부 PyTorch/SB3 의존성과 맞지 않을 수 있으므로 권장하지 않습니다.

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows에서 기본 `python`이 3.13이면 아래처럼 3.11로 가상환경을 만듭니다.

```powershell
py -3.11 -m venv .venv
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

이 smoke run은 실행 경로 확인용입니다. 학습을 하지 않으므로 `overall_pass=false`가 나올 수 있습니다.

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

생성 결과와 임시 학습 산출물은 `artifacts/` 아래에 만들고 Git에는 포함하지 않습니다.

## 발표 결과 재현

최종 발표 슬라이드의 결과 화면과 지표는 아래 패키지에서 재현합니다.

```powershell
powershell -ExecutionPolicy Bypass -File repro\presentation_20260531\run_reproduce_figures.ps1
```

해당 패키지에는 발표 슬라이드, 선택된 웨이트, 평가 로그, figure 재생성 스크립트가 포함되어 있습니다.
