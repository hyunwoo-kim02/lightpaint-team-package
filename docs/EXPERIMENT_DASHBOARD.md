# Experiment Dashboard

Phase B 실험을 브라우저에서 설정, 실행, 비교하는 도구입니다.

## 실행

프로젝트 루트에서:

```powershell
python -m src.train.experiment_dashboard --open
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_dashboard.ps1
```

대시보드는 자신을 실행한 Python으로 학습 subprocess를 실행합니다. 먼저 같은 환경에 `requirements.txt`를 설치해 둡니다.

## 사용 순서

1. `Smoke preset`으로 환경 실행 확인
2. `M0 corner sanity`로 짧은 학습 흐름 확인
3. `M0 full corner`로 본 학습
4. M0에서 저장된 model을 `load_model`로 M1/M2에 연결
5. letter/drawn path로 최종 painting 확인

Preset 버튼은 설정만 바꿉니다. 실제 실행은 `Run 시작`을 눌러야 합니다.

## Run 산출물

각 run은 아래에 저장됩니다.

```text
artifacts/dashboard/runs/<run-id>/
```

주요 파일:

- `dashboard_run.json`: 실행 설정
- `reward_overrides.json`: run별 reward override
- `run.log`: stdout/stderr 로그
- `summary.json`: 평가 결과
- `models/*.zip`: 저장된 PPO 모델

## 결과 해석

`complete`는 subprocess가 끝났다는 뜻입니다.
`overall_pass`는 평가 기준을 통과했다는 뜻입니다.

Smoke run은 짧게 실행하므로 `overall_pass=false`여도 정상일 수 있습니다.

주요 평가 기준:

- `corner_path_rmse_reduced_20pct`
- `corner_speed_reduced_5_to_20pct`
- `straight_path_rmse_not_worse_10pct`
- `painting_coverage_kept_90pct`
- `painting_precision_kept_90pct`
- `painting_iou_kept_90pct`
- `off_target_ratio_not_worse_5pct`
- `overall_pass`

대시보드에서 반드시 같이 볼 지표:

- `painted_pixel_coverage`
- `painted_pixel_precision`
- `painted_pixel_iou`
- `painted_pixel_dice`
- `off_target_pixel_ratio`
- `tracking_rmse_m`
- `corner_path_rmse_m`
- `corner_mean_speed_mps`
- `crash`

Painting mask threshold는 `summary.json`의 `evaluation.painting_threshold`에 기록됩니다. 현재 기본값은 cumulative intensity `> 0.3`입니다.
