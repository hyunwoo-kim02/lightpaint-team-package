# Lightpaint Team Package

드론 라이트 페인팅 RL 실험 프레임워크입니다.

흐름:

```text
square/letter/drawn path
-> LightPaintRef
-> PyBullet CF2X + DSLPIDControl
-> Phase B velocity/LED residual policy
-> summary.json 기준 평가
```

## 1. 설치

Python 3.10 또는 3.11을 권장합니다. Python 3.13은 `torch==2.2.0` 설치가 실패할 수 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows에서 기본 `python`이 3.13이면 첫 줄만 아래처럼 실행합니다.

```powershell
py -3.11 -m venv .venv
```

또는:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1
```

## 2. 설치 확인

```powershell
python -m pytest -q --basetemp artifacts\pytest_tmp_run -p no:cacheprovider
```

짧은 실행 확인:

```powershell
python -m src.train.train_phase_b --trajectory square --wind-mode M0 --max-steps 60 --total-timesteps 0 --bc-epochs 0 --output-dir artifacts\team_smoke_m0
```

`artifacts\team_smoke_m0\summary.json`이 생성되면 실행 경로는 정상입니다. 이 smoke는 학습을 하지 않으므로 `overall_pass=false`가 나올 수 있습니다.

## 3. 대시보드 실행

```powershell
python -m src.train.experiment_dashboard --open
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_dashboard.ps1
```

대시보드가 기본 실험 진입점입니다.

## 4. CLI 실험 예시

```powershell
python -m src.train.train_phase_b --trajectory square --wind-mode M0 --seed 7 --total-timesteps 2048 --output-dir artifacts\m0_square

python -m src.train.train_phase_b --trajectory letter --label L --wind-mode M1 --seed 7 --total-timesteps 2048 --output-dir artifacts\m1_letter_l

python -m src.train.train_phase_b --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --path-scale 0.8 --wind-mode M2 --seed 7 --total-timesteps 2048 --output-dir artifacts\m2_drawn
```

직접 그린 경로는 `tools/draw_path.html`에서 만들고 `data/drawn_paths/user/` 아래에 저장합니다.

## 5. 결과 기준

최종 판단은 각 run의 `summary.json` 기준으로 합니다.

중요 지표:

- `painted_pixel_coverage`
- `painted_pixel_precision`
- `painted_pixel_iou`
- `painted_pixel_dice`
- `off_target_pixel_ratio`
- `tracking_rmse_m`
- `corner_path_rmse_m`
- `corner_mean_speed_mps`
- `crash`
- `success_criteria.overall_pass`

Coverage만 보지 말고 precision, IoU, off-target ratio를 함께 봅니다.

## 6. 주요 진입점

| 목적 | 파일 |
|---|---|
| 대시보드 | `src/train/experiment_dashboard.py` |
| Phase B 학습/평가 | `src/train/train_phase_b.py` |
| Reference 생성 | `src/train/reference_factory.py` |
| 경로/문자 reference | `src/env/lightpaint_ref.py` |
| PyBullet 환경 | `src/env/light_paint_aviary_pyb.py` |
| Reward 설정 | `src/env/reward_config.py` |
| Reward 계산 | `src/env/reward_terms.py` |
| 직접 경로 그리기 | `tools/draw_path.html` |

## 7. 문서

- `docs/TEAM_EXPERIMENT_CODE_SPEC.md`: 실험 목표, 구조, 평가 기준
- `docs/EXPERIMENT_DASHBOARD.md`: 대시보드 사용법

## 8. 공유 제외

공유 제외:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `artifacts/` 내부 생성 결과물
- 학습 모델 `.zip`

모델을 이어서 학습해야 할 때만 해당 run의 `summary.json`, `reward_overrides.json`, `dashboard_run.json`, `models/*.zip`을 따로 공유합니다.
