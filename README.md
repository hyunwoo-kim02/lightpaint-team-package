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

## 4. CLI 실행 확인 예시

아래 명령은 개별 실행 경로 확인용입니다. 학습 성공 또는 최종 성능 근거로 쓰지 않습니다.
글자별 본 학습은 뒤의 `letters`, `standard`, `final` batch profile로 실행합니다.

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

## Final goal batch

최종 목표와 금지 가드레일은 `docs/FINAL_GOAL_SPEC.md`를 기준으로 합니다.

짧은 smoke가 아니라 여러 글자/외란/seed를 실제 학습 대상으로 계획하려면:

```powershell
python -m src.train.run_final_goal_batch --profile final --dry-run --output-dir artifacts\final_goal_batch_plan
```

실제 실행은 `--dry-run`을 제거합니다. 최종 제출 후보는 `final` profile을 사용합니다.
이 profile은 `square,L,DG,CAT,Pig,RL,drawn` 전체 trajectory, `M0,M1,M2` wind mode,
seed `7,11,17,23,29`, run당 `total_timesteps=300000`입니다.
`standard` profile은 같은 trajectory/wind에 seed `7,11,17`, run당 `100000` step인
중간 탐색용 profile이며 최종 완료 증거는 아닙니다.
M1은 M0 모델을, M2는 M1 모델을 이어받는 curriculum 명령을 생성합니다.
따라서 curriculum 학습에서는 wind 순서를 `M0,M1,M2`로 유지해야 합니다.
`M0,M2`처럼 M1을 건너뛰는 계획은 preflight에서 실패로 기록됩니다.
기본 최종 평가 rollout은 `--trained-action-filter corner_tangent_decel`과
`--teacher-window-m 0.15`를 사용합니다. raw policy 자체를 비교하려면 명시적으로
`--trained-action-filter none`을 붙여 별도 ablation으로 기록합니다.
실제 학습을 시작하는 첫 미완료 run 직전에 batch runner가 PyBullet/SB3 의존성을 한 번
검사합니다. 누락되면 `batch_summary.json`, `matrix_results.csv`, 해당 run의
`batch_run.log`에 현재 Python 경로와 누락 패키지를 남기고 중단합니다. 이미 완료된
`summary.json`을 재사용해 집계만 다시 만드는 경우에는 PyBullet 의존성이 없어도 동작합니다.

주요 산출물:

- `batch_summary.json`
- `metrics.csv`
- `matrix_results.csv`
- `best_model_manifest.json`

이미 완료된 하위 run에 `summary.json`이 있으면 기본적으로 재사용합니다. 다시 계산하려면
`--no-resume`을 붙입니다. 재사용 시에도 저장된 설정, `artifacts.model_path` 파일 존재,
`total_timesteps`, `max_steps_override`, `final` profile의 metric/criterion 일관성을 확인합니다.

### Full multi-letter training profiles

최종 목표용 학습은 짧은 smoke가 아니라 전체 trajectory/wind/seed matrix로 실행합니다.

```powershell
python -m src.train.run_final_goal_batch --profile letters --output-dir artifacts\letter_batch_standard

python -m src.train.run_final_goal_batch --profile standard --output-dir artifacts\final_goal_batch_standard

python -m src.train.run_final_goal_batch --profile final --output-dir artifacts\final_goal_batch_final
```

- `letters`: `L,DG,CAT,Pig,RL` x `M0,M1,M2` x seed `7,11,17`, run당 `100000` PPO step.
- `standard`: `square,L,DG,CAT,Pig,RL,drawn` x `M0,M1,M2` x seed `7,11,17`, run당 `100000` PPO step.
- `final`: 같은 trajectory/wind 전체 matrix에 seed `7,11,17,23,29`, run당 `300000` PPO step.
- `sanity`: 실행 확인용입니다. 학습 성공 또는 최종 목표 달성 근거로 쓰지 않습니다.

학습이 끝난 모델을 최종 평가 전용으로 다시 돌릴 때는 이전 batch 산출물의
`best_model_manifest.json`을 사용합니다.

```powershell
python -m src.train.run_final_goal_batch --profile final --eval-only --model-manifest artifacts\final_goal_batch_final\best_model_manifest.json --output-dir artifacts\final_goal_eval_final
```

대시보드에서는 `전체 Batch` 탭에서 `Eval-only 최종 평가`를 체크하고 `Model manifest`에
해당 `best_model_manifest.json` 경로를 넣습니다.
manifest 안의 모델 경로가 상대 경로이면 manifest 파일이 있는 폴더 기준으로 해석합니다.
따라서 학습 산출물 폴더를 팀원에게 통째로 공유해도 eval-only 경로가 덜 깨집니다.

최종 보고 전에는 verifier를 실행해 산출물 전체가 최종 목표 증거로 충분한지 확인합니다.

```powershell
python -m src.train.verify_final_goal_batch artifacts\final_goal_eval_final
```

기본 verifier는 `profile=final`, `batch_final_goal_pass=true`, 전체 matrix 완료, 모든 run
`final_goal_pass=true`, 필수 CSV/manifest 산출물 존재를 요구합니다.
각 run의 `final_goal_criteria`는 `summary.metrics`에서 재계산되어야 하므로 성공 flag만
고친 산출물은 통과하지 못합니다.
batch-level robustness도 `matrix_results.csv`에서 다시 계산되므로, M1/M2 성능 하락을
숨기기 위해 summary flag만 고친 산출물은 통과하지 못합니다.
`matrix_results.csv`, `metrics.csv`, per-run `summary.json`의 숫자 metric도 서로 일치해야 합니다.
`aggregate_metrics.csv`의 group별 통계도 같은 matrix에서 재계산되어야 합니다.
또한 eval-only에 사용된 source training manifest를 따라가서 run당 `300000` step 학습과
M0→M1→M2 `load_model` curriculum 연결이 유지됐는지 확인합니다.
모델 경로는 존재만 보는 것이 아니라 Stable-Baselines zip 형식과 필수 항목(`data`, `policy.pth`)도
검증하므로 빈 파일이나 임의 zip으로 만든 결과는 통과하지 않습니다.

학습, eval-only 평가, verifier를 한 번에 연결하려면 pipeline runner를 사용합니다.

```powershell
python -m src.train.run_final_goal_pipeline --profile final --output-dir artifacts\final_goal_pipeline --run-name final_candidate
```

먼저 명령 구조만 확인하려면 `--dry-run`을 붙입니다. dry-run은 train/eval batch 계획 산출물을
만들고 verifier는 실행하지 않습니다. 실제 실행은 이 명령을 실행한 Python 환경에 PyBullet,
gym-pybullet-drones, SB3 의존성이 설치되어 있어야 합니다. pipeline은 시작 전에
`pipeline_summary.json`의 `preflight`에 현재 Python, 누락 의존성, `drawn` JSON 존재 여부를
기록하고 문제가 있으면 train을 시작하지 않습니다.

기본 `drawn` 경로는 `data\drawn_paths\user\user_drawn_path.json`입니다. 다른 파일을 쓰려면
pipeline과 batch 모두에 아래처럼 명시합니다.

```powershell
python -m src.train.run_final_goal_pipeline --profile final --drawn-path data\drawn_paths\user\my_path.json --output-dir artifacts\final_goal_pipeline --run-name final_candidate
```

축소 검증이나 특정 글자 sweep을 먼저 확인할 때는 batch override를 함께 넘길 수 있습니다.

```powershell
python -m src.train.run_final_goal_pipeline --profile letters --dry-run --trajectories L,DG --wind-modes M0 --seeds 7,11 --total-timesteps 100000 --run-name letters_probe
```
