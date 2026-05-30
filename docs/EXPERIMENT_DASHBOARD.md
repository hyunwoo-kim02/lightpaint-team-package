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
3. `전체 Batch` 탭에서 `letters` profile로 L/DG/CAT/Pig/RL 전체 장시간 학습을 실행
4. 더 넓은 기준이 필요하면 `standard`로 square/letter/drawn 전체 matrix 실행
5. 최종 목표 달성 주장은 `final` profile의 `batch_summary.json`으로 판단

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

## Final goal result fields

대시보드의 기존 `overall_pass`는 Phase B 중간 게이트입니다. 최종 목표 평가는
`summary.json`의 `final_goal_criteria.final_goal_pass`를 별도로 봅니다.

최종 목표 기준은 `docs/FINAL_GOAL_SPEC.md`에 고정되어 있으며, 핵심 필드는 다음과 같습니다.

- `painted_pixel_iou`, `painted_pixel_precision`, `painted_pixel_recall`, `painted_pixel_dice`
- `off_target_pixel_ratio`
- `path_rmse_m`, `path_max_m`
- `corner_path_rmse_m`, `corner_speed_ratio`, `corner_overshoot_m`
- `led_precision`, `led_recall`, `led_flicker_rate`
- `action_norm_mean`, `action_norm_max`, `action_rate_mean`, `action_rate_max`
- `rpm_saturation_ratio`, `crash_rate`, `out_of_bounds_rate`

여러 글자/외란/seed 전체 계획은 다음 명령으로 확인합니다.

```powershell
python -m src.train.run_final_goal_batch --dry-run --output-dir artifacts\final_goal_batch_plan
```

전체 batch의 최종 판정은 `batch_summary.json`의
`batch_final_goal_criteria.batch_final_goal_pass`를 봅니다. 이 값은 전체 matrix가
완성되고, 모든 run의 `final_goal_pass`가 true이며, M1/M2가 M0 대비
`painting_iou` 20% 이상 하락 또는 `path_rmse_m` 30% 이상 악화를 보이지 않을 때만 true입니다.
실패한 경우에는 `batch_final_goal_criteria.failed_final_goal_runs`에서 어떤
trajectory/wind/seed가 어떤 per-run 기준을 통과하지 못했는지 먼저 확인합니다.
`matrix_results.csv`는 모든 trajectory/wind/seed 조합을 한 행씩 고정해서 보여주므로
planned/missing/failed 조합을 빠르게 점검할 때 사용합니다.

Batch 실행은 기본적으로 `--resume` 동작을 사용합니다. 기존 run 폴더에 `summary.json`이 있으면
그 결과를 재사용해 집계 파일을 다시 만들고, 없는 run만 실행합니다.
재사용 전에 저장된 config, `artifacts.model_path` 파일 존재, `final` profile의
metric/criterion 일관성, `total_timesteps`, `max_steps_override`를 확인하므로 깨진 run 폴더나
짧은 smoke rollout을 조용히 이어받지 않습니다.
새 학습을 시작해야 하는 경우에는 batch runner가 PyBullet, pybullet_data,
gym-pybullet-drones, stable-baselines3 의존성을 먼저 확인합니다. 누락되면
`batch_summary.json`의 `dependency_preflight`와 해당 run의 `batch_run.log`에 현재 Python
경로와 설치 안내가 기록됩니다. 완료된 run을 재사용해 결과표만 다시 만드는 경우에는 이
preflight가 필요하지 않습니다.
curriculum 학습은 같은 trajectory/seed 안에서 `M0 -> M1 -> M2` 순서로 weight를 이어받습니다.
`M1`은 `M0` 뒤, `M2`는 `M1` 뒤에 있어야 하며, 잘못된 wind 순서나 누락된 drawn JSON은
`batch_preflight` 실패로 기록되어 학습을 시작하지 않습니다.

`corner_speed_ratio`는 corner 평균 속도를 reference speed로 나눈 값입니다. 기존 straight
구간 대비 속도 비교가 필요하면 `corner_speed_vs_straight_ratio`를 보조 진단값으로 봅니다.

## Eval-only final batch

장시간 학습 batch가 끝나면 그 산출물의 `best_model_manifest.json`을 이용해 같은 matrix를
학습 없이 다시 평가할 수 있습니다. 대시보드에서는 `전체 Batch` 탭에서
`Eval-only 최종 평가`를 체크하고 `Model manifest`에 이전 batch의
`best_model_manifest.json` 경로를 넣습니다. 이 모드는 `--eval-only --model-manifest ...`를
붙여 실행되며, 최종 목표 달성 주장은 이 평가 batch의
`batch_summary.json`과 `matrix_results.csv`를 기준으로 확인합니다.
manifest 안의 모델 경로가 상대 경로이면 manifest 파일이 있는 폴더 기준으로 해석합니다.
학습 산출물 폴더를 다른 팀원에게 공유할 때는 manifest와 `models` 폴더를 함께 유지합니다.

최종 보고 전에는 아래 명령을 실행합니다.

```powershell
python -m src.train.verify_final_goal_batch artifacts\final_goal_eval_final
```

이 verifier가 0으로 종료해야 최종 목표 산출물이 충분하다고 보고할 수 있습니다.
각 per-run `summary.json`의 `final_goal_criteria`는 `summary.metrics`에서 같은 수식으로
재계산되므로, 성공 flag만 수정한 산출물은 통과하지 못합니다.
batch-level robustness도 `matrix_results.csv`에서 다시 계산되므로, M1/M2가 M0 대비
무너진 결과는 `batch_final_goal_criteria` flag만 고쳐도 통과하지 못합니다.
또한 `matrix_results.csv`, `metrics.csv`, per-run `summary.json`의 숫자 metric이 서로
일치해야 하므로 stale CSV를 붙여 넣은 결과도 최종 증거로 인정되지 않습니다.
`aggregate_metrics.csv`의 group별 평균/표준편차/최소/최대도 같은 matrix에서 재계산됩니다.
`final` profile 검증은 eval batch만 보지 않고, eval에 사용된 source training
`best_model_manifest.json`도 따라가서 원본 학습 batch가 canonical final matrix와
run당 `300000` PPO step으로 실행됐는지 확인합니다. 짧게 학습한 모델을 나중에 final
eval 형식으로만 평가한 결과는 최종 목표 달성 증거가 아닙니다.
source training matrix의 curriculum도 확인하므로 M1은 같은 trajectory/seed의 M0 모델,
M2는 같은 trajectory/seed의 M1 모델을 `load_model`로 사용해야 합니다.
검증기는 기록된 model path가 실제 Stable-Baselines zip인지와 필수 항목(`data`, `policy.pth`)도
확인하므로 빈 파일이나 임의 파일명만으로 만든 산출물은 통과하지 않습니다.

CLI에서 전체 절차를 한 번에 연결하려면 아래 명령을 사용합니다.

```powershell
python -m src.train.run_final_goal_pipeline --profile final --output-dir artifacts\final_goal_pipeline --run-name final_candidate
```

먼저 `--dry-run`으로 train/eval batch 계획이 맞는지 확인한 뒤 실제 실행으로 전환합니다.
특정 글자나 seed만 먼저 줄여 확인하려면 `--trajectories`, `--wind-modes`, `--seeds`,
`--total-timesteps` 같은 batch override를 함께 넘깁니다.
pipeline은 실제 실행 전에 `pipeline_summary.json`의 `preflight`에 현재 Python 실행 파일,
PyBullet/SB3 의존성 누락 여부, `drawn` JSON 존재 여부를 기록합니다. `preflight.ok=false`이면
train을 시작하지 않고 `failed_step=preflight`로 종료합니다. 기본 `drawn` 경로는
`data\drawn_paths\user\user_drawn_path.json`이며, 다른 파일을 쓰려면 `--drawn-path`를 명시합니다.
실패 시에는 먼저 `pipeline_summary.json`의 `preflight.issues` 또는 `verification.issues`를
확인합니다. 여기에는 어떤 trajectory/wind/seed 또는 최종 기준이 통과하지 못했는지가 한국어로
구조화되어 기록됩니다.

## Full matrix training

대시보드의 단일 run preset은 개별 실험과 시각화 확인용입니다. 팀 공유용 성능 평가는 짧은
smoke가 아니라 batch runner로 여러 글자, 외란, seed를 모두 실행합니다.

```powershell
python -m src.train.run_final_goal_batch --profile letters --output-dir artifacts\letter_batch_standard

python -m src.train.run_final_goal_batch --profile standard --output-dir artifacts\final_goal_batch_standard

python -m src.train.run_final_goal_batch --profile final --output-dir artifacts\final_goal_batch_final
```

- `letters`: `L,DG,CAT,Pig,RL` x `M0,M1,M2` x seed `7,11,17`, run당 `100000` PPO step.
- `standard`: `square,L,DG,CAT,Pig,RL,drawn` x `M0,M1,M2` x seed `7,11,17`, run당 `100000` PPO step.
- `final`: 같은 matrix에 seed `7,11,17,23,29`, run당 `300000` PPO step.
- `sanity`: 실행 확인용이며 learning success로 보고하지 않습니다.

팀 공유용 학습 결과는 `letters`, `standard`, `final` 중 하나로 실행한 batch 산출물만 사용합니다.
단일 run preset이나 `sanity` 결과는 디버깅/환경 확인용이며, 여러 글자에 대한 학습 성공 근거로
보고하지 않습니다.
대시보드에서는 `letters 장시간 학습 적용` 또는 `standard 전체 학습 적용`을 누르면 dry-run이
해제된 실제 학습 설정으로 바뀝니다. `final 최종 계획 적용`은 매우 긴 실행이므로 먼저 dry-run
계획을 만들도록 둔 뒤, 계획을 확인하고 직접 dry-run 체크를 해제해 실행합니다.

대시보드 Batch 탭의 기본 최종 평가 설정은 `평가 Action filter=corner_tangent_decel`,
`Teacher window m=0.15`입니다. 이는 학습된 residual action을 없애는 mock이 아니라,
corner 영향권 안에서 reference tangent 방향 감속 성분만 남기는 배포/평가 정책입니다.
raw policy 자체를 비교할 때만 `none`으로 바꾸고, 결과 이름에 raw ablation임을 남기세요.

앞선 M0/M1 weight는 batch runner가 trajectory/seed별로 이어받습니다. 중간에 끊기면 기본
`--resume` 동작으로 완료된 run을 재사용하고 남은 run만 이어서 실행합니다.
