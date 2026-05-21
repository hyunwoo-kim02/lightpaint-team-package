# Experiment Dashboard

로컬 대시보드는 Phase B 실험 설정, reward weight 튜닝, run 실행, 로그 확인,
결과 비교를 한 화면에서 처리하는 브라우저 UI입니다.

Python 표준 라이브러리만 사용합니다. 학습 subprocess는 대시보드를 실행한
동일한 Python executable로 실행되므로, 팀원마다 Windows 가상환경 경로가
달라도 각자 환경에서 그대로 사용할 수 있습니다.

## Launch

From `lightpaint-team-package/`:

```powershell
python -m src.train.experiment_dashboard --open
```

Windows helpers:

```powershell
tools\run_dashboard.bat
powershell -ExecutionPolicy Bypass -File tools\run_dashboard.ps1
```

The Windows helpers first try the project infra environment at
`..\mini_script\teams\infra\.venv\Scripts\python.exe`. If that file is not
present, they fall back to `python` from the current shell.

If your PyBullet training dependencies are installed in a specific virtual
environment, activate that environment first or call its Python executable
directly:

```powershell
.\.venv\Scripts\python.exe -m src.train.experiment_dashboard --open
```

## 실행 상태 확인

상단 상태 바에서 현재 선택된 run의 상태를 바로 확인합니다.

- `대기`: 아직 선택된 run이 없거나 실행 전입니다.
- `실행 중`: 학습 subprocess가 아직 종료되지 않았습니다. 로그는 약 1.5초
  간격으로 자동 갱신됩니다.
- `완료`: subprocess가 exit code 0으로 끝났고 `summary.json`이 생성된
  상태입니다.
- `실패`: subprocess가 nonzero exit code로 종료되었습니다. PyBullet
  dependency가 없는 경우에도 이 상태가 됩니다.
- `중지됨`: 대시보드에서 선택 run 중지를 요청한 상태입니다.

`Live Log`와 `선택 Run Log`에는 `src.train.train_phase_b`의 stdout/stderr가
표시됩니다. PyBullet dependency가 없으면 기존 preflight 메시지가 여기에
나옵니다.

## 설정 저장 방식

대시보드는 세 종류의 값을 구분합니다.

- 기준값: 서버 코드의 `DEFAULT_CONFIG`와 `src/env/reward_config.py` 값입니다.
- 자동 draft: 브라우저 `localStorage`에 저장되는 현재 편집값입니다. 페이지를
  새로 열면 마지막 draft를 자동 복원합니다.
- backup: preset/import/reset 직전 값입니다. preset 버튼으로 좋은 값을
  덮어썼더라도 `이전 설정 복원`으로 되돌릴 수 있습니다.

`Smoke preset`, `Short PPO preset`, `M2 강건성 preset` 버튼은 실행 버튼이
아닙니다. 설정값만 적용하고 toast로 피드백을 표시합니다. 실제 실행은
`Run 시작` 버튼을 눌러야 시작됩니다.

## 추천 시작값

처음 공유받은 팀원이 값을 고를 때는 아래 범위에서 시작하는 것을 권장합니다.
대시보드 각 입력 항목에 마우스를 올리면 같은 추천값과 영향 범위가 한국어
도움말로 표시됩니다.

| 항목 | 추천 시작값 | 용도 |
| --- | --- | --- |
| `max_steps` | 학습/평가에서는 비움, smoke는 2~90, square 전체 평가는 350~400 | episode 길이입니다. 비우면 reference 길이, `speed`, `settle_time`, `ctrl_freq`로 자동 계산합니다. |
| `speed` | 0.30~0.35 m/s, 기본 0.35 | reference 진행 속도입니다. 너무 빠르면 코너 이탈이 커질 수 있습니다. |
| `settle_time` | 2.0s, 최종 영상/평가는 2~3s | reference 종료 뒤 마지막 목표점 주변에서 안정화할 시간입니다. |
| `ctrl_freq` / `pyb_freq` | 30Hz / 240Hz | 환경 step 주기와 PyBullet physics 주기입니다. |
| `gui` | 기본 off, 디버깅/최종 확인 때만 on | PyBullet GUI 창을 rollout 확인용으로 띄웁니다. PyBullet은 프로세스당 GUI 1개만 허용하므로 PPO/BC 학습 env는 DIRECT로 실행되고, 평가 rollout 창이 순차적으로 열립니다. |
| `total_timesteps` | smoke 0, 빠른 sanity 2048~8192, M0 후보 1만~5만, M1/M2 5만~20만 | PPO 학습량입니다. 한 episode의 비행 시간과는 다릅니다. |
| `n_steps` / `batch_size` | 64/32부터 시작, 안정 학습은 256~512와 그 이하 batch | PPO rollout buffer와 minibatch 크기입니다. `batch_size <= n_steps`여야 합니다. |
| `learning_rate` | 3e-4, 불안정하면 1e-4 | PPO optimizer learning rate입니다. |
| `bc_epochs` / `bc_episodes` | smoke 0, 시작점은 2 / 4, 안정 warm start는 3~5 / 8~16 | teacher 기반 behavior cloning warm start입니다. |
| `teacher_gain` / `teacher_window_m` | 0.10 / 0.15m | 코너 접근 시 감속 teacher action의 강도와 적용 거리입니다. |
| `corner_window_m` | 0.18m, 보통 0.15~0.25m | corner metric/reward/teacher 영향권입니다. 너무 크면 직선 구간이 섞입니다. |

## 실험 로드맵

`실험 로드맵` 탭은 팀원이 같은 순서로 실험을 진행하도록 만든 안내/설정
화면입니다.

1. `0. 환경 / API smoke`: PyBullet dependency, reset/step, dashboard
   subprocess가 끝까지 도는지 확인합니다. 이 단계는 매우 짧아서
   `overall_pass=false`가 나올 수 있으며, 실행 상태가 `완료`이고
   `summary.json`이 있으면 환경 검증 목적은 통과입니다.
2. `1. M0 corner sanity`: 외란이 없는 M0에서 짧은 PPO 설정으로 Phase B
   residual velocity와 reward/metric 흐름이 끝까지 도는지 확인합니다.
   이 단계는 실행 sanity라 `overall_pass=false`가 나올 수 있습니다.
3. `2. M0 full corner 학습`: 외란 없는 full horizon에서 코너 감속과 path
   tracking 개선을 실제 학습 목표로 확인합니다. 이 단계부터는 이전 sanity
   산출물이 아니라 새로 저장된 `model_path`를 다음 단계에 넘깁니다.
4. `3. M1 외란 강건성`: M0 full train에서 얻은 `model_path`를 `load_model`로
   재사용해 약한 외란 조건에서 강건성을 확인합니다.
5. `4. M2 외란 강건성`: 같은 방식으로 M1 weight를 이어받아 더 강한
   외란에서 평가합니다.
6. `5. 최종 light painting`: letter 또는 drawn path에서 실제 라이트 페인팅
   결과, 영상, CSV artifact를 함께 확인합니다. 권장 설정은 `eval_only=true`라
   선택한 모델을 추가 업데이트 없이 평가합니다.

각 단계의 `이 단계 설정 적용` 버튼은 권장 설정만 채웁니다. 이전 단계의
weight를 쓰려면 Results 탭에서 좋은 run을 선택한 뒤 `선택 모델로 이어서 학습`
버튼을 누르거나, `load_model`에 직접 `.zip` 경로를 입력합니다. 최종 단계도
같은 방식으로 선택 모델을 재사용하며, `선택 모델 평가만`을 누르면 바로 평가
전용 설정으로 전환됩니다.

## Weight 재사용

완료된 run의 `summary.json`에는 저장된 PPO 모델 경로가
`artifacts.model_path`로 기록됩니다. 대시보드는 이 값을 Results 탭에 읽어와
다음 실험의 `load_model` 입력으로 넘길 수 있게 합니다.

- `선택 모델로 이어서 학습`: 선택한 run의 `model_path`를 `load_model`에 넣고
  `eval_only=false`, `reset_num_timesteps=false`로 설정합니다.
- `선택 모델 평가만`: 같은 모델을 넣되 `eval_only=true`로 설정해 추가
  업데이트 없이 평가 run을 만듭니다.
- `reset_num_timesteps`: curriculum처럼 앞선 weight를 계속 키우는 실험에서는
  보통 끕니다. 독립 실험처럼 timestep 카운터를 새로 시작하고 싶을 때만 켭니다.

## What It Controls

- Reference source: square, rendered letter, or drawn JSON path.
- Wind mode: M0, M1, or M2.
- Runtime settings: seed, speed, max steps, control frequency, device, video.
- PyBullet GUI display toggle. 기본값은 off이며, GUI는 평가 rollout 디버깅/최종 확인용입니다.
- PPO settings: timesteps, n_steps, batch size, epochs, learning rate, gamma,
  GAE lambda, entropy coefficient, clip range, and policy log std init.
- BC/teacher settings: teacher gain/window, BC epochs, episodes, batch size,
  learning rate, and nonzero sample weight.
- Reward settings from `src/env/reward_config.py`. 각 항목에 마우스를 올리면
  한국어 도움말이 표시됩니다.

## Output Layout

Runs are stored under:

```text
artifacts/dashboard/runs/<run-id>/
```

Each run includes:

- `dashboard_run.json`: dashboard metadata, command, Python path, config.
- `reward_overrides.json`: per-run reward/action constants.
- `run.log`: stdout/stderr from the training subprocess.
- `summary.json`: normal `src.train.train_phase_b` summary when the run
  reaches the summary-writing stage.
- Normal training artifacts under the run's `artifacts/` and `models/`
  folders.

## Reward Tuning Contract

The dashboard does not edit `src/env/reward_config.py`. It writes a per-run
`reward_overrides.json` file and sets `LIGHTPAINT_REWARD_CONFIG` for the
training subprocess. `src/env/reward_config.py` loads that file at import time,
validates keys and finite numeric values, and exposes the active override path
inside the training summary.

This keeps the baseline source stable while making team experiments
reproducible.

## LED Reward 해석

현재 Phase B LED는 scripted reference 위에서 residual을 더하는 방식입니다.
나중에 full RL LED로 확장해도 reward를 재사용할 수 있도록 LED 관련 component는
아래처럼 분리해서 기록됩니다.

- `r_paint_outcome`: 실제 painting 결과 기반 보상입니다.
  `r_led_target + r_led_off + r_repaint`로 구성되며 scripted LED를 모방했는지와
  무관합니다. full RL LED에서도 그대로 유지해야 하는 핵심 항목입니다.
- `r_led_prior`: 정확히는 one-sided LED miss prior입니다.
  현재는 `r_led_miss = -W_LED_MISS * max(0, led_ref - brightness)`와 같은 값입니다.
  즉 scripted LED reference보다 어두운 false-off만 벌하고, `led_ref=0`인데 켜는
  false-on은 여기서 직접 벌하지 않습니다. false-on은 `r_led_off`, `r_repaint`,
  `r_paint_outcome`처럼 실제 painting 결과 항에서 판단합니다. residual 단계에서는
  학습을 빠르게 하지만, full RL LED 단계에서는 `W_LED_MISS`를 0에 가깝게 낮춰야
  정책이 scripted LED를 단순 모방하지 않습니다.
- `r_led_flicker`: LED brightness가 step 사이에서 급격히 변하는 것을 억제합니다.
  residual/full RL 양쪽에서 재사용 가능합니다.
- `r_led_scripted`: 이전 분석 코드와의 호환을 위해 남긴 legacy aggregate입니다.
  새 분석에서는 `r_paint_outcome`과 `r_led_prior`를 분리해서 보는 것을 권장합니다.

따라서 LED curriculum은 보통 `W_LED_MISS`를 유지한 residual 단계에서 시작하고,
full RL LED에 가까워질수록 `W_LED_MISS`를 낮추면서 `W_NEW_TARGET`,
`W_OFF_TARGET`, `W_REPAINT`, `W_LED_FLICKER` 중심으로 판단합니다. symmetric scripted
LED imitation이 필요한 별도 실험에서는 `abs(led_ref - brightness)` 형태의 새 prior를
추가하고 기본 reward와 분리하는 것이 안전합니다.

## Interpreting Results

The Results tab reads `summary.json` and highlights the existing success gate:

- `corner_path_rmse_reduced_20pct`:
  `phaseB_trained.corner_path_rmse_m <= phaseB_zero.corner_path_rmse_m * 0.80`
- `corner_speed_reduced_5_to_20pct`:
  `phaseB_zero.corner_mean_speed_mps * 0.80 <= phaseB_trained.corner_mean_speed_mps <= phaseB_zero.corner_mean_speed_mps * 0.95`
- `straight_path_rmse_not_worse_10pct`:
  `phaseB_trained.straight_path_rmse_m <= phaseB_zero.straight_path_rmse_m * 1.10`
- `painting_coverage_kept_90pct`:
  `phaseB_trained.painted_pixel_coverage >= phaseB_zero.painted_pixel_coverage * 0.90`
- `painting_precision_kept_90pct`:
  `phaseB_trained.painted_pixel_precision >= phaseB_zero.painted_pixel_precision * 0.90`
- `painting_iou_kept_90pct`:
  `phaseB_trained.painted_pixel_iou >= phaseB_zero.painted_pixel_iou * 0.90`
- `off_target_ratio_not_worse_5pct`:
  `phaseB_trained.off_target_pixel_ratio <= phaseB_zero.off_target_pixel_ratio + 0.05`
- `overall_pass`: 위 기준을 모두 만족해야 true입니다.

대시보드는 trained rollout의 주요 light-painting 지표도 함께 보여줍니다:
painted pixel coverage/recall, precision, IoU, Dice/F1, off-target ratio,
tracking RMSE, corner path RMSE, straight path RMSE, corner speed ratio,
reward sum, crash flag, `mean_r_paint_outcome`, `mean_r_led_prior`,
`mean_r_led_flicker`.
Painting mask는 `summary.json`의 `evaluation.painting_threshold` 기준으로
생성되며, 현재 기본값은 cumulative intensity `> 0.3`입니다.

`complete`와 `overall_pass`는 다릅니다. `complete`는 subprocess가 정상 종료된
상태이고, `overall_pass`는 실험 목표를 달성했다는 정량 기준입니다. 팀원 간
공유 시에는 `dashboard_run.json`, `reward_overrides.json`, `summary.json`,
`models/*.zip`을 함께 남기면 같은 조건을 다시 실행하거나 이어서 학습할 수
있습니다.
