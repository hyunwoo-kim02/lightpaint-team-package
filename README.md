# Lightpaint Team Package

현재 구현의 중심 흐름:

> square/letter/drawn 경로 -> DATT 방식 `LightPaintRef` -> PyBullet CF2X -> `DSLPIDControl` -> 학습 제어 action으로 velocity/LED 보정 -> M0/M1/M2 tracking 및 painting 검증

이 패키지는 PID-only 기준선과 학습 제어 실험을 같은 reference, 같은 산출물 형식으로 비교하기 위한 작업 공간이다. 현재 유효한 실험 범위는 사각형 경로, 이미지 기반 글자 경로, `tools/draw_path.html`에서 export한 직접 그린 경로, 그리고 M0/M1/M2 mode다.

2026-05-13 이후 업데이트: 이미지에서 생성한 글자 경로는 `LightPaintRef`에 들어가기 전에 arc-length resampling과 smoothing을 거친다. 이렇게 하면 PID/M0 기준선이 픽셀 격자 형태의 계단식 경로를 그대로 따라가려는 문제가 줄어든다. 글자 및 알파벳 rollout도 사각형 gate와 같은 시각화 산출물 묶음을 생성해야 한다: tracking CSV/metrics, 3D PNG, tracking PNG, frames, MP4, GIF, summary PNG, reference-path diagnostic PNG.

## 주요 파일

| 경로 | 역할 |
|---|---|
| `src/env/lightpaint_ref.py` | DATT 방식 `pos/vel/acc/yaw(t)` reference adapter 및 사각형 waypoint source. |
| `src/env/light_paint_aviary_pyb.py` | `DSLPIDControl`을 사용하는 활성 PyBullet CF2X 환경. |
| `src/env/pybullet_setup.py` | 한글/비ASCII 경로에서 PyBullet asset을 읽기 위한 mirror 우회 처리. |
| `src/train/train_phase_a_pid.py` | 현재 PID-only rollout 및 시각화 entrypoint. |
| `src/train/batch_alphabet_pid.py` | fixed-z x-y 및 z-motion x-z mode에 대한 A-Z PID batch rollout. |
| `src/train/visualize_flight.py` | PID rollout에서 사용하는 시각화 helper. |
| `src/env/reward_config.py` | 학습 제어 reward weight, action scale, LED threshold 정의. |
| `src/env/reward_terms.py` | PyBullet env와 fallback env가 공유하는 reward component 계산. |
| `tools/draw_path.html` | 브라우저에서 직접 stroke를 그리고 JSON으로 export하는 canvas 도구. |
| `tests/test_lightpaint_ref.py` | reference 계약 테스트. |
| `tests/test_phase_a_pyb_env.py` | 활성 PyBullet 환경 smoke 테스트. |

## 보관된 레거시 경로

기존 PPO-first 지시문과 레거시 코드는 아래로 이동되어 있다.

`../archive/2026-05-13_pre_square_pid_cleanup/`

보관된 `train_w1.py`, `eval_w1.py`, 예전 `mini_script/teams/*/code`, 예전 Claude prompt는 현재 구현 기준으로 사용하지 않는다.

## 현재 Gate

DATT 방식 reference adapter는 현재 아래 흐름으로 연결되어 있다.

`reference path -> LightPaintRef -> LightPaintAviaryPyB -> DSLPIDControl -> velocity/LED 보정 action`

controller는 원본 입력 형식이 아니라 시간 기준으로 parameterize된 reference를 사용한다. 학습 제어 단계에서는 PID가 따라갈 기준 속도와 LED 명령에 보정 action을 더해 결과를 비교한다.

## 빠른 확인

`lightpaint-team-package/`에서 실행:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m pytest tests/test_lightpaint_ref.py tests/test_phase_a_pyb_env.py -q
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_a_pid --trajectory square --wind-mode M0 --seed 42 --led-always-on
```

PyBullet 학습/rollout에는 `pybullet`, `pybullet_data`, `gym_pybullet_drones`, `stable_baselines3`가 필요하다. `src.train.train_phase_b`가 `런타임 의존성 오류`를 출력하면 현재 Python 환경에 학습 실행 의존성이 없는 상태이므로 먼저 설치 상태를 맞춘다.

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyBullet이 없는 환경에서는 reference, 설정 검증, 시각화 helper, standalone 로직처럼 PyBullet-free 테스트만 먼저 실행한다.

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m pytest -q tests\test_lightpaint_ref.py tests\test_reference_factory.py tests\test_train_phase_b_config.py tests\test_visualize_flight.py tests\test_lightpaint_geometry.py -p no:cacheprovider --basetemp artifacts\pytest_tmp_fallback
```

PyBullet 설치 환경에서는 전체 테스트와 짧은 학습 entry smoke를 실행한다.

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts\pytest_tmp_run -p no:cacheprovider
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_b --trajectory square --square-side 0.8 --wind-mode M0 --seed 7 --max-steps 3 --total-timesteps 0 --n-steps 8 --batch-size 8 --output-dir artifacts\validation_phase_b_smoke
```

PyBullet 창을 직접 보며 확인해야 할 때만 `--gui`를 추가한다. GUI는 평가 rollout 확인용이며, PyBullet의 프로세스당 GUI 1개 제한 때문에 PPO/BC 학습 env는 DIRECT로 유지된다. 대량 실험에서는 기본값인 off를 유지한다.

검수 산출물은 trajectory CSV, metrics CSV, corner-error CSV, 3D/summary/refpath/tracking plot, frames, GIF, PyBullet-view MP4를 포함한다. 최종 목표 검수에서는 `painted_pixel_coverage`, off-target 억제, tracking RMSE, crash 여부와 함께 LED split metric인 `mean_r_paint_outcome`, `mean_r_led_prior`, `mean_r_led_flicker`를 같이 확인한다. 여기서 `mean_r_led_prior`는 full scripted LED imitation이 아니라 one-sided LED miss prior이며, false-on은 painting outcome 항으로 판단한다.

## 직접 그린 경로

브라우저에서 `tools/draw_path.html`을 열어 하나 이상의 stroke를 그리고 JSON 파일로 export할 수 있다. 기본적으로 각 stroke는 LED ON으로 처리되며, stroke 사이를 연결하는 connector motion은 LED OFF로 자동 삽입된다. export된 JSON에는 `path_scale`을 포함할 수 있고, CLI의 `--path-scale`은 JSON 파일을 직접 수정하지 않고 같은 거리 배율을 적용한다.

직접 그린 JSON 경로를 사용하는 rollout 예시:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_a_pid --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --wind-mode M0 --seed 42
```

직접 그린 경로 파일은 `data/drawn_paths/user/` 및 `data/drawn_paths/examples/` 아래에 정리한다.

## 학습 제어 실험

공통 학습 명령은 `src.train.train_phase_b`를 사용한다. 이 명령은 PID 기준선과 같은 reference 입력을 받으며 M0, M1, M2 mode에서 실행할 수 있다.

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_b --trajectory square --square-side 0.8 --path-scale 1.0 --wind-mode M0 --seed 7 --total-timesteps 128
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_b --trajectory letter --label L --path-scale 0.8 --wind-mode M1 --seed 7 --total-timesteps 128
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.train_phase_b --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --path-scale 0.8 --wind-mode M2 --seed 7 --total-timesteps 128
```

smoke check에서는 `--total-timesteps`를 작게 둔다. full experiment에서는 reference source, `path_scale`, wind mode, seed, reward 변경 내용, PPO 설정, output directory를 기록해야 한다.

reward 실험에서 주로 수정할 위치는 `src/env/reward_config.py`와 `src/env/reward_terms.py`다. PyBullet env와 standalone fallback env는 같은 reward 계산을 공유하므로, reward weight나 component를 각 env 파일에서 따로 수정하지 않는다. PyBullet 물리, DSLPIDControl, URDF/asset, `applyExternalForce` 수정은 `src/env/light_paint_aviary_pyb.py`에서 하고, PyBullet 없는 fallback simple physics 수정은 `src/env/light_paint_aviary_standalone.py`에서 한다.

코드 정의와 실험 규칙은 `docs/TEAM_EXPERIMENT_CODE_SPEC.md`에 정리되어 있다.

## 알파벳 Batch

`lightpaint-team-package/`에서 실행:

```powershell
..\mini_script\teams\infra\.venv\Scripts\python.exe -m src.train.batch_alphabet_pid --letters ABCDEFGHIJKLMNOPQRSTUVWXYZ --modes xy_fixed_z xz_z_motion --output-dir artifacts\alphabet_pid --snapshot-count 4 --video-frames 10 --video-fps 5
```

산출물은 `artifacts/alphabet_pid/` 아래에 저장된다. 기본적으로 각 letter/mode run은 사각형 기준 실험과 같은 시각화 산출물 묶음을 생성한다. 짧은 debug MP4가 의도된 경우에만 `--quick-video`를 사용한다.
