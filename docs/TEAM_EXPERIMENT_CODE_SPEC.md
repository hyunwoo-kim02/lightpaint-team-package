# Light Painting Drone Experiment Definition

## 1. 실험 목표

이 프로젝트의 실험 목표는 드론이 주어진 경로를 따라 이동하면서 LED로 목표 이미지를 그리도록 학습시키는 것이다.

실험에서 다루는 경로 입력은 세 가지다.

| 입력 | 설명 |
|---|---|
| `square` | 사각형 기준 경로. 코너 감속과 기본 추종 성능 확인에 사용한다. |
| `letter` | 글자 또는 짧은 문자열에서 생성한 경로. 예: `L`, `R`, `DG`, `CAT`. |
| `drawn` | `tools/draw_path.html`에서 직접 그려 저장한 JSON 경로. |

실험에서 다루는 외란 조건은 세 가지다.

| mode | 의미 |
|---|---|
| `M0` | 외란 없음 |
| `M1` | 약한 외란 |
| `M2` | 강한 외란 |

최종적으로 확인해야 할 것은 다음이다.

| 목표 | 확인할 내용 |
|---|---|
| 경로 추종 | 실제 드론 위치가 reference path를 얼마나 잘 따라가는가 |
| 속도 조절 | 코너, 곡선, stroke 연결부에서 필요한 감속이 생기는가 |
| LED 제어 | LED가 목표 stroke에서 켜지고 불필요한 구간에서 꺼지는가 |
| 외란 대응 | M1/M2에서 성능이 얼마나 유지되거나 개선되는가 |
| 경로 일반화 | square에서만 되는 것이 아니라 letter/drawn에도 적용되는가 |

## 2. 학습 제어 단계의 의미

이 실험의 학습 제어 단계는 기존 PID 제어를 완전히 버리는 단계가 아니다.

기본 위치 추종은 PID가 수행하고, 강화학습 정책은 PID가 따라갈 목표 속도와 LED 밝기에 보정값을 더한다.

| 항목 | 의미 |
|---|---|
| 기본 제어 | reference 위치와 속도를 따라가기 위한 PID 제어 |
| 학습 제어 | PID 목표 속도에 더할 속도 보정값과 LED 보정값을 출력하는 정책 |
| 속도 보정값 | `[delta_v_x, delta_v_y, delta_v_z]` |
| LED 보정값 | `delta_LED` |
| 최종 action | `[delta_v_x, delta_v_y, delta_v_z, delta_LED]` |

즉, 이 단계의 목표는 드론을 처음부터 직접 조종하는 것이 아니라, 기존 PID 기준 움직임 위에 필요한 보정을 학습하는 것이다.

## 3. 현재 확실한 부분

| 항목 | 상태 |
|---|---|
| 학습 제어 action 의미 | `[delta_v_x, delta_v_y, delta_v_z, delta_LED]` |
| 학습 제어 action space | 4차원 보정 action |
| reference 내부 형식 | `LightPaintRef` |
| 지원 reference | `square`, `letter`, `drawn JSON` |
| `draw_path.html` JSON 입력 | 사용 가능 |
| 경로 크기 조절 | `path_scale` 사용 |
| 지원 mode | `M0`, `M1`, `M2` |
| 기본 학습 진입점 | `src.train.train_phase_b` |
| 저장 weight 불러오기 | `--load-model` 사용 |
| 학습 없이 평가 | `--eval-only` 사용 |
| 불러온 weight에서 추가 학습 | `--load-model`과 `--total-timesteps` 사용 |
| 기본 산출물 | metrics CSV, trajectory CSV, tracking PNG, summary JSON |

## 4. 아직 불확실한 부분

| 항목 | 불확실한 점 |
|---|---|
| PPO 안정성 | PPO 학습이 BC 또는 zero-action 기준보다 항상 좋아지는지 확인되지 않았다. |
| M0 to M1/M2 전이 성능 | M0에서 학습한 policy/checkpoint가 M1/M2에서 얼마나 좋은지는 확인해야 한다. |
| reward 설계 | 현재 reward가 모든 경로와 모든 외란 조건에서 원하는 행동을 유도하는지 불확실하다. |
| LED 학습 | LED 보정값이 실제 painting 품질 개선으로 이어지는지 추가 검증이 필요하다. |
| letter/drawn 성능 기준 | square와 같은 기준을 그대로 적용할지, 별도 기준이 필요한지 결정해야 한다. |

## 5. 코드 구성

| 파일 | 역할 |
|---|---|
| `src/train/train_phase_b.py` | 속도/LED 보정 정책 학습 실행 진입점 |
| `src/train/train_phase_b_m0_corner.py` | PPO/BC 학습, rollout, metric 저장 |
| `src/train/reference_factory.py` | `square`, `letter`, `drawn` reference 생성 |
| `src/train/train_phase_a_pid.py` | 기존 PID-only 기준 실행 |
| `src/train/visualize_flight.py` | 기존 시각화 산출물 생성 |
| `src/env/light_paint_aviary_pyb.py` | PyBullet 환경, DSLPIDControl, URDF/asset, external force, action 처리 |
| `src/env/light_paint_aviary_standalone.py` | PyBullet 없는 환경에서 계약을 확인하는 fallback env |
| `src/env/reward_config.py` | reward weight, action scale, LED threshold, reward key 정의 |
| `src/env/reward_terms.py` | PyBullet/fallback이 공유하는 reward component 계산 |
| `src/env/lightpaint_ref.py` | 경로를 `LightPaintRef`로 변환 |
| `src/env/wind_modes.py` | M0/M1/M2 외란 정의 |
| `src/train/experiment_dashboard.py` | 브라우저 기반 실험 설정/실행/결과 확인 대시보드 |
| `tools/draw_path.html` | 사용자 직접 경로 JSON 생성 |

## 6. 수정해야 하는 부분

실험에서 주로 수정할 부분은 다음이다.

| 수정 대상 | 파일 | 목적 |
|---|---|---|
| reward weight/action scale | 대시보드 또는 `src/env/reward_config.py` | 원하는 행동의 보상 강도와 Phase B residual scale 조절 |
| reward component | `src/env/reward_terms.py` | 경로 추종, 감속, LED, 외란 대응 항목 수정 |
| PPO 설정 | `src/train/train_phase_b_m0_corner.py` | 학습 안정성 조정 |
| BC teacher policy | `src/train/train_phase_b_m0_corner.py` | 초기 행동 가이드 수정 |
| action filter | `src/train/train_phase_b_m0_corner.py` | 비정상 보정 action 제한 |
| curriculum | `src/train/train_phase_b_m0_corner.py` | M0에서 M1/M2로 확장하는 절차 설계 |

reward 실험에서는 `src/env/light_paint_aviary_pyb.py`와 `src/env/light_paint_aviary_standalone.py`의 reward 계산을 직접 갈라서 수정하지 않는다. 두 환경은 `src/env/reward_config.py`와 `src/env/reward_terms.py`를 공유해야 한다. 단순 weight/action-scale 튜닝은 먼저 `python -m src.train.experiment_dashboard --open` 대시보드에서 run별 `reward_overrides.json`으로 수행한다. PyBullet env 파일은 PyBullet state 읽기, DSLPIDControl, URDF/asset setup, `_physics()`, `applyExternalForce` 같은 backend 로직을 수정할 때만 건드린다. Standalone env 파일은 PyBullet 없이 확인하는 simple physics와 fallback PID 계약을 수정할 때만 건드린다.

실험 중 유지해야 하는 부분은 다음이다.

| 유지 대상 | 이유 |
|---|---|
| 학습 제어 action 의미 | action 의미가 바뀌면 기존 metric과 모델 비교가 불가능하다. |
| `LightPaintRef` contract | 모든 reference 입력이 같은 env로 들어가기 위한 공통 형식이다. |
| 기존 metric column 이름 | 실험 결과 비교를 유지하기 위해 필요하다. |
| M0/M1/M2 mode 이름 | 외란 조건의 의미를 고정하기 위해 필요하다. |

## 7. 실험 과정

### 7.1 경로 선택

경로는 하나를 선택한다.

| 경로 | 주요 인자 |
|---|---|
| `square` | `--trajectory square --square-side 0.8` |
| `letter` | `--trajectory letter --label L` |
| `drawn` | `--trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json` |

경로 크기는 `--path-scale`로 조절한다.

```powershell
python -m src.train.train_phase_b --trajectory drawn --drawn-path data\drawn_paths\user\user_drawn_path.json --path-scale 0.8
```

### 7.2 baseline 실행

M0에서는 다음을 비교한다.

| baseline | 의미 |
|---|---|
| 기존 PID-only | 학습 보정 없이 PID만 사용하는 기준 |
| 보정 없는 학습 제어 | 속도/LED 보정 action을 0으로 둔 기준 |
| 학습된 보정 제어 | 학습된 정책이 속도/LED 보정값을 출력하는 경우 |

M1/M2에서는 기존 PID-only 실행을 baseline으로 쓰지 않는다. 현재 PID-only 기준 실행은 M0 전용이다.

M1/M2에서는 다음을 비교한다.

| baseline | 의미 |
|---|---|
| 보정 없는 학습 제어 | 외란이 있는 상태에서 속도/LED 보정 action이 없는 기준 |
| 학습된 보정 제어 | 외란이 있는 상태에서 학습된 정책이 보정값을 출력하는 경우 |

### 7.3 학습 실행

기본 실행:

```powershell
python -m src.train.train_phase_b --trajectory square --square-side 0.8 --wind-mode M0 --seed 7
```

짧은 smoke:

```powershell
python -m src.train.train_phase_b --trajectory letter --label L --wind-mode M2 --max-steps 5 --total-timesteps 0
```

PPO 실험:

```powershell
python -m src.train.train_phase_b --trajectory square --square-side 0.8 --wind-mode M0 --seed 7 --total-timesteps 1024
```

BC warm-start 실험:

```powershell
python -m src.train.train_phase_b --trajectory square --square-side 0.8 --wind-mode M0 --seed 7 --total-timesteps 0 --bc-epochs 300 --bc-episodes 2 --bc-batch-size 128 --bc-learning-rate 0.001 --bc-nonzero-weight 100 --trained-action-filter corner_tangent_decel
```

## 8. 실험 인자 기준

### 8.1 경로 및 환경 인자

| 인자 | 의미 | 시작값 | 조정 방향 | 주의점 |
|---|---|---:|---|---|
| `--trajectory` | 사용할 reference 종류 | `square` | `letter`, `drawn`으로 확장 | 먼저 `square`에서 reward 동작을 확인한다. |
| `--square-side` | 사각형 한 변 길이 | `0.8` | 너무 쉬우면 증가, 너무 불안정하면 감소 | `path_scale`과 곱해져 실제 크기가 결정된다. |
| `--label` | 글자 또는 문자열 | `L` | `R`, `DG`, `CAT` 등으로 확장 | 복잡한 문자열일수록 경로가 길고 어려워진다. |
| `--drawn-path` | 직접 그린 JSON 경로 | `data/drawn_paths/user/user_drawn_path.json` | 실험 경로 파일로 변경 | `tools/draw_path.html` 출력 형식을 사용한다. |
| `--path-scale` | 전체 경로 크기 배율 | `1.0` | `0.6~1.2`에서 시작 | 작을수록 쉬우나 너무 작으면 의미 있는 조향이 줄어든다. |
| `--max-waypoints` | 경로 waypoint 수 제한 | drawn은 JSON 설정, letter는 `240` | 필요 시 명시 | CLI에 쓰면 JSON 안의 값보다 우선한다. |
| `--smooth-ref` / `--no-smooth-ref` | reference smoothing 사용 여부 | drawn은 JSON 설정, letter는 켜짐 | 경로가 계단식이면 켬, 원본 stroke 검증이면 끔 | CLI에 쓰면 JSON 안의 값보다 우선한다. |
| `--smooth-window-m` | smoothing 거리 창 | drawn은 JSON 설정, letter는 `0.08` | 경로가 거칠면 증가 | 너무 크면 코너/획 형태가 둔해진다. |
| `--speed` | reference 진행 속도 | `0.35` | `0.25~0.45` 범위에서 조정 | 빠를수록 코너 오차와 외란 영향이 커진다. |
| `--wind-mode` | 외란 조건 | `M0` | `M0 -> M1 -> M2` 순서 | M1/M2는 M0 성능 확인 후 사용한다. |
| `--seed` | 초기 상태와 난수 고정 | `7` | `7, 13, 21` 등 복수 seed | 단일 seed 결과만으로 결론 내리지 않는다. |
| `--max-steps` | rollout 최대 step | 자동 | smoke는 `3~10`, full은 자동 | 학습 검증은 full rollout을 사용한다. |
| `--settle-time` | 경로 종료 후 추가 시간 | `2.0` | 보통 유지 | 너무 짧으면 끝부분 painting/추종 확인이 어렵다. |
| `--led-always-on` / `--no-led-always-on` | reference LED를 항상 켤지 여부 | 꺼짐 | LED schedule을 무시한 추종 실험에서만 켬 | 기본값은 stroke의 LED ON/OFF와 connector OFF를 유지한다. |

### 8.2 학습 인자

| 인자 | 의미 | 시작값 | 조정 방향 | 주의점 |
|---|---|---:|---|---|
| `--total-timesteps` | PPO 학습 step 수 | smoke `128`, 실험 `1024+` | 안정하면 증가 | 짧은 smoke는 성능 판단용이 아니다. |
| `--n-steps` | PPO rollout buffer 길이 | `64` | `64~256` | 너무 작으면 advantage 추정이 불안정할 수 있다. |
| `--batch-size` | PPO minibatch 크기 | `32` | `32~128` | `n_steps`와 나누어 떨어지는 값이 좋다. |
| `--n-epochs` | PPO update 반복 수 | `2` | `2~5` | 너무 크면 기존 행동이 급격히 망가질 수 있다. |
| `--learning-rate` | PPO 학습률 | `3e-4` | 불안정하면 `1e-4`로 감소 | reward가 흔들리면 먼저 낮춘다. |
| `--gamma` | 장기 reward 할인율 | `0.99` | 보통 유지 | 짧은 경로에서도 과도하게 낮추지 않는다. |
| `--gae-lambda` | advantage smoothing | `0.95` | 보통 유지 | PPO 기본 안정성에 영향이 크다. |
| `--ent-coef` | exploration 보상 | `0.0` | 필요 시 소폭 증가 | 현재는 과한 탐색보다 안정성이 우선이다. |
| `--clip-range` | PPO policy update 제한 | `0.2` | 불안정하면 `0.1` | policy가 급변하면 낮춘다. |
| `--log-std-init` | 초기 action 분산 | `-2.0`, 안정 실험 `-3.0` | action이 크면 더 낮춤 | 보정 action이 과하면 경로가 망가진다. |
| `--device` | 학습 장치 | `cpu` | GPU 가능 시 변경 | 현재 smoke와 짧은 실험은 CPU로 충분하다. |
| `--load-model` | 저장된 PPO weight 경로 | 없음 | 기존 `.zip`을 지정 | 다른 mode나 경로에서 재활용할 때 사용한다. |
| `--eval-only` | 학습 없이 평가만 수행 | 꺼짐 | weight 평가 시 사용 | `--load-model`과 함께 사용한다. |
| `--reset-num-timesteps` | 이어 학습 시 timestep 카운터 초기화 | 꺼짐 | 새 run처럼 기록할 때 켬 | weight 자체를 초기화하는 옵션은 아니다. |

### 8.3 BC warm-start 인자

| 인자 | 의미 | 시작값 | 조정 방향 | 주의점 |
|---|---|---:|---|---|
| `--bc-epochs` | teacher 행동 모방 학습 epoch | `0`, 안정 실험 `300` | teacher가 유효하면 증가 | `0`이면 BC 없이 PPO만 사용한다. |
| `--bc-episodes` | BC 데이터 수집 episode 수 | `2~4` | 데이터 부족하면 증가 | 많을수록 느려진다. |
| `--bc-batch-size` | BC minibatch 크기 | `128` | `64~256` | 데이터 수에 비해 너무 크면 효과가 줄 수 있다. |
| `--bc-learning-rate` | BC 학습률 | `0.001` | 불안정하면 감소 | policy 초기값에 직접 영향을 준다. |
| `--bc-nonzero-weight` | 보정 action이 0이 아닌 sample 가중치 | `100` | teacher 행동이 약하면 증가 | 너무 크면 특정 구간 행동만 과하게 배운다. |
| `--teacher-gain` | teacher 감속 강도 | `0.10` | 감속 부족하면 증가 | 너무 크면 코너 전 속도가 과도하게 줄어든다. |
| `--teacher-window-m` | 코너 전 teacher 적용 거리 | `0.15` | 코너 대응이 늦으면 증가 | 경로가 짧으면 너무 크게 잡지 않는다. |
| `--trained-action-filter` | 학습 action 제한 방식 | `none`, 안정 실험 `corner_tangent_decel` | 코너 감속 실험에서는 filter 사용 | filter는 행동 공간을 제한하므로 목적과 맞을 때만 사용한다. |

### 8.4 인자 선택 순서

| 단계 | 권장 선택 |
|---|---|
| 1 | `square`, `M0`, `speed=0.35`, `path_scale=1.0`으로 reward 동작 확인 |
| 2 | BC warm-start로 원하는 행동이 나오는지 확인 |
| 3 | PPO를 짧게 붙였을 때 BC 행동이 망가지지 않는지 확인 |
| 4 | 같은 설정을 `letter`와 `drawn`에 적용 |
| 5 | M0 checkpoint 또는 설정을 M1로 확장 |
| 6 | M2에서 reward와 curriculum을 다시 조정 |

### 8.5 weight 재활용 인자

| 목적 | 명령 구성 |
|---|---|
| 저장된 weight를 같은 경로에서 평가 | `--load-model <model.zip> --eval-only` |
| 저장된 weight를 다른 경로에서 평가 | `--load-model <model.zip> --eval-only --trajectory letter --label L` |
| 저장된 weight를 M1에서 평가 | `--load-model <model.zip> --eval-only --wind-mode M1` |
| 저장된 weight에서 추가 학습 | `--load-model <model.zip> --total-timesteps 1024` |
| M0 weight를 M1에서 추가 학습 | `--load-model <m0_model.zip> --wind-mode M1 --total-timesteps 1024` |

weight를 불러온 실험은 `summary.json`에 `load_model`과 `source_model_path`가 기록되어야 한다.

## 9. 검증 기준

기본 검증은 같은 reference, 같은 seed, 같은 mode에서 baseline 대비 성능이 개선되는지 확인한다.

| metric | 의미 |
|---|---|
| `tracking_rmse_m` | 시간 기준 reference 위치와 실제 위치의 RMSE |
| `corner_path_rmse_m` | heading-change threshold를 통과한 급격한 방향 변화 구간의 path distance RMSE |
| `straight_path_rmse_m` | 직선 또는 완만한 구간의 path distance RMSE |
| `corner_mean_speed_mps` | 코너 구간 평균 속도 |
| `painted_pixel_coverage` / `painted_pixel_recall` | 목표 mask 중 실제로 칠해진 비율 |
| `painted_pixel_precision` | 칠해진 픽셀 중 목표 mask 안에 들어간 비율 |
| `painted_pixel_iou` | 칠해진 mask와 목표 mask의 intersection-over-union |
| `painted_pixel_dice` | 칠해진 mask와 목표 mask의 Dice/F1 overlap |
| `off_target_pixel_ratio` | 칠해진 픽셀 중 목표 mask 밖에 있는 비율 |
| `reward_sum` | episode 전체 reward 합 |
| `crash` | 충돌 또는 조기 종료 여부 |

Painting mask 기준 threshold는 `summary.json`의 `evaluation.painting_threshold`에
기록한다. 현재 기본값은 cumulative intensity `> 0.3`이다.

M0에서 우선 확인할 기준:

| 조건 | 목표 |
|---|---|
| 경로 오차 | 학습된 보정 제어가 보정 없는 기준보다 낮아야 한다. |
| 코너 속도 | 코너 구간에서 필요한 감속이 생겨야 한다. |
| 직선 구간 | 직선 구간 오차가 크게 악화되면 안 된다. |
| painting | coverage/recall, precision, IoU가 baseline 대비 크게 떨어지면 안 되고 off-target ratio가 크게 늘면 안 된다. |
| crash | 없어야 한다. |

M1/M2에서 확인할 기준:

| 조건 | 목표 |
|---|---|
| 외란 대응 | 학습된 보정 제어가 보정 없는 기준보다 경로 오차를 줄이는지 확인 |
| 정책 전이 | M0에서 얻은 policy/checkpoint가 성능 유지에 도움이 되는지 확인 |
| reward 적합성 | 외란 관련 reward가 실제 보정 행동을 만들고 있는지 확인 |

## 10. 시각화 기준

시각화는 기존 artifact 형식을 사용한다.

| 산출물 | 확인할 내용 |
|---|---|
| tracking PNG | reference path와 실제 경로 차이 |
| trajectory CSV | step별 위치, reference, brightness, RPM |
| metrics CSV | run 단위 성능 요약 |
| corner diagnostics CSV | 급격한 방향 변화 구간에서 속도와 오차 변화 |
| summary JSON | 실험 설정, metric, artifact 경로 |

시각화에서 반드시 확인할 것:

| 항목 | 확인 이유 |
|---|---|
| 경로가 끝까지 수행되는가 | 짧게 끊긴 rollout이면 학습 결과로 볼 수 없다. |
| reference와 actual이 같은 scale인가 | `path_scale` 적용 오류 확인 |
| 코너/곡선에서 속도가 줄어드는가 | 목표 행동 확인 |
| LED가 목표 stroke와 맞는가 | painting 목표 확인 |
| M1/M2에서 경로가 얼마나 밀리는가 | 외란 영향 확인 |

## 11. 결과 기록

각 실험 결과에는 다음을 기록한다.

| 항목 | 예시 |
|---|---|
| reference | `square`, `letter:L`, `drawn:user_drawn_path.json` |
| path_scale | `0.8` |
| wind_mode | `M0`, `M1`, `M2` |
| seed | `7` |
| total_timesteps | `1024` |
| reward 수정 내용 | `corner speed penalty 증가` |
| PPO 설정 | `n_steps=64, batch_size=32, learning_rate=3e-4` |
| baseline | `보정 없는 학습 제어` |
| 비교 결과 | RMSE, coverage/recall, precision, IoU, off-target ratio, reward, crash |
| artifact 경로 | `artifacts/.../summary.json` |

## 12. 다음 실험 목표

우선순위는 다음 순서로 둔다.

| 순서 | 목표 |
|---|---|
| 1 | M0 square에서 코너 감속과 경로 오차 개선을 안정적으로 재현 |
| 2 | M0 letter에서 같은 reward가 유효한지 확인 |
| 3 | M0 drawn JSON에서 같은 reward가 유효한지 확인 |
| 4 | M0 checkpoint를 M1로 가져가 성능 유지 여부 확인 |
| 5 | M1 reward/curriculum 수정 |
| 6 | M2로 확장 |

현재 핵심 질문은 다음이다.

| 질문 | 의미 |
|---|---|
| M0에서 reward가 원하는 행동을 만들 수 있는가 | 기본 학습 가능성 |
| M0에서 얻은 policy를 M1/M2에 재사용할 수 있는가 | 학습 효율 |
| LED 보정값을 학습시키는 것이 painting 품질을 높이는가 | 최종 목표와의 연결 |
| letter/drawn에서도 같은 metric으로 비교 가능한가 | 경로 일반화 |
