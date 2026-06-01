# 발표 결과 재현 패키지

이 폴더는 최종 발표 슬라이드의 결과 화면을 재현하기 위한 최소 데이터, 평가 로그, 웨이트, 스크립트를 묶은 패키지입니다.

대상 슬라이드:

- 9: DG 경로의 M0/M1/M2 궤적 비교
- 10: DG 경로의 light-painting 비교
- 11: user-drawn 경로의 M0/M1/M2 궤적 비교
- 12: user-drawn 경로의 light-painting 비교
- 13: Path RMSE, Corner Path RMSE
- 14: Painted Pixel IoU, Off-target ratio

## 구성

| 경로 | 내용 |
| --- | --- |
| `data/eval/` | DG 경로 평가 로그와 지표 |
| `data/user_drawn_eval/` | user-drawn 경로 평가 로그와 지표 |
| `data/waypoints/` | 발표 결과 생성에 사용한 기준 경로 |
| `weights/` | 발표 결과의 최종 PPO residual policy 웨이트 |
| `paper_figures/` | 발표용 figure 재생성 스크립트 |
| `slides/` | 발표 슬라이드 원본 |

## 실행 환경

프로젝트 루트에서 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.10 또는 3.11을 권장합니다.

## 발표 figure 재생성

프로젝트 루트에서 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File repro\presentation_20260531\run_reproduce_figures.ps1
```

생성 위치:

- `repro/presentation_20260531/paper_figures/generated_20260531/`
- `repro/presentation_20260531/paper_figures/presentation_charts/`

## 데이터 검증

아래 명령은 웨이트 해시, 필수 로그 파일, 발표 지표 값을 확인합니다.

```powershell
python repro\presentation_20260531\verify_repro.py
```

figure까지 생성한 뒤에는 다음처럼 확인합니다.

```powershell
python repro\presentation_20260531\verify_repro.py --check-generated
```

## 재현 기준

이 패키지는 저장된 평가 로그와 선택된 웨이트를 기준으로 발표 결과를 재생성합니다. 발표 수치와 그림은 `data/eval/selected_phase_metrics.csv` 및 각 조건의 trajectory CSV에서 계산됩니다.

시뮬레이션을 처음부터 다시 평가해 같은 로그를 만들려면 평가 코드 상태도 결과와 함께 고정되어야 합니다. 발표 결과 생성에 사용한 평가 코드는 Git 이력의 `b2f0bfc` 커밋을 기준으로 합니다.
