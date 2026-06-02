# 결과 재현

프로젝트 루트에서 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File repro\presentation_20260531\run_reproduce_figures.ps1
```

결과를 확인합니다.

```powershell
python repro\presentation_20260531\verify_repro.py --check-generated
```
