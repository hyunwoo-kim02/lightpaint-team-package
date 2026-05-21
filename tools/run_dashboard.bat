@echo off
setlocal
cd /d "%~dp0\.."
set "INFRA_PY=%CD%\..\mini_script\teams\infra\.venv\Scripts\python.exe"
if exist "%INFRA_PY%" (
    "%INFRA_PY%" -m src.train.experiment_dashboard --open %*
) else (
    python -m src.train.experiment_dashboard --open %*
)
