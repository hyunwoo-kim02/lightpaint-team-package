@echo off
setlocal
cd /d "%~dp0\.."
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" -m src.train.experiment_dashboard --open %*
) else (
    python -m src.train.experiment_dashboard --open %*
)
