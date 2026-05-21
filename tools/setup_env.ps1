param(
    [string]$Python = "python"
)

Set-Location (Join-Path $PSScriptRoot "..")

$version = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($version -notin @("3.10", "3.11")) {
    Write-Error "Python 3.10 or 3.11 is required. Current version: $version"
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$venvPython = Resolve-Path ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
