param(
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

Set-Location (Join-Path $PSScriptRoot "..")
$infraPython = Join-Path (Get-Location) "..\mini_script\teams\infra\.venv\Scripts\python.exe"
$dashboardArgs = @("-m", "src.train.experiment_dashboard", "--host", $HostName, "--port", $Port, "--open") + $ExtraArgs
if (Test-Path $infraPython) {
    & $infraPython @dashboardArgs
} else {
    python @dashboardArgs
}
