$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PackageRoot "..\..")
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Push-Location $RepoRoot
try {
    & $Python "repro\presentation_20260531\paper_figures\generate_dg_trajectory_comparison.py"
    & $Python "repro\presentation_20260531\paper_figures\generate_user_drawn_trajectory_comparison.py"
    & $Python "repro\presentation_20260531\paper_figures\generate_led_always_on_trajectory_comparisons.py"
    & $Python "repro\presentation_20260531\paper_figures\generate_metric_charts.py"
    & $Python "repro\presentation_20260531\verify_repro.py" "--check-generated"
} finally {
    Pop-Location
}
