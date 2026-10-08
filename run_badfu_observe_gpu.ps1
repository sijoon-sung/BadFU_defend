# Use an activated CUDA-enabled Python environment, or set $env:PYTHON.
$ErrorActionPreference = 'Stop'
$experimentPython = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $experimentPython -u -m experiments.request_purify --dataset badfu --device cuda:0 --observe --probe-size 64 --arms none detected oracle --post-rounds 0 @args
    $experimentExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $experimentExitCode
