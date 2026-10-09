$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& "C:\Users\Amberly\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m dsdnet_receipt.cli readiness --config (Join-Path $ProjectRoot "configs\default.json")

