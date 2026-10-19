$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& (Join-Path $ProjectRoot ".venv-chapter4\Scripts\python.exe") -c "from dsdnet_receipt.evaluation_ui import launch; launch(r'$ProjectRoot\configs\default.json')"
