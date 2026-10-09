$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& "C:\Users\Amberly\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -c "from dsdnet_receipt.ui import launch; launch(r'$ProjectRoot\configs\default.json')"

