$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& (Join-Path $ProjectRoot ".venv-chapter4\Scripts\python.exe") (Join-Path $ProjectRoot "scripts\launch_web_app.py")
