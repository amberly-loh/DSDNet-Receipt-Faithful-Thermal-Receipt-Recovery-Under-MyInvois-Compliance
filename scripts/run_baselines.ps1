param([switch]$OCR)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Arguments = @("-m", "dsdnet_receipt.cli", "baselines", "--config", (Join-Path $ProjectRoot "configs\default.json"))
if ($OCR) { $Arguments += "--ocr" }
& "C:\Users\Amberly\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" @Arguments

