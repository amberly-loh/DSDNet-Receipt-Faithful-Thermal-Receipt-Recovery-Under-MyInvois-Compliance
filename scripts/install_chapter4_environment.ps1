$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv-chapter4\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.10 -m venv (Join-Path $PSScriptRoot '..\.venv-chapter4')
}
& $python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap package installation failed.' }
& $python -m pip install -f 'https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html' -r (Join-Path $PSScriptRoot '..\requirements-chapter4.txt')
if ($LASTEXITCODE -ne 0) { throw 'Chapter 4 dependency installation failed.' }
& $python -m pip install --no-deps -r (Join-Path $PSScriptRoot '..\requirements-chapter4-override.txt')
if ($LASTEXITCODE -ne 0) { throw 'OpenCV protocol override failed.' }
& $python -m pip check
# A non-zero check is expected only for PaddleOCR's declared OpenCV upper bound;
# imports and the OCR smoke test are the operational compatibility gate.
