@echo off
rem Speed work, 8 November 2026: install the GPU text reader into a separate folder,
rem check it, then time the app on validation receipts under three configurations.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe
set GPUPKG=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs
set LOG=artifacts\speed_run_20261108.log
set PYTHONPATH=%CD%\src
echo started %date% %time% > %LOG%
if not exist "%GPUPKG%\paddle" (
  echo installing paddle gpu >> %LOG%
  "%PY%" -m pip install --no-deps --target "%GPUPKG%" paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html >> artifacts\gpu_setup.log 2>&1
  if not exist "%GPUPKG%\paddle" "%PY%" -m pip install --no-deps --target "%GPUPKG%" paddlepaddle-gpu==2.5.2 >> artifacts\gpu_setup.log 2>&1
)
echo gpu install step finished %time% >> %LOG%
set PYTHONPATH=%GPUPKG%;%CD%\src
"%PY%" scripts\check_gpu_ocr.py >> %LOG% 2>&1
set PYTHONPATH=%CD%\src
set RECEIPT_TILE_BATCH=16
"%PY%" scripts\speed_check_20261108.py --label batch16_cpu >> %LOG% 2>&1
set PYTHONPATH=%GPUPKG%;%CD%\src
set RECEIPT_OCR_GPU=1
"%PY%" scripts\speed_check_20261108.py --label batch16_gpu >> %LOG% 2>&1
set RECEIPT_TILE_BATCH=
set RECEIPT_OCR_GPU=
set PYTHONPATH=%CD%\src
"%PY%" scripts\speed_check_20261108.py --label current >> %LOG% 2>&1
echo finished %date% %time% >> %LOG%
echo done > artifacts\speed_run_20261108.done
