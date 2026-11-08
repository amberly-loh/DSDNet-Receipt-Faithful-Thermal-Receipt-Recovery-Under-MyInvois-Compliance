@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe
set GPUPKG=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs
set LOG=artifacts\speed_run2_20261108.log
echo started %date% %time% > %LOG%
set PYTHONPATH=%GPUPKG%;%CD%\src
"%PY%" scripts\check_gpu_ocr.py >> %LOG% 2>&1
set RECEIPT_TILE_BATCH=16
set RECEIPT_OCR_GPU=1
"%PY%" scripts\speed_check_20261108.py --label final_gpu >> %LOG% 2>&1
echo finished %date% %time% >> %LOG%
