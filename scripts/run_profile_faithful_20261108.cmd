@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe" scripts\profile_faithful_20261108.py SROIE-MY-TRAIN-0009/V03,SROIE-MY-TRAIN-0033/V02,SROIE-MY-TEST-0262/V03,SROIE-MY-TRAIN-0227/V01,SROIE-MY-TEST-0160/V03 > artifacts\profile_faithful_20261108.log 2>&1
