@echo off
rem Development data for the post-evaluation RQ1 revision: validation receipts only (8 Nov 2026).
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
if not exist artifacts\dev_validation_20261108 mkdir artifacts\dev_validation_20261108
set LOG=artifacts\dev_validation_20261108\run_log.txt
powercfg /change standby-timeout-ac 0 >> %LOG% 2>&1
echo === start %date% %time% === >> %LOG%
git add scripts\dev_validation_20261108.py scripts\run_dev_validation_20261108.cmd >> %LOG% 2>&1
git commit -m "RQ1 revision: record validation readings view by view (no test receipts)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > artifacts\dev_validation_20261108\git_head.txt
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\dev_validation_20261108.py > artifacts\dev_validation_20261108\console.txt 2>&1
echo dev exit %errorlevel% %time% >> %LOG%
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === end %date% %time% === >> %LOG%
