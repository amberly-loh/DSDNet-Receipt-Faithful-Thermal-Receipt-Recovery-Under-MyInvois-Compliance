@echo off
rem Independent replication on the 53 never-used test receipts (8 Nov 2026). Plan: artifacts\replication_20261108\plan.json
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
if not exist artifacts\replication_20261108 mkdir artifacts\replication_20261108
set LOG=artifacts\replication_20261108\run_log.txt
echo === start %date% %time% === >> %LOG%
git add scripts\replication_20261108.py scripts\run_replication_20261108.cmd >> %LOG% 2>&1
git commit -m "Independent replication on the 53 never-used test receipts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > artifacts\replication_20261108\git_head.txt
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\replication_20261108.py > artifacts\replication_20261108\console.txt 2>&1
echo replication exit %errorlevel% %time% >> %LOG%
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === end %date% %time% === >> %LOG%
