@echo off
rem 8 Nov 2026: server-path fix check, then the Fast-preset measurement, then restart the app.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_run_20261108
set LOG=%OUTDIR%\run_log_fast.txt
echo === start %date% %time% === > %LOG%
git add src\dsdnet_receipt\inference_worker.py scripts\smoke_server_path_20261108.py scripts\fast_preset_20261108.py scripts\profile_faithful_20261108.py scripts\run_fast_20261108.cmd >> %LOG% 2>&1
git commit -m "Server worker: allow the GPU text-reader helper (non-daemon worker) and multi-threaded image steps with the GPU reader; smoke test; Fast-preset measurement" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
set PYTHONPATH=%CD%\src
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_final2.xml > %OUTDIR%\pytest_console_final2.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\smoke_server_path_20261108.py > %OUTDIR%\smoke_console.txt 2>&1
echo smoke exit %errorlevel% %time% >> %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\unstick2_20261108.ps1 >> %LOG% 2>&1
%PY% scripts\fast_preset_20261108.py > %OUTDIR%\fast_console.txt 2>&1
echo fast exit %errorlevel% %time% >> %LOG%
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === end %date% %time% === >> %LOG%
