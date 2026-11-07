@echo off
rem Final locked evaluation of the final configuration (8 Nov 2026). Plan: artifacts\final_run_20261108\plan.json
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set GPUPKG=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs
set OUTDIR=artifacts\final_run_20261108
if not exist %OUTDIR% mkdir %OUTDIR%
set LOG=%OUTDIR%\run_log.txt
powercfg /change standby-timeout-ac 0 >> %LOG% 2>&1
echo === final run start %date% %time% === >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add configs\default.json src\dsdnet_receipt\__init__.py src\dsdnet_receipt\ocr.py src\dsdnet_receipt\receipt_context.py src\dsdnet_receipt\restoration.py src\dsdnet_receipt\web_app.py scripts\speed_check_20261108.py scripts\check_gpu_ocr.py scripts\profile_restoration_20261108.py scripts\export_support_audit_20261108.py scripts\final_run_20261108.py scripts\run_final_20261108.cmd scripts\run_server_persistent.cmd scripts\restart_app.ps1 >> %LOG% 2>&1
git commit -m "Speed: GPU text reader in a helper process, cached reading-copy tidy-up, restoration tile batch 16, stage timing; support audit export; final locked evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head.txt
set PYTHONPATH=%CD%\src
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_final.xml > %OUTDIR%\pytest_console_final.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
set PYTHONPATH=%GPUPKG%;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\final_run_20261108.py > %OUTDIR%\console.txt 2>&1
echo final run exit %errorlevel% %time% >> %LOG%
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === final run end %date% %time% === >> %LOG%
