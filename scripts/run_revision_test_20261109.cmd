@echo off
rem RQ1 revision: one-time test run of the frozen changes (9 Nov 2026). Plan: artifacts\revision_test_20261109\plan.json
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\revision_test_20261109
if not exist %OUTDIR% mkdir %OUTDIR%
set LOG=%OUTDIR%\run_log.txt
powercfg /change standby-timeout-ac 0 >> %LOG% 2>&1
echo === start %date% %time% === >> %LOG%
git add src\dsdnet_receipt\ocr.py src\dsdnet_receipt\receipt_context.py tests\test_rq1_revision.py scripts\revision_test_20261109.py scripts\run_revision_test_20261109.cmd >> %LOG% 2>&1
git commit -m "RQ1 revision (chosen on validation only): keep the spaced reading of identical letters; reading copy keeps lines the reader accepted (>= 0.50)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head.txt
set PYTHONPATH=%CD%\src
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results.xml > %OUTDIR%\pytest_console.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\revision_test_20261109.py > %OUTDIR%\console.txt 2>&1
echo revision exit %errorlevel% %time% >> %LOG%
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === end %date% %time% === >> %LOG%
