@echo off
rem Confirmatory evaluation of the improved app (7 Nov 2026, evening).
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
powercfg /change standby-timeout-ac 0 >> %LOG% 2>&1
powercfg /change monitor-timeout-ac 0 >> %LOG% 2>&1
echo === confirmatory start %date% %time% === >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add src\dsdnet_receipt\preprocessing.py src\dsdnet_receipt\web_app.py tests\test_photo_crop.py scripts\confirmatory_evaluation_20261107.py scripts\run_confirmatory_20261107.cmd >> %LOG% 2>&1
git commit -m "Photos: cut by the print when the paper cannot be told from a white background; app also reads the DSDNet restoration view; confirmatory evaluation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head_confirmatory.txt
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_confirmatory.xml > %OUTDIR%\pytest_console_confirmatory.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
%PY% scripts\confirmatory_evaluation_20261107.py > %OUTDIR%\confirmatory_console.txt 2>&1
echo confirmatory exit %errorlevel% %time% >> %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === confirmatory end %date% %time% === >> %LOG%
