@echo off
rem Wrap-up (7 Nov 2026): after the transcription pass, commit the photo crop/upright fix,
rem run the full test suite for the final register, and restart the app with the new code.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
echo === wrap-up queued %date% %time% === >> %LOG%
:wait
if exist %OUTDIR%\DONE_texts.txt goto go
timeout /t 60 /nobreak > nul
goto wait
:go
echo === wrap-up start %date% %time% === >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add src\dsdnet_receipt\preprocessing.py src\dsdnet_receipt\web_app.py tests\test_photo_crop.py scripts\run_final_wrapup_20261107.cmd >> %LOG% 2>&1
git commit -m "Phone photos: separate paper from beige backgrounds by whiteness, turn sideways receipts upright, check upside down by reading" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head_final.txt
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_release.xml > %OUTDIR%\pytest_console_release.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo done > %OUTDIR%\DONE_wrapup.txt
echo === wrap-up end %date% %time% === >> %LOG%
