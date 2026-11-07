@echo off
rem Transcription pass (7 Nov 2026): waits for run 3 to finish, commits the reading-order fix,
rem re-runs the tests, then stores and scores every method's text on the same 120 test images.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
echo === transcription pass queued %date% %time% === >> %LOG%
:wait
if exist %OUTDIR%\DONE_v3.txt goto go
timeout /t 60 /nobreak > nul
goto wait
:go
echo === transcription pass start %date% %time% === >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add src\dsdnet_receipt\receipt_context.py tests\test_receipt_context.py scripts\final_evaluation_20261107.py scripts\diagnose_reading_copy_20261107.py scripts\run_diagnose_reading_copy.cmd scripts\run_diagnose_one.cmd scripts\run_final_evaluation_texts_20261107.cmd >> %LOG% 2>&1
git commit -m "Reading copy: order pieces of one printed row left to right; transcription pass with word F1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head_texts.txt
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_final.xml > %OUTDIR%\pytest_console_final.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
%PY% scripts\final_evaluation_20261107.py texts --test-sources 40 > %OUTDIR%\texts_console.txt 2>&1
echo texts exit %errorlevel% %time% >> %LOG%
echo done > %OUTDIR%\DONE_texts.txt
echo === transcription pass end %date% %time% === >> %LOG%
