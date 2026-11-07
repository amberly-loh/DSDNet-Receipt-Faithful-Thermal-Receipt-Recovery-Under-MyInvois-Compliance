@echo off
rem Final evaluation, run 3 (7 Nov 2026): stops earlier runs, keeps their partial records,
rem runs the regression tests after the merchant fix, then the seeded-sample experiment.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
echo === run 3 start %date% %time% === >> %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='cmd.exe'\" | Where-Object { ($_.CommandLine -like '*run_final_evaluation_20261107.cmd*' -or $_.CommandLine -like '*run_final_evaluation_corrected_20261107.cmd*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> %LOG% 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*final_evaluation_20261107.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> %LOG% 2>&1
timeout /t 8 /nobreak > nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*final_evaluation_20261107.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> %LOG% 2>&1
timeout /t 3 /nobreak > nul
if exist %OUTDIR%\records.jsonl move /y %OUTDIR%\records.jsonl %OUTDIR%\pilot_records_sample_before_merchant_fix.jsonl >> %LOG% 2>&1
if exist %OUTDIR%\DONE.txt del /q %OUTDIR%\DONE.txt

echo --- git backup of corrections --- >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add src\dsdnet_receipt\fields.py tests\test_core.py scripts\run_final_evaluation_v3_20261107.cmd >> %LOG% 2>&1
git commit -m "Merchant selection: skip address and registration lines, join two-line company names, rank by printed row" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head.txt

echo --- pytest after merchant fix --- >> %LOG%
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results.xml > %OUTDIR%\pytest_console.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
echo --- usability static audit --- >> %LOG%
%PY% scripts\evaluate_usability_static.py > %OUTDIR%\usability_console.txt 2>&1
copy /y artifacts\training_candidate_20261102\usability_static_audit.json %OUTDIR%\ >> %LOG% 2>&1

echo --- experiments --- >> %LOG%
%PY% scripts\final_evaluation_20261107.py collect --test-sources 40 --validation-sources 40 > %OUTDIR%\collect_console.txt 2>&1
echo collect exit %errorlevel% %time% >> %LOG%
%PY% scripts\final_evaluation_20261107.py analyse > %OUTDIR%\analyse_console.txt 2>&1
echo analyse exit %errorlevel% %time% >> %LOG%
echo done > %OUTDIR%\DONE_v3.txt
echo === end %date% %time% === >> %LOG%
