@echo off
rem Final evaluation, corrected run (7 Nov 2026): stops the earlier full-population run,
rem keeps its partial records as a pilot, then runs the seeded-sample experiment.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
echo === corrected run start %date% %time% === >> %LOG%
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*final_evaluation_20261107.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> %LOG% 2>&1
timeout /t 8 /nobreak > nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*final_evaluation_20261107.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >> %LOG% 2>&1
timeout /t 3 /nobreak > nul
if exist %OUTDIR%\records.jsonl move /y %OUTDIR%\records.jsonl %OUTDIR%\pilot_records_full_population.jsonl >> %LOG% 2>&1
if exist %OUTDIR%\DONE.txt del /q %OUTDIR%\DONE.txt

echo --- git backup of corrections --- >> %LOG%
git switch app-update-20261107 >> %LOG% 2>&1
git add scripts\final_evaluation_20261107.py scripts\run_final_evaluation_corrected_20261107.cmd web\templates\index.html >> %LOG% 2>&1
git commit -m "Final evaluation: seeded source sample, reading-copy transcription for the proposed system; date button label" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head.txt

echo --- usability static audit (after label fix) --- >> %LOG%
%PY% scripts\evaluate_usability_static.py > %OUTDIR%\usability_console.txt 2>&1
copy /y artifacts\training_candidate_20261102\usability_static_audit.json %OUTDIR%\ >> %LOG% 2>&1

echo --- experiments --- >> %LOG%
%PY% scripts\final_evaluation_20261107.py collect --test-sources 40 --validation-sources 40 > %OUTDIR%\collect_console.txt 2>&1
echo collect exit %errorlevel% %time% >> %LOG%
%PY% scripts\final_evaluation_20261107.py analyse > %OUTDIR%\analyse_console.txt 2>&1
echo analyse exit %errorlevel% %time% >> %LOG%
echo done > %OUTDIR%\DONE_corrected.txt
echo === end %date% %time% === >> %LOG%
