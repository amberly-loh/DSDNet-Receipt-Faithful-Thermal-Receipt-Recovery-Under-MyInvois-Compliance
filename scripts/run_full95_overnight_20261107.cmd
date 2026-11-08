@echo off
rem Overnight run (7-8 Nov 2026): complete the planned evaluation on all 95 test sources.
rem The first 40 sources are already done and are skipped automatically; the same frozen system is used.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
set LOG=%OUTDIR%\run_log.txt
powercfg /change standby-timeout-ac 0 >> %LOG% 2>&1
powercfg /change monitor-timeout-ac 0 >> %LOG% 2>&1
echo === full-95 run start %date% %time% === >> %LOG%
git rev-parse HEAD > %OUTDIR%\git_head_full95.txt
copy /y %OUTDIR%\final_evaluation_report.json %OUTDIR%\final_evaluation_report_40sources.json >> %LOG% 2>&1
%PY% scripts\final_evaluation_20261107.py collect --test-sources 95 --validation-sources 40 > %OUTDIR%\collect_full95_console.txt 2>&1
echo collect exit %errorlevel% %time% >> %LOG%
%PY% scripts\final_evaluation_20261107.py texts --test-sources 95 > %OUTDIR%\texts_full95_console.txt 2>&1
echo texts exit %errorlevel% %time% >> %LOG%
%PY% scripts\final_evaluation_20261107.py analyse > %OUTDIR%\analyse_full95_console.txt 2>&1
echo analyse exit %errorlevel% %time% >> %LOG%
echo done > %OUTDIR%\DONE_full95.txt
echo === full-95 run end %date% %time% === >> %LOG%
