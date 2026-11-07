@echo off
rem Final evaluation run, 7 November 2026: backup, full tests, audits, experiments.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_evaluation_20261107
if not exist %OUTDIR% mkdir %OUTDIR%
set LOG=%OUTDIR%\run_log.txt
echo === start %date% %time% === >> %LOG%
if exist web\static\_selftest_lapan.png del /q web\static\_selftest_lapan.png

echo --- git backup --- >> %LOG%
git switch -c app-update-20261107 >> %LOG% 2>&1 || git switch app-update-20261107 >> %LOG% 2>&1
git add src web tests scripts configs README.md pyproject.toml requirements.txt requirements-chapter4.txt requirements-chapter4-override.txt .gitignore >> %LOG% 2>&1
git commit -m "App update 7 Nov 2026: local workspace, phone scanning, photo crop, receipt recognition, final evaluation scripts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git rev-parse HEAD > %OUTDIR%\git_head.txt
git status --porcelain -- src web tests scripts configs > %OUTDIR%\git_status_code.txt

echo --- keep previous evidence --- >> %LOG%
if not exist %OUTDIR%\previous_20261102 mkdir %OUTDIR%\previous_20261102
copy /y artifacts\training_candidate_20261102\*.json %OUTDIR%\previous_20261102\ >> %LOG% 2>&1

echo --- pytest --- >> %LOG%
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results.xml > %OUTDIR%\pytest_console.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%

echo --- audits and benchmarks --- >> %LOG%
%PY% scripts\audit_source_privacy.py > %OUTDIR%\audit_privacy_console.txt 2>&1
%PY% scripts\evaluate_usability_static.py > %OUTDIR%\usability_console.txt 2>&1
%PY% scripts\benchmark_local_api.py > %OUTDIR%\benchmark_local_console.txt 2>&1
%PY% scripts\benchmark_concurrent_api.py > %OUTDIR%\benchmark_concurrent_console.txt 2>&1
%PY% scripts\evaluate_end_to_end_synthetic.py > %OUTDIR%\e2e_synthetic_console.txt 2>&1
copy /y artifacts\training_candidate_20261102\source_privacy_audit.json %OUTDIR%\ >> %LOG% 2>&1
copy /y artifacts\training_candidate_20261102\usability_static_audit.json %OUTDIR%\ >> %LOG% 2>&1
copy /y artifacts\training_candidate_20261102\local_api_performance.json %OUTDIR%\ >> %LOG% 2>&1
copy /y artifacts\training_candidate_20261102\concurrent_api_performance.json %OUTDIR%\ >> %LOG% 2>&1
copy /y artifacts\e2e_synthetic_20261102\results.json %OUTDIR%\e2e_synthetic_results.json >> %LOG% 2>&1
echo audits done %time% >> %LOG%

echo --- experiments --- >> %LOG%
%PY% scripts\final_evaluation_20261107.py collect > %OUTDIR%\collect_console.txt 2>&1
echo collect exit %errorlevel% %time% >> %LOG%
%PY% scripts\final_evaluation_20261107.py analyse > %OUTDIR%\analyse_console.txt 2>&1
echo analyse exit %errorlevel% %time% >> %LOG%
echo done > %OUTDIR%\DONE.txt
echo === end %date% %time% === >> %LOG%
