@echo off
rem 8 Nov 2026: half precision for the Fast preset. Validation decides; the test run only times an adopted setting.
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PY="C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe"
set OUTDIR=artifacts\final_run_20261108
set LOG=%OUTDIR%\run_log_fp16.txt
echo === start %date% %time% === > %LOG%
git add src\dsdnet_receipt\restoration.py src\dsdnet_receipt\settings.py src\dsdnet_receipt\web_app.py scripts\speed_check_20261108.py scripts\fast_preset_20261108.py scripts\decide_fp16_20261108.py scripts\run_fp16_20261108.cmd >> %LOG% 2>&1
git commit -m "Fast preset: optional half-precision restoration; validation decision rule; timing run" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
set PYTHONPATH=%CD%\src
%PY% -m pytest tests -q -p no:cacheprovider --junitxml=%OUTDIR%\pytest_results_final3.xml > %OUTDIR%\pytest_console_final3.txt 2>&1
echo pytest exit %errorlevel% >> %LOG%
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;%CD%\src
set RECEIPT_OCR_GPU=1
%PY% scripts\speed_check_20261108.py --label fast_fp32 --preset fast --fp32 >> %LOG% 2>&1
%PY% scripts\speed_check_20261108.py --label fast_fp16 --preset fast >> %LOG% 2>&1
%PY% scripts\decide_fp16_20261108.py artifacts\final_evaluation_20261107\speed_20261108 >> %LOG% 2>&1
if errorlevel 1 goto skip
%PY% scripts\fast_preset_20261108.py fast_fp16_test > %OUTDIR%\fast16_console.txt 2>&1
echo fast16 exit %errorlevel% %time% >> %LOG%
:skip
set RECEIPT_OCR_GPU=
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\restart_app.ps1 >> %LOG% 2>&1
echo === end %date% %time% === >> %LOG%
