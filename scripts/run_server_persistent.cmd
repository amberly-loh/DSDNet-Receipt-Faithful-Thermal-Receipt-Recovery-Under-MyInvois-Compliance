@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
:restart
rem GPU text reader (helper process) and its packages, adopted 8 Nov 2026 after the validation no-harm check
set PYTHONPATH=C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\gpu_ocr_pkgs;C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\src
set RECEIPT_OCR_GPU=1
"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe" "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\scripts\serve_production.py" 1>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stdout.log" 2>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stderr.log"
echo Server exited with code %errorlevel% at %date% %time%>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stderr.log"
timeout /t 5 /nobreak >nul
goto restart
