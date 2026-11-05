@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
:restart
"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe" "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\scripts\serve_production.py" 1>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stdout.log" 2>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stderr.log"
echo Server exited with code %errorlevel% at %date% %time%>>"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\server.stderr.log"
timeout /t 5 /nobreak >nul
goto restart
