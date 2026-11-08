@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PYTHONPATH=%CD%\src
"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe" scripts\export_support_audit_20261108.py > artifacts\support_audit_export.log 2>&1
echo exit %errorlevel% >> artifacts\support_audit_export.log
