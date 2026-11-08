@echo off
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set PYTHONPATH=%CD%\src
"C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\.venv-chapter4\Scripts\python.exe" scripts\profile_restoration_20261108.py > artifacts\profile_restoration_20261108.log 2>&1
