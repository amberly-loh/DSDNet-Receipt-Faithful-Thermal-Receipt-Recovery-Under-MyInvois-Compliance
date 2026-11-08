@echo off
rem Push the latest app code to GitHub (code and documents only; no environments, packages or receipt data).
cd /d "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
set LOG=artifacts\push_to_github_20261109.txt
echo === push %date% %time% === > %LOG%
findstr /x /c:"gpu_ocr_pkgs/" .gitignore >nul || (
  echo.>> .gitignore
  echo # Large local folders and run sandboxes>> .gitignore
  echo gpu_ocr_pkgs/>> .gitignore
  echo tmp/>> .gitignore
  echo output/>> .gitignore
  echo artifacts/*/sandbox*/>> .gitignore
  echo artifacts/*/*.sqlite3>> .gitignore
)
git add .gitignore src tests scripts web configs README.md DATASET_REGISTER.md DEPLOYMENT_AUDIT_READINESS.md SUPERVISOR_CONFORMANCE_MATRIX.md CHAPTER4_ENVIRONMENT.md CHAPTER4_VIDEO_FIGURE_REVIEW.md requirements.txt requirements-chapter4.txt requirements-chapter4-override.txt pyproject.toml .env.example >> %LOG% 2>&1
git status --short >> %LOG% 2>&1
git commit -m "Latest application code (9 Nov 2026)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" >> %LOG% 2>&1
git log --oneline -5 >> %LOG% 2>&1
echo --- pushing --- >> %LOG%
git push -u origin app-update-20261107 >> %LOG% 2>&1
echo push exit %errorlevel% >> %LOG%
type %LOG%
echo.
echo If it says "push exit 0", the code is on GitHub. Press any key to close.
pause >nul
