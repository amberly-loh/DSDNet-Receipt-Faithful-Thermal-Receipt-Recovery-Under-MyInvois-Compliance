# Stops any running receipt app server and starts it again with the latest code.
$Root = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
$log = Join-Path $Root "artifacts\restart_app.log"
"=== restart $(Get-Date -Format s) ===" | Out-File $log -Append

# 1. stop the auto-restart loop and the Python server (only this project's processes)
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -like "*run_server_persistent*" -or
        $_.CommandLine -like "*serve_production.py*" -or
        $_.CommandLine -like "*launch_web_app.py*" -or
        $_.CommandLine -like "*serve_web_app.py*")
} | ForEach-Object {
    "stopping PID $($_.ProcessId): $($_.CommandLine)" | Out-File $log -Append
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
# the GPU text-reader helper processes of the stopped server (spawned by multiprocessing)
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -like "*spawn_main*" -and $_.CommandLine -like "*receipt_recovery*"
} | ForEach-Object {
    "stopping helper PID $($_.ProcessId)" | Out-File $log -Append
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
# anything else still holding port 5055
Get-NetTCPConnection -LocalPort 5055 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
    "stopping port owner PID $($_.OwningProcess)" | Out-File $log -Append
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

# One-off (12 Nov 2026): the final run skipped publishing because one test still expected the removed admin page.
# When the flag file exists, run the tests again and publish only if they all pass. Then remove the flag.
$flag = Join-Path $Root "artifacts\final_20261112\PUBLISH_PENDING"
if (Test-Path $flag) {
    Remove-Item $flag -Force
    Push-Location $Root
    $env:PYTHONPATH = "$Root\src"
    & "$Root\.venv-chapter4\Scripts\python.exe" -m pytest tests -q -p no:cacheprovider --junitxml="$Root\artifacts\final_20261112\pytest_results_2.xml" *> "$Root\artifacts\final_20261112\pytest_console_2.txt"
    $code = $LASTEXITCODE
    "retest exit $code" | Out-File $log -Append
    if ($code -eq 0) {
        git add tests\test_evaluation_security.py scripts\restart_app.ps1 scripts\nfr1_retime_20261112.py 2>&1 | Out-File $log -Append
        git commit -m "Security test: the removed admin page now answers 404" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" 2>&1 | Out-File $log -Append
        powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\scripts\publish_github_20261112.ps1"
        "publish exit $LASTEXITCODE" | Out-File $log -Append
    }
    Pop-Location
}

# One-off (12 Nov 2026): time the Fast preset on the 55 primary test sources once, then remove the flag.
$fastFlag = Join-Path $Root "artifacts\final_20261112\FAST_PENDING"
if (Test-Path $fastFlag) {
    Remove-Item $fastFlag -Force
    Push-Location $Root
    $env:PYTHONPATH = "$Root\gpu_ocr_pkgs;$Root\src"
    $env:RECEIPT_OCR_GPU = "1"
    & "$Root\.venv-chapter4\Scripts\python.exe" "$Root\scripts\nfr1_retime_20261112.py" fast --primary *> "$Root\artifacts\final_20261112\nfr1_fast_console.txt"
    "fast retime exit $LASTEXITCODE" | Out-File $log -Append
    $env:RECEIPT_OCR_GPU = ""
    Pop-Location
}

# 2. start the persistent server (restarts itself if it exits)
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$Root\scripts\run_server_persistent.cmd`"" -WindowStyle Minimized
"started run_server_persistent.cmd" | Out-File $log -Append

# 3. wait until it answers
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:5055/" -UseBasicParsing -TimeoutSec 3
        "server answered HTTP $($r.StatusCode) after $(2*($i+1))s" | Out-File $log -Append
        break
    } catch { }
}
