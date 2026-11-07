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
