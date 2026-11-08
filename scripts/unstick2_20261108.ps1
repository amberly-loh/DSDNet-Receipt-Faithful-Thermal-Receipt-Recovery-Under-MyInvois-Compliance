# Stop text-reader helper processes whose parent has already exited (orphans), nothing else.
$log = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\unstick_20261108.log"
"=== $(Get-Date -Format s) ===" | Out-File $log -Append
$alive = @{}; Get-CimInstance Win32_Process | ForEach-Object { $alive[[int]$_.ProcessId] = $true }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*spawn_main*" -and $_.CommandLine -like "*receipt_recovery*" } | ForEach-Object {
  if (-not $alive.ContainsKey([int]$_.ParentProcessId) -or $_.CommandLine -like "*parent_pid=32456*") {
    "stopping orphan helper PID $($_.ProcessId) parent $($_.ParentProcessId)" | Out-File $log -Append
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*receipt_recovery*" -and $_.Name -like "python*" } | ForEach-Object {
  "remaining PID $($_.ProcessId) parent $($_.ParentProcessId): $($_.CommandLine)" | Out-File $log -Append }
