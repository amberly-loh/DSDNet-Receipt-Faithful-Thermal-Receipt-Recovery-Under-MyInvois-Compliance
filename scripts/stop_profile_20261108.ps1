$log = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\unstick_20261108.log"
"=== stop profiler $(Get-Date -Format s) ===" | Out-File $log -Append
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*profile_restoration_20261108*" } | ForEach-Object {
  "stopping PID $($_.ProcessId)" | Out-File $log -Append; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
