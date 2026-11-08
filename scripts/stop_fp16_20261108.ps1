$log = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\unstick_20261108.log"
"=== stop fp16 timing $(Get-Date -Format s) ===" | Out-File $log -Append
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*fast_preset_20261108*" } | ForEach-Object {
  "stopping PID $($_.ProcessId)" | Out-File $log -Append; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*spawn_main*" -and $_.CommandLine -like "*receipt_recovery*" } | ForEach-Object {
  $p = Get-Process -Id $_.ParentProcessId -ErrorAction SilentlyContinue
  if (-not $p) { "stopping orphan helper $($_.ProcessId)" | Out-File $log -Append; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
