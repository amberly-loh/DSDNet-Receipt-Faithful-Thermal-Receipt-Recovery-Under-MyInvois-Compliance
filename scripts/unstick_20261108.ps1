# Stop only the stuck GPU-check process and its text-reader helper (nothing else).
$log = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery\artifacts\unstick_20261108.log"
"=== $(Get-Date -Format s) ===" | Out-File $log -Append
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*check_gpu_ocr*" } | ForEach-Object {
  "stopping check PID $($_.ProcessId)" | Out-File $log -Append; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*spawn_main*" -and $_.ExecutablePath -like "*receipt_recovery*" } | ForEach-Object {
  "stopping helper PID $($_.ProcessId) parent $($_.ParentProcessId)" | Out-File $log -Append; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine -like "*receipt_recovery*" -and $_.Name -like "python*" } | ForEach-Object {
  "remaining PID $($_.ProcessId): $($_.CommandLine)" | Out-File $log -Append }
