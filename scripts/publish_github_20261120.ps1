# Publishes the final code (no datasets, no receipt images, no model weights, no local data) to the
# project's GitHub repository, branch main, together with the experiment outputs (text formats only). Uses the existing clone at ..\github_publish so history is kept.
# Nothing is deployed online. Log: artifacts\publish_github_log.txt
$ErrorActionPreference = "Continue"
$Root = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
$Pub  = "C:\Users\Amberly\Desktop\FYP\Task 7\github_publish"
$log  = Join-Path $Root "artifacts\publish_github_log.txt"
function Log($m) { "$(Get-Date -Format s) $m" | Out-File $log -Append -Encoding utf8 }
"=== publish $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
try {
    Set-Location $Pub
    git fetch origin 2>&1 | Out-File $log -Append -Encoding utf8
    git checkout main 2>&1 | Out-File $log -Append -Encoding utf8
    git pull --ff-only origin main 2>&1 | Out-File $log -Append -Encoding utf8

    # empty the working tree except git metadata, then copy the clean export in
    Get-ChildItem -Force $Pub | Where-Object { $_.Name -notin @(".git", ".gitattributes") } | Remove-Item -Recurse -Force
    foreach ($dir in @("src", "web", "tests", "configs")) {
        robocopy (Join-Path $Root $dir) (Join-Path $Pub $dir) /E /XD __pycache__ .pytest_cache /XF *.pyc *.sqlite3 /NFL /NDL /NJH /NJS | Out-Null
    }
    New-Item -ItemType Directory -Force (Join-Path $Pub "scripts") | Out-Null
    $scripts = @("serve_production.py", "run_server_persistent.cmd", "restart_app.ps1", "restart_app.cmd",
                 "final_evaluation_20261107.py", "final_run_20261108.py", "replication_20261108.py",
                 "parser_revision_20261112.py", "nfr1_retime_20261112.py", "capture_check_sweep_20261112.py",
                 "reading_agreement_run_20261113.py", "fresh_test_20261113.py", "fresh_test2_20261113.py",
                 "fast_preset_20261108.py", "speed_check_20261108.py", "prepare_sroie_protocol.py",
                 "generate_controlled_corpus.py", "collect_experiment_outputs_20261120.ps1",
                 "publish_github_20261120.ps1", "publish_private_data_20261120.ps1")
    foreach ($s in $scripts) { $p = Join-Path $Root "scripts\$s"; if (Test-Path $p) { Copy-Item $p (Join-Path $Pub "scripts\$s") } }
    foreach ($f in @("requirements.txt", "requirements-chapter4.txt", "requirements-chapter4-override.txt", "pyproject.toml", ".env.example", ".gitignore")) {
        $p = Join-Path $Root $f; if (Test-Path $p) { Copy-Item $p (Join-Path $Pub $f) }
    }
    Copy-Item (Join-Path $Root "README_PUBLIC.md") (Join-Path $Pub "README.md")
    Copy-Item (Join-Path $Root "DATASET_REGISTER.md") (Join-Path $Pub "DATASET_REGISTER.md")
    $exp = Join-Path $Root "experiments_public"
    if (-not (Test-Path $exp)) { Log "refused, experiments_public missing (run the collector first)"; exit 4 }
    robocopy $exp (Join-Path $Pub "experiments") /E /NFL /NDL /NJH /NJS | Out-Null
    $expMB = (Get-ChildItem -Recurse -File (Join-Path $Pub "experiments") | Measure-Object Length -Sum).Sum / 1MB
    if ($expMB -gt 300) { Log ("refused, experiment outputs too large: {0:N0} MB" -f $expMB); exit 5 }

    # never publish data: refuse if any image, database, checkpoint or secret-looking value slipped in
    $bad = Get-ChildItem -Recurse -File $Pub | Where-Object { $_.FullName -notlike "*\.git\*" -and
        ($_.Extension -in @(".jpg", ".jpeg", ".png", ".webp", ".heic", ".sqlite3", ".db", ".pt", ".pth", ".ckpt", ".npz", ".zip") -or $_.Name -eq ".env") }
    $bad = $bad | Where-Object { $_.FullName -notlike "*\web\static\*" }
    if ($bad) { Log ("refused, data files found: " + (($bad | ForEach-Object { $_.FullName }) -join "; ")); exit 2 }
    $secret = Select-String -Path (Get-ChildItem -Recurse -File $Pub | Where-Object { $_.FullName -notlike "*\.git\*" }).FullName `
        -Pattern 'ghp_[A-Za-z0-9]{20,}', 'GOCSPX-[A-Za-z0-9_-]{10,}', 'AIza[0-9A-Za-z_-]{30,}', 'sk-[A-Za-z0-9]{20,}', '-----BEGIN [A-Z ]*PRIVATE KEY' -ErrorAction SilentlyContinue
    if ($secret) { Log ("refused, secret-looking text: " + (($secret | ForEach-Object { "$($_.Path):$($_.LineNumber)" }) -join "; ")); exit 3 }

    git add -A 2>&1 | Out-File $log -Append -Encoding utf8
    git commit -m "ClearReceipt 1.2.0: field and line-item checks, estimates with options, crop fix, experiment outputs and Appendix A test register (datasets, receipt images and model weights not included)" `
        -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017Qg8pS6fYkqPsBFfPns9pd" 2>&1 | Out-File $log -Append -Encoding utf8
    git push origin main 2>&1 | Out-File $log -Append -Encoding utf8
    Log "push exit $LASTEXITCODE"
    git log -1 --stat --format="%H %s" 2>&1 | Out-File $log -Append -Encoding utf8
} catch {
    Log "error: $_"
    exit 1
}
