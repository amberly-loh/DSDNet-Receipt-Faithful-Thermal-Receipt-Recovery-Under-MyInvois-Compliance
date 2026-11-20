# Pushes the datasets (real-life receipts, SROIE-Malaysia, controlled degradations), the fresh real-receipt
# tests and the full experiment outputs to a SEPARATE PRIVATE GitHub repository. Nothing here goes to the
# public code repository. Log: artifacts\publish_private_log.txt
$ErrorActionPreference = "Continue"
$Root  = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
$Data  = "C:\Users\Amberly\Desktop\FYP\Task 7\FYP datasets"
$Priv  = "C:\Users\Amberly\Desktop\FYP\Task 7\private_data_repo"
$Owner = "amberly-loh"; $Name = "DSDNet-Receipt-Data-Private"
$Url   = "https://github.com/$Owner/$Name.git"
$log   = Join-Path $Root "artifacts\publish_private_log.txt"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -FilePath $log -Append }
"=== private publish $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8

# 1. the private repository must exist and must be private
$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
    gh repo view "$Owner/$Name" --json visibility 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Log "creating private repository $Owner/$Name"
        gh repo create "$Owner/$Name" --private --description "Private datasets and experiment outputs for DSDNet-Receipt" 2>&1 | Out-File $log -Append -Encoding utf8
    }
    $vis = (gh repo view "$Owner/$Name" --json visibility -q .visibility 2>$null)
    if ($vis -ne "PRIVATE") { Log "STOPPED: $Owner/$Name is not private (visibility: $vis). Nothing was pushed."; exit 6 }
} else {
    git ls-remote $Url 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Log "STOPPED: private repository not found. Create it at https://github.com/new"
        Log "  name: $Name   visibility: PRIVATE   (no README, no .gitignore, no licence) - then run this again."
        exit 7
    }
    $ans = Read-Host "Is https://github.com/$Owner/$Name set to PRIVATE? Type YES to continue"
    if ($ans -ne "YES") { Log "STOPPED: privacy not confirmed. Nothing was pushed."; exit 8 }
}

# 2. local clone of the private repository
if (-not (Test-Path (Join-Path $Priv ".git"))) {
    New-Item -ItemType Directory -Force $Priv | Out-Null
    Set-Location $Priv
    git init -b main 2>&1 | Out-File $log -Append -Encoding utf8
    git remote add origin $Url
    git fetch origin 2>&1 | Out-File $log -Append -Encoding utf8
    git ls-remote --exit-code --heads origin main 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { git reset --soft origin/main 2>&1 | Out-File $log -Append -Encoding utf8 }
}
Set-Location $Priv
git config http.postBuffer 524288000

function Part($label, $src, $dst, [string[]]$files, [string[]]$extra) {
    if (-not (Test-Path $src)) { Log "skip $label (not found: $src)"; return }
    $rc = @($src, (Join-Path $Priv $dst)) + $files + @("/E", "/NFL", "/NDL", "/NJH", "/NJS") + $extra
    robocopy @rc | Out-Null
    git add -A -- $dst 2>&1 | Out-File $log -Append -Encoding utf8
    git commit -q -m "Add $label" 2>&1 | Out-File $log -Append -Encoding utf8
    git push -u origin main 2>&1 | Out-File $log -Append -Encoding utf8
    Log "$label pushed (exit $LASTEXITCODE)"
}

@"
# DSDNet-Receipt private data (do not make public)

Datasets and full experiment outputs for the DSDNet-Receipt final year project (Amberly Loh, TAR UMT).
The code is in the public repository amberly-loh/DSDNet-Receipt-Faithful-Thermal-Receipt-Recovery-Under-MyInvois-Compliance.

| Folder | Contents |
|---|---|
| datasets/real_life/ | Self-collected Malaysian receipt photos (personal data; keep private) |
| datasets/sroie_malaysia/ | SROIE-Malaysia images, annotations and manifests (CC BY 4.0; see its LICENSE and NOTICE). The two parquet files are left out: they repeat the images and exceed GitHub's 100 MB file limit |
| datasets/controlled_degraded/ | Controlled degradations used for validation and test, with the generation manifest |
| datasets/replication_degraded/ | Degraded images of the 53 replication sources, with their manifest |
| datasets/combined_malaysian_receipts_manifest.csv | Combined receipt manifest |
| fresh_tests/ | Both fresh real-receipt tests: selected photos, labels and results |
| experiment_outputs/ | Every experiment and test output (same as the public experiments folder) |

CORD is not included: it was cited in the literature review but not used in any experiment.
"@ | Out-File (Join-Path $Priv "README.md") -Encoding utf8
git add README.md; git commit -q -m "README" 2>&1 | Out-File $log -Append -Encoding utf8

$ex = @("/XD", ".git", "__pycache__")
Part "experiment outputs"   (Join-Path $Root "experiments_public") "experiment_outputs" @() $ex
Part "receipt manifest"     $Data "datasets" @("combined_malaysian_receipts_manifest.csv") @("/LEV:1")
Part "real-life receipts"   (Join-Path $Data "Real life") "datasets\real_life" @() $ex
Part "SROIE-Malaysia"       (Join-Path $Data "External\SROIE-Malaysia") "datasets\sroie_malaysia" @() @("/XD", ".git", "/XF", "*.parquet")
Part "controlled test"      (Join-Path $Root "artifacts\controlled_degradation_sroie\test") "datasets\controlled_degraded\test" @() $ex
Part "controlled validation" (Join-Path $Root "artifacts\controlled_degradation_sroie\validation") "datasets\controlled_degraded\validation" @() $ex
Part "generation manifest"  (Join-Path $Root "artifacts\controlled_degradation_sroie") "datasets\controlled_degraded" @("generation_manifest.jsonl", "generation_summary.json") @("/LEV:1")
Part "replication images"   (Join-Path $Root "artifacts\replication_20261108\degraded") "datasets\replication_degraded" @() $ex
Part "replication manifest" (Join-Path $Root "artifacts\replication_20261108") "datasets\replication_degraded" @("replication_manifest.jsonl") @("/LEV:1")
Part "fresh test 1"         (Join-Path $Root "fresh_test_20261113") "fresh_tests\fresh_test_20261113" @() $ex
Part "fresh test 2"         (Join-Path $Root "fresh_test_2_20261113") "fresh_tests\fresh_test_2_20261113" @() $ex

git log --oneline -15 2>&1 | Out-File $log -Append -Encoding utf8
$st = git status --porcelain
if ($st) { Log "warning: uncommitted files remain in the private clone" } else { Log "private repository up to date" }
