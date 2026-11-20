# Collects the machine-readable outputs of every experiment and test run into experiments_public\
# (text formats only: csv, json, jsonl, txt, xml, xlsx, md). No images, databases, weights or sandboxes.
$ErrorActionPreference = "Continue"
$Root = "C:\Users\Amberly\Desktop\FYP\Task 7\receipt_recovery"
$Out  = Join-Path $Root "experiments_public"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force $Out | Out-Null

$folders = @("final_evaluation_20261107", "final_run_20261108", "replication_20261108", "dev_validation_20261108",
             "revision_test_20261109", "parser_revision_20261112", "agreement_20261113", "support_audit_20261108",
             "support_audit2_20261113", "support_audit2b_20261113", "gsup_recal_20261113", "validation_restoration",
             "validation_fixed_ocr", "claim_dependent_analysis", "system_verification", "test_evidence_20261103",
             "readiness", "release_audit", "protocol_lock")
foreach ($f in $folders) {
    $src = Join-Path $Root "artifacts\$f"
    if (Test-Path $src) {
        robocopy $src (Join-Path $Out $f) *.csv *.json *.jsonl *.txt *.xml *.xlsx *.md /S /MAX:50000000 `
            /XD sandbox sandbox_* degraded photos images figures __pycache__ /NFL /NDL /NJH /NJS | Out-Null
    }
}
$cd = Join-Path $Out "controlled_degradation_sroie"
New-Item -ItemType Directory -Force $cd | Out-Null
foreach ($f in @("generation_manifest.jsonl", "generation_summary.json")) {
    $p = Join-Path $Root "artifacts\controlled_degradation_sroie\$f"; if (Test-Path $p) { Copy-Item $p $cd }
}
$p = Join-Path $Root "artifacts\capture_check_sweep.json"; if (Test-Path $p) { Copy-Item $p $Out }
Copy-Item (Join-Path $Root "experiments_export\*") $Out -Recurse -Force
Rename-Item (Join-Path $Out "README_experiments.md") "README.md" -Force

# remove empty folders left by the filters
Get-ChildItem $Out -Recurse -Directory | Sort-Object { $_.FullName.Length } -Descending |
    Where-Object { -not (Get-ChildItem $_.FullName -Force) } | Remove-Item -Force
$n = (Get-ChildItem $Out -Recurse -File).Count
$mb = [math]::Round(((Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
"collected $n files, $mb MB"
