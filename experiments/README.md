# Experiment outputs

Machine-readable outputs (CSV, JSON, JSONL, TXT, XML, XLSX) of every experiment and test run reported in
the thesis. Images, databases, model weights and run sandboxes are not included. Outputs that contain text
read from the author's own real receipts (the fresh real-receipt tests) are kept in the private data
repository only.

| File or folder | What it holds |
|---|---|
| `appendix_A_test_register.xlsx` / `.csv` | Appendix A executed test-case register (101 cases: functional, integration, usability, privacy and security, performance, experiment) with a summary sheet |
| `pytest_report.xml` | JUnit report of the automated test suite, produced when this folder was exported |
| `final_evaluation_20261107/` | First and confirmatory evaluation runs, RQ1 tables, RQ3 field records, API and concurrency performance, static usability and privacy audits |
| `final_run_20261108/` | Locked final run on the 95 test sources |
| `replication_20261108/` | Replication run on 53 further test sources |
| `dev_validation_20261108/`, `revision_test_20261109/` | Disclosed reading revision: validation development and the single test run |
| `parser_revision_20261112/` | Shared field-parser revision run (all methods) |
| `agreement_20261113/` | Final reading-agreement run (validation gate, locked test and replication records), NFR1 timing logs |
| `support_audit_20261108/`, `support_audit2_20261113/`, `support_audit2b_20261113/` | Human-versus-rule support audits (AC2) |
| `gsup_recal_20261113/` | Calibration and acceptance-threshold recalibration on validation |
| `validation_restoration/`, `validation_fixed_ocr/` | Validation restoration and fixed-OCR runs |
| `claim_dependent_analysis/` | Claim-dependent analysis outputs |
| `system_verification/`, `test_evidence_20261103/`, `readiness/`, `release_audit/` | System test and release evidence |
| `protocol_lock/` | Locked generation manifest of the 611 controlled sources (430 train, 86 validation, 95 test) and its SHA-256 |
| `controlled_degradation_sroie/` | Generation manifest and summary of the controlled degradations |
| `capture_check_sweep.json` | Capture-check threshold sweep |
