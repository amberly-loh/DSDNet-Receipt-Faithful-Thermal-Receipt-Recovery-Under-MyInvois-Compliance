# DSDNet Receipt Recovery Research System

This repository contains two deliberately separate applications.

## Research evaluation system

The evaluation interface is for experiments only. It compares Direct OCR, Otsu, Sauvola, the perceptual filter, and, after a valid checkpoint is available, DSDNet under shared preprocessing and OCR. It records metrics, runtime, per-method outputs, and an audit file. It is not the final user interface. Launch it with `scripts/run_evaluation_ui.ps1`.

## Final user application

The final application accepts a receipt through file upload, batch upload or camera capture; corrects document geometry; applies the configured eight-stage, four-maxout-branch DSDNet restoration path; constructs a source-supported faithful view; runs multi-view OCR; and extracts primary receipt fields, line items, tags and accounting evidence. It records readiness decisions, human corrections and an audit package. It has no baseline selector and OCR is never used to typeset or redraw words on the recovered image. If required evidence is missing or ambiguous, the record remains for human confirmation. Launch it with `scripts/run_user_app.ps1` or `scripts/run_ui.ps1`.

The browser application now provides a persistent local workspace rather than a single-use scan page:

- account registration and sign-in with scrypt password hashing and an eight-hour signed session;
- per-user receipt ownership, searchable history and grid/list gallery;
- dashboard counts for total, review-required and reviewed records;
- editable primary fields, tags, private notes, accounting/MyInvois-support fields, line items and explicit human-review status;
- side-by-side original/recovered image comparison;
- per-receipt audit PDF, accounting workbook, document-image downloads and receipt-history CSV/Excel exports;
- SQLite persistence at `artifacts/app_data/receipts.sqlite3`.

Mutating requests use a session-bound CSRF token, and receipt records and images are checked against the signed-in user's ownership. The local HTTP prototype is intended for thesis evaluation on one trusted computer. Public deployment would additionally require HTTPS, secure-cookie enforcement, production hosting, encrypted backups, account recovery and operational monitoring.

### Google sign-in

Google authentication is optional and handles identity only. Receipt images, extracted fields and audit records remain in the local SQLite and audit stores. Configure these environment variables before launching the app:

```text
GOOGLE_CLIENT_ID=<OAuth web client ID>
GOOGLE_CLIENT_SECRET=<OAuth client secret>
GOOGLE_REDIRECT_URI=http://127.0.0.1:5055/auth/google/callback
```

The Google Cloud OAuth client must list the same callback as an authorised redirect URI. Never commit OAuth credentials or local receipt data.

## Malaysian datasets

- SROIE-Malaysia: 987 Malaysian English-language scanned receipts with OCR and key-field annotations for controlled degradation and evaluation.
- Self-collected set: 646 real-life Malaysian receipt images, provisionally grouped into 573 source groups, for natural-domain robustness after permissions and annotations are confirmed.

Run `python scripts/prepare_sroie_protocol.py`, then `python scripts/generate_controlled_corpus.py`. The source-level split uses seed 56 before variants are created. Receipt/invoice number is not supplied as a SROIE entity and remains ineligible until manually annotated.

The application is an accounting-evidence and MyInvois-preparation aid, not a MyInvois submission client. It does not generate a legally complete UBL e-Invoice, apply an authorised digital signature, submit to the MyInvois API, or return a validation UUID. Buyer identity, legal classification/tax codes and other absent mandatory data must be provided through an authorised issuer workflow.

The thesis experiments use the recorded iteration-43,000 checkpoint on the locked development-validation evidence described in Chapter 5. The planned one-million-iteration run and a fresh independent held-out test were not completed and are not claimed. After the semantic extraction, readiness and evidence-interface changes, the full automated regression suite was rerun on 5 November 2026: **132 tests passed, 0 failed**. The JUnit result, full test-case register, alignment matrix, MyInvois coverage matrix and SHA-256 provenance manifest are stored under `artifacts/consistency_audit_20261105/`.
