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

### Degradation-aware finishing of the faithful view

After the faithful view is selected, `receipt_cleanup.finish_faithful_view` applies a final source-anchored stage to degraded neutral receipts (degradation profile `faded` or `blurred`, or small low-resolution captures). Clean and colour receipts are returned unchanged.

- Faded A4/A5 invoices use `severe_document.make_severe_document_view`: paper normalisation, stroke-scale black-hat evidence, hysteresis growth of confident ink, removal of fold ridges, speckle and diagonal fold lines away from glyphs, and a locally normalised ink tone.
- Small, faint captures (shorter side under 450 px) use `make_faint_document_view`: the image is enlarged and shown as a contrast-normalised grey view whose filter width follows the measured stroke width, so water-stain tide lines and fold shadows drop out.
- `suppress_receipt_artifacts` then works on every degraded view. Torn or punched holes are shown as flat grey. Outside OCR- or geometry-localised text lines it returns speckle, crease webs, long traces and stain bands to paper, while keeping logos, halftone pictures, table grids, dotted rules, signatures and punctuation. Inside OCR text lines, faint print is normalised to each line's strongest source strokes.

OCR is used for localisation only; no recognised text is drawn. The stage records its decisions in the audit under `native_refinement.degradation_finishing`.

### MyInvois e-Invoice draft

Each receipt page has a MyInvois panel built on `myinvois.py`, following the LHDN MyInvois SDK Invoice v1.0 data structure and code lists (state, payment mode, tax type and the 45 classification codes).

- **Draft from the receipt.** Supplier name, BRN, SST number, address split into lines, postcode, city and state, E.164 phone, e-mail, document number, transaction date and time, payment mode, line items and totals are prefilled. Each line item is checked against its printed row so that quantity × unit price = amount.
- **Source of every value.** Each field is labelled as coming from the receipt, a saved profile, a rule default, the consolidated rule or the user.
- **Profiles.** Supplier details that receipts do not print (TIN, MSIC code, business activity) are remembered per supplier. The user's own business is remembered as the buyer, so later receipts are completed automatically.
- **Consolidated mode.** This applies the General Public buyer: TIN EI00000000010, "NA" identifiers and address, state 17 and classification 004.
- **Validation.** Mandatory fields, lengths, code lists, TIN, BRN, MyKad, SST, MSIC, phone and e-mail formats, and the arithmetic chain are checked: line totals, total excluding tax, plus tax, plus rounding, equals the receipt total.
- **Export.** Once valid, the draft downloads as an unsigned MyInvois Invoice v1.0 UBL 2.1 JSON document. The e-Invoice issue date and time are set at export. The supplier, as issuer, signs (required for document version 1.1) and submits it with their MyInvois credentials. The app does not call the MyInvois API.

## Malaysian datasets

- SROIE-Malaysia: 987 Malaysian English-language scanned receipts with OCR and key-field annotations for controlled degradation and evaluation.
- Self-collected set: 646 real-life Malaysian receipt images, provisionally grouped into 573 source groups, for natural-domain robustness after permissions and annotations are confirmed.

Run `python scripts/prepare_sroie_protocol.py`, then `python scripts/generate_controlled_corpus.py`. The source-level split uses seed 56 before variants are created. Receipt/invoice number is not supplied as a SROIE entity and remains ineligible until manually annotated.

The application is an accounting-evidence and MyInvois-preparation aid, not a MyInvois submission client. It does not generate a legally complete UBL e-Invoice, apply an authorised digital signature, submit to the MyInvois API, or return a validation UUID. Buyer identity, legal classification/tax codes and other absent mandatory data must be provided through an authorised issuer workflow.

The thesis experiments use the recorded iteration-43,000 checkpoint on the locked development-validation evidence described in Chapter 5. The planned one-million-iteration run and a fresh independent held-out test were not completed and are not claimed. After the semantic extraction, readiness and evidence-interface changes, the full automated regression suite was rerun on 5 November 2026: **132 tests passed, 0 failed**. The JUnit result, full test-case register, alignment matrix, MyInvois coverage matrix and SHA-256 provenance manifest are stored under `artifacts/consistency_audit_20261105/`.
