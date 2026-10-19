# DSDNet Receipt Recovery Research System

This repository contains two deliberately separate applications.

## Research evaluation system

The evaluation interface is for experiments only. It compares Direct OCR, Otsu, Sauvola, the perceptual filter, and, after a valid checkpoint is available, DSDNet under shared preprocessing and OCR. It records metrics, runtime, per-method outputs, and an audit file. It is not the final user interface. Launch it with `scripts/run_evaluation_ui.ps1`.

## Final user application

The final application accepts a receipt through file upload or camera capture, applies the configured DSDNet restoration model, runs PaddleOCR, extracts the three thesis fields, and records review decisions and an audit package. It has no baseline selector. If the checkpoint or source-trace support is unavailable, it does not substitute a baseline or automatically accept a field; it requires human review. Launch it with `scripts/run_user_app.ps1` or `scripts/run_ui.ps1`.

The browser application now provides a persistent local workspace rather than a single-use scan page:

- account registration and sign-in with scrypt password hashing and an eight-hour signed session;
- per-user receipt ownership, searchable history and grid/list gallery;
- dashboard counts for total, review-required and reviewed records;
- editable fields, tags, private notes and explicit human-review status;
- side-by-side original/recovered image comparison;
- per-receipt JSON audit download and receipt-history CSV export; and
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

Formal DSDNet results require a trained, recorded checkpoint and the exact Chapter 4 environment. The eight-stage configuration exceeds the available 4 GB development GPU during training, so current interface and baseline outputs are development evidence, not final DSDNet performance claims. Training must be completed on a larger compatible GPU before the held-out test partition is opened.
