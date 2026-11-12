# DSDNet-Receipt (ClearReceipt)

**DSDNet-Receipt: Evidence-Aware Thermal Receipt Recovery in the Malaysian MyInvois Context**
Final Year Project, Amberly Loh. Version 1.0.0.

DSDNet-Receipt recovers faded thermal receipts and reads the details that matter for accounting and
MyInvois: merchant, date, receipt number, total and line items. It adapts the DSDNet restoration
network (Kuo et al., 2025) to receipts. The recovered view only uses evidence that is still in the
photo; no text is redrawn. Every extracted value can be checked against the original before it is
saved. The desktop app is shown to users as **ClearReceipt**.

The app runs entirely on your own computer. Receipts, images and records stay in a local SQLite
database and are never uploaded anywhere. This repository is code only. Datasets, receipt images and
trained model weights are not included.

## What the app does

- Scan from a file, several files at once, or a phone on the same Wi-Fi. On the phone you can turn and
  crop the photo before sending.
- Restore the receipt and read it, then show the original and the recovered view side by side.
- Fill in merchant, date, receipt number, total, line items and accounting fields. Every value shows
  where it was read from, and you confirm it before saving.
- Ask you to scan again if a photo looks turned the wrong way or cut off.
- Keep a searchable history, finance summaries, and CSV and Excel exports.
- Prepare a MyInvois e-Invoice draft (unsigned UBL 2.1 JSON). The app does not submit anything to
  LHDN.

## Running it

Requirements: Python 3.10+, PaddleOCR 2.7.0.3 and PyTorch. A GPU is optional.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
set PYTHONPATH=src
python scripts\serve_production.py   # or double-click scripts\run_server_persistent.cmd
```

Then open http://127.0.0.1:5055. The paths to the model checkpoints are set in `configs/default.json`.
The checkpoints are not in this repository. Train them with the scripts in `scripts/`, or place your
own at the configured paths.

## Repository layout

| Path | Contents |
|---|---|
| `src/dsdnet_receipt/` | Restoration, OCR, field extraction, evidence support, MyInvois draft, web app |
| `web/` | App pages (desktop and phone) |
| `tests/` | Automated tests (`pytest tests`) |
| `scripts/` | App launch scripts and the final evaluation scripts |
| `configs/` | Configuration |

## Evaluation

The final evaluation compares the proposed system with five baselines on controlled degradations of
SROIE Malaysian receipts: A direct OCR, B Otsu, B Sauvola, C perceptual filter and D DSDNet alone.
For field acceptance it also compares two policies, E (OCR and rule score) and F (OCR confidence
only). The full method and results are in the thesis.
