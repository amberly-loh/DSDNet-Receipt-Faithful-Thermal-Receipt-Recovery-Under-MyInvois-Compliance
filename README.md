# DSDNet Receipt Faithful Recovery

This repository contains a prototype for receipt restoration, OCR extraction, and baseline evaluation using local open-source tools.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Tesseract locally if you want `pytesseract` support:
   ```bash
   sudo apt update && sudo apt install tesseract-ocr
   ```

3. Place clean receipt images into:
   - `Datasets/Real life/clean/`

## Dataset Preparation

Generate a partial synthetic dataset of degraded receipts:

```bash
python dataset_utils.py
```

This creates:
- `Datasets/Real life/synthetic/clean/`
- `Datasets/Real life/synthetic/degraded/`
- `Datasets/Real life/synthetic/masks/`
- `Datasets/Real life/synthetic/metadata.csv`

## Baselines and Evaluation

Run the evaluation script to produce a results table in `results/`:

```bash
python run_evaluation.py
```

Output files:
- `results/baseline_summary.csv`
- `results/baseline_summary.md`

Open `results/baseline_summary.md` in VS Code for a rendered table view.

## FastAPI Demo

Start the local API server:

```bash
python app.py
```

Then open `http://127.0.0.1:8000/` in your browser to upload a receipt image.

## Notes

- The current baseline implementations are prototypes and use local heuristics.
- Replace `_fake_dsdnet_restore` and `_restoration_stub` with your actual DSDNet model and generative restoration model for full experiments.
- Use `paddleocr` or `pytesseract` for OCR text extraction.
