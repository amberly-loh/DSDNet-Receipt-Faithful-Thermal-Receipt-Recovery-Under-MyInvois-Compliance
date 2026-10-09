# Dataset register and evidence boundaries

## CORD-v2

- Official source: https://huggingface.co/datasets/naver-clova-ix/cord-v2
- Publisher repository: https://github.com/clovaai/cord
- License: CC BY 4.0
- Size: 1,000 independently collected receipt images (800 train, 100 validation, 100 test).
- Language/domain: Indonesian receipts in Latin script.
- Intended project role: external controlled-development and benchmark corpus, including clean-source synthetic degradation, OCR evaluation, and field-level development where the annotation mapping is valid.
- Prohibited interpretation: CORD-v2 is not Malaysian evidence and must not be used to claim performance on naturally degraded Malaysian thermal receipts.

The publisher's 800/100/100 split is retained as provenance. To match Chapter 3, the project assigns the 1,000 sources once to an exact deterministic 700/150/150 train/validation/test partition using seed 56, before creating synthetic variants. Synthetic variants inherit the project partition and source identifier of their original receipt; they are nested observations and never increase the independent-source count.

CORD's released semantic categories do not directly guarantee all three thesis target fields. Full OCR/line annotations are available, but transaction date, receipt/invoice number, and total amount must be mapped and manually verified before field-accuracy experiments. A missing target field is recorded as ineligible rather than treated as an OCR error.

## Self-collected Malaysian receipts

- Intended project role: natural-domain robustness checks, OCR results, qualitative failures, review outcomes, and conservative routing when evidence support is unavailable.
- Restrictions: permission, merchant/template grouping, language, clean-reference status, and field annotation must be confirmed before formal use.
- Prohibited interpretation: naturally degraded receipts do not provide pixel-level ground truth for historical recovery, authenticity, or source-trace support.

## Alignment with Chapters 1–4

The two-corpus design preserves the existing scope: controlled data supports known-reference restoration and evidence experiments, while local Malaysian receipts test natural robustness. All preprocessing, PaddleOCR 2.7.0 English PP-OCRv3 settings, DSDNet configuration, metrics, and decision logic remain those specified in Chapters 1–4. Dataset provenance and the external-domain limitation must be reported in Chapter 5.
