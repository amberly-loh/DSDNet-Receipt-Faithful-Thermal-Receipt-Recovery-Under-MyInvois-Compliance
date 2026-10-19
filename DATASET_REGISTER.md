# Dataset register and evidence boundaries

## SROIE-Malaysia

- Source: ICDAR2019 SROIE distribution of Malaysian English-language receipts.
- License: CC BY 4.0.
- Size: 987 scanned receipt images with OCR words/bounding boxes and company, date, address, and total entities.
- Project role: controlled clean-source synthetic degradation, OCR evaluation, restoration metrics, and eligible field evaluation.
- Project split: deterministic 70:15:15 at source level with seed 56 (691 train, 148 validation, 148 test). All synthetic variants inherit their source partition.
- Limitation: receipt/invoice number is not an entity annotation and requires manual annotation before confirmatory analysis.

## Self-collected Malaysian receipts

- Size: 646 images, provisionally grouped into 573 source groups.
- Intended project role: natural-domain robustness checks, OCR results, qualitative failures, review outcomes, and conservative routing when evidence support is unavailable.
- Restrictions: permission, merchant/template grouping, language, clean-reference status, and field annotation must be confirmed before formal use.
- Prohibited interpretation: naturally degraded receipts do not provide pixel-level ground truth for historical recovery, authenticity, or source-trace support.

## Alignment with Chapters 1–4

Together these collections provide at least 1,000 Malaysian receipt images without counting nested images or degradation variants as independent sources. Controlled SROIE data supports known-reference restoration and evidence experiments, while real-life receipts test natural robustness. All preprocessing, PaddleOCR 2.7.0 English PP-OCRv3 settings, DSDNet configuration, metrics, and decision logic remain those specified in Chapters 1–4.
