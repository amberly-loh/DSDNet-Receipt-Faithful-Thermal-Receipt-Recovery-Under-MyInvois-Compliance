# Dataset register and evidence boundaries

## SROIE-Malaysia

- Source: ICDAR2019 SROIE distribution of Malaysian English-language receipts.
- License: CC BY 4.0.
- Size: 987 scanned receipt images with OCR words/bounding boxes and company, date, address, and total entities.
- Project role: controlled clean-source synthetic degradation, OCR evaluation, restoration metrics, and eligible field evaluation.
- Project split: deterministic 70:15:15 at source level with seed 56 (691 train, 148 validation, 148 test). All synthetic variants inherit their source partition.
- Controlled corpus actually generated and used (locked manifest `artifacts/protocol_lock/generation_manifest_611_sources.jsonl`, SHA-256 `83ca768c7dd5e77c0f72828f9ebe51ff94163fae7044fd1b5a87bd28d71ebd8f`): 611 sources, 430 train, 86 validation and 95 test, each with light, moderate and severe variants. The final evaluation used 40 of the 86 validation sources for calibration and threshold selection, the 95 locked test sources (primary analysis on positions 41-95, 55 sources) and 53 further test sources generated for replication (`artifacts/replication_20261108`).
- Limitation: receipt/invoice number is not an entity annotation and requires manual annotation before confirmatory analysis.

## Self-collected Malaysian receipts

- Size: 646 images, provisionally grouped into 573 source groups.
- Intended project role: natural-domain robustness checks, OCR results, qualitative failures, review outcomes, and conservative routing when evidence support is unavailable.
- Restrictions: permission, merchant/template grouping, language, clean-reference status, and field annotation must be confirmed before formal use.
- Prohibited interpretation: naturally degraded receipts do not provide pixel-level ground truth for historical recovery, authenticity, or source-trace support.

## Alignment with Chapters 1–4

Together these collections provide at least 1,000 Malaysian receipt images without counting nested images or degradation variants as independent sources. Controlled SROIE data supports known-reference restoration and evidence experiments, while real-life receipts test natural robustness. All preprocessing, PaddleOCR 2.7.0 English PP-OCRv3 settings, DSDNet configuration, metrics, and decision logic remain those specified in Chapters 1–4.
