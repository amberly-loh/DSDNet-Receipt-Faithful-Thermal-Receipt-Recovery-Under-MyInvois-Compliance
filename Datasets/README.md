# Datasets

This folder holds dataset preparation assets and synthetic degradation tools.

## Included receipt data

- `Real life/` contains the project receipt collection and generated development assets.
- `SROIE-Malaysia/` contains 987 extracted ICDAR 2019 SROIE receipt images: 626 train records and 361 test records. The folder also includes OCR words, bounding boxes, key-field annotations, a SHA-256 manifest, provenance documentation, and the CC BY 4.0 license.

The SROIE release is used as a Malaysian controlled-source candidate corpus. Exact duplicates, near duplicates, clean-reference eligibility, thermal-receipt relevance, target-field coverage, and the project-level 70:15:15 split must be checked before formal experiments. The original SROIE split is retained only as source provenance.

## Instructions

1. Place clean receipt images in `Datasets/Real life/clean/`.
2. Use `dataset_utils.py` to generate synthetic degraded pairs and metadata.
3. The generated dataset will be stored under `Datasets/Real life/synthetic/`.
