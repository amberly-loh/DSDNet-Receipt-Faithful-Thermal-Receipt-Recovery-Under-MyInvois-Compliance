# DSDNet Receipt Research Prototype

This repository follows the supervisor-required evidence sequence:

1. dataset inventory, source grouping, deterministic partitioning, and leakage checks;
2. reproducible synthetic degradation with parameters and masks;
3. Baselines A, B, and C under shared preprocessing and OCR;
4. a gated adapter for the official DSDNet source;
5. support, calibration, and routing only after restoration is validated.

Raw receipt images remain in the local dataset folder. Outputs under `artifacts/` are provisional until consent, source grouping, clean-reference status, language, and field annotations are manually confirmed. Provisional outputs must not be reported as formal Chapter 5 evidence.

The external CORD-v2 corpus is registered separately as 1,000 independent controlled/benchmark receipt sources. It must not be described as Malaysian evidence. See `DATASET_REGISTER.md` and `SUPERVISOR_CONFORMANCE_MATRIX.md`.

Run the PowerShell scripts in `scripts/` in order: `run_readiness.ps1`, `run_pilot.ps1`, then `run_baselines.ps1`.

Formal OCR and DSDNet work must use the isolated Chapter 4 environment described in `CHAPTER4_ENVIRONMENT.md`. The pipeline refuses to treat the currently installed PaddleOCR 3.7.0 as the required PaddleOCR 2.7 and PP-OCRv3 configuration.

Import CORD-v2 with `python scripts/import_cord_v2.py --output "../FYP datasets/External/CORD-v2"`. The importer preserves original image bytes, annotations, publisher partitions, Chapter 3 project partitions, source identifiers, hashes, license provenance, and deletes each temporary parquet shard after extraction to limit disk use.
