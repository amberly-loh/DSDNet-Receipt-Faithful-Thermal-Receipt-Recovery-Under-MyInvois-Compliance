# DSDNet Receipt Development Readiness Package

## Current status

The project has completed the first executable readiness pass for dataset inventory, provisional source grouping, deterministic partitioning, synthetic degradation, leakage checking, and image-processing Baselines A to C. These outputs are development evidence only. They are not yet formal Chapter 5 evidence because consent, source grouping, clean-reference eligibility, merchant/template family, language, and field annotations still require confirmation.

## Dataset manifest and quantity report

- Local images inventoried: 646 (493 JPG and 153 PNG).
- Provisional independent source groups: 573.
- Provisional source-group split: 389 train, 99 validation, and 85 test.
- Image counts after group assignment: 460 train, 100 validation, and 86 test.
- Confirmed clean controlled sources: 0.
- Naturally collected images: 646 pending eligibility and consent confirmation.
- Verified canonical field annotations: 0.
- Verified high-risk field annotations: 0.
- Exact duplicate or source-group partition leakage detected: none in the provisional manifest.

The current grouping uses exact hashes and a conservative perceptual-hash screen. It cannot replace manual confirmation that repeated photographs, receipt series, and merchant/template families have been assigned correctly. The test partition must remain untouched after the grouping is approved.

## Pilot degradation evidence

Ten visually readable source candidates were used only to test the implementation. They are not yet accepted as clean formal references. Thirty variants were generated: one light, one moderate, and one severe variant for each candidate. Every variant records the source identifier, variant identifier, severity, deterministic seed, degradation parameters, clean-foreground mask, removal mask, observable-trace mask, and operational support target.

The final reproducibility check regenerated all 30 variants and confirmed byte-identical degraded outputs. An earlier run exposed an OpenCV Boolean-mask error and a performance problem in abrasion generation; both were corrected and retained as development history.

## Baselines A to C

The shared baseline runner implements:

- Baseline A: direct extraction input without enhancement.
- Baseline B1: Otsu binarisation.
- Baseline B2: Sauvola binarisation.
- Baseline C: unsharp masking, bilateral filtering, and local contrast enhancement.

The 30 pilot variants produced 120 method records. Overall pilot means were:

| Method | PSNR | SSIM | Processing time seconds |
| --- | ---: | ---: | ---: |
| A Direct | 18.5248 | 0.6041 | 0.0002 |
| B Otsu | 13.4547 | 0.6917 | 0.0007 |
| B Sauvola | 12.6494 | 0.7310 | 0.0138 |
| C Perceptual | 18.2527 | 0.6695 | 0.0078 |

These values demonstrate that the runner and metrics operate consistently; they do not establish a research result because the source candidates and annotations have not passed the formal data gate.

## Repository and environment record

- Official DSDNet source: `setsunil/DSDNet`.
- Recorded DSDNet commit: `8a2822e44ea1e00970ed429a8230e4116034e2f9`.
- Current inspected machine: Windows 11, Python 3.12.14, NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB, driver 566.07.
- Available baseline packages include NumPy 2.3.5 and OpenCV.
- Currently installed OCR environment: PaddleOCR 3.7.0 with PaddlePaddle 3.3.1. This environment is not the formal OCR configuration.
- Required fixed OCR configuration from Chapter 4: PaddleOCR 2.7.0 with English PP-OCRv3 detection and recognition.
- PyTorch is not installed in this runtime.

This machine record is not assumed to be the final training environment. Every device used for a run must generate its own environment and provenance record.

## Current gates and risks

1. The ten formal clean controlled sources have not been confirmed.
2. Consent or usage permission is not yet recorded per local image.
3. Canonical OCR text and field annotations are absent.
4. The installed PaddleOCR 3.7.0 environment differs from the Chapter 4 PaddleOCR 2.7.0 and PP-OCRv3 configuration.
5. The Chapter 4 OCR environment must be created and pass a smoke test before OCR outcomes are reported as formal evidence.
6. DSDNet training must not begin until the Baseline A to C OCR comparison is operational.
7. The current machine has 4 GB VRAM, so the planned batch size and training resolution must be tested rather than assumed.

## Required work before the next supervision meeting

1. Confirm ten clean, permitted receipt sources for the controlled pilot.
2. Review and correct the provisional source groups, template families, language labels, permissions, and partitions.
3. Complete canonical full-text and target-field annotations for the ten controlled sources.
4. Establish one fixed OCR environment and record its exact version and settings.
5. Rerun Baselines A to C and report CER, WER, field accuracy, PSNR, SSIM, runtime, and failed OCR count.
6. Only after the baseline evidence is complete, configure the eight-stage grayscale DSDNet adaptation and test training feasibility.

## Answers currently available for the next meeting

- Independent clean sources available: none formally confirmed; ten candidates are awaiting review.
- Synthetic variants currently generated: three per pilot source, comprising light, moderate, and severe variants.
- Split timing: provisional source-group assignment was completed before the final pilot generation.
- Naturally degraded images available: 646 files, with 573 provisional source groups; permission requires confirmation.
- DSDNet commit: `8a2822e44ea1e00970ed429a8230e4116034e2f9`.
- Baselines A to C: image-processing paths are operational; the fixed OCR and annotated evaluation are not yet operational.
- Support-proxy ground truth: the operational target is the clean-foreground mask intersected with the observable-trace mask on controlled synthetic data.
- Synthetic-only results: pixel-level support AUC, Trace-F1 against the operational target, paired PSNR/SSIM, and claims requiring a known clean state.
- Test-set status: no model, calibration, quantile, support scale, or threshold selection has been performed on the provisional test partition.
