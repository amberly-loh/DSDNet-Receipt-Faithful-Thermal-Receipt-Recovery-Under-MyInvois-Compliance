# DSDNet Receipt Development Readiness Package

## Current status

The project has completed the first executable readiness pass for dataset inventory, deterministic partitioning, synthetic degradation, leakage checking, image-processing Baselines A to C, and the exact Chapter 4 OCR smoke test. CORD-v2 now provides 1,000 licensed external controlled sources; the self-collected Malaysian set remains a separate natural-robustness corpus. Formal Chapter 5 claims remain gated by target-field annotation review and complete protocol runs.

## External controlled corpus

- Dataset: CORD-v2, CC BY 4.0, 1,000 independent receipt images.
- Publisher partition recorded: 800 train, 100 validation, 100 test.
- Chapter 3 project partition assigned before degradation: 700 train, 150 validation, 150 test using seed 56.
- Controlled variants generated: 3,000 (1,000 light, 1,000 moderate, 1,000 severe).
- Each variant retains its source ID, project partition, deterministic seed, parameters, clean reference, degraded image, removal mask, observable-trace mask, foreground mask, support target, and output hash.
- CORD is Indonesian/Latin-script external controlled data and is not Malaysian natural-domain evidence.
- Full-text OCR references are present for all 1,000 sources; 972 total-amount values map directly. Transaction date and receipt/invoice number require manual source-grounded verification.

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

The fixed-OCR development gate subsequently evaluated ten project-training CORD sources at all three severities (30 variants and 120 method records). It completed with zero OCR failures and persisted CER, WER, PSNR, SSIM, total-amount exact accuracy, preprocessing time, and OCR time. This satisfies the supervisor's operational gate for beginning DSDNet adaptation, but it is not untouched-test evidence and covers only the total-amount target field.

## DSDNet adaptation readiness

The pinned DSDNet architecture now has a thin receipt adapter with one-channel input, eight stages, four maxout branches, Gaussian-kernel construction from the recorded blur parameter, retained stage states, and retained tensors entering the inherited maxout pools. A reduced 16×16 CPU structural test passed at commit `8a2822e44ea1e00970ed429a8230e4116034e2f9`. The earlier 64×64 CPU attempt exceeded the execution window, confirming that training should use a CUDA-enabled environment or another recorded device. Random-weight smoke output is not restoration evidence.

## Repository and environment record

- Official DSDNet source: `setsunil/DSDNet`.
- Recorded DSDNet commit: `8a2822e44ea1e00970ed429a8230e4116034e2f9`.
- Current inspected machine: Windows 11, NVIDIA GeForce RTX 3050 Laptop GPU with 4096 MiB, driver 566.07.
- Available baseline packages include NumPy 2.3.5 and OpenCV.
- General runtime: Python 3.12 with PaddleOCR 3.7.0; it is not accepted for formal evidence.
- Isolated Chapter 4 runtime: Python 3.10.11, PaddleOCR 2.7.0.3, PaddlePaddle 2.5.2, English PP-OCRv3 detection and recognition, OpenCV 4.8.1, and PyTorch 2.1.2 CPU build.
- OCR smoke test: PASS on a real CORD validation receipt (12 recognized lines).
- Recorded deviations: Python 3.10.11 rather than nominal 3.10.13; PaddleOCR's package metadata requests OpenCV <=4.6 although Chapter 4 records 4.8.1. The operational import/OCR test passed after the documented override.

This machine record is not assumed to be the final training environment. Every device used for a run must generate its own environment and provenance record.

## Current gates and risks

1. CORD target-field mappings for transaction date and receipt/invoice number still require manual review.
2. Consent or usage permission is not yet recorded per self-collected local image.
3. Baselines A to C require an exact-environment OCR development run before DSDNet adaptation begins.
4. Full formal baseline results must not be inferred from the smoke test.
5. The current machine has 4 GB VRAM and the installed PyTorch build is CPU-only; final training-device provenance must be recorded on the selected device.

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
