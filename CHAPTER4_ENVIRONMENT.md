# Chapter 4 Execution Environment

Formal OCR and DSDNet runs must use an isolated Python 3.10 environment created by `scripts/install_chapter4_environment.ps1`.

The fixed OCR configuration is PaddleOCR 2.7.0.3 in the PaddleOCR 2.7 family, PaddlePaddle 2.5.2, English language mode, PP-OCRv3 detection and recognition, angle classification disabled, and identical OCR settings for Baselines A to F and the proposed method.

The available official Windows Python runtime is 3.10.11 rather than the report's nominal 3.10.13. This patch-level deviation is recorded in Chapter 5 provenance and does not authorize any change to the OCR model, preprocessing, or decision protocol.

PaddleOCR 2.7.0.3 declares `opencv-python<=4.6.0.66`, whereas Chapter 4 records OpenCV 4.8.1.78. The installation therefore resolves PaddleOCR with 4.6 first and then applies the recorded 4.8.1.78 override without dependency resolution. This is a known package-metadata conflict and must be included in the environment report. Formal use requires a successful import and OCR smoke test after the override; the version record alone is insufficient.

The current Python 3.12 runtime with PaddleOCR 3.7.0 is suitable for inspecting the workspace but is not accepted for formal OCR evidence. The pipeline records a version mismatch instead of silently substituting the newer OCR model.

The DSDNet source is the official `setsunil/DSDNet` repository at commit `8a2822e44ea1e00970ed429a8230e4116034e2f9`. PyTorch and CUDA compatibility must be checked on the device selected for training. Each device must produce a separate environment record, configuration hash, checkpoint hash, seed record, and run identifier.
