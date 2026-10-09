# Chapter 4 Execution Environment

Formal OCR and DSDNet runs must use an isolated Python 3.10.13 environment created from `requirements-chapter4.txt`.

The fixed OCR configuration is PaddleOCR 2.7.0.3 in the PaddleOCR 2.7 family, PaddlePaddle 2.6.2, English language mode, PP-OCRv3 detection and recognition, angle classification disabled, and identical OCR settings for Baselines A to F and the proposed method.

The current Python 3.12 runtime with PaddleOCR 3.7.0 is suitable for inspecting the workspace but is not accepted for formal OCR evidence. The pipeline records a version mismatch instead of silently substituting the newer OCR model.

The DSDNet source is the official `setsunil/DSDNet` repository at commit `8a2822e44ea1e00970ed429a8230e4116034e2f9`. PyTorch and CUDA compatibility must be checked on the device selected for training. Each device must produce a separate environment record, configuration hash, checkpoint hash, seed record, and run identifier.

