# DSDNet Receipt Recovery

Research prototype for local receipt restoration and review. The repository folders and optional source archive contain application source, configuration, tests, selected experiment scripts and the upstream DSDNet architecture and licence.

## Setup

Use Python 3.10 and install dependencies using requirements-ocr.txt for the locked OCR version. The optional OpenCV override in requirements-opencv-override.txt was recorded in development but may conflict with PaddleOCR; check the environment before applying it. Start with `python scripts/serve_production.py` after supplying model checkpoints and configuring paths. Run `python -m pytest tests/test_core.py tests/test_web_app.py` for selected tests.

## Missing files and data

Model checkpoints and datasets are not in this archive. The configuration references local weights and images; obtain or retrain compatible files and update the paths. The archive alone cannot reproduce the reported trained-model results. Do not publish receipts containing personal or payment information without permission, or third-party datasets without checking their licences.

This prototype does not submit to MyInvois. Research results and deployment readiness require further validation.

## Handwriting dataset

The exploratory handwriting pilot used ORAND-CAR-2014, distributed by its creator at https://www.orand.cl/icfhr2014-hdsr/ under CC BY-NC-ND 4.0. Download the original archive from the official source and retain its attribution and licence. It is for non-commercial use. The local original archive SHA-256 is 7c4b3d9e820bbcdfd291f5679b74013bd055b4f771dc64ebd8b8ce599ff4a61a. This exploratory dataset does not establish accuracy on receipt handwriting.
