from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config
from dsdnet_receipt.restoration import make_readability_view


def main() -> None:
    config, _ = load_config(ROOT / "configs" / "default.json")
    refreshed, skipped = 0, 0
    for run in (Path(config["artifacts_root"]) / "user_runs").iterdir():
        source_path, recovery_path, target_path = run / "input.png", run / "ocr_input.png", run / "restored.png"
        if not source_path.is_file() or not recovery_path.is_file():
            skipped += 1; continue
        source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        recovery = cv2.imread(str(recovery_path), cv2.IMREAD_GRAYSCALE)
        if source is None or recovery is None:
            skipped += 1; continue
        backup = run / "restored_thresholded_backup.png"
        if target_path.is_file() and not backup.exists(): shutil.copy2(target_path, backup)
        if not cv2.imwrite(str(target_path), make_readability_view(source, recovery)):
            raise RuntimeError(f"Could not update {target_path}")
        refreshed += 1
    print({"refreshed": refreshed, "skipped": skipped})


if __name__ == "__main__":
    main()
