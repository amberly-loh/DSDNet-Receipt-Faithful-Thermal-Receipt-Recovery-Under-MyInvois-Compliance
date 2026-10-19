from __future__ import annotations
import json, sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess

config, _ = load_config(ROOT / "configs" / "default.json")
all_files = [p for p in Path(config["dataset_root"]).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = int(sys.argv[2]) if len(sys.argv) > 2 else len(all_files)
files = all_files[offset:offset + count]
result = {"total": len(files), "decoded": 0, "failed": [], "boundary": {}, "coarse_rotated": 0, "deskewed": 0}
started = time.time()
for index, path in enumerate(files, 1):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        result["failed"].append({"file": str(path), "error": "decode failed"}); continue
    try:
        _, record = preprocess(image, config); result["decoded"] += 1
        status = record["boundary_detection"]["status"]; result["boundary"][status] = result["boundary"].get(status, 0) + 1
        result["coarse_rotated"] += int(record["orientation_correction"].get("coarse_rotation_degrees", 0) != 0)
        result["deskewed"] += int(abs(record["orientation_correction"].get("deskew_degrees", 0)) >= .35)
    except Exception as error:
        result["failed"].append({"file": str(path), "error": f"{type(error).__name__}: {error}"})
    if index % 50 == 0: print(f"checked {index}/{len(files)}", flush=True)
result["seconds"] = time.time() - started
target = Path(config["artifacts_root"]) / "system_verification" / f"natural_preprocessing_{offset}_{offset + len(files)}.json"
target.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
