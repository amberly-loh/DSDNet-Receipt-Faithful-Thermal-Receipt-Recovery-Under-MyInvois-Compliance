from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    config_path = REPO / "configs" / "dsdnet_receipt_train.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_manifest = (config_path.parent / config["dataset_manifest"]).resolve()
    generation_manifest = (config_path.parent / config["generation_manifest"]).resolve()
    cuda = torch.cuda.is_available()
    report = {
        "status": "PASS" if cuda else "BLOCKED_NO_CUDA",
        "reason": "CUDA training device available" if cuda else "The current PyTorch 2.1.2 build is CPU-only; the eight-stage 64x64 smoke test exceeded the execution window.",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "config_sha256": file_hash(config_path),
        "dataset_manifest_sha256": file_hash(dataset_manifest),
        "generation_manifest_sha256": file_hash(generation_manifest),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_commit_required": config["network"]["source_commit"],
        "training_iterations": config["training"]["total_iterations"],
        "mixed_precision": config["training"]["mixed_precision"],
        "gradient_clipping": config["training"]["gradient_clipping"],
    }
    target = REPO / "artifacts" / "readiness" / "dsdnet_training_preflight.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
