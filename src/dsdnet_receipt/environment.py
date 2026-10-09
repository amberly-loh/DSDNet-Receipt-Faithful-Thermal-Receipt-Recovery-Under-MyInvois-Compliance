from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import write_json


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except Exception:
        return None


def collect_environment(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    git_source = Path(config["dsdnet"]["source_path"])
    source_commit = command_output(["git", "-C", str(git_source), "rev-parse", "HEAD"])
    gpu = command_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": package_version("torch"),
            "opencv-python": package_version("opencv-python"),
            "paddlepaddle": package_version("paddlepaddle"),
            "paddleocr": package_version("paddleocr"),
            "numpy": package_version("numpy"),
        },
        "cuda_driver_report": gpu,
        "dsdnet_source": str(git_source),
        "dsdnet_commit": source_commit,
        "expected_dsdnet_commit": config["dsdnet"]["source_commit"],
        "base_seed": config["base_seed"],
        "configuration_file": str(config_path),
        "formal_run_ready": bool(
            package_version("torch")
            and source_commit == config["dsdnet"]["source_commit"]
            and package_version("paddleocr") == config["ocr"]["required_version"]
        ),
    }

