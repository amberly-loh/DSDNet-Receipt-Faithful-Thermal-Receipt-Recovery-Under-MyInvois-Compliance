from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    for key in ("dataset_root", "artifacts_root"):
        config[key] = str((base / config[key]).resolve())
    config["dsdnet"]["source_path"] = str((base / config["dsdnet"]["source_path"]).resolve())
    checkpoint = config["dsdnet"].get("checkpoint")
    if checkpoint:
        config["dsdnet"]["checkpoint"] = str((base / checkpoint).resolve())
    support_calibration = config["dsdnet"].get("evidence_support_calibration")
    if support_calibration:
        config["dsdnet"]["evidence_support_calibration"] = str((base / support_calibration).resolve())
    artifact_mask = config.get("artifact_mask", {})
    artifact_checkpoint = artifact_mask.get("checkpoint")
    if artifact_checkpoint:
        artifact_mask["checkpoint"] = str((base / artifact_checkpoint).resolve())
    native = config.get('native_refinement', {})
    if native.get('checkpoint'):
        native['checkpoint'] = str((base / native['checkpoint']).resolve())
        native['source_path'] = config['dsdnet']['source_path']
    return config, config_path


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

