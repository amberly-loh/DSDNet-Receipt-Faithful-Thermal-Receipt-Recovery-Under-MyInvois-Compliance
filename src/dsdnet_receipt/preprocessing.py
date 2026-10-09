from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def preprocess(image: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    original_shape = list(gray.shape)
    limit = int(config["preprocessing"]["max_long_edge"])
    scale = min(1.0, limit / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, {
        "original_shape": original_shape,
        "processed_shape": list(gray.shape),
        "scale": scale,
        "auto_crop": False,
        "normalization": "uint8 grayscale [0,255]",
    }

