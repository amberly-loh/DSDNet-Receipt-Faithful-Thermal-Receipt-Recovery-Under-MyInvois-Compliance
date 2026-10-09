from __future__ import annotations

import cv2
import numpy as np


def direct_extraction(image: np.ndarray) -> np.ndarray:
    return image.copy()


def otsu(image: np.ndarray) -> np.ndarray:
    _, output = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return output


def sauvola(image: np.ndarray, window: int = 31, k: float = 0.2, dynamic_range: float = 128.0) -> np.ndarray:
    if window % 2 == 0:
        raise ValueError("Sauvola window size must be odd")
    source = image.astype(np.float32)
    mean = cv2.boxFilter(source, cv2.CV_32F, (window, window), normalize=True)
    mean_square = cv2.boxFilter(source * source, cv2.CV_32F, (window, window), normalize=True)
    deviation = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    threshold = mean * (1.0 + k * (deviation / dynamic_range - 1.0))
    return np.where(source > threshold, 255, 0).astype(np.uint8)


def perceptual_filter(image: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    unsharp = cv2.addWeighted(image, 1.6, smooth, -0.6, 0)
    bilateral = cv2.bilateralFilter(unsharp, d=7, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(bilateral)


BASELINES = {
    "A_direct": direct_extraction,
    "B_otsu": otsu,
    "B_sauvola": sauvola,
    "C_perceptual": perceptual_filter,
}

