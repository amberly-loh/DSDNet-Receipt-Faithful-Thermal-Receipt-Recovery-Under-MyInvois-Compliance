from __future__ import annotations

import math

import cv2
import numpy as np


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = float(np.mean((reference.astype(np.float64) - estimate.astype(np.float64)) ** 2))
    if error == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(error))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    left = reference.astype(np.float64)
    right = estimate.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    sigma_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    sigma_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    numerator = (2 * mu_left * mu_right + c1) * (2 * covariance + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (sigma_left + sigma_right + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, source in enumerate(reference, start=1):
        current = [i]
        for j, target in enumerate(hypothesis, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target))
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    return levenshtein(list(reference), list(hypothesis)) / max(len(reference), 1)


def wer(reference: str, hypothesis: str) -> float:
    return levenshtein(reference.split(), hypothesis.split()) / max(len(reference.split()), 1)

