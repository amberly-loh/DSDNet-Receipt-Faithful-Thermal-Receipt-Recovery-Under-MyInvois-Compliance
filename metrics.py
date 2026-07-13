from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


HIGH_RISK_FIELDS = ["date", "receipt_no", "tax_id", "tax_amount", "total_amount"]


@dataclass
class FieldMetrics:
    cer: float
    wer: float
    exact_match: float
    high_risk_accuracy: float
    overall_accuracy: float


@dataclass
class FaithfulnessMetrics:
    trace_f1: float
    faithfulness_auc: float
    ece: float


@dataclass
class AbstentionMetrics:
    abstention_rate: float
    accepted_field_accuracy: float
    unsupported_acceptance_rate: float
    precision_on_accepted_high_risk: float


def compute_image_metrics(clean: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    psnr = peak_signal_noise_ratio(clean, restored, data_range=255)
    # Flexible SSIM for channel dimensions
    win_size = min(7, min(clean.shape[:2]))
    if win_size % 2 == 0:
        win_size -= 1
    
    channel_axis = 2 if clean.ndim == 3 else None
    ssim = structural_similarity(
        clean, 
        restored, 
        win_size=win_size, 
        channel_axis=channel_axis, 
        data_range=255
    )
    return {"psnr": float(psnr), "ssim": float(ssim)}


def compute_text_metrics(gt_text: str, pred_text: str) -> dict[str, float]:
    if not gt_text.strip():
        return {"cer": 0.0, "wer": 0.0}
    try:
        output = jiwer.process_words(gt_text, pred_text)
        cer = jiwer.cer(gt_text, pred_text)
        return {"cer": float(cer), "wer": float(output.wer)}
    except Exception:
        return {"cer": 1.0, "wer": 1.0}


def compute_field_accuracy(gt_fields: dict[str, str], pred_fields: dict[str, str]) -> FieldMetrics:
    total = len(gt_fields)
    correct = sum(1 for k in gt_fields if gt_fields[k].strip().lower() == pred_fields.get(k, "").strip().lower())
    high_risk_total = len(HIGH_RISK_FIELDS)
    high_risk_correct = sum(
        1
        for k in HIGH_RISK_FIELDS
        if gt_fields.get(k, "").strip().lower() == pred_fields.get(k, "").strip().lower()
    )

    # Compute CER/WER across aggregated field strings
    gt_full = " ".join(gt_fields.values())
    pred_full = " ".join(pred_fields.values())
    text_mets = compute_text_metrics(gt_full, pred_full)

    return FieldMetrics(
        cer=text_mets["cer"],
        wer=text_mets["wer"],
        exact_match=correct / max(1, total),
        high_risk_accuracy=high_risk_correct / max(1, high_risk_total),
        overall_accuracy=correct / max(1, total),
    )


def compute_trace_f1(gt_mask: np.ndarray, pred_map: np.ndarray, threshold: float = 0.3) -> float:
    pred_mask = (pred_map >= threshold).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    tp = np.logical_and(pred_mask, gt_binary).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_binary)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_binary).sum()
    if tp == 0:
        return 0.0
    return float(tp / max(1, tp + 0.5 * (fp + fn)))


def compute_faithfulness_auc(gt_mask: np.ndarray, pred_map: np.ndarray) -> float:
    values = pred_map.ravel()
    labels = (gt_mask.ravel() > 127).astype(int)
    if np.unique(labels).size < 2:
        return 0.0
    try:
        prob_true, prob_pred = calibration_curve(labels, values, n_bins=10, strategy="uniform")
        return float(auc(prob_pred, prob_true))
    except Exception:
        return 0.0


def compute_ece(gt_mask: np.ndarray, pred_map: np.ndarray, n_bins: int = 10) -> float:
    labels = (gt_mask.ravel() > 127).astype(int)
    confidences = pred_map.ravel()
    if np.unique(labels).size < 2:
        return 0.0
    try:
        prob_true, prob_pred = calibration_curve(labels, confidences, n_bins=n_bins, strategy="uniform")
        return float(np.abs(prob_true - prob_pred).mean())
    except Exception:
        return 0.0


def compute_abstention_metrics(
    gt_fields: dict[str, str],
    pred_fields: dict[str, str],
    abstained: list[str],
) -> AbstentionMetrics:
    total_fields = len(pred_fields)
    accepted_fields = [k for k in pred_fields if k not in abstained]
    accepted_correct = sum(
        1
        for k in accepted_fields
        if gt_fields.get(k, "").strip().lower() == pred_fields.get(k, "").strip().lower()
    )
    unsupported_accept = sum(
        1
        for k in accepted_fields
        if gt_fields.get(k, "").strip() == "" or gt_fields.get(k, "").strip().lower() != pred_fields.get(k, "").strip().lower()
    )
    high_risk_accepted = [k for k in accepted_fields if k in HIGH_RISK_FIELDS]
    high_risk_prec = (
        sum(
            1
            for k in high_risk_accepted
            if gt_fields.get(k, "").strip().lower() == pred_fields.get(k, "").strip().lower()
        )
        / max(1, len(high_risk_accepted))
    )
    return AbstentionMetrics(
        abstention_rate=len(abstained) / max(1, total_fields),
        accepted_field_accuracy=accepted_correct / max(1, len(accepted_fields)),
        unsupported_acceptance_rate=unsupported_accept / max(1, len(accepted_fields)),
        precision_on_accepted_high_risk=high_risk_prec,
    )