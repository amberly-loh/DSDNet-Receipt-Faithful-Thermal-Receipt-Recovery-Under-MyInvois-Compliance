from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).reshape(-1)
    return np.array([points[np.argmin(total)], points[np.argmin(difference)], points[np.argmax(total)], points[np.argmax(difference)]], dtype=np.float32)


def _rectify_receipt(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidate = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < 0.20 * height * width:
            break
        polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            candidate = polygon.reshape(4, 2)
            break
    if candidate is None:
        return image, {"status": "not_detected", "corners": None}
    top_left, top_right, bottom_right, bottom_left = _order_points(candidate)
    target_width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    target_height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    if target_width < 96 or target_height < 96:
        return image, {"status": "rejected_small_region", "corners": candidate.tolist()}
    destination = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left]), destination)
    border = (255, 255, 255) if image.ndim == 3 else 255
    rectified = cv2.warpPerspective(image, transform, (target_width, target_height), borderValue=border)
    return rectified, {"status": "rectified", "corners": candidate.tolist(), "output_shape": [target_height, target_width]}


def _projection_score(gray: np.ndarray) -> float:
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    margin_y, margin_x = max(1, gray.shape[0] // 40), max(1, gray.shape[1] // 40)
    binary[:margin_y] = 0; binary[-margin_y:] = 0; binary[:, :margin_x] = 0; binary[:, -margin_x:] = 0
    rows, columns = binary.sum(axis=1).astype(np.float64), binary.sum(axis=0).astype(np.float64)
    return float(np.var(rows / max(gray.shape[1], 1)) / (np.var(columns / max(gray.shape[0], 1)) + 1e-9))


def _rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width, new_height = int(height * sine + width * cosine), int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    border = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _orient_and_deskew(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    score_0 = _projection_score(gray)
    rotated_90, gray_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    score_90 = _projection_score(gray_90)
    coarse_rotation = 90 if score_90 > score_0 * 1.20 else 0
    oriented, oriented_gray = (rotated_90, gray_90) if coarse_rotation else (image, gray)
    edges = cv2.Canny(oriented_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(40, oriented_gray.shape[1] // 8), maxLineGap=20)
    angles: list[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            while angle <= -90: angle += 180
            while angle > 90: angle -= 180
            if abs(angle) <= 25: angles.append(angle)
    hough_skew = float(np.median(angles)) if len(angles) >= 3 else 0.0
    # Receipt edges are often torn or missing, while the collective text region
    # still provides a stable dominant axis.  Estimate that axis from plausible
    # connected ink components and use it when it is stronger than Hough lines.
    binary = cv2.adaptiveThreshold(oriented_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    component_mask = np.zeros_like(binary)
    for label, (x, y, width, height, area) in enumerate(stats[1:], start=1):
        if 5 < area < 2500 and height < max(100, oriented_gray.shape[0] // 12) and width < max(250, oriented_gray.shape[1] // 2):
            component_mask[labels == label] = 255
    linked = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    component_contours, _ = cv2.findContours(linked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    component_points = [contour.reshape(-1, 2) for contour in component_contours if cv2.contourArea(contour) > 20]
    component_skew = 0.0
    if component_points:
        (_, _), (rect_width, rect_height), rect_angle = cv2.minAreaRect(np.vstack(component_points))
        component_skew = float(rect_angle - 90.0 if rect_width >= rect_height else rect_angle)
        if component_skew > 45: component_skew -= 90
        if component_skew < -45: component_skew += 90
    # Do not rotate an already level receipt because a textured page can bias
    # the connected-component axis. Prefer the measured rule/text-line angle;
    # accept the component estimate only when line evidence is unavailable or
    # both estimators agree in direction and magnitude.
    skew = hough_skew
    if len(angles) < 3:
        skew = component_skew if 0.5 <= abs(component_skew) <= 25 else 0.0
    elif 0.5 <= abs(component_skew) <= 25 and abs(hough_skew) >= 0.35:
        same_direction = np.sign(component_skew) == np.sign(hough_skew)
        if same_direction and abs(component_skew - hough_skew) <= 1.0:
            skew = float(np.median([component_skew, hough_skew]))
    corrected = _rotate_bound(oriented, -skew) if abs(skew) >= 0.35 else oriented
    return corrected, {"status": "applied", "coarse_rotation_degrees": coarse_rotation, "deskew_degrees": round(-skew, 3), "orientation_scores": {"0": round(score_0, 4), "90": round(score_90, 4)}, "hough_skew_degrees": round(hough_skew, 3), "component_skew_degrees": round(component_skew, 3), "line_count": len(angles)}


def _content_crop(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if 6 < area < 2500 and height < max(100, gray.shape[0] // 12) and width < max(300, gray.shape[1] // 2):
            boxes.append((x, y, x + width, y + height))
    if len(boxes) < 10:
        return image, {"status": "not_detected"}
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    pad_x, pad_y = max(20, int((right - left) * 0.10)), max(20, int((bottom - top) * 0.06))
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(gray.shape[1], right + pad_x), min(gray.shape[0], bottom + pad_y)
    if (right - left) * (bottom - top) < 0.12 * gray.size:
        return image, {"status": "rejected_small_region"}
    return image[top:bottom, left:right], {"status": "content_fallback", "box": [left, top, right, bottom]}


def preprocess(image: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if image is None or image.size == 0 or image.ndim not in {2, 3}:
        raise ValueError("The uploaded image is empty or unsupported.")
    original_shape = list(image.shape)
    settings = config.get("preprocessing", {})
    working = image.copy()
    boundary = {"status": "disabled", "corners": None}
    if settings.get("boundary_detection", True): working, boundary = _rectify_receipt(working)
    orientation = {"status": "disabled", "coarse_rotation_degrees": 0, "deskew_degrees": 0.0}
    if settings.get("orientation_correction", True): working, orientation = _orient_and_deskew(working)
    if boundary.get("status") != "rectified" and settings.get("boundary_detection", True):
        working, fallback = _content_crop(working)
        if fallback.get("status") == "content_fallback": boundary = fallback
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
    limit = int(settings["max_long_edge"])
    scale = min(1.0, limit / max(gray.shape))
    if scale < 1.0: gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, {"original_shape": original_shape, "processed_shape": list(gray.shape), "scale": scale, "boundary_detection": boundary, "orientation_correction": orientation, "grayscale": "BT.601 OpenCV conversion", "normalization": "uint8 grayscale [0,255]", "generative_processing": False}
