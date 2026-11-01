from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).reshape(-1)
    return np.array([points[np.argmin(total)], points[np.argmin(difference)], points[np.argmax(total)], points[np.argmax(difference)]], dtype=np.float32)


def _crop_retains_marks(gray: np.ndarray, polygon: np.ndarray) -> bool:
    """Reject internal boxes that would discard source print outside the crop."""
    inside = np.zeros_like(gray,np.uint8)
    cv2.fillConvexPoly(inside,cv2.convexHull(polygon.astype(np.int32)),1)
    inside = cv2.dilate(inside,np.ones((11,11),np.uint8)) > 0
    darkness = cv2.GaussianBlur(gray,(0,0),3).astype(np.float32)-gray.astype(np.float32)
    marks = (darkness >= 8).astype(np.uint8)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(marks,8)
    keep = np.zeros(count,np.uint8)
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]>=4:
            keep[index]=1
    measured=keep[labels]>0
    # Black scanner backing outside a receipt is not document content. Its
    # high-contrast boundary otherwise vetoes a valid perspective correction.
    _, backing_labels, backing_stats, _ = cv2.connectedComponentsWithStats((gray<30).astype(np.uint8),8)
    frame_labels=np.unique(np.concatenate((backing_labels[0],backing_labels[-1],backing_labels[:,0],backing_labels[:,-1])))
    backing=np.zeros_like(gray,np.uint8)
    for label in frame_labels:
        if label and backing_stats[label,cv2.CC_STAT_AREA]>.01*gray.size:
            backing[backing_labels==label]=1
    backing=cv2.dilate(backing,np.ones((11,11),np.uint8))>0
    measured &= ~backing
    outside=int(np.count_nonzero(measured & ~inside))
    return outside <= 8


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
        if len(polygon) == 4 and cv2.isContourConvex(polygon) and _crop_retains_marks(gray,polygon.reshape(4,2)):
            candidate = polygon.reshape(4, 2)
            break
    if candidate is None:
        # A page touching the scan boundary has no closed Canny contour.
        # Separate midtone paper from white/black exterior, then use its hull.
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        # White paper texture alone is not proof of a page boundary. Require
        # a sizeable dark exterior touching the frame; otherwise the hull can
        # cut faint text off an already-cropped receipt.
        _, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats((gray < 25).astype(np.uint8), 8)
        edge_labels = np.unique(np.concatenate((dark_labels[0], dark_labels[-1], dark_labels[:, 0], dark_labels[:, -1])))
        exterior_area = max((int(dark_stats[label, cv2.CC_STAT_AREA]) for label in edge_labels if label != 0), default=0)
        if exterior_area < .03 * gray.size:
            return image, {"status": "not_detected", "corners": None}
        silhouette = ((smooth > 30) & (smooth < 244)).astype(np.uint8) * 255
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        paper_contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        exterior_fraction = float(np.mean((border < 20) | (border > 250)))
        for contour in sorted(paper_contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(contour) < .35 * height * width or exterior_fraction < .15:
                continue
            hull = cv2.convexHull(contour)
            for epsilon in (.015, .025, .04, .06):
                polygon = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
                if len(polygon) == 4 and cv2.isContourConvex(polygon) and _crop_retains_marks(gray,polygon.reshape(4,2)):
                    candidate = polygon.reshape(4, 2)
                    break
            if candidate is not None:
                break
        if candidate is None:
            # A curved/clipped sheet may not admit a safe quadrilateral. Keep
            # its full geometry and clear only black scanner backing outside
            # the content-retaining paper hull, rather than cutting edge text.
            for contour in sorted(paper_contours,key=cv2.contourArea,reverse=True)[:3]:
                if cv2.contourArea(contour)<.35*gray.size:
                    continue
                hull=cv2.convexHull(contour).reshape(-1,2)
                directions=[]
                for start,end in zip(hull,np.roll(hull,-1,axis=0)):
                    delta=end-start
                    if np.linalg.norm(delta)<min(gray.shape)*.25:
                        continue
                    angle=abs(float(np.degrees(np.arctan2(delta[1],delta[0]))))%90
                    directions.append(min(angle,90-angle))
                if sum(5<=angle<=35 for angle in directions)<2:
                    continue
                paper_mask=np.zeros_like(gray,np.uint8)
                cv2.fillConvexPoly(paper_mask,hull.astype(np.int32),1)
                paper_mask=cv2.dilate(paper_mask,np.ones((5,5),np.uint8))
                backing=np.zeros_like(gray,bool)
                for label in edge_labels:
                    if label and dark_stats[label,cv2.CC_STAT_AREA]>.03*gray.size:
                        backing |= dark_labels==label
                removable=backing & (paper_mask==0)
                if np.count_nonzero(removable)<.01*gray.size:
                    continue
                result=image.copy()
                result[removable]=255
                return result,{'status':'scanner_backing_removed','corners':None,'page_outline':hull.tolist(),'removed_backing_pixels':int(removable.sum())}
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
    # A portrait receipt must not be turned sideways merely because creases or
    # reverse-side print produce a stronger projection score after rotation.
    # Coarse 90-degree correction is reserved for genuinely landscape uploads.
    landscape_upload = gray.shape[1] > gray.shape[0] * 1.10
    coarse_rotation = 90 if landscape_upload and score_90 > score_0 * 1.20 else 0
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
    component_labels=np.zeros(len(stats),np.uint8)
    for label, (x, y, width, height, area) in enumerate(stats[1:], start=1):
        if 5 < area < 2500 and height < max(100, oriented_gray.shape[0] // 12) and width < max(250, oriented_gray.shape[1] // 2):
            component_labels[label]=255
    component_mask=component_labels[labels]
    linked = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    component_contours, _ = cv2.findContours(linked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    component_points = [contour.reshape(-1, 2) for contour in component_contours if cv2.contourArea(contour) > 20]
    component_skew = 0.0
    if component_points:
        (_, _), (rect_width, rect_height), rect_angle = cv2.minAreaRect(np.vstack(component_points))
        component_skew = float(rect_angle - 90.0 if rect_width >= rect_height else rect_angle)
        if component_skew > 45: component_skew -= 90
        if component_skew < -45: component_skew += 90
    # Estimate each linked text row independently. A receipt crease may be a
    # stronger edge than its print, but it does not produce several parallel
    # character rows. The weighted median therefore ignores isolated folds.
    row_angles: list[float] = []
    row_weights: list[float] = []
    for contour in component_contours:
        x, y, row_width, row_height = cv2.boundingRect(contour)
        if row_width < max(45, oriented_gray.shape[1] // 10):
            continue
        if row_height <= 0 or row_width / row_height < 2.5:
            continue
        if row_height > max(55, oriented_gray.shape[0] // 16):
            continue
        (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
        row_angle = float(angle - 90.0 if rw < rh else angle)
        while row_angle <= -45: row_angle += 90
        while row_angle > 45: row_angle -= 90
        if abs(row_angle) <= 15:
            row_angles.append(row_angle)
            row_weights.append(float(row_width))
    text_row_skew = 0.0
    if len(row_angles) >= 3:
        order = np.argsort(row_angles)
        ordered_angles = np.asarray(row_angles)[order]
        ordered_weights = np.asarray(row_weights)[order]
        cumulative = np.cumsum(ordered_weights)
        text_row_skew = float(
            ordered_angles[np.searchsorted(cumulative, cumulative[-1] / 2.0)]
        )
    # Do not rotate an already level receipt because a textured page can bias
    # the connected-component axis. Prefer the measured rule/text-line angle;
    # accept the component estimate only when line evidence is unavailable or
    # both estimators agree in direction and magnitude.
    skew = text_row_skew if len(row_angles) >= 3 else hough_skew
    if len(row_angles) >= 3:
        # Avoid unnecessary resampling for effectively level text.
        skew = text_row_skew if abs(text_row_skew) >= 0.25 else 0.0
    elif len(angles) < 3:
        # The enclosing content box can align with a fold or stain when faint
        # text provides no independent line evidence. Do not rotate on that
        # single measurement alone.
        skew = 0.0
    elif 0.5 <= abs(component_skew) <= 25 and abs(hough_skew) >= 0.35:
        same_direction = np.sign(component_skew) == np.sign(hough_skew)
        if same_direction and abs(component_skew - hough_skew) <= 1.0:
            skew = float(np.median([component_skew, hough_skew]))
    projection_skew = None
    projection_probe = {}
    supporting_angles = [angle for angle in angles if abs(angle-component_skew)<=2.0]
    corroborated_component = (5.0<=abs(component_skew)<=25
        and len(supporting_angles)>=max(3,int(np.ceil(len(angles)*.2))))
    if corroborated_component and abs(component_skew-skew)>3:
        # Horizontal linking biases sloping rows toward zero. Multiple
        # independent raw Hough segments agreeing with the component axis
        # corroborate the real tilt even when crease lines dominate the median.
        skew=float(np.median(supporting_angles))
    if not corroborated_component and abs(component_skew - skew) > 3.0 and abs(component_skew) <= 25:
        # Horizontal closing can bias sloping text rows toward zero. Score
        # the independent glyph mask at several rotations before trusting it.
        probe_binary = cv2.adaptiveThreshold(oriented_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        _, probe_labels, probe_stats, _ = cv2.connectedComponentsWithStats(probe_binary)
        probe_mask = np.zeros_like(probe_binary)
        selected_probe=np.zeros(len(probe_stats),np.uint8)
        for label, (px, py, pw, ph, pa) in enumerate(probe_stats[1:], 1):
            if (3 <= pa <= 1200 and 2 <= pw <= max(30, oriented_gray.shape[1] * .08)
                    and 3 <= ph <= max(25, oriented_gray.shape[0] * .04)):
                selected_probe[label]=255
        probe_mask=selected_probe[probe_labels]
        scale = min(1.0, 650.0 / max(probe_mask.shape))
        small = cv2.resize(probe_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        side = int(np.ceil(np.hypot(*small.shape))) + 8
        canvas = np.zeros((side, side), np.uint8)
        top, left = (side - small.shape[0]) // 2, (side - small.shape[1]) // 2
        canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
        def alignment_score(angle):
            matrix = cv2.getRotationMatrix2D((side / 2, side / 2), float(angle), 1.0)
            rotated = cv2.warpAffine(canvas, matrix, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)
            rows = rotated.sum(axis=1).astype(np.float64)
            rows = np.convolve(rows, np.ones(7) / 7.0, mode="same")
            return float(np.sum(rows * rows) / max(np.sum(rows) ** 2, 1))
        candidates = np.arange(-20.0, 20.01, 0.5)
        scores = [alignment_score(angle) for angle in candidates]
        best = float(candidates[int(np.argmax(scores))])
        projection_probe = {"best": best, "gain": max(scores) / max(alignment_score(skew), 1e-9)}
        corroborated = abs(best - component_skew) < 2.0
        if max(scores) > alignment_score(skew) * (1.01 if corroborated else 1.12):
            fine_angles = np.arange(best - .5, best + .51, .1)
            projection_skew = float(max(fine_angles, key=alignment_score))
            skew = projection_skew
    # Image-coordinate slopes increase clockwise; OpenCV positive rotations
    # counteract that slope. Negating it doubles the original tilt.
    minimum_skew = 1.5 if min(oriented_gray.shape) < 250 else 0.35
    if abs(skew) < minimum_skew:
        skew = 0.0
    corrected = _rotate_bound(oriented, skew) if abs(skew) >= minimum_skew else oriented
    return corrected, {"status": "applied", "coarse_rotation_degrees": coarse_rotation, "deskew_degrees": round(skew, 3), "projection_skew_degrees": projection_skew, "projection_probe": projection_probe, "orientation_scores": {"0": round(score_0, 4), "90": round(score_90, 4)}, "hough_skew_degrees": round(hough_skew, 3), "component_skew_degrees": round(component_skew, 3), "text_row_skew_degrees": round(text_row_skew, 3), "line_count": len(angles), "text_row_count": len(row_angles)}


def _content_crop(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if 6 < area < 2500 and height < max(100, gray.shape[0] // 12) and width < max(300, gray.shape[1] // 2):
            boxes.append(tuple(int(value) for value in (x,y,x+width,y+height)))
    if len(boxes) < 10:
        return image, {"status": "not_detected"}
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    pad_x, pad_y = max(20, int((right - left) * 0.10)), max(20, int((bottom - top) * 0.06))
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(gray.shape[1], right + pad_x), min(gray.shape[0], bottom + pad_y)
    if (right - left) * (bottom - top) < 0.12 * gray.size:
        return image, {"status": "rejected_small_region"}
    crop_polygon=np.array([[left,top],[right,top],[right,bottom],[left,bottom]],np.int32)
    if not _crop_retains_marks(gray,crop_polygon):
        return image, {"status": "rejected_content_loss", "box": [left,top,right,bottom]}
    return image[top:bottom, left:right], {"status": "content_fallback", "box": [left, top, right, bottom]}


def preprocess(image: np.ndarray, config: dict[str, Any], *, preserve_color: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
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
        backing_removed=boundary.get('status')=='scanner_backing_removed'
        working, fallback = _content_crop(working)
        if fallback.get("status") == "content_fallback":
            if boundary.get('status')=='scanner_backing_removed':
                fallback['scanner_backing_removal']=boundary
            boundary = fallback
        if backing_removed or fallback.get('status')=='rejected_content_loss':
            trim_gray=cv2.cvtColor(working,cv2.COLOR_BGR2GRAY) if working.ndim==3 else working
            ys,xs=np.where(trim_gray<253)
            if len(xs):
                left,top=max(0,int(xs.min())-8),max(0,int(ys.min())-8)
                right,bottom=min(trim_gray.shape[1],int(xs.max())+9),min(trim_gray.shape[0],int(ys.max())+9)
                working=working[top:bottom,left:right]
                boundary['blank_frame_trim']=[left,top,right,bottom]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
    limit = int(settings["max_long_edge"])
    scale = min(1.0, limit / max(gray.shape))
    result = working if preserve_color else gray
    if scale < 1.0: result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return result, {"original_shape": original_shape, "processed_shape": list(result.shape), "scale": scale, "boundary_detection": boundary, "orientation_correction": orientation, "grayscale": "registered color retained" if preserve_color else "BT.601 OpenCV conversion", "normalization": "uint8 [0,255]", "generative_processing": False}
