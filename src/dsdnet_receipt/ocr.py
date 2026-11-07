from __future__ import annotations

import importlib.metadata
import os
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def ocr_on_gpu() -> bool:
    """The text reader runs on the GPU when RECEIPT_OCR_GPU=1 and a GPU build is importable."""
    return os.environ.get("RECEIPT_OCR_GPU", "0") == "1"


# The GPU build of PaddlePaddle and PyTorch cannot share one process (both register the
# same CUDA types with pybind11; whichever loads second fails). The GPU text reader
# therefore runs in one helper process that never imports PyTorch; the application
# sends it images and receives the same predictions PaddleOCR returns in-process.
_GPU_POOL = None
_GPU_ENGINE = None


def _gpu_worker_init(settings: dict) -> None:
    global _GPU_ENGINE
    os.environ["RECEIPT_OCR_GPU_WORKER"] = "1"
    _prepare_gpu_runtime()
    from paddleocr import PaddleOCR
    _GPU_ENGINE = PaddleOCR(use_angle_cls=False, show_log=False, use_gpu=True, **settings)


def _gpu_worker_ocr(image: np.ndarray, det: bool):
    if det:
        return _GPU_ENGINE.ocr(image, cls=False)
    return _GPU_ENGINE.ocr(image, det=False, cls=False)


def shutdown_gpu_worker() -> None:
    """Stop the GPU text-reader helper process (also registered to run at exit), so a
    finished script never waits on it."""
    global _GPU_POOL
    pool, _GPU_POOL = _GPU_POOL, None
    if pool is None:
        return
    for process in list(getattr(pool, "_processes", {}).values()):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _gpu_pool(settings: dict):
    global _GPU_POOL
    if _GPU_POOL is None:
        import atexit
        atexit.register(shutdown_gpu_worker)
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        _GPU_POOL = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_gpu_worker_init, initargs=(settings,))
    return _GPU_POOL


class _GpuEngine:
    """Same .ocr() interface as PaddleOCR, served by the GPU helper process."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings

    def ocr(self, image, cls=False, det=True):
        return _gpu_pool(self.settings).submit(_gpu_worker_ocr, image, det).result()


def _prepare_gpu_runtime() -> None:
    # On Windows the GPU build of PaddlePaddle needs the CUDA 11 / cuDNN 8 libraries;
    # the PyTorch CUDA 11.8 wheel already ships them, so reuse that folder. PyTorch itself
    # must not be imported first: both libraries register the same GPU type with pybind11,
    # so PaddlePaddle has to be imported before PyTorch (see dsdnet_receipt/__init__.py).
    if os.name == "nt":
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            lib = os.path.join(os.path.dirname(spec.origin), "lib")
            if os.path.isdir(lib):
                os.add_dll_directory(lib)
                os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    # Share the 4 GB laptop GPU with DSDNet: grow on demand instead of reserving most of it.
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")


@dataclass
class OCRResult:
    text: str
    lines: list[dict[str, Any]]
    engine_version: str
    status: str
    error: str | None = None


class PaddleOCRAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.version = importlib.metadata.version("paddleocr")
        self.required_version = str(config["ocr"]["required_version"])
        self._engine = None

    @property
    def version_matches_protocol(self) -> bool:
        return self.version == self.required_version

    def _load(self) -> Any:
        if self._engine is None:
            gpu = ocr_on_gpu() or bool(self.config["ocr"].get("use_gpu", False))
            if gpu and self.version.startswith("2.7"):
                self._engine = _GpuEngine({
                    "lang": self.config["ocr"]["language"],
                    "ocr_version": self.config["ocr"]["ocr_version"],
                    "gpu_mem": int(self.config["ocr"].get("gpu_mem", 800)),
                })
                return self._engine
            from paddleocr import PaddleOCR

            if self.version.startswith("2.7"):
                self._engine = PaddleOCR(
                    lang=self.config["ocr"]["language"],
                    ocr_version=self.config["ocr"]["ocr_version"],
                    use_angle_cls=False,
                    show_log=False,
                    enable_mkldnn=(not gpu) and bool(self.config["ocr"].get("enable_mkldnn", False)) and os.getenv("FLAGS_use_mkldnn", "1") != "0",
                    cpu_threads=int(self.config["ocr"].get("cpu_threads", 1)),
                    use_gpu=gpu,
                    gpu_mem=int(self.config["ocr"].get("gpu_mem", 800)),
                )
            else:
                self._engine = PaddleOCR(
                    lang=self.config["ocr"]["language"],
                    use_doc_orientation_classify=self.config["ocr"]["use_doc_orientation_classify"],
                    use_doc_unwarping=self.config["ocr"]["use_doc_unwarping"],
                    use_textline_orientation=self.config["ocr"]["use_textline_orientation"],
                )
        return self._engine

    def run(self, image: np.ndarray) -> OCRResult:
        try:
            if image.ndim == 2:
                import cv2

                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            engine = self._load()
            if self.version.startswith("2.7"):
                prediction = engine.ocr(image, cls=False)
                lines: list[dict[str, Any]] = []
                for page in prediction or []:
                    for item in page or []:
                        box, recognition = item
                        text, score = recognition
                        lines.append({"text": str(text), "confidence": float(score), "box": box})
                return OCRResult(
                    text="\n".join(line["text"] for line in lines),
                    lines=lines,
                    engine_version=self.version,
                    status="ok" if self.version_matches_protocol else "protocol_version_mismatch",
                )
            prediction = engine.predict(image)
            lines: list[dict[str, Any]] = []
            for page in prediction:
                payload = page.json if hasattr(page, "json") else {}
                result = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                boxes = result.get("rec_boxes", [])
                for index, text in enumerate(texts):
                    lines.append(
                        {
                            "text": str(text),
                            "confidence": float(scores[index]) if index < len(scores) else None,
                            "box": boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else (boxes[index] if index < len(boxes) else None),
                        }
                    )
            return OCRResult(
                text="\n".join(line["text"] for line in lines),
                lines=lines,
                engine_version=self.version,
                status="ok" if self.version_matches_protocol else "protocol_version_mismatch",
            )
        except Exception as error:
            return OCRResult("", [], self.version, "failed", f"{type(error).__name__}: {error}")

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Read the whole image as one text line (no line detection)."""
        try:
            if image.ndim == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            engine = self._load()
            if self.version.startswith("2.7"):
                prediction = engine.ocr(image, det=False, cls=False)
                text, score = prediction[0][0]
                return str(text), float(score)
        except Exception:
            pass
        return "", 0.0

    def run_views(self, views: list[tuple[str, np.ndarray]]) -> OCRResult:
        """Run the fixed OCR configuration on faithful views and merge duplicates."""
        merged: list[dict[str, Any]] = []
        errors: list[str] = []
        seen_views: set[tuple] = set()
        for view_name, image in views:
            signature=(image.shape,str(image.dtype),hashlib.sha256(image.tobytes()).digest())
            if signature in seen_views:
                continue
            seen_views.add(signature)
            result = self.run(image)
            if result.error:
                errors.append(f"{view_name}: {result.error}")
            for line in result.lines:
                normalized = _normalise_text(line.get("text"))
                if not normalized:
                    continue
                candidate = dict(line)
                candidate["source_view"] = view_name
                duplicate_index = next(
                    (index for index, old in enumerate(merged)
                     if _normalise_text(old.get("text")) == normalized
                     and _same_physical_line(old.get("box"), candidate.get("box"))),
                    None,
                )
                if duplicate_index is None:
                    merged.append(candidate)
                elif float(candidate.get("confidence") or 0) > float(merged[duplicate_index].get("confidence") or 0):
                    merged[duplicate_index] = candidate
        lines = sorted(merged, key=lambda item: _line_position(item.get("box")))
        return OCRResult("\n".join(str(line["text"]) for line in lines), lines, self.version,
                         "ok" if lines else "failed", "; ".join(errors) if errors and not lines else None)


def deduplicate_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate previously persisted or externally supplied OCR records."""
    merged: list[dict[str, Any]] = []
    for line in lines:
        normalized = _normalise_text(line.get("text"))
        if not normalized: continue
        index = next((i for i, old in enumerate(merged) if _normalise_text(old.get("text")) == normalized and _same_physical_line(old.get("box"), line.get("box"))), None)
        if index is None: merged.append(line)
        elif float(line.get("confidence") or 0) > float(merged[index].get("confidence") or 0): merged[index] = line
    return sorted(merged, key=lambda item: _line_position(item.get("box")))


def _line_position(box: Any) -> tuple[float, float]:
    try:
        return (min(float(point[1]) for point in box), min(float(point[0]) for point in box))
    except (TypeError, ValueError, IndexError):
        return (float("inf"), float("inf"))


def _normalise_text(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _box_bounds(box: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        if not xs or not ys: return None
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _same_physical_line(first: Any, second: Any) -> bool:
    """Match the same OCR line across views without merging repeated values."""
    a, b = _box_bounds(first), _box_bounds(second)
    if a is None or b is None:
        return first is None and second is None
    ax, ay, ar, ab = a; bx, by, br, bb = b
    aw, ah = max(ar - ax, 1.0), max(ab - ay, 1.0)
    bw, bh = max(br - bx, 1.0), max(bb - by, 1.0)
    centre_dx = abs((ax + ar) / 2 - (bx + br) / 2)
    centre_dy = abs((ay + ab) / 2 - (by + bb) / 2)
    return centre_dx <= max(12.0, .18 * max(aw, bw)) and centre_dy <= max(8.0, .45 * max(ah, bh))
