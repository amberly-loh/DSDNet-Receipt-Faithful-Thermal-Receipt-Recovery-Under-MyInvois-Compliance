from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import load_config, write_json
from .fields import extract_fields, route_fields
from .images import read_gray, write_image
from .ocr import PaddleOCRAdapter
from .preprocessing import preprocess
from .restoration import RestorationEngine


class UserReceiptApp(tk.Tk):
    """Final user workflow; research baselines are intentionally absent."""

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config, _ = load_config(config_path)
        self.title("DSDNet Receipt Recovery and Review")
        self.geometry("1280x820")
        self.minsize(1020, 680)
        self.image: np.ndarray | None = None
        self.input_path: Path | None = None
        self.preview_refs: list[ImageTk.PhotoImage] = []
        self.status = tk.StringVar(value="Upload a receipt image or capture one with the camera.")
        self._build()

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="Upload receipt", command=self.upload).pack(side="left", padx=4)
        ttk.Button(controls, text="Capture from camera", command=self.capture).pack(side="left", padx=4)
        ttk.Button(controls, text="Restore and review", command=self.process).pack(side="left", padx=16)
        ttk.Button(controls, text="Clear", command=self.clear).pack(side="left", padx=4)
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=12, pady=8)
        self.original_panel = self._panel(panes, "Degraded receipt")
        self.output_panel = self._panel(panes, "Restored receipt")
        panes.add(self.original_panel[0], weight=1)
        panes.add(self.output_panel[0], weight=1)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.ocr_text = self._text_tab(notebook, "OCR text")
        self.decision_text = self._text_tab(notebook, "Fields and decisions")
        self.audit_text = self._text_tab(notebook, "Audit summary")
        ttk.Label(self, textvariable=self.status, padding=(12, 4)).pack(fill="x")

    def _panel(self, parent: ttk.Panedwindow, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, anchor="center")
        label.pack(fill="both", expand=True)
        return frame, label

    def _text_tab(self, notebook: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text=title)
        widget = tk.Text(frame, height=9, wrap="word")
        widget.pack(fill="both", expand=True)
        return widget

    def upload(self) -> None:
        name = filedialog.askopenfilename(filetypes=[("Receipt images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if name:
            self.input_path = Path(name)
            self.image = read_gray(name)
            self._show(self.original_panel[1], self.image)
            self.status.set(f"Loaded {self.input_path.name}")

    def capture(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Camera unavailable", "The default camera could not be opened.")
            return
        captured = None
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.imshow("Receipt capture - Space to capture, Esc to cancel", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                captured = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                break
            if key == 27:
                break
        camera.release()
        cv2.destroyAllWindows()
        if captured is not None:
            self.image, self.input_path = captured, None
            self._show(self.original_panel[1], self.image)
            self.status.set("Camera image captured.")

    def process(self) -> None:
        if self.image is None:
            messagebox.showwarning("No receipt", "Upload or capture a receipt first.")
            return
        started = time.perf_counter()
        processed, preprocessing_record = preprocess(self.image, self.config)
        restoration = RestorationEngine(self.config).run(processed)
        self._show(self.output_panel[1], restoration.image)
        ocr = PaddleOCRAdapter(self.config).run(restoration.image)
        fields = route_fields(
            extract_fields(ocr.text, ocr.lines),
            support_available=False,
            model_available=restoration.model_available,
        )
        run_id = f"USER-{time.strftime('%Y%m%d-%H%M%S')}"
        root = Path(self.config["artifacts_root"]) / "user_runs" / run_id
        write_image(root / "input.png", processed)
        write_image(root / "restored.png", restoration.image)
        audit = {
            "run_id": run_id,
            "source": str(self.input_path) if self.input_path else "camera capture",
            "runtime_seconds": time.perf_counter() - started,
            "restoration_status": restoration.status,
            "restoration_error": restoration.error,
            "checkpoint": restoration.checkpoint,
            "ocr_status": ocr.status,
            "ocr_version": ocr.engine_version,
            "support_status": "unavailable",
            "fields": [field.as_dict() for field in fields],
            "preprocessing": preprocessing_record,
            "formal_evidence": False,
        }
        write_json(root / "audit.json", audit)
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.insert("end", ocr.text or f"OCR unavailable: {ocr.error or ocr.status}")
        self.decision_text.delete("1.0", "end")
        for field in fields:
            self.decision_text.insert("end", f"{field.name}: {field.value or 'Not extracted'}\nDecision: {field.decision}\nReason: {field.reason}\n\n")
        self.audit_text.delete("1.0", "end")
        self.audit_text.insert("end", json.dumps(audit, indent=2, ensure_ascii=False))
        message = "Natural-source support is unavailable; human review is required."
        if not restoration.model_available:
            message = "DSDNet checkpoint unavailable; no baseline was substituted."
        self.status.set(f"Review package saved as {run_id}. {message}")

    def clear(self) -> None:
        self.image, self.input_path = None, None
        self.original_panel[1].configure(image="")
        self.output_panel[1].configure(image="")
        for widget in (self.ocr_text, self.decision_text, self.audit_text):
            widget.delete("1.0", "end")
        self.status.set("Upload a receipt image or capture one with the camera.")

    def _show(self, label: ttk.Label, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((560, 380), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo)
        self.preview_refs.append(photo)


def launch(config_path: str | Path) -> None:
    UserReceiptApp(config_path).mainloop()
