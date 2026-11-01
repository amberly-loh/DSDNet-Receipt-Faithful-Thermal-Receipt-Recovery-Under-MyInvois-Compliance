from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .baselines import BASELINES
from .config import load_config, write_json
from .images import read_gray, write_image
from .ocr import PaddleOCRAdapter
from .preprocessing import preprocess


class ReceiptApp(tk.Tk):
    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config, _ = load_config(config_path)
        self.title("DSDNet Receipt Development Prototype")
        self.geometry("1180x760")
        self.minsize(960, 640)
        self.image: np.ndarray | None = None
        self.input_path: Path | None = None
        self.preview_refs: list[ImageTk.PhotoImage] = []
        self.method = tk.StringVar(value="C_perceptual")
        self.status = tk.StringVar(value="Select a receipt image or capture one from a camera.")
        self._build()

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="Upload receipt", command=self.upload).pack(side="left", padx=4)
        ttk.Button(controls, text="Capture from camera", command=self.capture).pack(side="left", padx=4)
        ttk.Label(controls, text="Development method").pack(side="left", padx=(24, 4))
        ttk.Combobox(controls, textvariable=self.method, values=list(BASELINES), state="readonly", width=18).pack(side="left")
        ttk.Button(controls, text="Process", command=self.process).pack(side="left", padx=8)

        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=12, pady=8)
        self.original_panel = self._image_panel(panes, "Input receipt")
        self.output_panel = self._image_panel(panes, "Processed receipt")
        panes.add(self.original_panel[0], weight=1)
        panes.add(self.output_panel[0], weight=1)

        output_frame = ttk.LabelFrame(self, text="OCR development output", padding=8)
        output_frame.pack(fill="both", padx=12, pady=(0, 8))
        self.text = tk.Text(output_frame, height=8, wrap="word")
        self.text.pack(fill="both", expand=True)
        ttk.Label(self, textvariable=self.status, padding=(12, 4)).pack(fill="x")

    def _image_panel(self, parent: ttk.Panedwindow, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, anchor="center")
        label.pack(fill="both", expand=True)
        return frame, label

    def upload(self) -> None:
        name = filedialog.askopenfilename(filetypes=[("Receipt images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not name:
            return
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
            self.image = captured
            self.input_path = None
            self._show(self.original_panel[1], self.image)
            self.status.set("Camera image captured. Select Process to continue.")

    def process(self) -> None:
        if self.image is None:
            messagebox.showwarning("No receipt", "Upload or capture a receipt first.")
            return
        processed, preprocessing = preprocess(self.image, self.config)
        method = BASELINES[self.method.get()]
        started = time.perf_counter()
        output = method(processed)
        runtime = time.perf_counter() - started
        self._show(self.output_panel[1], output)

        ocr = PaddleOCRAdapter(self.config)
        color = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        result = ocr.run(color)
        self.text.delete("1.0", "end")
        self.text.insert("end", result.text or f"OCR unavailable: {result.error or result.status}")

        run_id = f"DEV-{time.strftime('%Y%m%d-%H%M%S')}"
        root = Path(self.config["artifacts_root"]) / "ui_runs" / run_id
        write_image(root / "input.png", processed)
        write_image(root / "output.png", output)
        write_json(
            root / "audit.json",
            {
                "run_id": run_id,
                "formal_evidence": False,
                "method": self.method.get(),
                "runtime_seconds": runtime,
                "preprocessing": preprocessing,
                "ocr_version": result.engine_version,
                "ocr_status": result.status,
                "ocr_error": result.error,
                "source": str(self.input_path) if self.input_path else "camera capture",
            },
        )
        self.status.set(f"Development run saved as {run_id}. This interface does not create formal evidence.")

    def _show(self, label: ttk.Label, image: np.ndarray) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((520, 420), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo)
        self.preview_refs.append(photo)


def launch(config_path: str | Path) -> None:
    ReceiptApp(config_path).mainloop()

