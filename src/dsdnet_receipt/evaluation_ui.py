from __future__ import annotations

import csv
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
from .metrics import psnr, ssim
from .ocr import PaddleOCRAdapter
from .preprocessing import preprocess
from .restoration import RestorationEngine


class EvaluationApp(tk.Tk):
    """Research-only baseline comparison interface."""

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config, _ = load_config(config_path)
        self.title("DSDNet Receipt Baseline Evaluation")
        self.geometry("1280x800")
        self.image: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.input_path: Path | None = None
        self.reference_path: Path | None = None
        self.outputs: dict[str, np.ndarray] = {}
        self.preview_refs: list[ImageTk.PhotoImage] = []
        self.status = tk.StringVar(value="Research interface: load a degraded receipt and optional clean reference.")
        self._build()

    def _build(self) -> None:
        controls = ttk.Frame(self, padding=12)
        controls.pack(fill="x")
        ttk.Button(controls, text="Load degraded image", command=self.load_degraded).pack(side="left", padx=4)
        ttk.Button(controls, text="Load clean reference", command=self.load_reference).pack(side="left", padx=4)
        ttk.Button(controls, text="Run all baselines", command=self.run_all).pack(side="left", padx=16)
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=12, pady=8)
        self.input_panel = self._panel(panes, "Degraded input")
        self.output_panel = self._panel(panes, "Selected baseline output")
        panes.add(self.input_panel[0], weight=1)
        panes.add(self.output_panel[0], weight=1)
        frame = ttk.LabelFrame(self, text="Baseline comparison", padding=8)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        columns = ("method", "psnr", "ssim", "runtime", "ocr_status", "ocr_lines")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for column, width in zip(columns, (170, 100, 100, 120, 170, 100)):
            self.table.heading(column, text=column.replace("_", " ").title())
            self.table.column(column, width=width, anchor="center")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.show_selected)
        ttk.Label(self, textvariable=self.status, padding=(12, 4)).pack(fill="x")

    def _panel(self, parent: ttk.Panedwindow, title: str):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        label = ttk.Label(frame, anchor="center")
        label.pack(fill="both", expand=True)
        return frame, label

    def load_degraded(self) -> None:
        name = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if name:
            self.input_path = Path(name)
            self.image = read_gray(name)
            self._show(self.input_panel[1], self.image)

    def load_reference(self) -> None:
        name = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if name:
            self.reference_path = Path(name)
            self.reference = read_gray(name)
            self.status.set(f"Clean reference loaded: {self.reference_path.name}")

    def run_all(self) -> None:
        if self.image is None:
            messagebox.showwarning("No input", "Load a degraded image first.")
            return
        processed, preprocessing_record = preprocess(self.image, self.config)
        reference = self.reference
        if reference is not None and reference.shape != processed.shape:
            reference = cv2.resize(reference, (processed.shape[1], processed.shape[0]))
        run_id = f"EVAL-{time.strftime('%Y%m%d-%H%M%S')}"
        root = Path(self.config["artifacts_root"]) / "evaluation_runs" / run_id
        ocr = PaddleOCRAdapter(self.config)
        rows = []
        self.outputs.clear()
        for item in self.table.get_children():
            self.table.delete(item)
        for method_id, method in BASELINES.items():
            started = time.perf_counter()
            output = method(processed)
            runtime = time.perf_counter() - started
            result = ocr.run(output)
            self.outputs[method_id] = output
            write_image(root / f"{method_id}.png", output)
            record = {
                "method": method_id,
                "psnr": psnr(reference, output) if reference is not None else None,
                "ssim": ssim(reference, output) if reference is not None else None,
                "runtime_seconds": runtime,
                "ocr_status": result.status,
                "ocr_lines": len(result.lines),
            }
            rows.append(record)
            self.table.insert("", "end", iid=method_id, values=(method_id, _fmt(record["psnr"]), _fmt(record["ssim"]), f"{runtime:.4f}", result.status, len(result.lines)))
        started = time.perf_counter()
        restored = RestorationEngine(self.config).run(processed)
        runtime = time.perf_counter() - started
        if restored.model_available:
            result = ocr.run(restored.image)
            self.outputs["D_proposed_dsdnet"] = restored.image
            write_image(root / "D_proposed_dsdnet.png", restored.image)
            record = {
                "method": "D_proposed_dsdnet",
                "psnr": psnr(reference, restored.image) if reference is not None else None,
                "ssim": ssim(reference, restored.image) if reference is not None else None,
                "runtime_seconds": runtime,
                "ocr_status": result.status,
                "ocr_lines": len(result.lines),
            }
            rows.append(record)
            self.table.insert("", "end", iid="D_proposed_dsdnet", values=(record["method"], _fmt(record["psnr"]), _fmt(record["ssim"]), f"{runtime:.4f}", result.status, len(result.lines)))
        else:
            self.table.insert("", "end", iid="D_proposed_unavailable", values=("D_proposed_dsdnet", "N/A", "N/A", f"{runtime:.4f}", restored.status, 0))
        root.mkdir(parents=True, exist_ok=True)
        with (root / "baseline_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(root / "run.json", {"run_id": run_id, "formal_evidence": False, "input": str(self.input_path), "reference": str(self.reference_path) if self.reference_path else None, "preprocessing": preprocessing_record})
        first = next(iter(BASELINES))
        self.table.selection_set(first)
        self._show(self.output_panel[1], self.outputs[first])
        self.status.set(f"Evaluation run saved as {run_id}. This is separate from the final user system.")

    def show_selected(self, _event=None) -> None:
        selected = self.table.selection()
        if selected and selected[0] in self.outputs:
            self._show(self.output_panel[1], self.outputs[selected[0]])

    def _show(self, label: ttk.Label, image: np.ndarray) -> None:
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        pil.thumbnail((560, 360), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.configure(image=photo)
        self.preview_refs.append(photo)


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    if np.isinf(value):
        return "inf"
    return f"{value:.4f}"


def launch(config_path: str | Path) -> None:
    EvaluationApp(config_path).mainloop()
