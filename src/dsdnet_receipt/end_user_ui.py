from __future__ import annotations

import json
import os
import threading
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


NAVY = "#12304A"
BLUE = "#176B9C"
LIGHT_BLUE = "#EAF4FA"
PALE = "#F5F8FA"
WHITE = "#FFFFFF"
TEXT = "#17232D"
MUTED = "#607381"
GREEN = "#237A57"
AMBER = "#A96500"
RED = "#A33A3A"
BORDER = "#D6E0E6"


class ReceiptRecoveryApp(tk.Tk):
    """Guided end-user receipt workflow. Research baselines are excluded."""

    def __init__(self, config_path: str | Path) -> None:
        super().__init__()
        self.config, _ = load_config(config_path)
        self.title("Receipt Recovery Assistant")
        self.geometry("1420x900")
        self.minsize(1120, 720)
        self.configure(bg=PALE)
        self.image: np.ndarray | None = None
        self.input_path: Path | None = None
        self.latest_audit_dir: Path | None = None
        self.preview_refs: dict[str, ImageTk.PhotoImage] = {}
        self.is_processing = False
        self.status_text = tk.StringVar(value="Choose a receipt image or capture one using the camera.")
        self.file_text = tk.StringVar(value="No receipt selected")
        self._configure_styles()
        self._build()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), foreground=WHITE, background=BLUE, padding=(15, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#0E587F"), ("disabled", "#A7BAC6")])
        style.configure("Secondary.TButton", font=("Segoe UI", 10, "bold"), foreground=NAVY, background=WHITE, padding=(14, 9), bordercolor=BORDER)
        style.map("Secondary.TButton", background=[("active", LIGHT_BLUE)])
        style.configure("Quiet.TButton", font=("Segoe UI", 9), foreground=BLUE, background=WHITE, padding=(9, 6), borderwidth=0)
        style.configure("Results.Treeview", font=("Segoe UI", 10), rowheight=42, background=WHITE, fieldbackground=WHITE, bordercolor=BORDER)
        style.configure("Results.Treeview.Heading", font=("Segoe UI", 10, "bold"), foreground=NAVY, background="#DDECF5", padding=8)
        style.map("Results.Treeview", background=[("selected", "#DCEEF8")], foreground=[("selected", TEXT)])
        style.configure("TNotebook", background=PALE, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(16, 9), background="#E4EBEF", foreground=MUTED)
        style.map("TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", NAVY)])
        style.configure("Horizontal.TProgressbar", background=BLUE, troughcolor="#DCE5EA", borderwidth=0)

    def _build(self) -> None:
        self._build_header()
        self._build_steps()
        workspace = tk.Frame(self, bg=PALE)
        workspace.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        workspace.grid_columnconfigure(0, weight=0, minsize=290)
        workspace.grid_columnconfigure(1, weight=1)
        workspace.grid_rowconfigure(0, weight=1)
        self._build_control_card(workspace)
        self._build_workspace(workspace)
        self._build_status_bar()

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=NAVY, height=92)
        header.pack(fill="x")
        header.pack_propagate(False)
        text = tk.Frame(header, bg=NAVY)
        text.pack(side="left", padx=28, pady=18)
        tk.Label(text, text="Receipt Recovery Assistant", font=("Segoe UI", 21, "bold"), bg=NAVY, fg=WHITE).pack(anchor="w")
        tk.Label(text, text="Recover, inspect and verify important receipt fields", font=("Segoe UI", 10), bg=NAVY, fg="#CFE2EE").pack(anchor="w", pady=(3, 0))
        badge = tk.Label(header, text="HUMAN REVIEW ENABLED", font=("Segoe UI", 9, "bold"), bg="#D8EFE5", fg=GREEN, padx=13, pady=7)
        badge.pack(side="right", padx=28)

    def _build_steps(self) -> None:
        strip = tk.Frame(self, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        strip.pack(fill="x", padx=24, pady=16)
        self.step_labels = []
        for number, title, detail in (
            (1, "Add receipt", "Upload or camera"),
            (2, "Recover text", "Restoration and OCR"),
            (3, "Verify fields", "Review before use"),
        ):
            block = tk.Frame(strip, bg=WHITE)
            block.pack(side="left", expand=True, fill="x", padx=18, pady=11)
            circle = tk.Label(block, text=str(number), font=("Segoe UI", 10, "bold"), width=3, height=1, bg="#DCE8EF", fg=NAVY)
            circle.pack(side="left", padx=(0, 10))
            labels = tk.Frame(block, bg=WHITE)
            labels.pack(side="left")
            tk.Label(labels, text=title, font=("Segoe UI", 10, "bold"), bg=WHITE, fg=TEXT).pack(anchor="w")
            tk.Label(labels, text=detail, font=("Segoe UI", 8), bg=WHITE, fg=MUTED).pack(anchor="w")
            self.step_labels.append(circle)
        self._set_step(1)

    def _build_control_card(self, parent: tk.Frame) -> None:
        card = tk.Frame(parent, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        tk.Label(card, text="Add a receipt", font=("Segoe UI", 15, "bold"), bg=WHITE, fg=TEXT).pack(anchor="w", padx=20, pady=(20, 4))
        tk.Label(card, text="Use a clear photo with the full receipt visible.", wraplength=240, justify="left", font=("Segoe UI", 9), bg=WHITE, fg=MUTED).pack(anchor="w", padx=20, pady=(0, 18))
        ttk.Button(card, text="Upload receipt image", style="Primary.TButton", command=self.upload).pack(fill="x", padx=20, pady=5)
        ttk.Button(card, text="Capture with camera", style="Secondary.TButton", command=self.capture).pack(fill="x", padx=20, pady=5)
        file_box = tk.Frame(card, bg=PALE, highlightbackground=BORDER, highlightthickness=1)
        file_box.pack(fill="x", padx=20, pady=18)
        tk.Label(file_box, text="SELECTED RECEIPT", font=("Segoe UI", 8, "bold"), bg=PALE, fg=MUTED).pack(anchor="w", padx=12, pady=(10, 2))
        tk.Label(file_box, textvariable=self.file_text, wraplength=220, justify="left", font=("Segoe UI", 9), bg=PALE, fg=TEXT).pack(anchor="w", padx=12, pady=(0, 10))
        self.process_button = ttk.Button(card, text="Recover and analyse", style="Primary.TButton", command=self.process, state="disabled")
        self.process_button.pack(fill="x", padx=20, pady=(4, 8))
        self.progress = ttk.Progressbar(card, mode="indeterminate")
        self.progress.pack(fill="x", padx=20, pady=(0, 14))
        tk.Frame(card, bg=BORDER, height=1).pack(fill="x", padx=20, pady=6)
        tk.Label(card, text="Safety note", font=("Segoe UI", 10, "bold"), bg=WHITE, fg=NAVY).pack(anchor="w", padx=20, pady=(10, 3))
        tk.Label(card, text="Always compare extracted values with the receipt image. The app routes uncertain or unsupported fields for human review.", wraplength=245, justify="left", font=("Segoe UI", 9), bg=WHITE, fg=MUTED).pack(anchor="w", padx=20)
        self.audit_button = ttk.Button(card, text="Open latest audit folder", style="Quiet.TButton", command=self.open_audit, state="disabled")
        self.audit_button.pack(anchor="w", padx=12, pady=18)

    def _build_workspace(self, parent: tk.Frame) -> None:
        area = tk.Frame(parent, bg=PALE)
        area.grid(row=0, column=1, sticky="nsew")
        area.grid_columnconfigure(0, weight=1)
        area.grid_rowconfigure(0, weight=3)
        area.grid_rowconfigure(1, weight=2)
        images = tk.Frame(area, bg=PALE)
        images.grid(row=0, column=0, sticky="nsew")
        images.grid_columnconfigure(0, weight=1)
        images.grid_columnconfigure(1, weight=1)
        images.grid_rowconfigure(0, weight=1)
        self.original_canvas = self._image_card(images, 0, "Original receipt", "Input image")
        self.restored_canvas = self._image_card(images, 1, "Recovered receipt", "DSDNet output")
        notebook = ttk.Notebook(area)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        results_frame = tk.Frame(notebook, bg=WHITE)
        text_frame = tk.Frame(notebook, bg=WHITE)
        audit_frame = tk.Frame(notebook, bg=WHITE)
        notebook.add(results_frame, text="Important fields")
        notebook.add(text_frame, text="Recognised text")
        notebook.add(audit_frame, text="Run details")
        columns = ("field", "value", "confidence", "action", "reason")
        self.results = ttk.Treeview(results_frame, columns=columns, show="headings", style="Results.Treeview")
        headings = {"field": "Field", "value": "Extracted value", "confidence": "OCR confidence", "action": "Action", "reason": "Reason"}
        widths = {"field": 170, "value": 190, "confidence": 120, "action": 190, "reason": 430}
        for name in columns:
            self.results.heading(name, text=headings[name])
            self.results.column(name, width=widths[name], anchor="w", stretch=name == "reason")
        self.results.pack(fill="both", expand=True, padx=12, pady=12)
        self.ocr_text = tk.Text(text_frame, wrap="word", relief="flat", font=("Consolas", 10), bg=WHITE, fg=TEXT, padx=14, pady=14)
        self.ocr_text.pack(fill="both", expand=True)
        self.audit_text = tk.Text(audit_frame, wrap="word", relief="flat", font=("Consolas", 9), bg=WHITE, fg=TEXT, padx=14, pady=14)
        self.audit_text.pack(fill="both", expand=True)

    def _image_card(self, parent: tk.Frame, column: int, title: str, subtitle: str) -> tk.Canvas:
        card = tk.Frame(parent, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
        header = tk.Frame(card, bg=WHITE)
        header.pack(fill="x", padx=16, pady=(13, 8))
        tk.Label(header, text=title, font=("Segoe UI", 12, "bold"), bg=WHITE, fg=TEXT).pack(side="left")
        tk.Label(header, text=subtitle, font=("Segoe UI", 8), bg=LIGHT_BLUE, fg=BLUE, padx=8, pady=3).pack(side="right")
        canvas = tk.Canvas(card, bg="#EFF3F5", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        canvas.bind("<Configure>", lambda _event, c=canvas: self._redraw_canvas(c))
        canvas.image_array = None
        self._draw_placeholder(canvas, "Receipt preview will appear here")
        return canvas

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x", side="bottom")
        self.status_dot = tk.Label(bar, text="●", font=("Segoe UI", 11), bg=WHITE, fg=BLUE)
        self.status_dot.pack(side="left", padx=(24, 8), pady=9)
        tk.Label(bar, textvariable=self.status_text, font=("Segoe UI", 9), bg=WHITE, fg=TEXT).pack(side="left", pady=9)
        tk.Label(bar, text="Local processing", font=("Segoe UI", 8), bg=WHITE, fg=MUTED).pack(side="right", padx=24)

    def upload(self) -> None:
        name = filedialog.askopenfilename(title="Choose a receipt image", filetypes=[("Receipt images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not name:
            return
        try:
            self.image = read_gray(name)
        except Exception as error:
            messagebox.showerror("Image could not be opened", str(error))
            return
        self.input_path = Path(name)
        self.file_text.set(f"{self.input_path.name}\n{self.image.shape[1]} × {self.image.shape[0]} pixels")
        self._show(self.original_canvas, self.image)
        self._clear_output()
        self.process_button.configure(state="normal")
        self._set_step(1)
        self._status("Receipt loaded. Select Recover and analyse to continue.", BLUE)

    def capture(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Camera unavailable", "The default camera could not be opened. Check camera permissions or upload an image instead.")
            return
        captured = None
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            cv2.putText(frame, "SPACE: capture   ESC: cancel", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 80), 2)
            cv2.imshow("Receipt camera", frame)
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
            self.file_text.set(f"Camera capture\n{captured.shape[1]} × {captured.shape[0]} pixels")
            self._show(self.original_canvas, captured)
            self._clear_output()
            self.process_button.configure(state="normal")
            self._set_step(1)
            self._status("Camera image captured. Select Recover and analyse.", BLUE)

    def process(self) -> None:
        if self.image is None or self.is_processing:
            return
        self.is_processing = True
        self.process_button.configure(state="disabled", text="Processing receipt…")
        self.progress.start(12)
        self._set_step(2)
        self._status("Recovering the receipt and recognising text. Please wait…", BLUE)
        image = self.image.copy()
        source = str(self.input_path) if self.input_path else "camera capture"
        threading.Thread(target=self._process_worker, args=(image, source), daemon=True).start()

    def _process_worker(self, image: np.ndarray, source: str) -> None:
        try:
            started = time.perf_counter()
            processed, preprocessing_record = preprocess(image, self.config)
            restoration = RestorationEngine(self.config).run(processed)
            ocr = PaddleOCRAdapter(self.config).run(restoration.image)
            fields = route_fields(extract_fields(ocr.text, ocr.lines), support_available=False, model_available=restoration.model_available)
            run_id = f"USER-{time.strftime('%Y%m%d-%H%M%S')}"
            root = Path(self.config["artifacts_root"]) / "user_runs" / run_id
            write_image(root / "input.png", processed)
            write_image(root / "restored.png", restoration.image)
            audit = {
                "run_id": run_id,
                "source": source,
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
            self.after(0, lambda: self._finish_process(restoration.image, ocr.text, ocr.error, fields, audit, root))
        except Exception as error:
            self.after(0, lambda: self._fail_process(error))

    def _finish_process(self, image, text, ocr_error, fields, audit, root: Path) -> None:
        self.is_processing = False
        self.progress.stop()
        self.process_button.configure(state="normal", text="Recover and analyse")
        self.latest_audit_dir = root
        self.audit_button.configure(state="normal")
        self._show(self.restored_canvas, image)
        for item in self.results.get_children():
            self.results.delete(item)
        labels = {"transaction_date": "Transaction date", "receipt_invoice_number": "Receipt / invoice number", "total_amount": "Total amount"}
        for field in fields:
            confidence = "Not available" if field.ocr_confidence is None else f"{field.ocr_confidence:.1%}"
            self.results.insert("", "end", values=(labels.get(field.name, field.name), field.value or "Not found", confidence, field.decision, field.reason))
        self.ocr_text.delete("1.0", "end")
        self.ocr_text.insert("end", text or f"OCR unavailable: {ocr_error or 'No text detected'}")
        self.audit_text.delete("1.0", "end")
        self.audit_text.insert("end", json.dumps(audit, indent=2, ensure_ascii=False))
        self._set_step(3)
        if audit["restoration_status"] == "ok":
            self._status("Analysis completed. Verify every extracted field before use.", AMBER)
        else:
            self._status("Review required: a trained DSDNet checkpoint is not available, and no baseline was substituted.", RED)

    def _fail_process(self, error: Exception) -> None:
        self.is_processing = False
        self.progress.stop()
        self.process_button.configure(state="normal", text="Recover and analyse")
        self._set_step(1)
        self._status("The receipt could not be processed.", RED)
        messagebox.showerror("Processing failed", f"{type(error).__name__}: {error}")

    def open_audit(self) -> None:
        if self.latest_audit_dir and self.latest_audit_dir.exists():
            os.startfile(self.latest_audit_dir)

    def _clear_output(self) -> None:
        self.restored_canvas.image_array = None
        self._draw_placeholder(self.restored_canvas, "Recovered image will appear here")
        for item in self.results.get_children():
            self.results.delete(item)
        self.ocr_text.delete("1.0", "end")
        self.audit_text.delete("1.0", "end")

    def _show(self, canvas: tk.Canvas, image: np.ndarray) -> None:
        canvas.image_array = image.copy()
        self._redraw_canvas(canvas)

    def _redraw_canvas(self, canvas: tk.Canvas) -> None:
        image = getattr(canvas, "image_array", None)
        if image is None:
            return
        width, height = max(canvas.winfo_width() - 24, 80), max(canvas.winfo_height() - 24, 80)
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        pil.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, image=photo, anchor="center")
        self.preview_refs[str(canvas)] = photo

    def _draw_placeholder(self, canvas: tk.Canvas, text: str) -> None:
        canvas.delete("all")
        canvas.create_text(max(canvas.winfo_width(), 300) // 2, max(canvas.winfo_height(), 220) // 2, text=text, fill=MUTED, font=("Segoe UI", 10))

    def _set_step(self, active: int) -> None:
        for index, label in enumerate(self.step_labels, start=1):
            label.configure(bg=BLUE if index == active else "#DCE8EF", fg=WHITE if index == active else NAVY)

    def _status(self, text: str, colour: str) -> None:
        self.status_text.set(text)
        self.status_dot.configure(fg=colour)


def launch(config_path: str | Path) -> None:
    ReceiptRecoveryApp(config_path).mainloop()
