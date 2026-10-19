from __future__ import annotations

import json
import shutil
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "RSW_AmberlyLohBintiMohdAzlanLoh_FullText.docx"
OUTPUT = ROOT.parent / "RSW_AmberlyLohBintiMohdAzlanLoh_FullText_Phase2_Ch5_Results.docx"


def shade(cell, fill):
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def set_borders(table):
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        line = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "4"), ("color", "B7B7B7")):
            line.set(qn(f"w:{key}"), value)
        borders.append(line)
    table._tbl.tblPr.append(borders)


def main():
    shutil.copy2(SOURCE, OUTPUT)
    doc = Document(OUTPUT)

    # Minimal Chapter 4 implementation clarification required by the executed run.
    parameters = doc.tables[19]
    additions = [
        ("DSDNet feature widths", "16 prior / 16 data channels", "Pinned repository runnable configuration; eight stages and four branches remain unchanged"),
        ("Deployment tiling", "128 x 128; 32-pixel overlap", "Matches the training crop and permits weighted full-receipt inference"),
        ("Source-preserving fusion", "alpha = 0.25", "Selected on validation by maximum field accuracy, then minimum CER; locked before test evaluation"),
    ]
    for row in additions:
        cells = parameters.add_row().cells
        for cell, value in zip(cells, row): cell.text = value

    start = next(p for p in doc.paragraphs if p.text.strip().startswith("Chapter 5: Experiments"))
    reference = next(p for p in doc.paragraphs if p.style.name == "Heading 1" and p.text.strip() == "References")
    # Remove every body element in the old Chapter 5/6 range, including its
    # placeholder tables, rather than deleting paragraphs alone.
    parent = start._p.getparent()
    current = start._p
    while current is not reference._p:
        following = current.getnext()
        parent.remove(current)
        current = following

    def para(text="", style=None, page_break=False):
        p = reference.insert_paragraph_before(style=style)
        if style and style.startswith("Heading"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent = 0
            p.paragraph_format.first_line_indent = 0
        if page_break: p.add_run().add_break(WD_BREAK.PAGE)
        if text: p.add_run(text)
        return p

    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers)); set_borders(t)
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for cell, value in zip(t.rows[0].cells, headers):
            cell.text = value; shade(cell, "D9EAF7")
            for run in cell.paragraphs[0].runs: run.bold = True
        for values in rows:
            cells = t.add_row().cells
            for cell, value in zip(cells, values): cell.text = value
        reference._p.addprevious(t._tbl); para("")

    summaries = []
    for path in (ROOT / "artifacts/validation_restoration").glob("checkpoint_*/summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries.append((data["checkpoint_iteration"], data["methods"]["Proposed_DSDNet_Receipt"]))
    summaries.sort()
    ocr = json.loads((ROOT / "artifacts/validation_fixed_ocr/checkpoint_0009000/summary.json").read_text(encoding="utf-8"))
    blend = json.loads((ROOT / "artifacts/validation_blend_selection/checkpoint_0009000/summary.json").read_text(encoding="utf-8"))
    logs = [json.loads(x) for x in (ROOT / "artifacts/training_development/training_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    blocks = []
    for start in range(0, 10000, 1000):
        rows = [r for r in logs if start < r["iteration"] <= start + 1000]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        blocks.append([f"{start+1}-{start+1000}", f"{mean('loss'):.4f}", f"{mean('l1'):.4f}", f"{1-mean('ssim_loss'):.4f}"])

    para("Chapter 5", "Heading 1"); para("Experiments, Results and Discussion", "Heading 1")
    para("5.1 Introduction", "Heading 2")
    para("This chapter reports the implemented system and completed development-stage experiments. Model selection used only the validation partition, and the held-out test partition was not opened. The findings guide refinement and demonstrate feasibility, but they are not confirmatory answers to the research questions.")

    para("5.2 Implemented Systems", "Heading 2"); para("5.2.1 Research Evaluation System", "Heading 3")
    para("The research evaluation system is separate from the end-user application. It executes Direct OCR, Otsu, Sauvola, the deterministic perceptual filter and DSDNet-Receipt under shared preprocessing and PaddleOCR settings. It records image metrics, OCR metrics, timing, outputs, configuration and audit identifiers. Baseline controls are not exposed as choices in the final user workflow.")
    para("5.2.2 Final User Application", "Heading 3")
    para("The final application accepts an uploaded image or camera capture, restores the receipt, performs OCR, extracts the transaction date, receipt or invoice number and total amount, and stores an audit package. For natural receipts, the true degradation operator is unavailable; source-trace support is marked unavailable and high-risk fields are routed to Human decision required. The application assists review and does not imply that uncertain fields are authentic or automatically approved.")

    para("5.3 Dataset and Evidence Boundary", "Heading 2")
    para("The project collection contains 987 Malaysian SROIE source receipts and 646 self-collected Malaysian real-life receipts. At this checkpoint, controlled degradation had completed for 611 SROIE sources, producing 1,833 variants. Source-level grouping produced 430 training sources (1,290 variants), 86 validation sources (258 variants), and 95 untouched test sources (285 variants). The locked manifest SHA-256 is 83ca768c7dd5e77c0f72828f9ebe51ff94163fae7044fd1b5a87bd28d71ebd8f. Natural receipts remain a separate robustness population and are not presented as paired clean-reference evidence.")
    table(["Population", "Sources", "Variants", "Role"], [["Controlled train", "430", "1,290", "Development training"], ["Controlled validation", "86", "258", "Checkpoint and fusion selection"], ["Controlled test", "95", "285", "Untouched; not reported here"], ["Self-collected natural", "646", "N/A", "Real-life workflow and future robustness evaluation"]])

    para("5.4 Training Implementation and Convergence", "Heading 2")
    para("The receipt adapter retained eight unrolled stages and four maxout branches. The runnable pinned DSDNet configuration uses 16 prior-feature and 16 data-feature channels; these widths were adopted because an earlier undocumented 49/48-channel class default created 633,960,496 parameters and could not fit the available 4 GB GPU. The corrected model contains 69,314,952 parameters. Training used AdamW, the specified L1 + 0.2(1-SSIM) objective, 128 x 128 grayscale crops, full precision, seed 56 and the recorded augmentation. A full-precision step reached approximately 1.92 GB peak CUDA memory.")
    para("The run completed 10,000 optimiser iterations and saved checkpoints every 1,000 iterations. This is explicitly a non-formal development run rather than the planned 1,000,000-iteration formal run. The decreasing block means show effective optimisation, but the difference in training budget remains a protocol deviation.")
    table(["Iterations", "Mean objective", "Mean L1", "Mean SSIM"], blocks)

    para("5.5 Validation Selection", "Heading 2")
    para("Checkpoint selection followed the predefined rule: minimum mean validation objective, followed by maximum SSIM and then the earliest iteration. Checkpoint 9,000 was selected with an objective of 0.05670, PSNR of 24.635 dB, SSIM of 0.8987 and L1 of 0.03644 over all 258 validation variants. Checkpoint 10,000 obtained a slightly higher SSIM but a worse objective, so it was not selected.")
    table(["Checkpoint", "Objective", "PSNR (dB)", "SSIM", "L1"], [[str(i), f"{m['mean_validation_objective']:.5f}", f"{m['mean_psnr']:.3f}", f"{m['mean_ssim']:.4f}", f"{m['mean_l1']:.4f}"] for i, m in summaries])

    para("5.6 Restoration and OCR Development Results", "Heading 2")
    para("On the complete validation image set, the selected checkpoint outperformed the best deterministic baseline objective. On a fixed five-source OCR subset containing 15 variants, it improved PSNR, SSIM, CER and WER relative to Direct OCR, while field accuracy was tied. Because CER and field accuracy are co-primary outcomes, the tie did not satisfy the development preference rule.")
    direct, raw = ocr["methods"]["A_direct"], ocr["methods"]["Proposed_DSDNet_Receipt"]
    chosen = blend["methods"][blend["selected"]]
    table(["Method", "PSNR", "SSIM", "CER", "WER", "Field accuracy"], [["Direct OCR", f"{direct['mean_psnr']:.3f}", f"{direct['mean_ssim']:.4f}", f"{direct['mean_cer']:.4f}", f"{direct['mean_wer']:.4f}", f"{direct['field_accuracy']:.4f}"], ["DSDNet, alpha=1.00", f"{raw['mean_psnr']:.3f}", f"{raw['mean_ssim']:.4f}", f"{raw['mean_cer']:.4f}", f"{raw['mean_wer']:.4f}", f"{raw['field_accuracy']:.4f}"], ["Proposed, alpha=0.25", f"{chosen['mean_psnr']:.3f}", f"{chosen['mean_ssim']:.4f}", f"{chosen['mean_cer']:.4f}", f"{chosen['mean_wer']:.4f}", f"{chosen['field_accuracy']:.4f}"]])
    para("A source-preserving fusion grid (alpha = 0.25, 0.50, 0.75 and 1.00) was evaluated on validation only. Alpha = 0.25 was selected by maximum field accuracy, then minimum CER. It achieved field accuracy 0.7333 compared with 0.6333 for Direct OCR, and CER 0.6042 compared with 0.6490. This result is promising but preliminary: it uses only five validation sources, has no confidence interval, and cannot establish statistical significance or test-set generalisation.")

    para("5.7 System Verification and Practical Context", "Heading 2")
    para("All six automated tests passed. A real-life Malaysian receipt completed the end-to-end application path with restoration status ok, 363 OCR characters and an audit record. Model loading and full OCR produced an initial runtime of approximately 19.9 seconds; a restoration-only repeat after caching took 0.88 seconds on a 384 x 512 input. These measurements demonstrate feasibility on the development machine, not a general latency guarantee.")
    para("The system is relevant to SMEs and staff who need to digitise faded purchase receipts or invoices for record checking, expense documentation and workflows adjacent to MyInvois. Its practical value is assistance: it makes weak text easier to inspect, exposes extracted values and preserves review decisions. User interest, usability, accessibility, reviewer accuracy and operational safety require a separate evaluation and are not demonstrated by the interface alone.")

    para("5.8 Research Question Status", "Heading 2")
    table(["Research question", "Development verdict", "Boundary"], [["RQ1 restoration and extraction", "Promising validation support", "Both co-primary metrics improved after validation-selected fusion on five sources; formal held-out inference remains pending"], ["RQ2 evidence support", "Not yet assessable", "Support maps, Trace-F1, AUC and calibration are not yet reported"], ["RQ3 selective routing", "Safety path implemented", "Natural inputs abstain when support is unavailable; held-out UAR and coverage remain pending"]])
    para("5.9 Limitations and Completion Work", "Heading 2")
    para("The evidence is limited by a 10,000-iteration development budget, the small OCR subset, incomplete controlled generation relative to all SROIE sources, absent cluster confidence intervals, and the lack of formal RQ2 and RQ3 results. Receipt or invoice number annotation and natural-domain ground truth remain incomplete. Before a formal superiority claim, the protocol must be completed, settings locked, the formal-run checker passed, and the held-out test partition evaluated once with source-group aggregation and uncertainty reporting.")
    para("5.10 Chapter Summary", "Heading 2")
    para("Phase 2 produced separate evaluation and end-user systems, a feasible eight-stage DSDNet-Receipt implementation, a completed 10,000-iteration development run and validation-selected source-preserving fusion. Development results outperform Direct OCR on CER and field accuracy in the fixed subset and improve image quality, while the app conservatively requires human review when source support is unavailable. These results justify continued formal evaluation but do not replace it.")

    chapter6 = para("Chapter 6", "Heading 1")
    chapter6.paragraph_format.page_break_before = True
    para("Conclusion and Recommendations", "Heading 1")
    para("6.1 Conclusion", "Heading 2")
    para("The project has implemented a Malaysian thermal-receipt recovery and evidence-aware review framework. The experimental system compares baselines with DSDNet-Receipt, while the final application supports camera or file input, restoration, OCR, field extraction, audit recording and conservative human routing. The selected development model, combined with validation-selected source-preserving fusion, improved both CER and field accuracy relative to Direct OCR on the fixed validation subset.")
    para("This conclusion is deliberately limited. The result is development evidence from five validation sources, not a formal held-out or statistically significant superiority result. Evidence-support discrimination, calibration and risk-coverage performance have not yet been established.")
    para("6.2 Objective Status", "Heading 2")
    table(["Objective", "Status", "Evidence"], [["Implement restoration and review framework", "Achieved at prototype level", "Separate systems; checkpoint-backed restoration; OCR, routing and audit path"], ["Improve restoration and extraction", "Partially achieved", "Validation gains in image metrics, CER and field accuracy; formal test pending"], ["Validate source-trace support", "Not yet achieved", "Required RQ2 metrics remain pending"], ["Validate selective safety", "Partially achieved", "Hard abstention verified; UAR and coverage remain pending"]])
    para("6.3 Contributions", "Heading 2")
    para("The current contribution is a reproducible adaptation of an unrolled DSDNet backbone to grayscale receipt recovery, combined with an auditable distinction between restoration confidence and source support. The work also provides a Malaysian controlled-data protocol, a source-preserving fusion selected for downstream OCR performance, and a user workflow that does not convert unavailable evidence into automatic acceptance.")
    para("6.4 Recommendations", "Heading 2")
    para("The next priority is to complete the planned training and controlled corpus, lock all validation-selected settings, run the conformance gate, and open the test partition once for the final paired analysis. Chapter 5 should then add source-group confidence intervals, failure cases, Trace-F1, AUC, ECE, Brier score, UAR and achieved coverage. Receipt-number and natural-receipt annotations should be completed with permission records. A small SME reviewer study may assess task completion, correction rate, perceived usefulness and accessibility; interface screenshots alone should not be treated as usability evidence.")
    para("6.5 Final Statement", "Heading 2")
    para("The development results support the feasibility and practical direction of DSDNet-Receipt: the system restores real receipt images, improves the selected validation metrics over the direct baseline, and preserves a human decision whenever the evidence required for safe acceptance is unavailable. The remaining work is to convert that encouraging development result into a protocol-conformant held-out evaluation.")

    reference.paragraph_format.page_break_before = True
    doc.save(OUTPUT); print(OUTPUT)


if __name__ == "__main__": main()
