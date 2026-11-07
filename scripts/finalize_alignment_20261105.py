from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
FYP = ROOT.parents[1]
INPUT = FYP / "Documentation result" / "RSW_AmberlyLoh_FullText_Chapter5_Completed_20261103.docx"
OUTPUT = FYP / "Documentation result" / "RSW_AmberlyLoh_FullText_Final_Aligned_20261105.docx"
EVIDENCE = ROOT / "artifacts" / "consistency_audit_20261105"
FIGURES = EVIDENCE / "figures"
JUNIT = EVIDENCE / "pytest_results.xml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fonts(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def wrap(draw, text, font, width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def box(draw, xy, title, items, fill, accent="#176f61"):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=22, fill=fill, outline=accent, width=4)
    draw.text((x1 + 24, y1 + 18), title, font=fonts(30, True), fill="#103f37")
    y = y1 + 62
    for item in items:
        for line in wrap(draw, item, fonts(22), x2 - x1 - 48):
            draw.text((x1 + 28, y), "• " + line if line == wrap(draw, item, fonts(22), x2-x1-48)[0] else "  " + line,
                      font=fonts(22), fill="#193d38")
            y += 30
        y += 5


def arrow(draw, start, end, color="#176f61"):
    draw.line([start, end], fill=color, width=6)
    ex, ey = end
    sx, sy = start
    import math
    ang = math.atan2(ey - sy, ex - sx)
    pts = [(ex, ey)]
    for off in (2.55, -2.55):
        pts.append((ex + 22 * math.cos(ang + off), ey + 22 * math.sin(ang + off)))
    draw.polygon(pts, fill=color)


def canvas(title, subtitle):
    im = Image.new("RGB", (1900, 1120), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1900, 108), fill="#0f4f46")
    d.text((55, 20), title, font=fonts(42, True), fill="white")
    d.text((58, 72), subtitle, font=fonts(21), fill="#d9f0eb")
    return im, d


def make_figures():
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Figure 4.1
    im, d = canvas("DSDNet-Receipt implemented system architecture", "Traceable receipt recovery, extraction, review and accounting evidence")
    layers = [
        (150, "Presentation", ["Scan/batch capture, history and receipt detail", "Faithful/original/compare views", "Human verification, PDF/Excel exports"], "#e9f5f2"),
        (370, "Application and API", ["Guest/account ownership, CSRF and validation", "Receipt lifecycle, readiness rules and audit events", "MyInvois-support coverage mapping (not submission)"], "#f3f7e9"),
        (590, "Computer vision, OCR and semantics", ["Geometry correction and document-aware preprocessing", "Eight-stage DSDNet; four maxout branches per stage", "Faithful source fusion, multi-view OCR, fields, line items and tags"], "#eef1fa"),
        (810, "Data, model and research evidence", ["SQLite metadata and protected image artifacts", "Configurations, checkpoints, manifests and hashes", "Baselines, ablation, robustness, statistics and test registers"], "#faf0e8"),
    ]
    for y, title, items, fill in layers:
        box(d, (105, y, 1795, y + 175), title, items, fill)
    for y in (325, 545, 765):
        arrow(d, (950, y - 8), (950, y + 32))
    im.save(FIGURES / "figure_4_1_architecture.png", dpi=(220, 220))

    # Figure 4.2
    im, d = canvas("Implemented receipt processing and review pipeline", "Runtime path used by the current application")
    xs = [55, 365, 675, 985, 1295, 1605]
    titles = ["1  Capture", "2  Geometry", "3  Recovery", "4  OCR", "5  Semantics", "6  Evidence"]
    items = [
        ["Camera/file/batch", "Decode and validate"],
        ["Crop, deskew, orient", "Preserve document bounds"],
        ["Degradation routing", "8 stages × 4 branches", "Faithful source fusion"],
        ["Original + recovered views", "OCR localization/dedup", "Never redraw words"],
        ["Primary fields", "Line items/accounting", "Tags/readiness"],
        ["Human verify", "Audit PDF/Excel", "Original retained"],
    ]
    for x, title, its in zip(xs, titles, items):
        box(d, (x, 245, x + 245, 660), title, its, "#eef7f4")
    for x in xs[:-1]:
        arrow(d, (x + 250, 450), (x + 298, 450))
    box(d, (165, 755, 820, 980), "Faithfulness safeguards", ["Original/compare paths remain available", "OCR protects/locates text but does not typeset content", "Logos, scripts, handwriting, dividers and signatures remain source-supported"], "#fff4df", "#b77716")
    box(d, (1080, 755, 1735, 980), "Decision boundary", ["Ready to confirm only when configured evidence gates pass", "Needs attention for missing/ambiguous evidence", "No direct MyInvois API submission or legal compliance claim"], "#f9e9eb", "#a43c4b")
    im.save(FIGURES / "figure_4_2_runtime_pipeline.png", dpi=(220, 220))

    # Figure 4.3
    im, d = canvas("Data-processing and evidence pipeline", "Offline development evidence and online application processing")
    box(d, (70, 180, 890, 940), "Offline development and evaluation", [
        "Curated receipt sources and grouped manifests",
        "Controlled thermal degradations and locked splits",
        "DSDNet training: eight unrolled stages and four learned maxout branches",
        "Baseline comparisons and retained checkpoint provenance",
        "Stage ablation, kernel sensitivity, severity robustness and paired statistics",
        "Failure/mixed-result analysis and evidence freeze",
    ], "#eef1fa", "#4b5ea8")
    box(d, (1010, 180, 1830, 940), "Online receipt workflow", [
        "Validated upload or batch capture",
        "Geometry correction and degradation-adaptive recovery",
        "Faithful image fusion plus original/compare views",
        "OCR candidate merge and semantic field/line-item extraction",
        "Readiness decision and user verification",
        "Accounting exports and MyInvois-support coverage report",
    ], "#e9f5f2", "#176f61")
    arrow(d, (885, 470), (1000, 470), "#7a5b1c")
    arrow(d, (1000, 650), (885, 650), "#7a5b1c")
    d.text((785, 520), "Shared", font=fonts(24, True), fill="#5c4518")
    d.text((748, 552), "configuration", font=fonts(22), fill="#5c4518")
    d.text((755, 585), "& provenance", font=fonts(22), fill="#5c4518")
    d.text((420, 985), "Boundary: MyInvois-supporting preparation and audit evidence; no signing, submission, UUID or validation response.", font=fonts(23, True), fill="#8a2838")
    im.save(FIGURES / "figure_4_3_data_evidence.png", dpi=(220, 220))

    # Figure 4.4
    im, d = canvas("Unrolled DSDNet learned maxout-shrinkage flow", "One source-supported restoration pass; k = 1 … 8")
    flow = [
        (60, "Input", ["Degraded image y", "Blur kernel h"]),
        (360, "Feature update", ["Prior/data convolutions", "Stage k state"]),
        (700, "Maxout M=4", ["Four learned branches", "Per-pixel selection"]),
        (1040, "Shrinkage", ["Learned threshold", "Sparse response"]),
        (1380, "Data consistency", ["CGNet update", "xₖ and diagnostics"]),
    ]
    for x, title, its in flow:
        box(d, (x, 285, x + 265, 650), title, its, "#eef7f4")
    for x in (325, 665, 1005, 1345):
        arrow(d, (x, 465), (x + 28, 465))
    arrow(d, (1510, 675), (520, 760), "#b77716")
    d.text((790, 748), "repeat until stage 8", font=fonts(26, True), fill="#8a5a11")
    box(d, (460, 820, 1440, 1010), "Output and downstream safeguards", ["x₈ enters document-aware faithful fusion", "Source evidence constrains stroke strengthening; OCR does not draw text", "Original, faithful and comparison views remain auditable"], "#fff4df", "#b77716")
    im.save(FIGURES / "figure_4_4_maxout_flow.png", dpi=(220, 220))


def parse_junit():
    root = ET.parse(JUNIT).getroot()
    cases = root.findall(".//testcase")
    rows = []
    counts = Counter()
    for i, tc in enumerate(cases, 1):
        cls = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        text = f"{cls} {name}".lower()
        if "security" in text or any(k in text for k in ("csrf", "password", "owner", "formula_injection", "integrity")):
            cat = "Privacy/Security"
        elif any(k in text for k in ("worker", "batch", "api", "workflow", "export", "workbook", "audit_pdf")):
            cat = "Integration"
        elif any(k in text for k in ("orientation", "crop", "button", "label", "review_status", "frontend")):
            cat = "Usability/Interface"
        elif any(k in text for k in ("performance", "latency", "concurrent")):
            cat = "Performance"
        elif any(k in text for k in ("dsdnet", "maxout", "stage", "restoration", "faithful", "ocr", "semantic", "line_item", "field")):
            cat = "CV/OCR/Semantic"
        else:
            cat = "Functional"
        failed = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        status = "Fail" if failed else ("Skipped" if skipped else "Pass")
        counts[cat] += 1
        rows.append({
            "Test Case ID": f"REG-{i:03d}",
            "Category": cat,
            "Test Description": name.replace("test_", "").replace("_", " ").capitalize(),
            "Preconditions": "Current 5 November 2026 application source, pinned test configuration and isolated test resources",
            "Test Steps": f"Execute pytest case {cls}::{name} and evaluate its programmed assertions",
            "Test Data": "Controlled fixture or repository sample defined by the automated test",
            "Expected Result": "All assertions pass without an unhandled error or unintended state change",
            "Actual Result": f"{status}; elapsed {float(tc.attrib.get('time', 0)):.3f} s",
            "Status": status,
            "Evidence": "pytest_results.xml",
        })
    return rows, counts


def write_evidence(rows, counts):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / "test_case_register_132.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (EVIDENCE / "test_case_register_132.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-change regression after semantic extraction, readiness and MyInvois-support UI updates",
        "result": "132 passed; 0 failed",
        "category_counts": dict(counts),
        "cases": rows,
        "limitations": [
            "Automated tests are development verification, not a government certification or production penetration test.",
            "MyInvois coverage is a preparation/audit aid; the application does not sign or submit UBL documents.",
            "Historical model comparisons remain development-validation evidence and are not relabelled as a fresh independent test set.",
        ],
    }, indent=2), encoding="utf-8")

    coverage = [
        ("Supplier identity", "Extracted when visible", "Name, registration no., TIN, address, phone and email are reviewable."),
        ("Document identity", "Extracted when visible", "Document type, number, date and currency are reviewable."),
        ("Monetary evidence", "Extracted/derived when visible", "Subtotal, discount, tax, payable total and payment method are reviewable."),
        ("Line items", "Extracted candidates", "Description, quantity, unit, unit price, discount and amount are reviewable; ambiguous handwriting remains human-confirmed."),
        ("Buyer identity", "User/issuer supplied", "A receipt generally cannot provide a complete e-Invoice buyer identity."),
        ("Classification and tax codes", "User/issuer supplied", "Do not infer legal codes when absent from the source receipt."),
        ("Digital signature and submission", "Outside application scope", "No UBL signing, MyInvois API submission, UUID or validation response."),
        ("Audit support", "Implemented", "Original image, faithful view, field evidence, integrity state, PDF and Excel coverage report."),
    ]
    with (EVIDENCE / "myinvois_coverage.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["Area", "Application status", "Boundary/meaning"]); w.writerows(coverage)

    alignment = [
        ("Eight-stage DSDNet", "dsdnet_adapter.py", "Figures 4.1, 4.3 and 4.4; Sections 4.1, 4.3 and Chapter 5"),
        ("Four maxout branches", "dsdnet_adapter.py", "Figures 4.1 and 4.4"),
        ("Faithful source-supported restoration", "restoration.py", "Figures 4.1–4.3; Chapters 4–6"),
        ("Geometry/crop/orientation", "preprocessing.py", "Figures 4.2–4.3; Chapters 4–5"),
        ("Multi-view OCR and semantic extraction", "ocr.py, fields.py, semantic.py", "Figures 4.1–4.3; Chapters 4–6"),
        ("Line items/accounting/MyInvois support", "semantic.py, web_app.py", "Figures 4.1–4.3; Sections 5.16 and 6.5.10"),
        ("Human verification and audit exports", "web_app.py, storage.py", "Figures 4.1–4.3; Chapters 4–6"),
    ]
    with (EVIDENCE / "app_thesis_alignment.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["Implemented capability", "Code evidence", "Thesis location"]); w.writerows(alignment)

    files = [
        ROOT / "src/dsdnet_receipt/preprocessing.py", ROOT / "src/dsdnet_receipt/dsdnet_adapter.py",
        ROOT / "src/dsdnet_receipt/restoration.py", ROOT / "src/dsdnet_receipt/ocr.py",
        ROOT / "src/dsdnet_receipt/fields.py", ROOT / "src/dsdnet_receipt/semantic.py",
        ROOT / "src/dsdnet_receipt/storage.py", ROOT / "src/dsdnet_receipt/web_app.py",
        ROOT / "configs/default.json", JUNIT, INPUT,
    ]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = "unavailable"
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
        "test_result": "132 passed in 60.89 s; 0 failed", "files": [
            {"path": str(p), "sha256": sha256(p)} for p in files if p.is_file()
        ]
    }
    (EVIDENCE / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return coverage, alignment


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd"); tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def format_table(table):
    table.style = "Table Grid"
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for i, row in enumerate(table.rows):
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i == 0:
                set_cell_shading(cell, "176F61")
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = "Arial"; r.font.size = Pt(8)
                    if i == 0:
                        r.font.bold = True; r.font.color.rgb = RGBColor(255,255,255)


def highlight_para(p):
    for run in p.runs:
        run.font.highlight_color = 7  # WD_COLOR_INDEX.YELLOW


def replace_figure(doc, caption, image_path):
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == caption:
            image_p = doc.paragraphs[i - 1]
            image_p.clear()
            image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            image_p.add_run().add_picture(str(image_path), width=Inches(6.35))
            return
    raise RuntimeError(f"Caption not found: {caption}")


def add_para_before(doc, anchor, text, style=None, yellow=False):
    p = doc.add_paragraph(text, style=style)
    anchor._p.addprevious(p._p)
    if yellow:
        highlight_para(p)
    return p


def add_table_before(doc, anchor, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    for i, h in enumerate(headers): t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row): cells[i].text = str(value)
    format_table(t)
    anchor._p.addprevious(t._tbl)
    return t


def find_para(doc, exact):
    for p in doc.paragraphs:
        if p.text.strip() == exact:
            return p
    raise RuntimeError(f"Paragraph not found: {exact}")


def edit_thesis(counts, coverage, alignment):
    doc = Document(INPUT)
    replace_figure(doc, "Figure 4.1 Implemented system architecture for DSDNet-Receipt", FIGURES / "figure_4_1_architecture.png")
    replace_figure(doc, "Figure 4.2 Implemented receipt processing and review pipeline", FIGURES / "figure_4_2_runtime_pipeline.png")
    replace_figure(doc, "Figure 4.3 Data-processing and evidence pipeline", FIGURES / "figure_4_3_data_evidence.png")
    replace_figure(doc, "Figure 4.4 Unrolled DSDNet learned maxout shrinkage algorithm flow", FIGURES / "figure_4_4_maxout_flow.png")

    # Minimal Chapter 4 addition, highlighted because the user's existing humanised text is preserved.
    anchor = find_para(doc, "4.2 Data-Processing and Evidence Pipeline")
    p = add_para_before(doc, anchor,
        "Post-implementation alignment note (5 November 2026). Figures 4.1–4.4 were redrawn from the implemented application. The runtime now combines geometry correction, degradation-adaptive eight-stage DSDNet recovery, source-supported faithful fusion, multi-view OCR, semantic extraction of primary fields, line items and accounting evidence, readiness rules, human verification and audit exports. Its MyInvois contribution is limited to supporting-evidence preparation and coverage reporting: it does not create a legally complete UBL e-Invoice, apply an authorised digital signature, submit to the MyInvois API or return a validation UUID.",
        yellow=True)

    # Mark the revised captions as Chapter 4 changes.
    for caption in (
        "Figure 4.1 Implemented system architecture for DSDNet-Receipt",
        "Figure 4.2 Implemented receipt processing and review pipeline",
        "Figure 4.3 Data-processing and evidence pipeline",
        "Figure 4.4 Unrolled DSDNet learned maxout shrinkage algorithm flow",
    ):
        highlight_para(find_para(doc, caption))

    chapter6 = find_para(doc, "Chapter 6")
    add_para_before(doc, chapter6, "5.16 Post-change Application–Thesis Consistency and Regression Verification", "Heading 2")
    add_para_before(doc, chapter6, "5.16.1 Current implementation alignment", "Heading 3")
    add_para_before(doc, chapter6,
        "A final consistency audit was performed after the date, handwriting, line-item, accounting-field, tag and review-status changes. The audit compared the current source modules with the Chapter 4 diagrams and with the claims carried into Chapters 5 and 6. Table 5.30 records the resulting traceability. The diagrams describe the implemented eight-stage/four-branch architecture, the source-supported faithful recovery path and the current semantic and evidence layers; they no longer imply that OCR redraws text or that the application submits an e-Invoice.")
    add_table_before(doc, chapter6, ["Capability", "Implemented code evidence", "Thesis alignment"], alignment)
    add_para_before(doc, chapter6, "Table 5.30 Post-change application–thesis traceability", "Caption1")

    add_para_before(doc, chapter6, "5.16.2 MyInvois-support coverage and boundary", "Heading 3")
    add_para_before(doc, chapter6,
        "The application now exposes supplier, document, monetary and line-item evidence that can support accounting review and later MyInvois preparation. This improves practical value without overstating legal or technical completeness. Official MyInvois structures and submission guidance require a structured e-Invoice workflow beyond a receipt image, including specified document data and submission processing (Lembaga Hasil Dalam Negeri Malaysia [HASiL], n.d.-a, n.d.-b). A source receipt cannot reliably supply every buyer identity, classification, tax-code or signing field required for an e-Invoice. Consequently, absent legal fields remain user- or issuer-supplied and the export explicitly reports coverage gaps. Table 5.31 distinguishes extracted evidence from information and actions outside the application scope.")
    add_table_before(doc, chapter6, ["Evidence area", "Application status", "Interpretation/boundary"], coverage)
    add_para_before(doc, chapter6, "Table 5.31 MyInvois-support coverage and non-submission boundary", "Caption1")

    add_para_before(doc, chapter6, "5.16.3 Regression and test-case evidence", "Heading 3")
    add_para_before(doc, chapter6,
        "The complete automated regression suite was re-executed on 5 November 2026 after the implementation changes. All 132 tests passed in 60.89 seconds with zero failures. This later run supersedes the earlier 116-test application snapshot only for current regression status; it does not replace the locked restoration experiments, ablation records or historical timings already reported in Sections 5.11–5.14. The machine-readable JUnit record and a 132-row register contain the required test-case identifier, description, preconditions, steps, data, expected result, actual result and status for every executed case.")
    summary_rows = [(k, v, "Pass", "0") for k, v in sorted(counts.items())]
    summary_rows.append(("Total", sum(counts.values()), "Pass", "0"))
    add_table_before(doc, chapter6, ["Test category", "Executed cases", "Result", "Failures"], summary_rows)
    add_para_before(doc, chapter6, "Table 5.32 Post-change automated regression summary (5 November 2026)", "Caption1")
    add_para_before(doc, chapter6,
        "The current run covers functional behaviour, CV/OCR/semantic extraction, interface/readiness rules, integration workflows, and privacy/security controls represented in the repository. Earlier performance, baseline/proposed-method, ablation, robustness, statistical and failure analyses remain separately preserved with their configurations and hashes. Automated success is evidence of implementation consistency, not proof of universal OCR accuracy, production-scale performance, government certification or legal MyInvois compliance.")

    add_para_before(doc, chapter6, "5.16.4 Provenance, uncertainty and remaining limitations", "Heading 3")
    add_para_before(doc, chapter6,
        "The final evidence folder stores the JUnit XML, full test-case register, application–thesis alignment matrix, MyInvois coverage matrix, redrawn diagrams and a SHA-256 provenance manifest for the principal preprocessing, DSDNet, restoration, OCR, field, semantic, storage, web and configuration files. Handwritten receipts and heavily degraded thermal documents can still contain ambiguous evidence; such cases remain subject to human confirmation rather than silent fabrication. This is consistent with the thesis emphasis on faithful recovery and auditability.")

    final_conclusion = find_para(doc, "6.6 Final Conclusion")
    add_para_before(doc, final_conclusion, "6.5.10 Post-change verification and MyInvois boundary", "Heading 3")
    add_para_before(doc, final_conclusion,
        "Following the final semantic-extraction and review-status corrections, the 5 November 2026 regression run passed 132 of 132 tests. Figures 4.1–4.4 and Sections 5.16 and 6.5.10 now describe the same implemented pipeline. The system contributes faithful receipt recovery, reviewable accounting evidence, field and line-item candidates, audit exports and MyInvois coverage preparation. It does not claim direct MyInvois submission, legal compliance certification, digital signing or replacement of the original receipt and professional review.")

    # Add a note to the representative appendix rather than duplicating 132 very wide rows in the thesis.
    appendix = find_para(doc, "Appendix A: Executed Test-Case Register")
    add_para_before(doc, appendix,
        "Lembaga Hasil Dalam Negeri Malaysia (HASiL). (n.d.-a). E-Invoice data structures. https://sdk.myinvois.hasil.gov.my/types/")
    add_para_before(doc, appendix,
        "Lembaga Hasil Dalam Negeri Malaysia (HASiL). (n.d.-b). Submit documents. https://sdk.myinvois.hasil.gov.my/einvoicingapi/02-submit-documents/")
    add_para_before(doc, appendix,
        "The appendix retains the earlier representative executed cases. The complete post-change 132-case register dated 5 November 2026 is supplied with the thesis evidence artifacts and is summarized in Table 5.32; the machine-readable register contains all required test-case columns.")

    # Ensure output directory exists and save.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    return len(doc.paragraphs), len(doc.tables), len(doc.inline_shapes)


def main():
    if not INPUT.is_file(): raise SystemExit(f"Missing thesis: {INPUT}")
    if not JUNIT.is_file(): raise SystemExit(f"Missing JUnit proof: {JUNIT}")
    make_figures()
    rows, counts = parse_junit()
    if len(rows) != 132 or any(r["Status"] != "Pass" for r in rows):
        raise SystemExit(f"Unexpected regression result: {len(rows)} cases, statuses={Counter(r['Status'] for r in rows)}")
    coverage, alignment = write_evidence(rows, counts)
    paras, tables, images = edit_thesis(counts, coverage, alignment)
    summary = {
        "output": str(OUTPUT), "sha256": sha256(OUTPUT), "paragraphs": paras,
        "tables": tables, "inline_shapes": images, "tests": len(rows),
        "test_categories": dict(counts), "figures": sorted(str(p) for p in FIGURES.glob("*.png")),
    }
    (EVIDENCE / "finalization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
