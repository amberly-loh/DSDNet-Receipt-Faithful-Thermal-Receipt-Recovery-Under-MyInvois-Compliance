from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

TASK = Path(__file__).resolve().parents[2]
SOURCE = TASK / "Group_1_Amberly_Loh_DSDNet_Receipt_English.docx"
OUTPUT = TASK / "Group_1_Amberly_Loh_DSDNet_Receipt_English_ANSWERED_UPDATED.docx"
UPLOAD_DIR = TASK / "THESIS_GROUP_ANALYSIS_12_GROUPS (answers)"

ANSWERS = {
    "How many independent clean receipt sources are currently available?": "The project now has 987 scanned Malaysian SROIE receipt images as controlled-source candidates, of which 971 are unique by exact file hash before near-duplicate and clean-reference screening. The separate real-life Malaysian collection contains 646 images grouped provisionally into 573 natural receipt sources. In total, 1,633 Malaysian receipt images are available, with 1,614 unique by exact hash. The controlled and natural collections retain the different evidence roles defined in Chapter 3.",
    "How many synthetic variants will be generated from each source?": "Three controlled variants will be generated from each eligible clean-reference SROIE source: one light, one moderate and one severe variant. If all 971 exact-unique candidates pass the remaining eligibility and near-duplicate checks, this produces 2,913 controlled variants. The 646 real-life receipts will not be treated as clean targets or synthetically degraded for the primary paired comparison.",
    "Was the source-group split completed before generating variants?": "Not yet for the expanded controlled corpus. The downloaded SROIE release retains its source split, but the thesis requires a deterministic project-level 70:15:15 source-group split. Exact and near duplicates, clean-reference eligibility and target-field coverage will therefore be confirmed first; the project split will then be recorded before any controlled variants are generated, and every variant will inherit its source partition.",
    "How many naturally degraded receipts are available, and is there permission to use them?": "The project includes 646 provided real-life Malaysian receipt images, grouped provisionally into 573 natural receipt sources. They are used for natural-condition robustness, OCR, qualitative failure and routing evaluation, not as pixel-level ground truth. Before formal analysis or displaying examples, usage permission must be recorded and personal or transaction identifiers must be reviewed and masked where necessary.",
    "Which DSDNet source-code commit is being used?": "The implementation uses the official setsunil/DSDNet repository at commit 8a2822e44ea1e00970ed429a8230e4116034e2f9. The project repository records this identifier in the configuration and run-provenance files.",
    "How was the original DSDNet non-blind operator adapted?": "The adapter uses one-channel grayscale input, eight unrolled stages, four maxout branches and CGNet refinement at every stage. For controlled data, the Gaussian forward kernel is reconstructed from each variant's recorded blur sigma. The inherited DSDNet shrinkage, convolution, padding and CGNet components are retained, while stage states and pre-maxout branch responses are exposed for later support analysis.",
    "Do controlled data and naturally degraded receipts use different operators?": "Yes. Controlled variants use the recorded Gaussian blur kernel for data consistency. For naturally degraded receipts, the true degradation operator is unknown; the implementation must therefore use the identity/observed-image path, mark source-trace residual support as unavailable and route affected high-risk decisions conservatively.",
    "Are Baselines A, B and C already operational?": "Yes. Direct extraction, Otsu, Sauvola and perceptual filtering are implemented under shared preprocessing and the same PaddleOCR configuration. Formal paired comparisons will use the screened controlled Malaysian SROIE sources, while the provided real-life Malaysian receipts will be evaluated separately for natural-condition robustness, in accordance with Chapter 3.",
    "Which exact OCR version and configuration are fixed?": "The isolated evaluation environment uses PaddleOCR 2.7.0.3 from the PaddleOCR 2.7 family with PaddlePaddle 2.5.2, English PP-OCRv3 detection and recognition, angle classification disabled and identical settings for every method. The OCR smoke test and all reported evaluations will be rerun on Malaysian receipts.",
    "What are the current CER, WER and field accuracy of the simplest baseline?": "No final values are reported yet for the expanded Malaysian corpus. Direct Extraction, Otsu, Sauvola and perceptual-filter baselines must be rerun after source screening, the project-level partition and target-field annotations are verified. Development outputs are not presented as formal test evidence.",
    "What is the ground truth for the support proxy?": "On controlled synthetic data, the operational support target is the clean-foreground mask intersected with the observable-trace mask after degradation. The removal, foreground, observable-trace and support-target masks are retained for every variant.",
    "Which results can only be reported using synthetic data?": "Pixel-level support AUC, Trace-F1, support-map agreement with known retained or removed traces, paired restoration metrics against a clean reference and claims requiring a known degradation state can only be reported from controlled synthetic data. Natural receipts are limited to robustness, OCR, qualitative failure and conservative-routing evidence.",
    "If DSDNet does not outperform Sauvola, how will the result be explained?": "The study will report the result directly and investigate operator mismatch, preprocessing, identity-path behaviour, OCR input resolution, patch text content and degradation realism. The support module will not be used to conceal weak restoration performance. If improvement is not established, the contribution will be narrowed to an evaluated adaptation and failure analysis.",
    "If the full support proxy does not outperform the residual-only version, how will the contribution be narrowed?": "The thesis will not claim added value from combining residual, stage-stability and branch-agreement signals. It will retain the best validated interpretable component, present the negative ablation result and narrow the contribution to the support signal that is actually supported by AUC, Trace-F1 and uncertainty estimates.",
    "Has the test set remained untouched for all parameter selection?": "Yes. No checkpoint, support scale, field quantile, calibration mapping or routing threshold has been selected using the Malaysian test partition. Its final membership will be confirmed at source-group level before controlled variants are generated, and it will remain reserved for final reporting after all validation choices are complete.",
}


def insert_after(paragraph: Paragraph, text: str) -> Paragraph:
    element = OxmlElement("w:p")
    paragraph._p.addnext(element)
    inserted = Paragraph(element, paragraph._parent)
    # Keep the answer beneath the numbered question without creating a new
    # numbered list item in Word.
    inserted.style = "Normal"
    lead = inserted.add_run("Answer: ")
    lead.bold = True
    inserted.add_run(text)
    inserted.paragraph_format.space_after = paragraph.paragraph_format.space_after
    return inserted


def main() -> None:
    document = Document(SOURCE)
    # Omit the previously rejected routing suggestion from the completed copy.
    for paragraph in list(document.paragraphs):
        if "one simple but frozen field-level routing experiment" in paragraph.text.lower():
            paragraph._element.getparent().remove(paragraph._element)
    found = set()
    for paragraph in list(document.paragraphs):
        question = paragraph.text.strip()
        if question in ANSWERS:
            insert_after(paragraph, ANSWERS[question])
            found.add(question)
    missing = set(ANSWERS) - found
    if missing:
        raise RuntimeError(f"Questions not found: {sorted(missing)}")
    document.save(OUTPUT)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, UPLOAD_DIR / OUTPUT.name)
    print(f"Answered {len(found)} questions")
    print(OUTPUT)
    print(UPLOAD_DIR / OUTPUT.name)


if __name__ == "__main__":
    main()
