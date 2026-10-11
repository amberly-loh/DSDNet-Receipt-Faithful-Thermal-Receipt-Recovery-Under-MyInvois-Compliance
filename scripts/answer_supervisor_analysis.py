from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph

TASK = Path(__file__).resolve().parents[2]
SOURCE = TASK / "Group_1_Amberly_Loh_DSDNet_Receipt_English.docx"
OUTPUT = TASK / "Group_1_Amberly_Loh_DSDNet_Receipt_English_ANSWERED_MALAYSIAN_ONLY.docx"
UPLOAD_DIR = TASK / "THESIS_GROUP_ANALYSIS_12_GROUPS (answers)"

ANSWERS = {
    "How many independent clean receipt sources are currently available?": "The provided Malaysian collection contains 646 real-life receipt images grouped provisionally into 573 independent receipt sources. These are the project's core data. Clean-reference eligibility still has to be confirmed image by image because naturally degraded receipts cannot automatically be treated as clean restoration targets. Only eligible Malaysian receipt sources will be used in the formal study.",
    "How many synthetic variants will be generated from each source?": "Three controlled variants will be generated from each Malaysian source that passes the clean-reference screening: one light, one moderate and one severe variant. If all 573 provisional sources are eligible, this yields 1,719 variants; the final number will be reported after duplicate, quality and privacy screening.",
    "Was the source-group split completed before generating variants?": "A deterministic source-group split has been prepared for the Malaysian collection before variant generation, and every generated variant will inherit its original receipt's partition. The split will be finalized after manual confirmation of the 573 provisional groups and clean-reference eligibility, followed by another automated leakage check.",
    "How many naturally degraded receipts are available, and is there permission to use them?": "The project includes all 646 provided real-life Malaysian receipt images, grouped provisionally into 573 natural receipt sources. They are used for Malaysian natural-condition robustness, OCR, qualitative failure and routing evaluation. Before formal publication or displaying example receipts, usage permission must be recorded and personal or transaction identifiers must be reviewed and masked where necessary.",
    "Which DSDNet source-code commit is being used?": "The implementation uses the official setsunil/DSDNet repository at commit 8a2822e44ea1e00970ed429a8230e4116034e2f9. The project repository records this identifier in the configuration and run-provenance files.",
    "How was the original DSDNet non-blind operator adapted?": "The adapter uses one-channel grayscale input, eight unrolled stages, four maxout branches and CGNet refinement at every stage. For controlled data, the Gaussian forward kernel is reconstructed from each variant's recorded blur sigma. The inherited DSDNet shrinkage, convolution, padding and CGNet components are retained, while stage states and pre-maxout branch responses are exposed for later support analysis.",
    "Do controlled data and naturally degraded receipts use different operators?": "Yes. Controlled variants use the recorded Gaussian blur kernel for data consistency. For naturally degraded receipts, the true degradation operator is unknown; the implementation must therefore use the identity/observed-image path, mark source-trace residual support as unavailable and route affected high-risk decisions conservatively.",
    "Are Baselines A, B and C already operational?": "Yes. Direct extraction, Otsu, Sauvola and perceptual filtering are implemented under shared preprocessing and the same PaddleOCR configuration. Their final development and evaluation runs will be repeated using only the screened Malaysian receipt sources; earlier external-dataset records are not formal results for this study.",
    "Which exact OCR version and configuration are fixed?": "The isolated evaluation environment uses PaddleOCR 2.7.0.3 from the PaddleOCR 2.7 family with PaddlePaddle 2.5.2, English PP-OCRv3 detection and recognition, angle classification disabled and identical settings for every method. The OCR smoke test and all reported evaluations will be rerun on Malaysian receipts.",
    "What are the current CER, WER and field accuracy of the simplest baseline?": "No final Malaysian baseline values are reported yet. The previously generated development figures came from an external dataset that is now excluded from the formal study. Direct Extraction, Otsu, Sauvola and perceptual-filter baselines must be rerun after the Malaysian sources and target-field annotations are verified.",
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
