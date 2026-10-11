# Chapter 4 video figure review

## Recommended figure

**Eight-stage exposed-state diagram, approximately 198.6 seconds.** This is the strongest reusable frame because it shows the degraded receipt passing through stages k=1 to k=8 and identifies the retained image estimates, shrinkage features and refinement variables. It can support the Chapter 4 architecture explanation, provided the caption states that it is a conceptual implementation diagram rather than an experimental result.

Suggested caption: **Figure 4.x Eight-stage DSDNet receipt-restoration sequence and retained intermediate states used by the evidence-support module.**

## Figures requiring caution

- **Large architecture pan, approximately 248–323 seconds:** do not use in its present form. The frame is only partially visible at each time and contains an “Authenticity Module.” The thesis explicitly excludes authenticity, forensic and legal verification claims.
- **Residual agreement, approximately 347.6 seconds:** the sampled frame contains only a heading and does not communicate the computation.
- **Stage stability, approximately 372.5 seconds:** acceptable only as a conceptual explanation of convergence across stages. It is not evidence of an observed result.
- **Branch agreement, approximately 397.3 seconds:** acceptable only as a simplified explanatory illustration. The repeated value 0.85 is illustrative and must not be reported as measured data.
- **Evidence-support threshold, approximately 521.4 seconds:** useful for explaining conservative routing, but the displayed curve and failed state are illustrative rather than experimental output.

## Recommendation

Use the eight-stage exposed-state frame as a supplementary Chapter 4 architecture figure. Retain the existing layered architecture and pipe-and-filter diagrams because they describe the overall software system and processing flow, while the video frame explains the internal DSDNet stage sequence. Redraw any other video concept as a clean vector diagram with thesis-consistent terminology before inclusion.
