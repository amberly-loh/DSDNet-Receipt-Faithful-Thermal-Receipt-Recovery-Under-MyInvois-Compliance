"""Visual regression probes for document content protection, not model scores."""
import json
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_faithful_content_view

folder = ROOT / 'artifacts' / 'faithful_docx_review'
for row in (11, 12, 13, 14):
    source = cv2.imread(str(folder / f'row_{row:02d}_original_00.png'))
    registered, geometry = preprocess(source, {'preprocessing': {'max_long_edge': 1600}}, preserve_color=True)
    gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    candidate = make_faithful_content_view(gray, gray, [], source_color=registered)
    cv2.imwrite(str(folder / f'row_{row:02d}_preservation_candidate.png'), candidate)
    (folder / f'row_{row:02d}_candidate_geometry.json').write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    print(row, candidate.shape, flush=True)
