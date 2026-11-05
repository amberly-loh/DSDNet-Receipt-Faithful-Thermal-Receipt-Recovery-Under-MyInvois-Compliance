"""Reproduce the five complaint cases without replacing saved images."""
import json
import sqlite3
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_faithful_content_view
from dsdnet_receipt.ocr import PaddleOCRAdapter

CASES = {
    'sin_lianhap': 'Document_20240511_0030 - Copy.jpg',
    'petronas': 'Screenshot 2026-07-13 080718.png',
    'mr_diy': 'Invoice_20241101_0003.jpg',
    'cowboy': 'Invoice_20241101_0021.jpg',
    'kuchai': 'misc_0077.jpg',
}
config, _ = load_config(ROOT / 'configs/default.json')
out = ROOT / 'artifacts' / 'complaint_regression'
out.mkdir(exist_ok=True)
db = sqlite3.connect(ROOT / 'artifacts/app_data/receipts.sqlite3')
db.row_factory = sqlite3.Row
records = []
ocr_engine = PaddleOCRAdapter(config)
for label, filename in CASES.items():
    row = db.execute('SELECT * FROM receipts WHERE source_name=? ORDER BY id DESC LIMIT 1', (filename,)).fetchone()
    original = cv2.imread(row['input_path'])
    registered, geometry = preprocess(original, config, preserve_color=True)
    source = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    root = Path(row['restored_path']).parent
    old_source = cv2.imread(str(root / 'rectified_input.png'), 0)
    model = cv2.imread(str(root / 'ocr_input.png'), 0)
    matched = old_source is not None and old_source.shape == source.shape and np.array_equal(old_source, source)
    audit = json.loads(row['audit_json'])
    lines = audit.get('ocr_lines', []) if matched else ocr_engine.run(source).lines
    candidate = make_faithful_content_view(source, model if matched else source, lines, source_color=registered)
    cv2.imwrite(str(out / f'{label}_candidate.png'), candidate)
    cv2.imwrite(str(out / f'{label}_registered.png'), registered)
    current = cv2.imread(str(root / 'faithful_content.png'))
    panels = []
    for name, image in [('ORIGINAL', original), ('CURRENT', current), ('CANDIDATE', candidate)]:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height = 900
        preview = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
        panel = np.full((height + 35, preview.shape[1], 3), 255, np.uint8)
        panel[35:] = preview
        cv2.putText(panel, name, (8, 25), 0, .65, (0, 0, 0), 1)
        panels.append(panel)
    cv2.imwrite(str(out / f'{label}_comparison.jpg'), np.concatenate(panels, axis=1))
    records.append({'case': label, 'run': row['public_id'], 'geometry': geometry, 'old_model_geometry_reused': matched})
    print(label, geometry['boundary_detection']['status'], geometry['orientation_correction']['deskew_degrees'], flush=True)
(out / 'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
