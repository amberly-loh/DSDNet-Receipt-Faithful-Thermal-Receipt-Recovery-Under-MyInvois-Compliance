"""Stage or publish all qualifying saved small-print receipts, from originals."""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_faithful_content_view, complete_document_preservation_mode

parser = argparse.ArgumentParser()
parser.add_argument('--publish', action='store_true')
args = parser.parse_args()
config, _ = load_config(ROOT / 'configs/default.json')
folder = ROOT / 'artifacts/small_print_revision'
folder.mkdir(exist_ok=True)
db = sqlite3.connect(ROOT / 'artifacts/app_data/receipts.sqlite3')
db.row_factory = sqlite3.Row
records, previews, cached = [], [], {}
for row in db.execute('SELECT * FROM receipts ORDER BY id DESC').fetchall():
    raw = cv2.imread(row['input_path'])
    if raw is None or raw.shape[1] / raw.shape[0] >= .7:
        continue
    digest = hashlib.sha256(raw.tobytes()).hexdigest()
    root = Path(row['restored_path']).parent
    audit = json.loads(row['audit_json'] or '{}')
    if digest not in cached:
        registered, geometry = preprocess(raw, config, preserve_color=True)
        if min(registered.shape[:2]) >= 250:
            continue
        if complete_document_preservation_mode(registered):
            continue
        source = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
        if float(np.percentile(source, 85) - np.percentile(source, 5)) >= 90:
            continue
        # Rebuild from measured pixels, not the already sharpened saved view.
        candidate = make_faithful_content_view(source, source, [], source_color=registered)
        path = folder / (digest + '.png')
        cv2.imwrite(str(path), candidate)
        cached[digest] = path, registered, geometry
        old = cv2.imread(str(root / 'faithful_content.png'), 0)
        panels = [source, old, candidate]
        if old is not None and len(previews) < 12:
            panels = [cv2.resize(im, (220, 460), interpolation=cv2.INTER_NEAREST) for im in panels]
            previews.append(np.concatenate(panels, axis=1))
    path, registered, geometry = cached[digest]
    if args.publish:
        backup = root / 'before_small_print_edge_revision'
        backup.mkdir(exist_ok=True)
        target = root / 'faithful_content.png'
        for name in ('faithful_content.png', 'audit.json', 'faithful_reference.png'):
            if (root / name).exists() and not (backup / name).exists():
                shutil.copy2(root / name, backup / name)
        if not (backup / 'database_audit.json').exists():
            (backup / 'database_audit.json').write_text(row['audit_json'] or '{}', encoding='utf-8')
        shutil.copy2(path, root / 'faithful_content.pending.png')
        (root / 'faithful_content.pending.png').replace(target)
        cv2.imwrite(str(root / 'faithful_reference.png'), registered)
        audit['small_print_edge_revision'] = {'method': 'native paper normalization and locally bounded glyph-scale sharpening', 'model_weights_changed': False, 'source_sha256': digest}
        audit['faithful_preprocessing'] = geometry
        for key, filename in [('faithful_content', 'faithful_content.png'), ('faithful_reference', 'faithful_reference.png')]:
            audit.setdefault('stored_file_sha256', {})[key] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        (root / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        db.execute('UPDATE receipts SET audit_json=? WHERE public_id=?', (json.dumps(audit), row['public_id']))
        db.commit()
    records.append({'run': row['public_id'], 'source': row['source_name'], 'candidate': str(path)})
if previews:
    cv2.imwrite(str(folder / 'comparison.jpg'), np.concatenate(previews, axis=0))
(folder / ('published.json' if args.publish else 'staged.json')).write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps({'saved_results': len(records), 'unique_receipts': len(cached), 'published': args.publish}))
