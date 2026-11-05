"""Publish reviewed candidates to pixel-identical saved copies, with backups."""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'artifacts' / 'complaint_regression'
records = json.loads((folder / 'records.json').read_text(encoding='utf-8'))
db = sqlite3.connect(ROOT / 'artifacts/app_data/receipts.sqlite3')
db.row_factory = sqlite3.Row
published = []
for record in records:
    reviewed = db.execute('SELECT * FROM receipts WHERE public_id=?', (record['run'],)).fetchone()
    source = cv2.imread(reviewed['input_path'])
    candidate_path = folder / (record['case'] + '_candidate.png')
    for row in db.execute('SELECT * FROM receipts WHERE source_name=?', (reviewed['source_name'],)).fetchall():
        other = cv2.imread(row['input_path'])
        if other is None or not np.array_equal(source, other):
            continue
        root = Path(row['restored_path']).parent
        backup = root / 'before_complaint_revision'
        backup.mkdir(exist_ok=True)
        for name in ('faithful_content.png', 'audit.json'):
            path = root / name
            if path.exists() and not (backup / name).exists():
                shutil.copy2(path, backup / name)
        if not (backup / 'database_audit.json').exists():
            (backup / 'database_audit.json').write_text(row['audit_json'] or '{}', encoding='utf-8')
        target = root / 'faithful_content.png'
        temporary = root / 'faithful_content.pending.png'
        shutil.copy2(candidate_path, temporary)
        temporary.replace(target)
        shutil.copy2(folder / (record['case'] + '_registered.png'), root / 'faithful_reference.png')
        audit = json.loads(row['audit_json'] or '{}')
        audit['faithful_preprocessing'] = record['geometry']
        audit['faithful_display_policy'] = 'continuous source-preserving document or thermal contrast'
        audit['faithful_processing'] = {
            'method': 'continuous source luminance enhancement',
            'model_geometry_reused_for_gap_evidence': record['old_model_geometry_reused'],
            'synthetic_text_rendering': False,
            'validation': 'five-case visual comparison and regression tests; residual folds remain',
        }
        audit.setdefault('stored_file_sha256', {})['faithful_content'] = hashlib.sha256(target.read_bytes()).hexdigest()
        audit['stored_file_sha256']['faithful_reference'] = hashlib.sha256((root / 'faithful_reference.png').read_bytes()).hexdigest()
        (root / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        db.execute('UPDATE receipts SET audit_json=? WHERE public_id=?', (json.dumps(audit), row['public_id']))
        db.commit()
        published.append({'run': row['public_id'], 'case': record['case']})
db.close()
(folder / 'published.json').write_text(json.dumps(published, indent=2), encoding='utf-8')
print(json.dumps({'updated': len(published), 'cases': len(records)}))
