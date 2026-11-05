"""Refresh the reviewed native-resolution case with reversible backups."""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
candidate = ROOT / 'artifacts/complaint_regression/petronas_sharp.png'
db = sqlite3.connect(ROOT / 'artifacts/app_data/receipts.sqlite3')
db.row_factory = sqlite3.Row
count = 0
reviewed_run = next(item['run'] for item in json.loads((ROOT / 'artifacts/complaint_regression/records.json').read_text()) if item['case'] == 'petronas')
reviewed = db.execute('SELECT input_path FROM receipts WHERE public_id=?', (reviewed_run,)).fetchone()
reviewed_pixels = cv2.imread(reviewed['input_path'])
for row in db.execute('SELECT * FROM receipts WHERE source_name=?', ('Screenshot 2026-07-13 080718.png',)).fetchall():
    if not np.array_equal(reviewed_pixels, cv2.imread(row['input_path'])):
        continue
    root = Path(row['restored_path']).parent
    target = root / 'faithful_content.png'
    backup = root / 'before_native_paper_revision'
    backup.mkdir(exist_ok=True)
    if not (backup / target.name).exists():
        shutil.copy2(target, backup / target.name)
        (backup / 'database_audit.json').write_text(row['audit_json'] or '{}', encoding='utf-8')
    shutil.copy2(candidate, root / 'faithful_content.pending.png')
    (root / 'faithful_content.pending.png').replace(target)
    audit = json.loads(row['audit_json'] or '{}')
    audit['native_print_revision'] = {'method': 'native-scale paper estimation and continuous contrast', 'trained_weights_changed': False}
    audit.setdefault('stored_file_sha256', {})['faithful_content'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    db.execute('UPDATE receipts SET audit_json=? WHERE public_id=?', (json.dumps(audit), row['public_id']))
    count += 1
db.commit()
print(f'Updated {count} Petronas saved views')
