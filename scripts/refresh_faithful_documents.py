"""Migrate only colored/ruled Faithful views; originals and Clean are untouched."""
import hashlib
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import complete_document_preservation_mode, make_complete_document_view

config, _ = load_config(ROOT / 'configs/default.json')
parser = argparse.ArgumentParser()
parser.add_argument('--run')
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()
connection = sqlite3.connect(ROOT / 'artifacts/app_data/receipts.sqlite3')
connection.row_factory = sqlite3.Row
records = connection.execute('SELECT public_id,input_path,restored_path,audit_json FROM receipts ORDER BY id DESC').fetchall()
report = []
for record in records:
    if args.run and record['public_id'] != args.run:
        continue
    original = cv2.imread(record['input_path'])
    if original is None:
        continue
    registered, geometry = preprocess(original, config, preserve_color=True)
    mode = complete_document_preservation_mode(registered)
    if mode is None:
        continue
    output = make_complete_document_view(registered)
    if args.dry_run:
        staging = ROOT / 'artifacts' / 'faithful_background_candidates'
        staging.mkdir(exist_ok=True)
        cv2.imwrite(str(staging / (record['public_id'] + '.png')), output)
        print(record['public_id'], 'staged only', flush=True)
        continue
    root = Path(record['restored_path']).parent
    target = root / 'faithful_content.png'
    backup = root / 'before_complete_document_preservation'
    backup.mkdir(exist_ok=True)
    for filename in ('faithful_content.png', 'audit.json'):
        existing = root / filename
        if existing.exists() and not (backup / filename).exists():
            shutil.copy2(existing, backup / filename)
    if not (backup / 'database_audit.json').exists():
        (backup / 'database_audit.json').write_text(record['audit_json'] or '{}', encoding='utf-8')
    temporary = root / 'faithful_content.pending.png'
    if not cv2.imwrite(str(temporary), output):
        raise RuntimeError(f'Cannot save {temporary}')
    temporary.replace(target)
    audit = json.loads(record['audit_json'] or '{}')
    audit['faithful_display_policy'] = mode
    audit['faithful_preprocessing'] = geometry
    audit['faithful_display_image'] = 'faithful_content.png'
    audit['faithful_processing'] = {
        'method': 'continuous source luminance enhancement',
        'model_pixels_used': False,
        'synthetic_text_rendering': False,
        'ocr_used_for_foreground_deletion': False,
        'color_retained': output.ndim == 3,
        'validation': 'preservation regression checks; full restoration quality not certified',
    }
    audit.setdefault('stored_file_sha256', {})['faithful_content'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    connection.execute('UPDATE receipts SET audit_json=? WHERE public_id=?', (json.dumps(audit), record['public_id']))
    connection.commit()
    report.append({'run': record['public_id'], 'mode': mode, 'image': str(target)})
    print(json.dumps(report[-1]), flush=True)
connection.close()
(ROOT / 'artifacts/faithful_docx_review/document_migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'updated': len(report), 'examined': len(records)}), flush=True)
