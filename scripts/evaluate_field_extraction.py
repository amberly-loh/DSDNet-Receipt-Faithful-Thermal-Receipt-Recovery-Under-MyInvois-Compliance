"""Evaluate field parsing using SROIE's reference OCR, not image OCR accuracy."""
from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.fields import extract_fields, _normalise_date
DATA = ROOT.parent / 'FYP datasets' / 'External' / 'SROIE-Malaysia'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--image-ocr', action='store_true', help='Run current PaddleOCR on original images instead of reference text')
    parser.add_argument('--limit', type=int, default=0, help='Limit evaluated receipts for a quick pilot')
    parser.add_argument('--records-output', type=Path, help='Write local per-field predictions and confidence to a new JSONL file')
    args = parser.parse_args()
    if args.records_output:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        if args.records_output.exists():
            raise SystemExit('Refusing to overwrite existing per-field records')
    with (DATA / 'sroie_protocol_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        permitted = {r['source_id'] for r in csv.DictReader(stream) if r['project_partition'] == args.partition}
    records = []
    for name in ('train', 'test'):
        with (DATA / 'annotations' / f'{name}.jsonl').open(encoding='utf-8') as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    counts = {'date': {'correct': 0, 'total': 0}, 'total': {'correct': 0, 'total': 0}}
    engine = None
    if args.image_ocr:
        from dsdnet_receipt.config import load_config
        from dsdnet_receipt.ocr import PaddleOCRAdapter
        engine = PaddleOCRAdapter(load_config(ROOT / 'configs/default.json')[0])
        import cv2
    seen = 0
    for record in records:
        if record['source_id'] not in permitted: continue
        if args.limit and seen >= args.limit: break
        seen += 1
        lines = []
        if engine:
            image = cv2.imread(str(DATA / record['image']))
            if image is None: raise RuntimeError(f"Missing image {record['source_id']}")
            reading = engine.run(image)
            if reading.error: raise RuntimeError(reading.error)
            lines = reading.lines
        else:
            for word, box in zip(record.get('words', []), record.get('bboxes', [])):
                x1,y1,x2,y2 = box
                lines.append({'text': word, 'box': [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], 'confidence': 1.0})
        fields = extract_fields('\n'.join(x['text'] for x in lines), lines)
        expected = record.get('entities', {})
        date = _normalise_date(str(expected.get('date') or '').strip())
        if date:
            counts['date']['total'] += 1
            counts['date']['correct'] += fields['transaction_date']['value'] == date
            if args.records_output:
                with args.records_output.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'source_id':record['source_id'],'field':'date','reference':date,'prediction':fields['transaction_date']['value'],'confidence':fields['transaction_date'].get('confidence'),'correct':fields['transaction_date']['value']==date})+'\n')
        try:
            total = Decimal(re.sub(r'[^\d.]', '', str(expected.get('total') or '')))
            if total.is_finite():
                counts['total']['total'] += 1
                counts['total']['correct'] += fields['total_amount']['value'] is not None and Decimal(fields['total_amount']['value']) == total
                if args.records_output:
                    prediction=fields['total_amount']['value']
                    with args.records_output.open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps({'source_id':record['source_id'],'field':'total','reference':str(total),'prediction':prediction,'confidence':fields['total_amount'].get('confidence'),'correct':prediction is not None and Decimal(prediction)==total})+'\n')
        except InvalidOperation: pass
    print(json.dumps({'partition': args.partition, 'receipts': seen, 'method': 'PaddleOCR on original images' if engine else 'reference OCR text and boxes; excludes image recognition', 'results': counts}, indent=2))

if __name__ == '__main__': main()
