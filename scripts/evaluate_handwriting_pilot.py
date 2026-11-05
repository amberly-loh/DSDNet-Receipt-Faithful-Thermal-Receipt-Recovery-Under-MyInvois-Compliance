"""Small provisional handwritten-field pilot; not a training or held-out benchmark."""
from pathlib import Path
import hashlib
import sys
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.fields import extract_fields
from dsdnet_receipt.ocr import PaddleOCRAdapter

DATA = ROOT.parent / 'FYP datasets' / 'Real life'
# Provisional visual readings; omit values that are ambiguous to the reviewer.
CASES = [
    ('Document_20240511_0008.jpg', '2024-03-19', '340.00'),
    ('Document_20240511_0070.jpg', '2024-03-26', '116.00'),
    ('Document_20240511_0081.jpg', '2024-07-13', '45.00'),
    ('Document_20250512_0027.jpg', '2024-11-21', '20.00'),
    ('Document_20250512_0041.jpg', '2024-06-26', '480.00'),
    ('Document_20250512_0042.jpg', '2024-07-15', '262.00'),
    ('Document_20250512_0060.jpg', '2024-06-25', None),
    ('Invoice_20240316_0035.jpg', '2024-02-27', '457.00'),
    ('Invoice_20250512_0032.jpg', '2024-11-30', None),
]

def main():
    engine = PaddleOCRAdapter(load_config(ROOT / 'configs/default.json')[0])
    seen = set()
    scores = {'date': [0, 0], 'total': [0, 0]}
    for filename, date, total in CASES:
        path = DATA / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        result = engine.run(cv2.imread(str(path)))
        if result.error: raise RuntimeError(result.error)
        fields = extract_fields(result.text, result.lines)
        values = (fields['transaction_date']['value'], fields['total_amount']['value'])
        for name, expected, observed in zip(('date', 'total'), (date, total), values):
            if expected is not None:
                scores[name][1] += 1
                scores[name][0] += expected == observed
        print(f'{filename}: date {values[0]} expected {date}; total {values[1]} expected {total or "unlabelled"}', flush=True)
    print('Provisional pilot scores (correct/labelled):', scores, flush=True)

if __name__ == '__main__': main()
