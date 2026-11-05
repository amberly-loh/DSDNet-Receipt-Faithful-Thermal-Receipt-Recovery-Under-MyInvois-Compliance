"""Extract only Faithful and Ori cells, preserving row/image associations."""
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'faithful_docx_review'
OUT.mkdir(parents=True, exist_ok=True)
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(r'C:\Users\Amberly\Downloads\Faithful.docx') as archive:
    doc = ET.fromstring(archive.read('word/document.xml'))
    relationships = ET.fromstring(archive.read('word/_rels/document.xml.rels'))
    targets = {item.get('Id'): item.get('Target') for item in relationships}
    records = []
    for table in doc.findall('.//w:tbl', NS):
        rows = table.findall('w:tr', NS)
        for row_index, row in enumerate(rows[1:], 1):
            cells = row.findall('w:tc', NS)
            record = {'row': row_index}
            for column, label in ((0, 'faithful'), (2, 'original')):
                if column >= len(cells):
                    continue
                cell = cells[column]
                record[label + '_notes'] = '\n'.join(
                    ''.join(t.text or '' for t in p.findall('.//w:t', NS))
                    for p in cell.findall('.//w:p', NS)
                ).strip()
                images = []
                for index, blip in enumerate(cell.findall('.//a:blip', NS)):
                    target = targets.get(blip.get('{' + NS['r'] + '}embed'))
                    if not target:
                        continue
                    path = OUT / f'row_{row_index:02d}_{label}_{index:02d}{Path(target).suffix}'
                    path.write_bytes(archive.read('word/' + target))
                    images.append(str(path))
                record[label + '_images'] = images
            records.append(record)
    (OUT / 'rows.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
