"""Inspect actual saved Faithful outputs; never regenerate while auditing."""
import hashlib
import json
import sqlite3
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/release_audit'
folder.mkdir(exist_ok=True)
db=sqlite3.connect(ROOT/'artifacts/app_data/receipts.sqlite3')
db.row_factory=sqlite3.Row
seen=set(); records=[]; pairs=[]
for row in db.execute('SELECT * FROM receipts ORDER BY id DESC').fetchall():
    original=cv2.imread(row['input_path'])
    if original is None:
        continue
    digest=hashlib.sha256(original.tobytes()).hexdigest()
    if digest in seen:
        continue
    seen.add(digest)
    root=Path(row['restored_path']).parent
    faithful=cv2.imread(str(root/'faithful_content.png'))
    if faithful is None:
        records.append({'run':row['public_id'],'source':row['source_name'],'error':'missing faithful image'})
        continue
    reference=None
    for filename in ('faithful_reference.png','registered_color.png','rectified_input.png'):
        if not (root/filename).is_file():
            continue
        reference=cv2.imread(str(root/filename))
        if reference is not None:
            break
    if reference is None:
        reference=original
    aligned_aspect=abs(reference.shape[1]/reference.shape[0]-faithful.shape[1]/faithful.shape[0])<.01
    reference=cv2.resize(reference,(faithful.shape[1],faithful.shape[0]),interpolation=cv2.INTER_AREA)
    src=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    rec=cv2.cvtColor(faithful,cv2.COLOR_BGR2GRAY)
    sd=np.maximum(cv2.GaussianBlur(src,(0,0),7).astype(float)-src,0)
    rd=np.maximum(cv2.GaussianBlur(rec,(0,0),7).astype(float)-rec,0)
    ink=sd>=6
    paper=cv2.dilate((sd>=3).astype(np.uint8),np.ones((9,9),np.uint8))==0
    retention=float(np.mean(rd[ink]>=np.minimum(sd[ink]*.35,4))) if ink.any() else None
    shade=float(np.mean(rec[paper]<225)) if paper.any() else None
    source_chroma=np.ptp(reference.astype(np.int16),axis=2)
    output_chroma=np.ptp(faithful.astype(np.int16),axis=2)
    colored=source_chroma>12
    color_retention=float(np.mean(output_chroma[colored]>6)) if colored.any() else None
    record={'run':row['public_id'],'source':row['source_name'],'sha256':digest,
        'geometry_aspect_matches_reference':aligned_aspect,'measured_ink_retention_probe':retention,
        'paper_shading_fraction_probe':shade,'color_pixel_retention_probe':color_retention,
        'width':rec.shape[1],'height':rec.shape[0],
        'note':'heuristic triage only; paper/ink labels and geometric alignment are not ground truth'}
    records.append(record)
    h=500; width=max(160,round(rec.shape[1]*h/rec.shape[0])); width=min(width,480)
    pair=np.full((h+30,width*2,3),255,np.uint8)
    pair[30:,:width]=cv2.resize(reference,(width,h))
    pair[30:,width:]=cv2.resize(faithful,(width,h))
    cv2.putText(pair,str(len(records))+' '+row['source_name'][:36],(5,20),0,.5,(0,0,0),1)
    pairs.append((record,pair))
records.sort(key=lambda r:(r.get('paper_shading_fraction_probe') or 0),reverse=True)
for start in range(0,len(pairs),6):
    group=pairs[start:start+6]; width=max(p.shape[1] for _,p in group)
    sheet=np.full((530*len(group),width,3),235,np.uint8)
    for i,(_,pair) in enumerate(group):
        sheet[i*530:(i+1)*530,:pair.shape[1]]=pair
    cv2.imwrite(str(folder/f'page_{start//6+1:02}.jpg'),sheet)
summary={'unique_saved_sources':len(seen),'evaluated':len(pairs),
    'missing_faithful':sum('error' in r for r in records),
    'geometry_aspect_flags':sum(not r.get('geometry_aspect_matches_reference',True) for r in records),
    'ink_retention_flags':sum(r.get('measured_ink_retention_probe') is not None and r['measured_ink_retention_probe']<.97 for r in records),
    'background_shading_flags':sum(r.get('paper_shading_fraction_probe') is not None and r['paper_shading_fraction_probe']>.05 for r in records),
    'scope':'saved natural images; heuristic triage, not release certification'}
(folder/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
