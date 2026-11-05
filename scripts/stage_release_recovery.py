"""Rebuild all saved source formats; separate staging and reviewed publication."""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_faithful_content_view
from dsdnet_receipt.ocr import PaddleOCRAdapter
cv2.setNumThreads(1)
parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');parser.add_argument('--incremental',action='store_true');args=parser.parse_args()
folder=ROOT/'artifacts/release_candidates';folder.mkdir(exist_ok=True)
config,_=load_config(ROOT/'configs/default.json')
db=sqlite3.connect(ROOT/'artifacts/app_data/receipts.sqlite3');db.row_factory=sqlite3.Row
rows=db.execute('SELECT * FROM receipts ORDER BY id DESC').fetchall()
if not args.publish:
    seen=set();records=[];panels=[]
    ocr=PaddleOCRAdapter(config)
    for row in rows:
        raw=cv2.imread(row['input_path'])
        if raw is None:continue
        digest=hashlib.sha256(raw.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        audit=json.loads(row['audit_json'] or '{}')
        # Already reviewed source-bounded learned results remain intact.
        if audit.get('native_refinement',{}).get('applied'):continue
        source,geometry=preprocess(raw,config,preserve_color=True)
        gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
        old_reference=cv2.imread(str(folder/(digest+'_reference.png'))) if args.incremental else None
        same_pixels=old_reference is not None and np.array_equal(old_reference,source)
        candidate=cv2.imread(str(folder/(digest+'.png')),cv2.IMREAD_UNCHANGED) if same_pixels else None
        if candidate is None:
            recognised=ocr.run(gray)
            candidate=make_faithful_content_view(gray,gray,recognised.lines,source_color=source)
        cv2.imwrite(str(folder/(digest+'.png')),candidate)
        cv2.imwrite(str(folder/(digest+'_reference.png')),source)
        record={'run':row['public_id'],'source':row['source_name'],'digest':digest,'geometry':geometry,
                'renderer_sha256':hashlib.sha256((ROOT/'src/dsdnet_receipt/restoration.py').read_bytes()).hexdigest()}
        records.append(record)
        if candidate.ndim==2:candidate=cv2.cvtColor(candidate,cv2.COLOR_GRAY2BGR)
        w=max(160,min(420,round(candidate.shape[1]*450/candidate.shape[0])))
        panel=np.full((480,w*2,3),255,np.uint8)
        panel[30:,:w]=cv2.resize(raw,(w,450));panel[30:,w:]=cv2.resize(candidate,(w,450))
        cv2.putText(panel,str(len(records))+' '+row['source_name'][:35],(5,20),0,.5,(0,0,0),1)
        panels.append(panel)
        print(len(records),row['source_name'],flush=True)
    for start in range(0,len(panels),5):
        group=panels[start:start+5];width=max(p.shape[1] for p in group)
        page=np.full((480*len(group),width,3),235,np.uint8)
        for i,panel in enumerate(group):page[i*480:(i+1)*480,:panel.shape[1]]=panel
        cv2.imwrite(str(folder/f'page_{start//5+1:02}.jpg'),page)
    (folder/'staged.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print('Staged',len(records),'unique sources')
else:
    staged=json.loads((folder/'staged.json').read_text())
    renderer_digest=hashlib.sha256((ROOT/'src/dsdnet_receipt/restoration.py').read_bytes()).hexdigest()
    # Validate the complete generation before replacing any saved result.
    for record in staged:
        if record.get('renderer_sha256')!=renderer_digest:
            raise RuntimeError('Staged renderer is stale; rebuild before publication')
        candidate=cv2.imread(str(folder/(record['digest']+'.png')),cv2.IMREAD_UNCHANGED)
        reference=cv2.imread(str(folder/(record['digest']+'_reference.png')))
        if candidate is None or reference is None or candidate.shape[:2]!=reference.shape[:2]:
            raise RuntimeError('Missing or misaligned staged output: '+record['source'])
    backup_path=folder/'database_before_release.sqlite3'
    if not backup_path.exists():
        with sqlite3.connect(backup_path) as backup_db:db.backup(backup_db)
    by_digest={r['digest']:r for r in staged};updated=[]
    for row in rows:
        raw=cv2.imread(row['input_path'])
        if raw is None:continue
        digest=hashlib.sha256(raw.tobytes()).hexdigest()
        if digest not in by_digest:continue
        root=Path(row['restored_path']).parent
        backup=root/'before_release_content_fix';backup.mkdir(exist_ok=True)
        for name in ('faithful_content.png','faithful_reference.png','audit.json'):
            if (root/name).exists() and not (backup/name).exists():shutil.copy2(root/name,backup/name)
        if not (backup/'database_audit.json').exists():(backup/'database_audit.json').write_text(row['audit_json'] or '{}',encoding='utf-8')
        shutil.copy2(folder/(digest+'.png'),root/'faithful_content.pending.png')
        (root/'faithful_content.pending.png').replace(root/'faithful_content.png')
        shutil.copy2(folder/(digest+'_reference.png'),root/'faithful_reference.png')
        audit=json.loads(row['audit_json'] or '{}')
        audit['faithful_preprocessing']=by_digest[digest]['geometry']
        audit['release_content_fix']={'method':'content-retaining geometry and source-luminance normalization','model_pixels_used':False,'original_preserved':True}
        for key in ('faithful_content','faithful_reference'):audit.setdefault('stored_file_sha256',{})[key]=hashlib.sha256((root/(key+'.png')).read_bytes()).hexdigest()
        (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        db.execute('UPDATE receipts SET audit_json=? WHERE public_id=?',(json.dumps(audit),row['public_id']));db.commit()
        updated.append(row['public_id'])
    (folder/'published.json').write_text(json.dumps(updated,indent=2));print('Updated',len(updated),'saved results')
