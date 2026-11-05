"""Publish the five visually reviewed source-bounded candidates with backups."""
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsdnet_receipt.config import load_config
from dsdnet_receipt.preprocessing import preprocess

config,_ = load_config(ROOT/'configs/default.json')
folder = ROOT/'artifacts/small_print_revision'
rows = json.loads((folder/'staged.json').read_text())
db = sqlite3.connect(ROOT/'artifacts/app_data/receipts.sqlite3')
db.row_factory=sqlite3.Row
checkpoint = Path(config['native_refinement']['checkpoint'])
model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
published=[]
for item in rows:
    row=db.execute('SELECT * FROM receipts WHERE public_id=?',(item['run'],)).fetchone()
    raw=cv2.imread(row['input_path'])
    digest=hashlib.sha256(raw.tobytes()).hexdigest()
    base=Path(item['candidate'])
    if digest != base.stem:
        raise ValueError('Source pixels differ from reviewed candidate')
    candidate=base.with_name(base.stem+'_learned.png')
    pixels=cv2.imread(str(candidate),0)
    if pixels is None:
        raise ValueError('Missing reviewed candidate')
    registered,geometry=preprocess(raw,config,preserve_color=True)
    if pixels.shape != (registered.shape[0]*2,registered.shape[1]*2):
        raise ValueError('Learned view must share the registered source geometry at 2x')
    root=Path(row['restored_path']).parent
    backup=root/'before_supported_learned_edges'
    backup.mkdir(exist_ok=True)
    for name in ('faithful_content.png','faithful_reference.png','audit.json'):
        if (root/name).exists() and not (backup/name).exists():
            shutil.copy2(root/name,backup/name)
    if not (backup/'database_audit.json').exists():
        (backup/'database_audit.json').write_text(row['audit_json'] or '{}',encoding='utf-8')
    shutil.copy2(candidate,root/'faithful_content.pending.png')
    (root/'faithful_content.pending.png').replace(root/'faithful_content.png')
    cv2.imwrite(str(root/'faithful_reference.png'),cv2.resize(registered,(pixels.shape[1],pixels.shape[0]),interpolation=cv2.INTER_CUBIC))
    audit=json.loads(row['audit_json'] or '{}')
    audit['native_refinement']={'applied':True,'checkpoint':str(checkpoint),'checkpoint_sha256':model_hash,
        'method':'source-bounded learned edge refinement','model_output_scale':2,
        'background_policy':'pixels >=235 unchanged from interpolated source-based view',
        'change_bounds':{'maximum_darkening':45,'maximum_lightening':15},'synthetic_glyph_rendering':False}
    audit['faithful_preprocessing']=geometry
    for key in ('faithful_content','faithful_reference'):
        audit.setdefault('stored_file_sha256',{})[key]=hashlib.sha256((root/(key+'.png')).read_bytes()).hexdigest()
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    db.execute('UPDATE receipts SET audit_json=? WHERE public_id=?',(json.dumps(audit),row['public_id']))
    db.commit()
    published.append(item['run'])
(folder/'learned_published.json').write_text(json.dumps(published,indent=2))
print(f'Published {len(published)} saved receipts')
