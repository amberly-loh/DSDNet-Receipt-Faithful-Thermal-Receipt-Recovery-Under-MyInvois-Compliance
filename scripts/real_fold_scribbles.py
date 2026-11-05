"""Sparse visually marked fold strokes; remaining pixels are unlabelled.

Coordinates refer to the original image cells extracted from Faithful.docx.
Text negatives are weak local-contrast candidates inside the indicated text
rectangles, not human-verified character masks. Never treat these as gold data.
"""
from pathlib import Path
import json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/real_fold_scribbles';OUT.mkdir(exist_ok=True)
examples={
 1: {'split':'train','folds':[[(24,188),(62,173),(101,165)],[(235,86),(258,71),(277,50)]], 'text':[(20,201,292,465)]},
 3: {'split':'train','folds':[[(25,244),(80,235),(140,225),(200,223),(285,229)],[(35,489),(95,478),(145,459),(192,464)]], 'text':[(23,10,293,135),(25,333,285,435)]},
 9: {'split':'train','folds':[[(305,287),(330,287),(355,286)],[(305,149),(330,151),(358,151)]], 'text':[(45,105,320,218),(45,462,300,571)]},
 2: {'split':'validation','folds':[[(145,242),(195,263),(244,303),(286,338)],[(209,511),(249,531),(294,539),(330,555)]], 'text':[(32,20,400,192),(31,294,400,400)]},
 7: {'split':'validation','folds':[[(230,403),(300,411),(370,408),(439,414),(479,429)]], 'text':[(30,25,632,205),(33,545,632,605)]},
}
manifest=[]
for row,item in examples.items():
    path=ROOT/f'artifacts/faithful_docx_review/row_{row:02d}_original_00.png'
    gray=cv2.imread(str(path),0)
    if gray is None: raise ValueError(f'Missing image {path}')
    fold=np.zeros_like(gray)
    for line in item['folds']:cv2.polylines(fold,[np.array(line,np.int32)],False,255,3,cv2.LINE_8)
    region=np.zeros_like(gray)
    for x1,y1,x2,y2 in item['text']:cv2.rectangle(region,(x1,y1),(x2,y2),255,-1)
    detail=cv2.GaussianBlur(gray,(0,0),3).astype(np.int16)-gray.astype(np.int16)
    ink=(region>0)&(detail>=12)&(cv2.dilate(fold,np.ones((17,17),np.uint8))==0)
    labels=np.full_like(gray,255);labels[ink]=0;labels[fold>0]=1
    folder=OUT/f'row_{row:02d}';folder.mkdir(exist_ok=True)
    cv2.imwrite(str(folder/'image.png'),gray);cv2.imwrite(str(folder/'labels.png'),labels)
    overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);overlay[labels==0]=(0,180,0);overlay[labels==1]=(0,0,255);cv2.imwrite(str(folder/'overlay.png'),overlay)
    manifest.append({'row':row,'split':item['split'],'image':str((folder/'image.png').resolve()),'labels':str((folder/'labels.png').resolve()),'fold_pixels':int((labels==1).sum()),'weak_ink_pixels':int((labels==0).sum()),'notes':'manual fold centerline; weak automatically selected ink negatives; unlabeled=255'})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
